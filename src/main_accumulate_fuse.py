# -*- coding:utf8 -*-
# try:
#     from torchtext.legacy.data import Iterator, BucketIterator
#     from torchtext.legacy import data
# except ImportError:
#     from torchtext.data import Iterator, BucketIterator
#     from torchtext import data
from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,SGD
import os
import sys
import wandb
import random
import numpy as np
import json
import os
import argparse
from bilevel_tools.meta import MetaSGD
from bilevel_tools.tbtools import AverageMeter
import torch.nn.functional as F
import bilevel_tools.loss_utils as loss_utils
import math
import time
import datetime
import matplotlib.pyplot as plt
import logging
import jsonlines

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from utils.basic_utils import *
from utils.constant import MODEL_PATH, SELF_WEIGHT_ADJUST_EPOCH, FEW_SHOT_SAMPLE_TEMPLATE, FEW_SHOT_PROMPT
from utils.reweight_train import *
from utils.sample_selection import *
from utils.bert_dataset import *
from utils.weight_adjust import weight_decay, model_importance_estimation
from utils.kd_functions import kd_label, kd_label_iter, kd_label_dataset, kd_label_entropy, kd_label_entropy_aware, kd_label_aware
from utils.mlp import *
from utils.influence_utils import run_full_influence_functions
from llm_query import gen_evaluation, gen_syn_data_few_shot
from utils.models import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_SIZE = 300  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
DROPOUT_RATE = 0.5
LAYER_NUM = 1
EMBEDDING_SIZE = 100
vectors = None
freeze = False

# BETA = 1.5

SYN_DATA_PATH = 'data_new/'
# SYN_DATA_PATH = 'data_new_few_shot_ambiguous/'
# SYN_DATA_PATH = 'data_new_few_shot_easytolearn/'
# SYN_DATA_PATH = 'data_new_[10]/'
CUDA_LAUNCH_BLOCKING=1


def set_seed(seed = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construct_outer_subloader(args, train_data, indices = None, idx_to_order=None):
    if indices is None:
        num_use_samples_inner=len(train_data.examples)
        indices = np.random.choice(list(range(num_use_samples_inner)), args.num_use_samples_outer, replace=False)
    else:
        indices = [idx_to_order[idx] for idx in indices]
    if args.small_model_name.upper() == 'LSTM':
        dev_data = data.Dataset([train_data.examples[ix] for ix in indices], train_data.fields)
        subset_iter= BucketIterator.splits(
            (dev_data,),
            batch_sizes=(args.backward_batch_size,),
            device='cuda',
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=True,
        )
    elif 'bert' in args.small_model_name.lower():
        dev_data = TokenizedDataset(
            file_path=(''),
        )
        dev_data.text = [] # clear all the samples
        dev_data.ids = [] # clear all the samples
        dev_data.attention_mask = [] # clear all the samples
        dev_data.label = [] # clear all the samples
        dev_data.idx = [] # clear all the samples
        for i in range(args.len_LLM):
            dev_data.text += [train_data.text[ix] for ix in indices]
            dev_data.ids += [train_data.ids[ix] for ix in indices]
            dev_data.attention_mask += [train_data.attention_mask[ix] for ix in indices]
            dev_data.label += [train_data.label[ix] for ix in indices]
            dev_data.idx += [train_data.idx[ix] for ix in indices]
        # dev_data.ids = torch.stack(dev_data.ids).squeeze().to(args.device)
        # dev_data.attention_mask = torch.stack(dev_data.attention_mask).squeeze().to(args.device)
        # dev_data.label = torch.tensor(dev_data.label).long().to(args.device)
        # dev_data.idx = torch.tensor(dev_data.idx).long().to(args.device)
        dev_data.ids = torch.stack(dev_data.ids).squeeze()
        dev_data.attention_mask = torch.stack(dev_data.attention_mask).squeeze()
        dev_data.label = torch.tensor(dev_data.label).long()
        dev_data.idx = torch.tensor(dev_data.idx).long()
        subset_iter = [DataLoader(dev_data, batch_size=args.backward_batch_size, shuffle=True)]
    return subset_iter[0]


def file_choose(num_samples):
    bins = [0,1000,10000]
    file_samples = -1
    for j in range(1,len(bins)):
        if bins[j-1] < num_samples <= bins[j]:
            file_samples = bins[j]
    assert file_samples > 0, "too many samples, haven't generated enough"
    print(f"require #{num_samples}, use file under #{file_samples}")
    return file_samples 


def load_iters(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    if args.small_model_name.upper() == "LSTM":
        return load_iters_lstm(args, batch_size, backward_batch_size, device, gold_data_path, syn_data_path, vectors, use_tree, num_use_samples_inner, num_use_samples_outer, shuffle_train)
    elif 'bert' in args.small_model_name.lower():
        return load_iters_bert(args, batch_size, backward_batch_size, device, gold_data_path, syn_data_path, vectors, use_tree, num_use_samples_inner, num_use_samples_outer, shuffle_train)

def load_iters_lstm(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True, unk_token='<unk>')
    LABEL = data.LabelField(batch_first=True, use_vocab=False) # , use_vocab=True
    INDEX = data.RawField()
    fields = {'C': ('text', TEXT),
              'Y': ('label', LABEL),
              'idx': ('idx', INDEX)}

    # if args.query_input_file == None:
    #     args.query_input_file = []
    #     for i in range(args.len_LLM):
    #         args.query_input_file.append((f'{SYN_DATA_PATH}{args.task_name}/mix/{args.llms[i]}/{file_choose(args.separate_num_use_samples_inner[i])}/train.jsonl') if args.mix else (f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/train.jsonl'))
    #         print(f"args.query_input_file[-1]={args.query_input_file[-1]}")
 
    train_data_list = []
    small_train_data_list = []
    small_valid_data_list = []
    all_data_examples = []
    for i in range(args.len_LLM):
        if args.steps == 0:
            train_data_path = (f'{SYN_DATA_PATH}{args.task_name}/mix/{args.llms[i]}/{file_choose(args.separate_num_use_samples_inner[i])}/') if args.mix else (f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/')
        else:
            assert args.mix == False, "Setting error, --mix should be False with --steps > 0, but now --mix is True"
            train_data_path = f'{SYN_DATA_PATH}{args.gen_sample_select}/{args.model_name_sample}/{args.small_model_name}/{args.fuse_dataset_sample_selection}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.seed}/{args.llms[i]}/{args.num_use_samples_inner[i]}_{args.num_use_samples_init[i]}_{args.num_use_samples_each_step_extend[i]}/'
        train_data, _ = data.TabularDataset.splits(
            path=train_data_path,
            train='train.jsonl',
            test='train.jsonl',
            # train='test.jsonl',
            # test='test.jsonl',
            format='json',
            fields=fields,
            filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
        )
        traindataset = train_data.examples[:args.sample_each_llm[i]]
        for _i, data_item in enumerate(traindataset):
            data_item.idx = _i
        # for data_item in traindataset:
        #     print("in construct, total data", data_item.idx, data_item.text, data_item.label)
        # small_traindataset, small_validationdataset = copy.deepcopy(traindataset[:int(args.sample_each_llm[i]*args.train_ratio)]), copy.deepcopy(traindataset[int(args.sample_each_llm[i]*args.train_ratio):])
        all_data_examples += traindataset
        shuffled_traindataset = copy.deepcopy(traindataset)
        random.shuffle(shuffled_traindataset)
        # train-valid split
        small_traindataset, small_validationdataset = copy.deepcopy(shuffled_traindataset[:int(args.sample_each_llm[i]*args.train_ratio)]), copy.deepcopy(shuffled_traindataset[int(args.sample_each_llm[i]*args.train_ratio):])
        small_traindataset_idx = []
        for _i, data_item in enumerate(small_traindataset):
            small_traindataset_idx.append(data_item.idx)
            data_item.idx = _i
        # for data_item in small_traindataset:
        #     print("in construct, small data", data_item.idx, data_item.text, data_item.label)
        # random.shuffle(small_traindataset)
        for _i, data_item in enumerate(small_validationdataset):
            data_item.idx = _i
        # ############## construct all data and separate as train and test ##############
        train_data = data.Dataset(traindataset, train_data.fields)
        train_data_list.append(train_data)
        small_train_data = data.Dataset(small_traindataset, train_data.fields)
        small_train_data_list.append(small_train_data)
        small_valid_data = data.Dataset(small_validationdataset, train_data.fields)
        small_valid_data_list.append(small_valid_data)
        # ############## construct all data and separate as train and test ##############

        # save original text of small train dataset
        args.samples_text[i] = []
        total_sample_text = []
        with jsonlines.open(f'{train_data_path}train.jsonl', 'r') as reader:
            for json_obj in reader:
                total_sample_text.append(json_obj['C'])
        for _idx in small_traindataset_idx:
            args.samples_text[i].append(total_sample_text[_idx])
        print(f"[debug] sample_text has length {len(args.samples_text[i])}")

    fields_dev = {'text': ('text', TEXT),
                  'label': ('label', LABEL),
                  'idx': ('idx', INDEX)}
    dev_data, test_data = data.TabularDataset.splits(
        path=gold_data_path,
        validation='train.jsonl',
        test='test.jsonl',
        # test='test_small.jsonl',
        format='json',
        fields=fields_dev,
        # fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )

    print(f"[debug] args.use_dev_outer={args.use_dev_outer}, args.subset_outer={args.subset_outer}")
    if args.use_dev_outer:
        dev_data_all = data.Dataset(dev_data.examples, dev_data.fields)
        dev_data = data.Dataset(dev_data.examples[:num_use_samples_outer], dev_data.fields)
    else:
        if args.subset_outer: # currently use this one
            indices = np.random.choice(list(range(args.sample_each_llm[-1])), int(num_use_samples_outer//args.len_LLM), replace=False)
            print(f"[debug] len(train_data.examples)={len(train_data.examples)}")
            data_sample_list = []
            for i in range(args.len_LLM):
                data_sample_list = data_sample_list + [train_data_list[i].examples[ix] for ix in indices]
            dev_data = data.Dataset(data_sample_list, train_data.fields)
        else:
            dev_data=train_data
        dev_data_all=train_data

    print(f'[debug] len(all_data_examples) for train data is {len(all_data_examples)}')
    all_data_examples = all_data_examples + dev_data.examples + test_data.examples
    print(f'[debug] len(all_data_examples) for all data is {len(all_data_examples)}')
    all_data = data.Dataset(all_data_examples, train_data.fields)
    if vectors is not None:
        TEXT.build_vocab(all_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(all_data, max_size=500000)
    # print(f"[debug] see TEXT after build_vocab {TEXT}")
    LABEL.build_vocab(all_data)
    print(f"[debug] see LABEL after build_vocab {LABEL}")

    concat_of_data = train_data_list + train_data_list + small_train_data_list + small_valid_data_list + [dev_data]
    concat_of_data = tuple(concat_of_data)
    concat_of_batch_size = [batch_size]*args.len_LLM + [backward_batch_size]*args.len_LLM + [batch_size]*args.len_LLM + [batch_size]*args.len_LLM + [batch_size]
    concat_of_batch_size = tuple(concat_of_batch_size)

    iters = BucketIterator.splits(
        concat_of_data,
        batch_sizes=concat_of_batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=shuffle_train,
    )
    iters = list(iters)
    train_iter_list = iters[0:args.len_LLM]
    train_iter_backward_list = iters[args.len_LLM:2*args.len_LLM]
    small_train_iter_list = iters[2*args.len_LLM:3*args.len_LLM]
    small_valid_iter_list = iters[3*args.len_LLM:4*args.len_LLM]
    dev_iter = iters[-1]

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)
    
    print(f'[debug] before exiting load iter: len(train_iter_list)={len(train_iter_list)}, len(train_data_list)={len(train_data_list)}')
    return train_iter_list, small_train_iter_list, small_valid_iter_list, train_iter_backward_list, dev_iter, test_iter, TEXT, LABEL, train_data_list, small_train_data_list, small_valid_data_list, dev_data_all


def load_iters_bert(args, batch_size=32, backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    init_token = args.tokenizer.cls_token # [CLS]   
    eos_token = args.tokenizer.sep_token # [SEP]
    pad_token = args.tokenizer.pad_token # [PAD]
    unk_token = args.tokenizer.unk_token # [UNK]
    init_token_idx = args.tokenizer.convert_tokens_to_ids(init_token) # 101
    eos_token_idx = args.tokenizer.convert_tokens_to_ids(eos_token) # 102
    pad_token_idx = args.tokenizer.convert_tokens_to_ids(pad_token) # 0
    unk_token_idx = args.tokenizer.convert_tokens_to_ids(unk_token) # 100


    train_data_list = []
    small_train_data_list = []
    small_valid_data_list = []
    all_data_examples = []
    for i in range(args.len_LLM):
        print("train_dataset for", args.llms[i])
        # print("train_dataset for", args.llms[i], "file for data is", (file_choose(args.separate_num_use_samples_inner[i] if args.mix else file_choose(args.num_use_samples_inner[i]))))
        if args.steps == 0:
            train_data_path = (f'{SYN_DATA_PATH}{args.task_name}/mix/{args.llms[i]}/{file_choose(args.separate_num_use_samples_inner[i])}/train.jsonl') if args.mix else (f'{SYN_DATA_PATH}{args.task_name}/{args.llms[i]}/{file_choose(args.num_use_samples_inner[i])}/train.jsonl')
        else:
            assert args.mix == False, "Setting error, --mix should be False with --steps > 0, but now --mix is True"
            train_data_path = f'{SYN_DATA_PATH}{args.gen_sample_select}/{args.model_name_sample}/{args.small_model_name}/{args.fuse_dataset_sample_selection}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.seed}/{args.llms[i]}/{args.num_use_samples_inner[i]}_{args.num_use_samples_init[i]}_{args.num_use_samples_each_step_extend[i]}/train.jsonl'
        train_data = TokenizedDataset(
            file_path=train_data_path,
            text_column='C',
            label_column='Y',
            index_column='idx',
            tokenizer=args.tokenizer,
            max_length=args.max_input_length,
            # device=args.device,
            max_sample=args.sample_each_llm[i],
            small_dataset_shuffle=False,
        )
        # train_data.idx = torch.tensor([_i for _i in range(args.sample_each_llm[i])]).long().to(args.device)
        train_data.idx = torch.tensor([_i for _i in range(args.sample_each_llm[i])]).long()
        
        indices = list(range(args.sample_each_llm[i]))
        random.shuffle(indices)
        train_valid_pivot_point = int(args.sample_each_llm[i]*args.train_ratio)
        
        # ############## separate as train and test ##############
        # train-valid split
        small_train_data = TokenizedDataset(
            file_path=(''),
        )
        small_train_data.text = [copy.deepcopy(train_data.text[ix]) for ix in indices[:train_valid_pivot_point]]
        small_train_data.ids = copy.deepcopy(train_data.ids[indices[:train_valid_pivot_point]])
        small_train_data.attention_mask = copy.deepcopy(train_data.attention_mask[indices[:train_valid_pivot_point]])
        small_train_data.label = copy.deepcopy(train_data.label[indices[:train_valid_pivot_point]])
        # small_train_data.idx = torch.tensor([_i for _i in range(train_valid_pivot_point)]).long().to(args.device)
        small_train_data.idx = torch.tensor([_i for _i in range(train_valid_pivot_point)]).long()
        
        small_valid_data = TokenizedDataset(
            file_path=(''),
        )
        small_valid_data.text = [copy.deepcopy(train_data.text[ix]) for ix in indices[train_valid_pivot_point:]]
        small_valid_data.ids = copy.deepcopy(train_data.ids[indices[train_valid_pivot_point:]])
        small_valid_data.attention_mask = copy.deepcopy(train_data.attention_mask[indices[train_valid_pivot_point:]])
        small_valid_data.label = copy.deepcopy(train_data.label[indices[train_valid_pivot_point:]])
        # small_valid_data.idx = torch.tensor([_i for _i in range(args.sample_each_llm[i]-train_valid_pivot_point)]).long().to(args.device)
        small_valid_data.idx = torch.tensor([_i for _i in range(args.sample_each_llm[i]-train_valid_pivot_point)]).long()

        train_data_list.append(train_data)
        small_train_data_list.append(small_train_data)
        small_valid_data_list.append(small_valid_data)
        # ############## separate as train and test ##############
        
        # save original text of small train dataset
        args.samples_text[i] = [copy.deepcopy(text) for text in small_train_data.text]
        print(f"[debug] sample_text has length {len(args.samples_text[i])}")

    print("test dataset")
    test_data = TokenizedDataset(
        file_path=(gold_data_path+'test.jsonl'),
        # file_path=(gold_data_path+'test_small.jsonl'),
        text_column='text',
        label_column='label',
        index_column='idx',
        tokenizer=args.tokenizer,
        device=args.device,
        max_length=args.max_input_length,
        # max_sample=100 # use all that is provided in the dataset file
        max_sample=-1 # use all that is provided in the dataset file
    )

    print(f"[debug] args.use_dev_outer={args.use_dev_outer}, args.subset_outer={args.subset_outer}")
    if args.use_dev_outer:
        print("dev dataset")
        dev_data_all = TokenizedDataset(
            file_path=(gold_data_path+'train.jsonl'),
            text_column='text',
            label_column='label',
            index_column='idx',
            tokenizer=args.tokenizer,
            device=args.device,
            max_length=args.max_input_length,
            small_dataset_shuffle=False,
        )
        dev_data = TokenizedDataset(
            file_path=(gold_data_path+'train.jsonl'),
            text_column='text',
            label_column='label',
            index_column='idx',
            tokenizer=args.tokenizer,
            max_length=args.max_input_length,
            device=args.device,
            max_sample=num_use_samples_outer,
            small_dataset_shuffle=False,
        )
    else:
        if args.subset_outer: # currently use this one
            indices = np.random.choice(list(range(args.sample_each_llm[-1])), int(num_use_samples_outer//args.len_LLM), replace=False)
            print(f"[debug] len(train_data.ids)={len(train_data.ids)}")
            data_sample_list = []
            dev_data = TokenizedDataset(
                file_path=(''),
                # text_column='text',
                # label_column='label',
                # index_column='idx',
                # tokenizer=args.tokenizer,
                # max_length=args.max_input_length,
                # device=args.device,
                # max_sample=1
            )
            dev_data.text = [] # clear all the samples
            dev_data.ids = [] # clear all the samples
            dev_data.attention_mask = [] # clear all the samples
            dev_data.label = [] # clear all the samples
            dev_data.idx = [] # clear all the samples
            for i in range(args.len_LLM):
                print(f"[debug] {len(train_data_list[i])=}")
                dev_data.text += [train_data_list[i].text[ix] for ix in indices]
                dev_data.ids += [train_data_list[i].ids[ix] for ix in indices]
                dev_data.attention_mask += [train_data_list[i].attention_mask[ix] for ix in indices]
                dev_data.label += [train_data_list[i].label[ix] for ix in indices]
                dev_data.idx += [train_data_list[i].idx[ix] for ix in indices]
        else:
            dev_data=train_data
        dev_data_all=train_data

    train_iter_list = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train) for dataset in train_data_list]
    small_train_iter_list = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train) for dataset in small_train_data_list]
    small_valid_iter_list = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_train) for dataset in small_valid_data_list]
    train_iter_backward_list = [DataLoader(dataset, batch_size=backward_batch_size, shuffle=shuffle_train) for dataset in train_data_list]
    dev_iter = DataLoader(dev_data, batch_size=batch_size, shuffle=shuffle_train)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for im, small_train_data in enumerate(train_data_list):
        # if im != len(small_train_data_list)-1:
        if im != 0:
            continue
        train_iter = DataLoader(small_train_data, batch_size=batch_size, shuffle=shuffle_train)
        for batch in train_iter:
            inputs, attention_mask, labels, idx = batch
            print(labels, idx)
            for _idx, _label in zip(idx, labels):
                print(f"model#{im}: idx={_idx}, label={small_train_data.label[_idx]}, text={small_train_data.text[_idx]}")
                # print(f"batch: idx={_idx}, batch_used_text={_text}")
            break

    print(f'[debug] before exiting load iter: len(train_iter_list)={len(train_iter_list)}, len(train_data_list)={len(train_data_list)}')
    return train_iter_list, small_train_iter_list, small_valid_iter_list, train_iter_backward_list, dev_iter, test_iter, train_data_list, small_train_data_list, small_valid_data_list, dev_data_all


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(args, model, data_iter, name, epoch=None, use_soft_label=False):
    model.to(args.device)
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    all_labels = []
    if args.small_model_name.upper() == 'LSTM':
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                (inputs, lens), labels = batch.text, batch.label
                output = model(inputs, lens)
                all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                loss = loss_func(output, labels)
                total_loss += loss.item()
                correct_num += (predicts == labels).sum().item()
                err_num += (predicts != labels).sum().item()
    elif 'bert' in args.small_model_name.lower():
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)

                if use_soft_label == True:
                    target = copy.deepcopy(torch.unsqueeze(labels, 1))
                    eval_labels = torch.zeros(target.size(0), args.num_classes, device=args.device)
                    eval_labels.scatter_(1, target, 1)
                else:
                    eval_labels = labels
                # print(f"[debug] in eval() of bert, {inputs.shape=}, {attention_mask.shape=}, {labels.shape=}, {idx.shape=}")
                # print(f"[debug] in eval() of bert, {eval_labels.shape=}")
                # print(f"{inputs[:3]=}, {attention_mask[:3]=}, {labels[:3]=}, {idx[:3]=}")

                # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                loss = loss_func(output, labels)
                total_loss += loss.item()
                correct_num += (predicts == labels).sum().item()
                err_num += (predicts != labels).sum().item()

    acc = correct_num / (correct_num + err_num)
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, acc, total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, acc, total_loss))
    all_labels = torch.cat(all_labels)
    model.to("cpu")
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    return acc, total_loss/len(data_iter)


def train(args, model, train_iter, dev_iter, loss_func, optimizer, epochs, patience=5, clip=5):
    best_model = copy.deepcopy(model)
    best_acc = -1
    patience_counter = 0

    if 'bert' in args.small_model_name.lower():
        if args.optim =='Adam':
            optimizer = Adam(model.parameters(), lr=args.inner_lr)
        elif args.optim =='SGD':
            optimizer = SGD(model.parameters(), lr=args.inner_lr, momentum=0.9)
        # # for param in model.base_model.parameters():
        # for param in model.bert.parameters():
        #     # print(param)
        #     param.requires_grad = False
        # if args.optim =='Adam':
        #     optimizer = Adam(model.classifier.parameters(), lr=args.inner_lr)
        # elif args.optim =='SGD':
        #     optimizer = SGD(model.classifier.parameters(), lr=args.inner_lr, momentum=0.9)
    model.to(args.device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        if args.small_model_name.upper() == 'LSTM':
            for batch in tqdm(train_iter):
                (inputs, lens), labels = batch.text, batch.label
                labels = batch.label

                model.zero_grad()
                output = model(inputs, lens)
                loss = loss_func(output, labels)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))
        elif 'bert' in args.small_model_name.lower():
            for batch in tqdm(train_iter):
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)
                # model_copy.zero_grad() # this is equal to optimizer.zero_grad() when optimizer contains only this model's whole parameters
                optimizer.zero_grad()
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

        acc = eval(args, model, dev_iter, "Dev", epoch)
        if args.wandb:
            wandb.log({"loss": total_loss, "val_acc": acc})

        if acc<best_acc:
            patience_counter +=1
        else:
            best_acc = acc
            patience_counter = 0
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'best_model.ckpt')
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break
    model.to("cpu")
    best_model.to("cpu")
    return best_model


def train_to_converge(args, model, train_data, theta, epoch_converge, inner_obj, test_loader, soft_label=None):
    model_copy = copy.deepcopy(model)
    if args.optim =='Adam':
        optimizer = Adam(model_copy.parameters(), lr=args.inner_lr)
    elif args.optim =='SGD':
        optimizer = SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    losses = AverageMeter("Loss", ":.3f")
    model_weights_cache = []
    opt_checkpoints_cache = []
    diverged = False
    if args.small_model_name.upper() == 'LSTM':
        # for data in train_data.examples:
        #     print(data.idx, data.text, data.label)
        train_iter, = BucketIterator.splits(
            (train_data,),
            batch_sizes=(args.train_batch_size,),
            device=device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        train_iter = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
        # print(f"{train_data[0]=}, {train_data[1]=}")
    # print(f"[debug] in train_to_converge theta.shape={theta.shape}, len(train_data)={len(train_data)}")
    
    model_copy.to(args.device)
    # print(f'a model on gpu, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    # print(f"{theta.shape=}, {type(theta)=}")
    for epoch in range(epoch_converge):
        model_copy.train()
        top1 = AverageMeter("OuterAcc@1", ":6.2f")
        for batch in tqdm(train_iter):
            if args.small_model_name.upper() == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                idx = batch.idx
                # print(f"{idx=}")
            elif 'bert' in args.small_model_name.lower():
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)

            # if soft_label != None:
            #     labels = soft_label[idx]
            #     assert inner_obj != 'kl', "KL divergence loss does not support soft label currently"
            
            # print(f"[debug] in train_to_converge each batch, len(inputs)={len(inputs)}, len(labels)={len(labels)}")
            # model_copy.zero_grad() # this is equal to optimizer.zero_grad() when optimizer contains only this model's whole parameters
            optimizer.zero_grad()
            
            # print(f'after puting a batch on gpu, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
            if args.small_model_name.upper() == 'LSTM':
                output = model_copy(inputs, lens)
            elif 'bert' in args.small_model_name.lower():
                output = model_copy(inputs, attention_mask=attention_mask, labels=labels).logits
            # print(f"[debug] in train_to_converge each batch, len(outputs)={len(output)}")
            # print(f"[debug] in train_to_converge each batch, idx={idx}")
            if inner_obj == "ce":
                if not args.normalize:
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                else:
                    loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
            elif inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss = torch.mean(loss_vec*theta[idx])
            
            if soft_label != None:
                temperatured_output = F.softmax(output/args.kd_temperature, dim=-1)
                temperatured_labels = F.softmax(soft_label[idx]/args.kd_temperature, dim=-1)
                if not args.normalize:
                    loss = args.kd_alpha * loss + (1-args.kd_alpha) * torch.mean(F.cross_entropy(temperatured_output, temperatured_labels, reduction='none').flatten()*theta[idx])
                else:
                    loss = args.kd_alpha * loss + (1-args.kd_alpha) * torch.sum(F.cross_entropy(temperatured_output, temperatured_labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
            
            if soft_label == None:
                # print(f"{labels=}, {output=}, {labels.shape=}, {output.shape=}")
                acc = loss_utils.accuracy(output, labels)
            else:
                acc = loss_utils.accuracy(output, torch.argmax(soft_label[idx],dim=-1))
            top1.update(acc, labels.size(0))
            losses.update(loss.item(), labels.size(0))
            loss.backward()
            optimizer.step()
            # print(f'after a batch train, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
        
    opt_checkpoints_cache.append(optimizer.state_dict())
    model_copy.to("cpu")
    model_weights_cache.append(copy.deepcopy(model_copy.state_dict()))
    if math.isnan(loss.item()):
        diverged = True
    return model_copy, losses.avg, top1.avg, model_weights_cache, opt_checkpoints_cache, diverged


def train_to_converge_fused(args, model, train_data, theta, selected_sample_indexs, epoch_converge, inner_obj):
    for _theta in theta:
        _theta = _theta.detach()
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]
    model_copy = copy.deepcopy(model)
    if args.optim =='Adam':
        optimizer = Adam(model_copy.parameters(), lr=args.inner_lr)
    elif args.optim =='SGD':
        optimizer = SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    losses = AverageMeter("Loss", ":.3f")
    model_weights_cache = []
    opt_checkpoints_cache = []
    diverged = False
    selected_train_dataset = []
    _id = 0
    if args.small_model_name.upper() == 'LSTM':
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
            selected_train_dataset[_id].idx = _id
            _id += 1
        selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
        # print(selected_train_data.fields)
        # selected_train_data = torch.tensor(train_data)[selected_sample_rows,selected_sample_columns]
        theta = torch.stack(theta)[selected_sample_rows,selected_sample_columns]
        train_iter, = BucketIterator.splits(
            (selected_train_data,),
            batch_sizes=(args.train_batch_size,),
            device=device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        selected_train_data = TokenizedDataset(
            file_path=(''),
        )
        selected_train_data.text = [] # clear all the samples
        selected_train_data.ids = [] # clear all the samples
        selected_train_data.attention_mask = [] # clear all the samples
        selected_train_data.label = [] # clear all the samples
        selected_train_data.idx = [] # clear all the samples
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_data.text += [train_data[row].text[column]]
            selected_train_data.ids += [train_data[row].ids[column]]
            selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
            selected_train_data.label += [train_data[row].label[column] ]
            selected_train_data.idx += [_id]
            _id += 1
        theta = torch.tensor([theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
        train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    model_copy.to(args.device)
    for epoch in range(epoch_converge):
        model_copy.train()
        top1 = AverageMeter("OuterAcc@1", ":6.2f")
        for batch in tqdm(train_iter):
            if args.small_model_name.upper() == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                labels = batch.label
                idx = batch.idx
                # model_copy.zero_grad() # this is equal to optimizer.zero_grad() when optimizer contains only this model's whole parameters
                optimizer.zero_grad()
                output = model_copy(inputs, lens)
            elif 'bert' in args.small_model_name.lower():
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)                
                # model_copy.zero_grad() # this is equal to optimizer.zero_grad() when optimizer contains only this model's whole parameters
                optimizer.zero_grad()
                output = model_copy(inputs, attention_mask=attention_mask, labels=labels).logits
            if inner_obj == "ce":
                if not args.normalize:
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                else:
                    loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
            elif inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss = torch.mean(loss_vec*theta[idx])
            acc = loss_utils.accuracy(output, labels)
            top1.update(acc, labels.size(0))
            losses.update(loss.item(), labels.size(0))
            loss.backward()
            optimizer.step()
    model_copy.to("cpu")
    opt_checkpoints_cache.append(optimizer.state_dict())
    model_weights_cache.append(copy.deepcopy(model_copy.state_dict()))
    if math.isnan(loss.item()):
        diverged = True
    return model_copy, losses.avg, top1.avg, model_weights_cache, opt_checkpoints_cache, diverged


def train_to_converge_with_weight_adjust_fused(args, model, train_data, theta, selected_sample_indexs, epoch_adjust, epoch_converge, inner_obj, test_loader, outer_iter, soft_label=None):
    # print(type(theta),theta[0])
    for _theta in theta:
        _theta = _theta.detach()
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]
    model_copy = copy.deepcopy(model)
    if args.optim =='Adam':
        optimizer = Adam(model_copy.parameters(), lr=args.inner_lr)
    elif args.optim =='SGD':
        optimizer = SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    losses = AverageMeter("Loss", ":.3f")
    model_weights_cache = []
    opt_checkpoints_cache = []
    diverged = False
    if selected_sample_rows == None or selected_sample_columns == None and len(train_data) > 1:
        selected_sample_rows, selected_sample_columns = [], []
        if args.small_model_name.upper() == 'LSTM':
            for row in range(args.len_LLM):
                for column in range(len(train_data[row].examples)):
                    selected_sample_rows.append(row)
                    selected_sample_columns.append(column)
        elif 'bert' in args.small_model_name.lower():
            for row in range(args.len_LLM):
                for column in range(len(train_data[row].idx)):
                    selected_sample_rows.append(row)
                    selected_sample_columns.append(column)
    if selected_sample_rows != None and selected_sample_columns != None:
        if args.small_model_name.upper() == 'LSTM':
            selected_train_dataset = []
            _id = 0
            for row, column in zip(selected_sample_rows,selected_sample_columns):
                selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
                selected_train_dataset[_id].idx = _id
                _id += 1
            selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
            theta = torch.stack(theta)[selected_sample_rows,selected_sample_columns]
            init_theta = copy.deepcopy(theta)
            # print(theta)
            # theta = torch.stack(theta)
            # print(type(theta), theta.shape)
        elif 'bert' in args.small_model_name.lower():
            _id = 0
            selected_train_data = TokenizedDataset(
                file_path=('')
            )
            selected_train_data.text = [] # clear all the samples
            selected_train_data.ids = [] # clear all the samples
            selected_train_data.attention_mask = [] # clear all the samples
            selected_train_data.label = [] # clear all the samples
            selected_train_data.idx = [] # clear all the samples
            for row, column in zip(selected_sample_rows,selected_sample_columns):
                selected_train_data.text += [train_data[row].text[column]]
                selected_train_data.ids += [train_data[row].ids[column]]
                selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
                selected_train_data.label += [train_data[row].label[column]]
                selected_train_data.idx += [_id]
                _id += 1
            theta = torch.tensor([theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
            init_theta = copy.deepcopy(theta)
    elif len(train_data) == 1:
        selected_train_data = train_data[0]
        init_theta = copy.deepcopy(theta)


    best_theta_each_iter = []
    print(f"training with fused data using weight adjust")
    for epoch in range(epoch_adjust):
        current_outer_iter_trained_model = []
        theta_mapped = copy.deepcopy(theta)
        # print(type(theta_mapped),theta_mapped)
        best_theta = copy.deepcopy(theta)
            
        ##############  ##############
        diverged = True # diverged==True means loss==nan, which means the training failed
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge, args.inner_obj, test_loader, soft_label=soft_label)
            print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
            if epoch_adjust % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge_fully_train, args.inner_obj, test_loader, soft_label=soft_label)
        # print(f"[debug] {args.stochastic_outer and args.subset_outer} {args.stochastic_outer}, {args.subset_outer}")
        # print(f"[debug] {args.use_dev_outer}")
        if args.stochastic_outer and args.subset_outer:
            if args.use_dev_outer:
                valid_loader = construct_outer_subloader(args, dev_data_all)
            else:
                valid_loader = construct_outer_subloader(args, selected_train_data) # currently using this one

        current_outer_iter_trained_model.append(model_copy_converged)
        
        if epoch_adjust % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test", use_soft_label=(soft_label != None))
            # test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, train_loader[i], name="test")
            print(f"weight-adjust-on-fused-dataset: #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
            logging.info(f"weight-adjust-on-fused-dataset: , #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
            if args.wandb:
                wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})

        if args.kd_slm == 2: # self-iteration kd, no weight adjust for each sample
            theta_mapped = [theta_mapped]
        else:
            theta_mapped, model_total_acc = weight_decay(args, current_outer_iter_trained_model, [selected_train_data], [theta_mapped], beta=args.BETA, _type=args.weight_adjust_criterial, single_dataset=True, use_soft_label=(soft_label!=None))
        theta = copy.deepcopy(theta_mapped[0])
        theta_score = copy.deepcopy(theta)
        # print(f"new theta[j] = {theta[j]}")
        best_theta=theta_score
        best_theta_each_iter.append(copy.deepcopy(best_theta))

        if args.kd_slm == 2: # self-iterate kd, use trained student model as teacher model
            # construct new soft_label
            print(f'[debug], original_soft_label {soft_label}')
            _, soft_label, _, _ = kd_label_dataset(args, [model_copy_converged_ft], selected_train_data, torch.tensor([1.0]))
            print(f'[debug], new_soft_label {soft_label}')
    if selected_sample_rows != None and selected_sample_columns != None:
        torch.save((selected_sample_rows, selected_sample_columns, init_theta, torch.stack(best_theta_each_iter)), f"{args.result_file_path}/sample_selected_self_adjust_{outer_iter}.pth")
    else:
        selected_sample_rows = torch.tensor([0 for _ in range(len(train_data[0]))], dtype=torch.long).to(theta[0].device)
        selected_sample_columns = torch.tensor([_i for _i in range(len(train_data[0]))], dtype=torch.long).to(theta[0].device)
        torch.save((selected_sample_rows, selected_sample_columns, init_theta, torch.stack(best_theta_each_iter)), f"{args.result_file_path}/all_sample_used_self_adjust_{outer_iter}.pth")
    return model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft


def train_to_converge_with_weight_adjust_and_selection_fused(args, model, train_data, theta, selected_sample_indexs, epoch_adjust, epoch_converge, inner_obj, test_loader, outer_iter):
    theta_for_function_call = [copy.deepcopy(_theta) for _theta in theta]
    for _theta in theta:
        _theta = _theta.detach()
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]
    model_copy = copy.deepcopy(model)
    if args.optim =='Adam':
        optimizer = Adam(model_copy.parameters(), lr=args.inner_lr)
    elif args.optim =='SGD':
        optimizer = SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    losses = AverageMeter("Loss", ":.3f")
    # model_weights_cache = []
    # opt_checkpoints_cache = []
    diverged = False
    selected_train_dataset = []
    _id = 0
    if args.small_model_name.upper() == 'LSTM':
        # if use all, put all the index together
        if selected_sample_rows == None or selected_sample_columns == None:
            selected_sample_rows, selected_sample_columns = [], []
            for row in range(args.len_LLM):
                for column in range(len(train_data[row].examples)):
                    selected_sample_rows.append(row)
                    selected_sample_columns.append(column)
        # add samples with index indicating
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
            selected_train_dataset[_id].idx = _id
            _id += 1
        selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
        print(f"{len(selected_train_data.examples)=}")
        # selected_train_data = torch.tensor(train_data)[selected_sample_rows,selected_sample_columns]
        # theta = torch.stack(theta)[selected_sample_rows,selected_sample_columns]
        theta = torch.tensor([theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
        train_iter, = BucketIterator.splits(
            (selected_train_data,),
            batch_sizes=(args.train_batch_size,),
            device=device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        # if use all, put all the index together
        if selected_sample_rows == None or selected_sample_columns == None:
            selected_sample_rows, selected_sample_columns = [], []
            for row in range(args.len_LLM):
                for column in range(len(train_data[row].idx)):
                    selected_sample_rows.append(row)
                    selected_sample_columns.append(column)
        # add samples with index indicating
        selected_train_data = TokenizedDataset(
            file_path=(''),
        )
        selected_train_data.text = [] # clear all the samples
        selected_train_data.ids = [] # clear all the samples
        selected_train_data.attention_mask = [] # clear all the samples
        selected_train_data.label = [] # clear all the samples
        selected_train_data.idx = [] # clear all the samples
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_data.text += [train_data[row].text[column]]
            selected_train_data.ids += [train_data[row].ids[column]]
            selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
            selected_train_data.label += [train_data[row].label[column]]
            selected_train_data.idx += [_id]
            _id += 1
        theta = torch.tensor([theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
        train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
    init_theta = copy.deepcopy(theta)
    # print(theta)
    # theta = torch.stack(theta)
    # print(type(theta), theta.shape)
    
    for epoch in range(SELF_WEIGHT_ADJUST_EPOCH):
        print(f"epoch #{epoch} for self-weight select")
        current_outer_iter_trained_model = []
        theta_mapped = copy.deepcopy(theta)
        # print(type(theta_mapped),theta_mapped)
        # best_theta = copy.deepcopy(theta)
            
        ##############  ##############
        diverged = True # diverged==True means loss==nan, which means the training failed
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge, args.inner_obj, test_loader)
            print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
            if epoch_adjust % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), 1, args.inner_obj, test_loader)
                # model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge_fully_train, args.inner_obj)

        current_outer_iter_trained_model.append(model_copy_converged)
        
        if epoch_adjust % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
            print(f"weight-adjust-for-selection-on-fused-dataset: #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_selction_ft={loss_ft}, train_acc_selction_ft={train_acc_ft}, test_acc_selction_ft={test_acc1_ft}, test_loss_selction_ft={test_loss_ft}")
            logging.info(f"weight-adjust-for-selection-on-fused-dataset: , #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_selction_ft={loss_ft}, train_acc_selction_ft={train_acc_ft}, test_acc_selction_ft={test_acc1_ft}, test_loss_selction_ft={test_loss_ft}")
            if args.wandb:
                wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})

        theta_mapped, model_total_acc = weight_decay(args, current_outer_iter_trained_model, [selected_train_data], [theta_mapped], beta=args.BETA, _type=args.weight_adjust_criterial, single_dataset=True)
        theta = copy.deepcopy(theta_mapped[0])
    theta_delta = theta - init_theta
    print(f"ratio of elements whose delta in theta larger than 0.0 is {torch.sum(theta_delta >= 0.0).item()/theta_delta.numel()}, total {torch.sum(theta_delta >= 0.0).item()}/{theta_delta.numel()}")
    logging.info(f"ratio of elements whose delta in theta larger than 0.0 is {torch.sum(theta_delta >= 0.0).item()/theta_delta.numel()}, total {torch.sum(theta_delta >= 0.0).item()}/{theta_delta.numel()}")
    
    new_selected_sample_rows, new_selected_sample_columns = [], []
    for i, (r,c) in enumerate(zip(selected_sample_rows,selected_sample_columns)):
        if theta_delta[i] >= 0.0:
            new_selected_sample_rows.append(r)
            new_selected_sample_columns.append(c)
    print(f"selected sample should equal to the number of elements with theta delta larger than 0.0, that is {len(new_selected_sample_rows)} and {len(new_selected_sample_columns)} should == {torch.sum(theta_delta >= 0.0).item()}")
    assert(torch.sum(theta_delta >= 0.0).item()==len(new_selected_sample_rows) and len(new_selected_sample_rows)==len(new_selected_sample_columns))
    if args.kd_slm == 0:
        if 'Adjust' in args.fuse_dataset_weight:
            return train_to_converge_with_weight_adjust_fused(args, model, train_data, theta_for_function_call, (new_selected_sample_rows, new_selected_sample_columns), epoch_adjust, epoch_converge, inner_obj, test_loader, outer_iter)
        else:
            return train_to_converge_fused(args, model, train_data, theta_for_function_call, (new_selected_sample_rows, new_selected_sample_columns), args.epoch_converge_fully_train, args.inner_obj)
    else:
        # for kd_slm with sample quality separation by increasedTheta
        low_quality_sample_rows, low_quality_sample_columns = [], []
        for i, (r,c) in enumerate(zip(selected_sample_rows,selected_sample_columns)):
            if theta_delta[i] < 0.0:
                low_quality_sample_rows.append(r)
                low_quality_sample_columns.append(c)
                # print(i,r,c)
        return new_selected_sample_rows, new_selected_sample_columns, low_quality_sample_rows, low_quality_sample_columns



# ################################## vote for all ##################################
def train_to_converge_with_weight_adjust_and_label_flip_fused(args, model, train_data, theta, trained_models, selected_sample_indexs, epoch_adjust, epoch_converge, inner_obj, test_loader, outer_iter):
    theta_for_function_call = [copy.deepcopy(_theta) for _theta in theta]
    for _theta in theta:
        _theta = _theta.detach()
    _saved_theta = [copy.deepcopy(_theta) for _theta in theta]
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]  
    assert selected_sample_rows == None and selected_sample_columns == None, "For increased theta selection and flipping for all data, no previous selection should be passed"
    selected_sample_rows, selected_sample_columns = [], []
    if args.small_model_name.upper() == 'LSTM':
        for row in range(args.len_LLM):
            for column in range(len(train_data[row].examples)):
                selected_sample_rows.append(row)
                selected_sample_columns.append(column)
    elif 'bert' in args.small_model_name.lower():
        for row in range(args.len_LLM):
            for column in range(len(train_data[row].idx)):
                selected_sample_rows.append(row)
                selected_sample_columns.append(column)
    selected_sample_rows = torch.tensor(selected_sample_rows, dtype=torch.long).to(args.device)
    selected_sample_columns = torch.tensor(selected_sample_columns, dtype=torch.long).to(args.device)
    # print(f"{selected_sample_rows=}, {len(train_data[0].idx)=}, {selected_sample_rows.shape=}")
    
    model_copy = copy.deepcopy(model)
    losses = AverageMeter("Loss", ":.3f")
    
    # step (1), gather all the samples for selecting those with increased weight
    model_weights_cache = []
    opt_checkpoints_cache = []
    diverged = False
    selected_train_dataset = []
    index_pairs = set()
    _id = 0
    if args.small_model_name.upper() == 'LSTM':
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
            selected_train_dataset[_id].idx = _id
            _id += 1
        selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
        # print(f"{type(train_data[0].examples)=}")
        new_train_data = [data.Dataset(copy.deepcopy(train_data[_i].examples), train_data[_i].fields) for _i in range(args.len_LLM)]
        new_small_train_data = []
        new_small_valid_data = []
        change_count = 0
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            _id = args.accumulate_sampels[row].item()+column
            original_label = selected_train_dataset[_id].label
            print("before", selected_train_dataset[_id].label)
            selected_train_dataset[_id].label = majority_voting_label(args, train_data[row][column], trained_models, train_data[0].fields)
            new_train_data[row][column].label = selected_train_dataset[_id].label 
            print("after", selected_train_dataset[_id].label)
            print(selected_train_dataset[_id].text, f"{row.item()=}, {column.item()=}")
            if original_label != new_train_data[row][column].label:
                change_count += 1
        save_flipped_samples(args, new_train_data, f'train')
        
        for _i in range(args.len_LLM):
            traindataset = new_train_data[_i].examples[:args.sample_each_llm[_i]]
            random.shuffle(traindataset)
            # train-valid split
            small_traindataset, small_validationdataset = copy.deepcopy(traindataset[:int(args.sample_each_llm[_i]*args.train_ratio)]), copy.deepcopy(traindataset[int(args.sample_each_llm[_i]*args.train_ratio):])
            for _id, data_item in enumerate(small_traindataset):
                data_item.idx = _id
            random.shuffle(small_traindataset)
            for _id, data_item in enumerate(small_validationdataset):
                data_item.idx = _id
            # ############## construct all data and separate as train and test ##############
            new_small_train_data.append(data.Dataset(small_traindataset, new_train_data[_i].fields))
            new_small_valid_data.append(data.Dataset(small_validationdataset, new_train_data[_i].fields))
            # ############## construct all data and separate as train and test ##############
        
        print(f"#total data {len(selected_sample_rows)}, after majority voting, #label_changed_sample={change_count} with ratio={(change_count/len(selected_sample_rows))*100}%")
        logging.info(f"#total data {len(selected_sample_rows)}, after majority voting, #label_changed_sample={change_count} with ratio={(change_count/len(selected_sample_rows))*100}%")
        selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
        # selected_train_data = torch.tensor(train_data)[new_selected_sample_rows,new_selected_sample_columns]
        theta = torch.stack(_saved_theta)[selected_sample_rows,selected_sample_columns]
        train_iter, = BucketIterator.splits(
            (selected_train_data,),
            batch_sizes=(args.train_batch_size,),
            device=device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        # add samples with index indicating
        selected_train_data = TokenizedDataset(
            file_path=(''),
        )
        selected_train_data.text = [] # clear all the samples
        selected_train_data.ids = [] # clear all the samples
        selected_train_data.attention_mask = [] # clear all the samples
        selected_train_data.label = [] # clear all the samples
        selected_train_data.idx = [] # clear all the samples
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            # print(f"{_id=}, {row=}, {column=}")
            selected_train_data.text += [train_data[row].text[column]]
            selected_train_data.ids += [train_data[row].ids[column]]
            selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
            selected_train_data.label += [train_data[row].label[column]]
            selected_train_data.idx += [torch.tensor(_id, dtype=torch.long)]
            _id += 1
        new_train_data = []
        new_small_train_data = []
        new_small_valid_data = []
        for _i in range(args.len_LLM):
            new_train_data.append(TokenizedDataset(
                                    file_path=(''),
                                    ))
            new_train_data[_i].text = copy.deepcopy(train_data[_i].text)
            new_train_data[_i].ids = copy.deepcopy(train_data[_i].ids)
            new_train_data[_i].attention_mask = copy.deepcopy(train_data[_i].attention_mask)
            new_train_data[_i].label = copy.deepcopy(train_data[_i].label)
            new_train_data[_i].idx = copy.deepcopy(train_data[_i].idx)
        change_count = 0
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            _id = args.accumulate_sampels[row].item()+column
            # print(f"{_id=}, {row=}, {column=}")
            print("before", selected_train_data.label[_id])
            original_label = copy.deepcopy(selected_train_data.label[_id])
            # selected_train_data.label[_id] = torch.tensor(majority_voting_label(args, train_data[row][column], trained_models, origin_model_index=row, use_weight=True), dtype=torch.long)
            selected_train_data.label[_id] = torch.tensor(majority_voting_label(args, train_data[row][column], trained_models), dtype=torch.long)
            new_train_data[row].label[column] = selected_train_data.label[_id]
            print("after", selected_train_data.label[_id])
            print("after", selected_train_data.text[_id])
            if int(original_label) != int(new_train_data[row].label[column]):
                change_count += 1
        save_flipped_samples(args, new_train_data, f'train')

        # ############## separate as train and test ##############
        for _i in range(args.len_LLM):
            indices = list(range(args.sample_each_llm[_i]))
            random.shuffle(indices)
            train_valid_pivot_point = int(args.sample_each_llm[_i]*args.train_ratio)
            # train-valid split
            small_train_data = TokenizedDataset(
                file_path=(''),
            )
            small_train_data.text = [copy.deepcopy(new_train_data[_i].text[ix]) for ix in indices[:train_valid_pivot_point]]
            small_train_data.ids = copy.deepcopy(new_train_data[_i].ids[indices[:train_valid_pivot_point]])
            small_train_data.attention_mask = copy.deepcopy(new_train_data[_i].attention_mask[indices[:train_valid_pivot_point]])
            small_train_data.label = copy.deepcopy(new_train_data[_i].label[indices[:train_valid_pivot_point]])
            small_train_data.idx = torch.tensor([_i for _i in range(train_valid_pivot_point)]).long()
            small_valid_data = TokenizedDataset(
                file_path=(''),
            )
            small_valid_data.text = [copy.deepcopy(new_train_data[_i].text[ix]) for ix in indices[train_valid_pivot_point:]]
            small_valid_data.ids = copy.deepcopy(new_train_data[_i].ids[indices[train_valid_pivot_point:]])
            small_valid_data.attention_mask = copy.deepcopy(new_train_data[_i].attention_mask[indices[train_valid_pivot_point:]])
            small_valid_data.label = copy.deepcopy(new_train_data[_i].label[indices[train_valid_pivot_point:]])
            small_valid_data.idx = torch.tensor([_i for _i in range(args.sample_each_llm[_i]-train_valid_pivot_point)]).long()
            new_small_train_data.append(small_train_data)
            new_small_valid_data.append(small_valid_data)
        # ############## separate as train and test ##############
        
        print(f"#total data {len(selected_sample_rows)}, after majority voting, #label_changed_sample={change_count} with ratio={(change_count/len(selected_sample_rows))*100}%")
        logging.info(f"#total data {len(selected_sample_rows)}, after majority voting, #label_changed_sample={change_count} with ratio={(change_count/len(selected_sample_rows))*100}%")
        theta = torch.tensor([_saved_theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
        train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
    init_theta = copy.deepcopy(theta)
    
    for epoch in range(SELF_WEIGHT_ADJUST_EPOCH):
    # for epoch in range(args.max_outer_iter):
        print(f"epoch #{epoch} for self-weight select")
        current_outer_iter_trained_model = []
        theta_mapped = copy.deepcopy(theta)
        # print(type(theta_mapped),theta_mapped)
        # best_theta = copy.deepcopy(theta)
            
        diverged = True # diverged==True means loss==nan, which means the training failed
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge, args.inner_obj, test_loader)
            print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
            if epoch_adjust % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), 1, args.inner_obj, test_loader)
                # model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge_fully_train, args.inner_obj)

        current_outer_iter_trained_model.append(model_copy_converged)
        
        if epoch_adjust % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
            print(f"weight-adjust-for-selection-on-fused-dataset: #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_selction_ft={loss_ft}, train_acc_selction_ft={train_acc_ft}, test_acc_selction_ft={test_acc1_ft}, test_loss_selction_ft={test_loss_ft}")
            logging.info(f"weight-adjust-for-selection-on-fused-dataset: , #fused_adjust_iter={epoch}, beta({args.BETA}), train_loss_selction_ft={loss_ft}, train_acc_selction_ft={train_acc_ft}, test_acc_selction_ft={test_acc1_ft}, test_loss_selction_ft={test_loss_ft}")
            if args.wandb:
                wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})

        theta_mapped, model_total_acc = weight_decay(args, current_outer_iter_trained_model, [selected_train_data], [theta_mapped], beta=args.BETA, _type=args.weight_adjust_criterial, single_dataset=True)
        if epoch == SELF_WEIGHT_ADJUST_EPOCH-1:
            theta_delta = copy.deepcopy(theta - init_theta)
        theta = copy.deepcopy(theta_mapped[0])
    # theta_delta = theta - init_theta
    print(f"ratio of elements whose delta in theta larger than 0.0 is {torch.sum(theta_delta >= 0.0).item()/theta_delta.numel()}, total {torch.sum(theta_delta >= 0.0).item()}/{theta_delta.numel()}")
    logging.info(f"ratio of elements whose delta in theta larger than 0.0 is {torch.sum(theta_delta >= 0.0).item()/theta_delta.numel()}, total {torch.sum(theta_delta >= 0.0).item()}/{theta_delta.numel()}")
    
    # choose samples that has increased weight as "high quality sample"
    # print(f"{theta_delta=}")
    new_selected_sample_rows, new_selected_sample_columns = [], []
    for i, (r,c) in enumerate(zip(selected_sample_rows,selected_sample_columns)):
        if theta_delta[i] >= 0.0:
            new_selected_sample_rows.append(r)
            new_selected_sample_columns.append(c)
    # print(f"selected sample should equal to the number of elements with theta delta larger than 0.0, that is {len(new_selected_sample_rows)} and {len(new_selected_sample_columns)} should == {torch.sum(theta_delta >= 0.0).item()}")
    assert(torch.sum(theta_delta >= 0.0).item()==len(new_selected_sample_rows) and len(new_selected_sample_rows)==len(new_selected_sample_columns))
    low_quality_sample_rows, low_quality_sample_columns = [], []
    for i, (r,c) in enumerate(zip(selected_sample_rows,selected_sample_columns)):
        if theta_delta[i] < 0.0:
            low_quality_sample_rows.append(r)
            low_quality_sample_columns.append(c)
            # print(i,r,c)
    # print(f"{new_selected_sample_rows=}, {new_selected_sample_columns=}")
    # print(f"{low_quality_sample_rows=}, {low_quality_sample_columns=}")

    if args.kd_slm == 0: # train on dataset that flipped 
        if 'Adjust' in args.fuse_dataset_weight:
            return train_to_converge_with_weight_adjust_fused(args, model, new_train_data, theta_for_function_call, (None, None), epoch_adjust, epoch_converge, inner_obj, test_loader, outer_iter)
        else:
            return train_to_converge_fused(args, model, new_train_data, theta_for_function_call, (None, None), args.epoch_converge_fully_train, args.inner_obj)
    else:
        # for kd_slm with sample quality separation by increasedTheta
        return new_selected_sample_rows, new_selected_sample_columns, low_quality_sample_rows, low_quality_sample_columns, new_train_data, new_small_train_data, new_small_valid_data
# ################################## vote for all ##################################


def train_to_converge_with_selection_kd_fused(args, model, train_data, theta, quality_separate_sample_indexs, soft_labels, epoch_adjust, epoch_converge, inner_obj, train_loader_backward, valid_loader, test_loader, outer_iter, final_test=True):
    theta_for_function_call = [copy.deepcopy(_theta) for _theta in theta]
    for _theta in theta:
        _theta = _theta.detach()
    _saved_theta = copy.deepcopy(theta)
    # soft labels
    soft_label_total, soft_label_not_seen, soft_label_single_predict_bad = soft_labels[0], soft_labels[1], soft_labels[2]
    # print(f"[debug] in <train_to_converge_with_selection_kd_fused>, {soft_label_total.shape=}, {soft_label_not_seen.shape=}, {soft_label_single_predict_bad.shape=}")
    # train 2 model separately on high and low quality data, with one-hot label
    high_quality_sample_rows, high_quality_sample_columns = quality_separate_sample_indexs[0], quality_separate_sample_indexs[1]
    low_quality_sample_rows, low_quality_sample_columns = quality_separate_sample_indexs[2], quality_separate_sample_indexs[3]

    print(f'here1, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    model_copy = copy.deepcopy(model)
    high_quality_model_copy = copy.deepcopy(model)
    high_quality_kd_model_copy = copy.deepcopy(model)
    low_quality_model_copy = copy.deepcopy(model)
    print(f'here2, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

    
    # (1) train only on good sample with one-hot label
    if final_test:
        if 'Reweight' in args.fuse_dataset_weight:
            print_info = ["weight-adjust-for-selection-kd-fused-(good-one-hot)", "fused_adjust_iter", "good_train_loss_selction_ft", "good_train_acc_selction_ft", "good_test_acc_selction_ft", "good_test_loss_selction_ft"]
            high_quality_one_hot_theta, high_quality_one_hot_model_copy, _, _ = solve_reweight_v2(args, args.fused_model, train_data, _saved_theta, (high_quality_sample_rows, high_quality_sample_columns), train_loader_backward, valid_loader, test_loader, print_info=print_info, reweight_epoch=epoch_adjust//3, soft_label=None)
        else:
            diverged = False
            selected_train_dataset = []
            _id = 0
            if args.small_model_name.upper() == 'LSTM':
                for row, column in zip(high_quality_sample_rows, high_quality_sample_columns):
                    selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
                    selected_train_dataset[_id].idx = _id
                    _id += 1
                selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
                theta = torch.stack(_saved_theta)[high_quality_sample_rows, high_quality_sample_columns]
                train_iter, = BucketIterator.splits(
                    (selected_train_data,),
                    batch_sizes=(args.train_batch_size,),
                    device=device,
                    sort_key=lambda x: len(x.text),
                    sort_within_batch=True,
                    repeat=False,
                    shuffle=args.shuffle_train,
                )
            elif 'bert' in args.small_model_name.lower():
                selected_train_data = TokenizedDataset(
                    file_path=(''),
                )
                selected_train_data.text = [] # clear all the samples
                selected_train_data.ids = [] # clear all the samples
                selected_train_data.attention_mask = [] # clear all the samples
                selected_train_data.label = [] # clear all the samples
                selected_train_data.idx = [] # clear all the samples
                for row, column in zip(high_quality_sample_rows, high_quality_sample_columns):
                    selected_train_data.text += [train_data[row].text[column]]
                    selected_train_data.ids += [train_data[row].ids[column]]
                    selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
                    selected_train_data.label += [train_data[row].label[column]]
                    selected_train_data.idx += [_id]
                    _id += 1
                theta = torch.tensor([_saved_theta[row][col] for row,col in zip(high_quality_sample_rows, high_quality_sample_columns)]).to(args.device)
                train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
            # init_theta = copy.deepcopy(theta)
            for epoch in range(epoch_adjust//3):
                logging.debug(f"epoch #{epoch} for good-sample-one-hot")
                current_outer_iter_trained_model = []
                theta_mapped = copy.deepcopy(theta)
                # print(type(theta_mapped),theta_mapped)
                # best_theta = copy.deepcopy(theta)
                    
                ##############  ##############
                diverged = True # diverged==True means loss==nan, which means the training failed
                while diverged:
                    model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge, args.inner_obj, test_loader)
                    print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
                    if epoch_adjust % args.check_ft_every==0:
                        model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), 1, args.inner_obj, test_loader)
                        # model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge_fully_train, args.inner_obj)

                current_outer_iter_trained_model.append(model_copy_converged)
                
                if epoch_adjust % args.check_ft_every == 0:
                    test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
                    print(f"weight-adjust-for-selection-kd-fused-(good-one-hot): #fused_adjust_iter={epoch}, beta({args.BETA}), good_train_loss_selction_ft={loss_ft}, good_train_acc_selction_ft={train_acc_ft}, good_test_acc_selction_ft={test_acc1_ft}, good_test_loss_selction_ft={test_loss_ft}")
                    logging.info(f"weight-adjust-for-selection-kd-fused-(good-one-hot): , #fused_adjust_iter={epoch}, beta({args.BETA}), good_train_loss_selction_ft={loss_ft}, good_train_acc_selction_ft={train_acc_ft}, good_test_acc_selction_ft={test_acc1_ft}, good_test_loss_selction_ft={test_loss_ft}")
                    if args.wandb:
                        wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})

                theta_mapped, model_total_acc = weight_decay(args, current_outer_iter_trained_model, [selected_train_data], [theta_mapped], beta=args.BETA, _type=args.weight_adjust_criterial, single_dataset=True)
                theta = copy.deepcopy(theta_mapped[0])
            high_quality_one_hot_theta = copy.deepcopy(theta)
            high_quality_one_hot_model_copy = copy.deepcopy(model_copy_converged_ft)
            high_quality_one_hot_theta = high_quality_one_hot_theta.to("cpu")
            high_quality_one_hot_model_copy = high_quality_one_hot_model_copy.to("cpu")

    # ############### prepare total_data for later use ###############
    accumulate_sampels = [0]
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in train_data:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
            total_data += copy.deepcopy(_dataset_examples)
            accumulate_sampels.append(accumulate_sampels[-1]+len(_dataset_examples)) 
        for _i in range(len(total_data)):
            total_data[_i].idx = _i
    elif 'bert' in args.small_model_name.lower():
        _id = 0
        total_data = TokenizedDataset(
            file_path=(''),
        )
        total_data.text = [] # clear all the samples
        total_data.ids = [] # clear all the samples
        total_data.attention_mask = [] # clear all the samples
        total_data.label = [] # clear all the samples
        total_data.idx = [] # clear all the samples
        for row in range(args.len_LLM):
            accumulate_sampels.append(accumulate_sampels[-1]+len(train_data[row].idx)) 
            for column in range(len(train_data[row].idx)):
                total_data.text += [train_data[row].text[column]]
                total_data.ids += [train_data[row].ids[column]]
                total_data.attention_mask += [train_data[row].attention_mask[column]]
                total_data.label += [train_data[row].label[column]]
                total_data.idx += [_id]
                _id += 1
    accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)

    # (6) train on all the samples with 1-hot label
    if 'Reweight' in args.fuse_dataset_weight:
        print_info = ["weight-adjust-for-selection-kd-fused-(all-one-hot)", "fused_adjust_iter", "total_one_hot_train_loss_selction_ft", "total_one_hot_train_acc_selction_ft", "total_one_hot_test_acc_selction_ft", "total_one_hot_test_loss_selction_ft"]
        total_one_hot_theta, total_one_hot_model_copy, _, _ = solve_reweight_v2(args, args.fused_model, train_data, _saved_theta, (high_quality_sample_rows, high_quality_sample_columns), train_loader_backward, valid_loader, test_loader, print_info=print_info, reweight_epoch=epoch_adjust//3, soft_label=None)
    else:
        diverged = False
        if args.small_model_name.upper() == 'LSTM':
            selected_train_dataset = total_data
            selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
            theta = torch.stack(_saved_theta).view(-1)
            train_iter, = BucketIterator.splits(
                (selected_train_dataset,),
                batch_sizes=(args.train_batch_size,),
                device=device,
                sort_key=lambda x: len(x.text),
                sort_within_batch=True,
                repeat=False,
                shuffle=args.shuffle_train,
            )
        elif 'bert' in args.small_model_name.lower():
            selected_train_data = total_data
            theta = torch.stack(_saved_theta).view(-1)
            train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
            print(f"{len(selected_train_data)=}, {theta.shape=}")
        # init_theta = copy.deepcopy(theta)
        for epoch in range(epoch_adjust//3):
            logging.debug(f"epoch #{epoch} for all-sample-one-hot")
            current_outer_iter_trained_model = []
            theta_mapped = copy.deepcopy(theta)
            # print(type(theta_mapped),theta_mapped)
            # best_theta = copy.deepcopy(theta)
                
            # print(f"{theta_mapped.shape=}, {len(selected_train_data)=}, {soft_label_not_seen.shape=}")
            ##############  ##############
            diverged = True # diverged==True means loss==nan, which means the training failed
            while diverged:
                model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge, args.inner_obj, test_loader)
                print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
                if epoch_adjust % args.check_ft_every==0:
                    model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), 1, args.inner_obj, test_loader)
                    # model_copy_converged_ft, loss_ft, train_acc_ft, model_weights_cache_ft, opt_checkpoints_cache_ft, diverged_ft = train_to_converge(args, model_copy, selected_train_data, theta_mapped.detach(), args.epoch_converge_fully_train, args.inner_obj)

            current_outer_iter_trained_model.append(model_copy_converged)
            
            if epoch_adjust % args.check_ft_every == 0:
                test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
                print(f"weight-adjust-for-selection-kd-fused-(all-one-hot): #fused_adjust_iter={epoch}, beta({args.BETA}), total_one_hot_train_loss_selction_ft={loss_ft}, total_one_hot_train_acc_selction_ft={train_acc_ft}, total_one_hot_test_acc_selction_ft={test_acc1_ft}, total_one_hot_test_loss_selction_ft={test_loss_ft}")
                logging.info(f"weight-adjust-for-selection-kd-fused-(all-one-hot): , #fused_adjust_iter={epoch}, beta({args.BETA}), total_one_hot_train_loss_selction_ft={loss_ft}, total_one_hot_train_acc_selction_ft={train_acc_ft}, total_one_hot_test_acc_selction_ft={test_acc1_ft}, total_one_hot_test_loss_selction_ft={test_loss_ft}")
                if args.wandb:
                    wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})

            theta_mapped, model_total_acc = weight_decay(args, current_outer_iter_trained_model, [selected_train_data], [theta_mapped], beta=args.BETA, _type=args.weight_adjust_criterial, single_dataset=True)
            theta = copy.deepcopy(theta_mapped[0])
        total_one_hot_theta = copy.deepcopy(theta)
        total_one_hot_model_copy = copy.deepcopy(model_copy_converged_ft)
        # total_one_hot_theta = total_one_hot_theta.to("cpu")
        # total_one_hot_model_copy = total_one_hot_model_copy.to("cpu")

    torch.cuda.empty_cache()
    # return total_kd_alpha_model_copy, total_kd_alpha_theta
    return total_one_hot_model_copy, total_one_hot_theta



def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def train_separate_models(args, model, train_data, small_train_data, small_valid_data, train_loader_backward, valid_loader, test_loader, main_outer_iter):
    init_model = [copy.deepcopy(_model) for _model in model]
    theta = []
    # if args.use_sigmoid:
    #     for i in range(len(train_data)):
    #         theta.append(torch.full([len(train_data[i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
    # else:
    #     for i in range(len(train_data)):
    #         theta.append(torch.full([len(train_data[i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
    if args.use_sigmoid:
        for i in range(len(small_train_data)):
            theta.append(torch.full([len(small_train_data[i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
    else:
        for i in range(len(small_train_data)):
            theta.append(torch.full([len(small_train_data[i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
    if args.optim =='Adam':
        theta_opt = Adam(theta, lr=args.outer_lr)
    elif args.optim =='SGD':
        theta_opt = SGD(theta, lr=args.outer_lr, momentum=0.9)
    for _theta in theta:
        _theta.grad = torch.zeros_like(_theta)
    best_theta = theta # <list>
    # temp_use_sigmoid = args.use_sigmoid
    
    current_outer_iter_trained_model_iter0 = []
    current_outer_iter_trained_more_steps_model_iter0 = []
    # for outer_iter in range(args.max_outer_iter):
    # for outer_iter in range(1):
    for outer_iter in range(5):
        # args.use_sigmoid = temp_use_sigmoid
        # step (1): train each small local model for each syn dataset with args.epoch_converge iterations
        current_outer_iter_trained_model = []
        current_outer_iter_trained_more_steps_model = []
        theta_mapped = [copy.deepcopy(_theta) for _theta in theta]
        for i in range(args.len_LLM):
            print(f"training with new data from #{i} LLM {args.llms[i]} in iter=#{outer_iter}")
            if args.temp_anneal:
                temp = args.end_temp + (args.max_outer_iter*(args.len_LLM-i) - outer_iter)/(args.max_outer_iter*args.len_LLM) * (1-args.end_temp)
                print("[debug] temp", temp)
            else:
                temp = 1
            if args.use_sigmoid:
            # if args.use_sigmoid and outer_iter==0.0:
                theta_mapped[i] = torch.sigmoid(theta[i]/temp)
            else:
                theta_mapped[i] = theta[i]
            # print("theta[i]", theta[i], torch.sigmoid(torch.tensor([0.0])))
            # print("theta_mapped[i]", theta_mapped[i])
            # for i in range(args.len_LLM):
            #     print("***", theta[i].grad)
            if not args.disable_outer_scheduler:
                assign_learning_rate(theta_opt, 0.5 * (1 + np.cos(np.pi * (outer_iter+args.max_outer_iter*i) / args.max_outer_iter*args.len_LLM)) * args.outer_lr)

            ############## original weird version ##############
            diverged = True # diverged==True means loss==nan, which means the training failed
            while diverged:
                # model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model[i], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge, args.inner_obj, test_loader)
                model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model[0], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge, args.inner_obj, test_loader)
                print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
                if outer_iter % args.check_ft_every==0:
                    # model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge(args, model[i], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge_fully_train, args.inner_obj, test_loader)
                    model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge(args, model[0], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge_fully_train, args.inner_obj, test_loader)
            # print(f"[debug] {args.stochastic_outer and args.subset_outer} {args.stochastic_outer}, {args.subset_outer}")
            # print(f"[debug] {args.use_dev_outer}")
            if args.stochastic_outer and args.subset_outer:
                if args.use_dev_outer:
                    valid_loader = construct_outer_subloader(args, dev_data_all)
                else:
                    valid_loader = construct_outer_subloader(args, train_data) # currently using this one
            
            # grad_weights_on_full_train, top1_outer, loss_outer = get_grad_weights_on_valid(model_copy_converged, valid_loader, theta_mapped[i].detach())
            # print(f"outer acc {top1_outer}, loss_outer {loss_outer}")
            # grad_theta = repass_backward(model, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, train_loader_backward[i], theta_mapped[i], theta[i])
            # # print(f"[debug] grad_theta.shape {grad_theta.shape} {grad_theta} {theta[i].grad}")
            # theta_opt.zero_grad()
            # print(f"sum grads {sum([g for g in grad_theta])}")
            # with torch.no_grad():
            #     theta[i].grad += grad_theta.data
            # torch.nn.utils.clip_grad_norm_(theta[i], args.clip_constant)
            # theta_opt.step()
            # if not args.use_sigmoid:
            #     with torch.no_grad():
            #         theta[i].data.clamp_(min=0, max=args.theta_upper_lim)
            # torch.cuda.empty_cache()
            current_outer_iter_trained_model.append(model_copy_converged)
            current_outer_iter_trained_more_steps_model.append(model_copy_converged_ft)
            if outer_iter == 0:
                current_outer_iter_trained_model_iter0.append(model_copy_converged)
                current_outer_iter_trained_more_steps_model_iter0.append(model_copy_converged_ft)
            
            if outer_iter % args.check_ft_every == 0:
                test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
                # test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, train_loader[i], name="test")
                print(f"voted dataset crossing: LLM#{i}, #iter={outer_iter}, beta({args.BETA}), new_train_loss_ft={loss_ft}, new_train_acc_ft={train_acc_ft}, new_test_acc_ft={test_acc1_ft}, new_test_loss_ft={test_loss_ft}")
                logging.info(f"voted dataset crossing: LLM#{i}, #iter={outer_iter}, beta({args.BETA}), new_train_loss_ft={loss_ft}, new_train_acc_ft={train_acc_ft}, new_test_acc_ft={test_acc1_ft}, new_test_loss_ft={test_loss_ft}")
                if args.wandb:
                    wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})
            # if outer_iter % args.check_ft_every == 0:
            #     test_acc1, test_loss = eval(args, model_copy_converged, test_loader, name="test")
            #     # test_acc1, test_loss = eval(args, model_copy_converged, train_loader[i], name="test")
            #     print(f"crossing: LLM#{i}, #iter={outer_iter}, beta({args.BETA}), train_loss={loss}, train_acc={train_acc}, test_acc={test_acc1}, test_loss={test_loss}")
            #     logging.info(f"crossing: LLM#{i}, #iter={outer_iter}, beta({args.BETA}), train_loss={loss}, train_acc={train_acc}, test_acc={test_acc1}, test_loss={test_loss}")
            #     if args.wandb:
            #         wandb.log({"train_loss": loss,"train_acc":train_acc,"test_acc": test_acc1, "test_loss":test_loss})
            
            # theta_score=copy.deepcopy(theta[i])
            # print(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
            # logging.info(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
            # if args.wandb:
            #     wandb.log({"train_loss": loss, "loss_outer": loss_outer, "temp":temp})
            ############## original weird version ##############

        model_total_acc = model_importance_estimation(args, current_outer_iter_trained_more_steps_model, small_valid_data, _type=args.weight_adjust_criterial)
        # model_total_acc = model_importance_estimation(args, current_outer_iter_trained_more_steps_model, train_data, _type=args.weight_adjust_criterial)
        theta_mapped, _depricated_model_total_acc = weight_decay(args, current_outer_iter_trained_model, small_train_data, theta_mapped, beta=args.BETA, _type=args.weight_adjust_criterial)
        for j in range(args.len_LLM):
            theta[j] = copy.deepcopy(theta_mapped[j])
            theta_score = copy.deepcopy(theta[j])
            # print(f"new theta[j] = {theta[j]}")
            best_theta[j]=theta_score
        
        model_importance = model_total_acc / torch.sum(model_total_acc)  
        torch.save(tuple(current_outer_iter_trained_more_steps_model), f"{args.result_file_path}/iter{main_outer_iter}_inneriter{outer_iter}_separate_models.pth")
        
    if args.fuse_dataset_weight == 'new' or args.fuse_dataset_weight == 'Adjust' or args.fuse_dataset_weight == 'Reweight':
        new_theta = []
        if args.use_sigmoid:
            for _i in range(len(train_data)):
                new_theta.append(torch.full([len(train_data[_i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
        else:
            for _i in range(len(train_data)):
                new_theta.append(torch.full([len(train_data[_i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
        new_theta_mapped = [copy.deepcopy(_theta) for _theta in new_theta]
        if args.temp_anneal:
            temp = args.end_temp + (args.max_outer_iter - outer_iter)/(args.max_outer_iter) * (1-args.end_temp)
            print("[debug] temp", temp)
        else:
            temp = 1
        for _i in range(len(new_theta)):
            if args.use_sigmoid:
            # if args.use_sigmoid and outer_iter==0.0:
                new_theta_mapped[_i] = torch.sigmoid(new_theta[_i]/temp)
            else:
                new_theta_mapped[_i] = new_theta[_i]
    elif 'inheritModelAndSample' in args.fuse_dataset_weight:
        theta_values = model_importance.reshape(-1,1) * torch.stack(theta_mapped)
        new_theta_mapped = [theta_values[_i] for _i in range(theta_values.shape[0])]
    elif 'inheritModel' in args.fuse_dataset_weight:
        new_theta_mapped = [copy.deepcopy(_theta) for _theta in theta_mapped]
        for _i in range(args.len_LLM):
            # (torch.full([len(train_data[_i])], args.init_theta, dtype=torch.float, device=device))
            new_theta_mapped[_i] = torch.full([len(train_data[_i])], model_importance[_i], dtype=torch.float, device=device) #, requires_grad=True
    elif 'inheritSample' in args.fuse_dataset_weight:
        new_theta_mapped = [copy.deepcopy(_theta) for _theta in theta_mapped]
    
    return current_outer_iter_trained_model_iter0, current_outer_iter_trained_more_steps_model_iter0, best_theta, new_theta_mapped, new_theta_mapped
    # return current_outer_iter_trained_model, current_outer_iter_trained_more_steps_model, best_theta, new_theta_mapped, new_theta_mapped


def solve_with_local_cross_validation(args, model, train_data, small_train_data, small_valid_data, train_loader_backward, valid_loader, test_loader, perform_few_shot_gen=False):
    '''
        input parameters:
            - model: trainable model
            - train_data: <list>, [train_dataset_for_synthetic_data_from_each_LLM]
            - test_loader: <Object>
    '''
    # assert len(train_loader) == len(train_loader_backward) == args.len_LLM
    init_model = [copy.deepcopy(_model) for _model in model]
    theta = []
    # if args.use_sigmoid:
    #     for i in range(len(train_data)):
    #         theta.append(torch.full([len(train_data[i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
    # else:
    #     for i in range(len(train_data)):
    #         theta.append(torch.full([len(train_data[i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
    if args.use_sigmoid:
        for i in range(len(small_train_data)):
            theta.append(torch.full([len(small_train_data[i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
    else:
        for i in range(len(small_train_data)):
            theta.append(torch.full([len(small_train_data[i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
    if args.optim =='Adam':
        theta_opt = Adam(theta, lr=args.outer_lr)
    elif args.optim =='SGD':
        theta_opt = SGD(theta, lr=args.outer_lr, momentum=0.9)
    for _theta in theta:
        _theta.grad = torch.zeros_like(_theta)
    best_theta = theta # <list>
    # temp_use_sigmoid = args.use_sigmoid
    
    # for outer_iter in range(args.max_outer_iter):
    for outer_iter in range(1):
    # for outer_iter in range(5):
        current_outer_iter_trained_model_iter0 = []
        current_outer_iter_trained_more_steps_model_iter0 = []
        # args.use_sigmoid = temp_use_sigmoid
        for _outer_iter in range(1 if 'Flip' in args.fuse_dataset_sample_selection else 5): # this is for separate model WA
            # step (1): train each small local model for each syn dataset with args.epoch_converge iterations
            current_outer_iter_trained_model = []
            current_outer_iter_trained_more_steps_model = []
            theta_mapped = [copy.deepcopy(_theta) for _theta in theta]
            for i in range(args.len_LLM):
                print(f"training with data from #{i} LLM {args.llms[i]} in iter=#{_outer_iter}")
                if args.temp_anneal:
                    temp = args.end_temp + (args.max_outer_iter*(args.len_LLM-i) - _outer_iter)/(args.max_outer_iter*args.len_LLM) * (1-args.end_temp)
                    print("[debug] temp", temp)
                else:
                    temp = 1
                if args.use_sigmoid:
                # if args.use_sigmoid and outer_iter==0.0:
                    theta_mapped[i] = torch.sigmoid(theta[i]/temp)
                else:
                    theta_mapped[i] = theta[i]
                # print("theta[i]", theta[i], torch.sigmoid(torch.tensor([0.0])))
                # print("theta_mapped[i]", theta_mapped[i])
                # for i in range(args.len_LLM):
                #     print("***", theta[i].grad)
                if not args.disable_outer_scheduler:
                    assign_learning_rate(theta_opt, 0.5 * (1 + np.cos(np.pi * (_outer_iter+args.max_outer_iter*i) / args.max_outer_iter*args.len_LLM)) * args.outer_lr)

                ############## original weird version ##############
                if 'Reweight' in args.fuse_dataset_weight:
                    print_info = [f"crossing: LLM#{i}", "fused_adjust_iter", "train_loss_ft", "train_acc_ft", "test_acc_ft", "test_loss_ft"]
                    theta_mapped[i], model_copy_converged_ft, _, _ = solve_reweight_v2(args, args.fused_model, [small_train_data[i]], [theta_mapped[i].detach()], (None,None), train_loader_backward, valid_loader, test_loader, print_info=print_info, reweight_epoch=1, soft_label=None)
                    current_outer_iter_trained_more_steps_model.append(model_copy_converged_ft)
                    theta[i] = copy.deepcopy(theta_mapped[i])
                    theta_score = copy.deepcopy(theta[i])
                    # print(f"new theta[i] = {theta[i]}")
                    best_theta[i]=theta_score
                else:
                    diverged = True # diverged==True means loss==nan, which means the training failed
                    while diverged:
                        model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(args, model[i], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge, args.inner_obj, test_loader)
                        print(f"diverged={diverged}, loss={loss}, train_acc={train_acc}")
                        if _outer_iter % args.check_ft_every==0:
                            model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge(args, model[i], small_train_data[i], theta_mapped[i].detach(), args.epoch_converge_fully_train, args.inner_obj, test_loader)
                    # print(f"[debug] {args.stochastic_outer and args.subset_outer} {args.stochastic_outer}, {args.subset_outer}")
                    # print(f"[debug] {args.use_dev_outer}")
                    if args.stochastic_outer and args.subset_outer:
                        if args.use_dev_outer:
                            valid_loader = construct_outer_subloader(args, dev_data_all)
                        else:
                            valid_loader = construct_outer_subloader(args, train_data) # currently using this one
                    
                    # grad_weights_on_full_train, top1_outer, loss_outer = get_grad_weights_on_valid(model_copy_converged, valid_loader, theta_mapped[i].detach())
                    # print(f"outer acc {top1_outer}, loss_outer {loss_outer}")
                    # grad_theta = repass_backward(model, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, train_loader_backward[i], theta_mapped[i], theta[i])
                    # # print(f"[debug] grad_theta.shape {grad_theta.shape} {grad_theta} {theta[i].grad}")
                    # theta_opt.zero_grad()
                    # print(f"sum grads {sum([g for g in grad_theta])}")
                    # with torch.no_grad():
                    #     theta[i].grad += grad_theta.data
                    # torch.nn.utils.clip_grad_norm_(theta[i], args.clip_constant)
                    # theta_opt.step()
                    # if not args.use_sigmoid:
                    #     with torch.no_grad():
                    #         theta[i].data.clamp_(min=0, max=args.theta_upper_lim)
                    # torch.cuda.empty_cache()
                    current_outer_iter_trained_model.append(model_copy_converged)
                    current_outer_iter_trained_more_steps_model.append(model_copy_converged_ft)
                    if _outer_iter == 0:
                        current_outer_iter_trained_model_iter0.append(model_copy_converged)
                        current_outer_iter_trained_more_steps_model_iter0.append(model_copy_converged_ft)
                
                    if _outer_iter % args.check_ft_every == 0:
                        test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, test_loader, name="test")
                        # test_acc1_ft, test_loss_ft = eval(args, model_copy_converged_ft, train_loader[i], name="test")
                        print(f"crossing: LLM#{i}, #iter={_outer_iter}, beta({args.BETA}), train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
                        logging.info(f"crossing: LLM#{i}, #iter={_outer_iter}, beta({args.BETA}), train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
                        if args.wandb:
                            wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})
                # if _outer_iter % args.check_ft_every == 0:
                #     test_acc1, test_loss = eval(args, model_copy_converged, test_loader, name="test")
                #     # test_acc1, test_loss = eval(args, model_copy_converged, train_loader[i], name="test")
                #     print(f"crossing: LLM#{i}, #iter={_outer_iter}, beta({args.BETA}), train_loss={loss}, train_acc={train_acc}, test_acc={test_acc1}, test_loss={test_loss}")
                #     logging.info(f"crossing: LLM#{i}, #iter={_outer_iter}, beta({args.BETA}), train_loss={loss}, train_acc={train_acc}, test_acc={test_acc1}, test_loss={test_loss}")
                #     if args.wandb:
                #         wandb.log({"train_loss": loss,"train_acc":train_acc,"test_acc": test_acc1, "test_loss":test_loss})
                
                # theta_score=copy.deepcopy(theta[i])
                # print(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
                # logging.info(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
                # if args.wandb:
                #     wandb.log({"train_loss": loss, "loss_outer": loss_outer, "temp":temp})
                ############## original weird version ##############

            model_total_acc = model_importance_estimation(args, current_outer_iter_trained_more_steps_model, small_valid_data, _type=args.weight_adjust_criterial)
            # model_total_acc = model_importance_estimation(args, current_outer_iter_trained_more_steps_model, train_data, _type=args.weight_adjust_criterial)
            if 'Adjust' in args.fuse_dataset_weight:
                theta_mapped, _depricated_model_total_acc = weight_decay(args, current_outer_iter_trained_model, small_train_data, theta_mapped, beta=args.BETA, _type=args.weight_adjust_criterial)
                for j in range(args.len_LLM):
                    theta[j] = copy.deepcopy(theta_mapped[j])
                    theta_score = copy.deepcopy(theta[j])
                    # print(f"new theta[j] = {theta[j]}")
                    best_theta[j]=theta_score
            
            torch.save(tuple(current_outer_iter_trained_more_steps_model+[args.fused_model]), f"{args.result_file_path}/iter{_outer_iter}_main_separate_models.pth")
            torch.save((theta_mapped, model_total_acc),f"{args.result_file_path}/iter{_outer_iter}_main_theta_acc.pth")
        
        # use the first model for further calculation
        current_outer_iter_trained_model = current_outer_iter_trained_model_iter0
        current_outer_iter_trained_more_steps_model = current_outer_iter_trained_more_steps_model_iter0

        ######### train a new llm with new data #########
        if args.kd_slm == 1 and 'increasedTheta' in args.fuse_dataset_sample_selection:
            # use increasedTheta for separating high and low quality data and then use different kd methods for each
            print(f"[debug] use increasedTheta for separating high and low quality data")
            model_importance = model_total_acc / torch.sum(model_total_acc)

            if args.fuse_dataset_weight == 'new' or args.fuse_dataset_weight == 'Adjust' or args.fuse_dataset_weight == 'Reweight':
                new_theta = []
                if args.use_sigmoid:
                    for _i in range(len(train_data)):
                        new_theta.append(torch.full([len(train_data[_i])], 0, dtype=torch.float, device=device)) #, requires_grad=True
                else:
                    for _i in range(len(train_data)):
                        new_theta.append(torch.full([len(train_data[_i])], args.init_theta, dtype=torch.float, device=device)) #, requires_grad=True
                new_theta_mapped = [copy.deepcopy(_theta) for _theta in new_theta]
                if args.temp_anneal:
                    temp = args.end_temp + (args.max_outer_iter - outer_iter)/(args.max_outer_iter) * (1-args.end_temp)
                    print("[debug] temp", temp)
                else:
                    temp = 1
                for _i in range(len(new_theta)):
                    if args.use_sigmoid:
                    # if args.use_sigmoid and outer_iter==0.0:
                        new_theta_mapped[_i] = torch.sigmoid(new_theta[_i]/temp)
                    else:
                        new_theta_mapped[_i] = new_theta[_i]
            elif 'inheritModelAndSample' in args.fuse_dataset_weight:
                theta_values = model_importance.reshape(-1,1) * torch.stack(theta_mapped)
                new_theta_mapped = [theta_values[_i] for _i in range(theta_values.shape[0])]
            elif 'inheritModel' in args.fuse_dataset_weight:
                new_theta_mapped = [copy.deepcopy(_theta) for _theta in theta_mapped]
                for _i in range(args.len_LLM):
                    # (torch.full([len(train_data[_i])], args.init_theta, dtype=torch.float, device=device))
                    new_theta_mapped[_i] = torch.full([len(train_data[_i])], model_importance[_i], dtype=torch.float, device=device) #, requires_grad=True
            elif 'inheritSample' in args.fuse_dataset_weight:
                new_theta_mapped = [copy.deepcopy(_theta) for _theta in theta_mapped]

            print(f"[debug] separate high and low quality data with increasedTheta")
            # assert 0 == 1
            if 'increasedThetaFlip' in args.fuse_dataset_sample_selection:
                print(f'here1-1, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                # assert 1 == 0
                high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column, new_train_data, new_small_train_data, new_small_valid_data = train_to_converge_with_weight_adjust_and_label_flip_fused(args, args.fused_model, train_data, new_theta_mapped, current_outer_iter_trained_more_steps_model, (None, None), args.max_outer_iter, args.epoch_converge_fully_train, args.inner_obj, test_loader, outer_iter)
                # TODO: need a new trained backward data loader
                # print(f'here1-2, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                new_current_outer_iter_trained_model, new_current_outer_iter_trained_more_steps_model, new_best_theta, model_importance, new_theta_mapped = train_separate_models(args, model, new_train_data, new_small_train_data, new_small_valid_data, train_loader_backward, valid_loader, test_loader, outer_iter)
                # print(f'here1-3, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
            elif 'increasedTheta' in args.fuse_dataset_sample_selection:
                high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column = train_to_converge_with_weight_adjust_and_selection_fused(args, args.fused_model, train_data, new_theta_mapped, (None, None), args.max_outer_iter, args.epoch_converge_fully_train, args.inner_obj, test_loader, outer_iter)
            torch.save((high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column), f"{args.result_file_path}/iter{outer_iter}_quality_select.pth")
            # generate kd label awaring to sample qualtiy
            if args.kd_aggregate_weight == 'Equal' or args.kd_aggregate_weight == 'Entropy':
                model_sample = torch.tensor(args.sample_each_llm).to(args.device)
                model_importance = model_sample / torch.sum(model_sample)
            elif args.kd_aggregate_weight == 'Model':
                model_importance = model_total_acc / torch.sum(model_total_acc)
            elif args.kd_aggregate_weight == 'EqualModel':
                model_sample = torch.tensor(args.sample_each_llm).to(args.device)
                model_importance = model_total_acc * model_sample
                model_importance = model_total_acc / torch.sum(model_total_acc)
            else:
                assert args.kd_aggregate_weight == 'Equal', f"Not supported KD label aggregation weight: '{args.kd_aggregate_weight}'"
            if args.kd_aggregate_weight == 'Entropy':
                if 'Flip' in args.fuse_dataset_sample_selection:
                    # print(f'here0-1, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                    kd_labeled_data, logits_per_sample, loss_per_sample, error_per_sample, logits_per_sample_ns, loss_per_sample_ns, error_per_sample_ns, logits_per_sample_s, loss_per_sample_s, error_per_sample_s = kd_label_entropy_aware(args, new_current_outer_iter_trained_more_steps_model, new_train_data, model_importance, high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column)
                    # print(f'here0-2, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                    trained_model, trained_theta = train_to_converge_with_selection_kd_fused(args, args.fused_model, new_train_data, new_theta_mapped, (high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column), (logits_per_sample, logits_per_sample_ns, logits_per_sample_s), args.max_outer_iter, args.epoch_converge_fully_train, args.inner_obj, train_loader_backward, valid_loader, test_loader, outer_iter, final_test=(not perform_few_shot_gen))
                    # print(f'here0-3, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                else:
                    kd_labeled_data, logits_per_sample, loss_per_sample, error_per_sample, logits_per_sample_ns, loss_per_sample_ns, error_per_sample_ns, logits_per_sample_s, loss_per_sample_s, error_per_sample_s = kd_label_entropy_aware(args, current_outer_iter_trained_more_steps_model, train_data, model_importance, high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column)
                    trained_model, trained_theta = train_to_converge_with_selection_kd_fused(args, args.fused_model, train_data, new_theta_mapped, (high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column), (logits_per_sample, logits_per_sample_ns, logits_per_sample_s), args.max_outer_iter, args.epoch_converge_fully_train, args.inner_obj, train_loader_backward, valid_loader, test_loader, outer_iter, final_test=(not perform_few_shot_gen))
            else:
                assert args.kd_aggregate_weight == 'Entropy', 'Implementation Error, currently supporting only Entropy based method'
                if 'Flip' in args.fuse_dataset_sample_selection:
                    kd_labled_data, logits_per_sample, loss_per_sample, error_per_sample = kd_label_aware(args, new_current_outer_iter_trained_more_steps_model, new_train_data, model_importance, high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column)
                else:
                    kd_labled_data, logits_per_sample, loss_per_sample, error_per_sample = kd_label_aware(args, current_outer_iter_trained_more_steps_model, train_data, model_importance, high_qualtiy_sample_row, high_quality_sample_column, low_qualtiy_sample_row, low_quality_sample_column)
        ######### train a new llm with new data #########

        ############################ get a overall dataset based based on weight decay (end) ############################

        if perform_few_shot_gen == True:
            print(f'here0-5, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
            # TODO: select sample with ambiguous and easy to learn
            importance_score = [None] * args.len_LLM
            confidence_score = [None] * args.len_LLM
            variability_score = [None] * args.len_LLM
            # ########################### calculate influence score for top ambiguous and top easy-to-learn ###########################
            if 'Flip' in args.fuse_dataset_sample_selection:
                new_total_valid_data = merge_all_dataset(args, new_small_valid_data)
                for im in range(args.len_LLM):
                    confidence_score[im], variability_score[im] = run_divergence_calculation(args, new_current_outer_iter_trained_more_steps_model, new_small_train_data[im])
            else:
                total_valid_data = merge_all_dataset(args, small_valid_data)
                for im in range(args.len_LLM):
                    confidence_score[im], variability_score[im] = run_divergence_calculation(args, current_outer_iter_trained_more_steps_model, small_train_data[im])
            print(f"{confidence_score=}, {variability_score=}")
            top_ambiguous_easy_to_learn_idx = sample_dynamic_selection(confidence_score, variability_score, args.gen_few_shot_k, args.gen_few_shot_pool_size, ambiguous_ratio=args.gen_few_shot_ambiguous_ratio, is_random=(args.gen_sample_select.replace('influence','')))
            logging.info(f"#ambiguous & easy-to-learn samples of each PLM is {[len(top_ambiguous_easy_to_learn_idx[im]) for im in range(args.len_LLM)]}")
            print(f'here0-6(1), {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
            if 'influence' in args.gen_sample_select:
                if 'Flip' in args.fuse_dataset_sample_selection:
                    for im in range(args.len_LLM):
                        if len(top_ambiguous_easy_to_learn_idx[im]) == 0:
                            importance_score[im] = []
                            continue
                        if args.small_model_name.upper() == 'LSTM':
                            selected_train_dataset = []
                            for column in top_ambiguous_easy_to_learn_idx[im]:
                                selected_train_dataset.append(copy.deepcopy(new_small_train_data[im][column]))
                            selected_train_data = data.Dataset(selected_train_dataset, new_small_train_data[im].fields)
                        elif 'bert' in args.small_model_name.lower():
                            selected_train_data = TokenizedDataset(
                                file_path=(''),
                            )
                            selected_train_data.text = [] # clear all the samples
                            selected_train_data.ids = [] # clear all the samples
                            selected_train_data.attention_mask = [] # clear all the samples
                            selected_train_data.label = [] # clear all the samples
                            selected_train_data.idx = [] # clear all the samples
                            for column in top_ambiguous_easy_to_learn_idx[im]:
                                selected_train_data.text += [new_small_train_data[im].text[column]]
                                selected_train_data.ids += [new_small_train_data[im].ids[column]]
                                selected_train_data.attention_mask += [new_small_train_data[im].attention_mask[column]]
                                selected_train_data.label += [new_small_train_data[im].label[column]]
                                selected_train_data.idx += [new_small_train_data[im].idx[column]]
                        importance_score[im] = run_full_influence_functions(args, trained_model, new_total_valid_data, selected_train_data, num_examples_to_test=len(new_total_valid_data), s_test_num_samples=100)
                else:
                    for im in range(args.len_LLM):
                        if len(top_ambiguous_easy_to_learn_idx[im]) == 0:
                            importance_score[im] = []
                            continue
                        if args.small_model_name.upper() == 'LSTM':
                            selected_train_dataset = []
                            for column in top_ambiguous_easy_to_learn_idx[im]:
                                selected_train_dataset.append(copy.deepcopy(small_train_data[im][column]))
                            selected_train_data = data.Dataset(selected_train_dataset, small_train_data[im].fields)
                        elif 'bert' in args.small_model_name.lower():
                            selected_train_data = TokenizedDataset(
                                file_path=(''),
                            )
                            selected_train_data.text = [] # clear all the samples
                            selected_train_data.ids = [] # clear all the samples
                            selected_train_data.attention_mask = [] # clear all the samples
                            selected_train_data.label = [] # clear all the samples
                            selected_train_data.idx = [] # clear all the samples
                            for column in top_ambiguous_easy_to_learn_idx[im]:
                                selected_train_data.text += [small_train_data[im].text[column]]
                                selected_train_data.ids += [small_train_data[im].ids[column]]
                                selected_train_data.attention_mask += [small_train_data[im].attention_mask[column]]
                                selected_train_data.label += [small_train_data[im].label[column]]
                                selected_train_data.idx += [small_train_data[im].idx[column]]
                        importance_score[im] = run_full_influence_functions(args, trained_model, total_valid_data, selected_train_data, num_examples_to_test=len(total_valid_data), s_test_num_samples=100) 
                print(f'here0-6(2), {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                print(f"{importance_score=}, {top_ambiguous_easy_to_learn_idx=}")
                print(f"{len(importance_score[im])}, {len(top_ambiguous_easy_to_learn_idx[im])}")
                prompt_samples_idx = importance_selection_among_good_dynamic_samples(importance_score, top_ambiguous_easy_to_learn_idx, args.gen_few_shot_k)
            else:
                flat_top_ambiguous_easy_to_learn_idx = [x for xs in top_ambiguous_easy_to_learn_idx for x in xs]
                model_idx_list = []
                for _im in range(args.len_LLM):
                    for _ in range(len(top_ambiguous_easy_to_learn_idx[_im])):
                        model_idx_list.append(_im)
                random_sample_pos = random.sample(range(len(flat_top_ambiguous_easy_to_learn_idx)),args.gen_few_shot_k)
                prompt_samples_idx = [(model_idx_list[_g], flat_top_ambiguous_easy_to_learn_idx[_g]) for _g in random_sample_pos]
            print(f"{prompt_samples_idx=}")
            # prompt_samples_idx = [(1,0),(1,1),(0,2),(0,3)]
            
            for im in range(args.len_LLM):
                gen_task_file_dir = f'{args.working_prompt_dir[im]}{args.i_step}/'
                if not os.path.exists(gen_task_file_dir):
                    os.makedirs(gen_task_file_dir)
                args.gen_task_file = f'{gen_task_file_dir}task.json' # "A json file providing the instructions and other information required for dataset generation. "
                args.gen_output_dir = args.working_sample_dir[im] # "The output directory to which the generated dataset is saved"
                args.gen_model_name = args.llms[im] # "The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported."
                args.gen_num_entries_per_input = args.num_use_samples_each_step_extend[im]

                # prepare few-shot prompt
                few_shot_samples = ''
                for i_sample in range(args.gen_few_shot_k):
                    print(f"{i_sample=}, {prompt_samples_idx[i_sample][0]=}, {prompt_samples_idx[i_sample][1]=}")
                    print(f"prompt sample = {args.samples_text[prompt_samples_idx[i_sample][0]][prompt_samples_idx[i_sample][1]]}")
                    few_shot_samples += f'{FEW_SHOT_SAMPLE_TEMPLATE[args.task_name]}{args.samples_text[prompt_samples_idx[i_sample][0]][prompt_samples_idx[i_sample][1]]}\n'
                prompt = FEW_SHOT_PROMPT[args.task_name]
                for key in prompt["labels"].keys():
                    prompt["labels"][key]["instruction"] = prompt["labels"][key]["instruction"].format(few_shot_samples, few_shot_samples)
                with open(args.gen_task_file, "w") as task_file:
                    json.dump(prompt, task_file)
                # print(f"[debug] see the prompt \n*****\n{prompt}\n*****")
                torch.cuda.empty_cache()
                print(f'here0-7, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
                
                # gen_data and write data
                gen_syn_data_few_shot(args)
                torch.cuda.empty_cache()
            # do not perform things now, wait for gen to target number of samples
            return


    print("++++++++++++++++finished solving++++++++++++++++++++")
    return best_theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='summary generator')
    parser.add_argument('--seed', type=int, default=12345, metavar='seed', help='random seed (default: 0)')
    parser.add_argument('--method', type=str, default="probability_1step")
    parser.add_argument('--limit', default=1000, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--backward_batch_size', default=64, type=int)
    parser.add_argument('--outer_lr', default=5e-2, type=float)
    parser.add_argument('--inner_lr', default=1e-4, type=float)
    parser.add_argument('--div_tol', default=9, type=float)
    parser.add_argument('--outer_ratio', default=0.1, type=float)
    parser.add_argument('--theta_upper_lim', default=100, type=float)
    parser.add_argument('--outer_threshold', default=0, type=float)
    parser.add_argument('--max_outer_iter', default=250, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--sample_times', default=1, type=int)
    parser.add_argument('--inter_dim', default=32, type=int)
    parser.add_argument('--runs_name', default="ours", type=str)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--gold_data_path', default=None, type=str)
    parser.add_argument('--syn_data_path', default='data_new/', type=str)
    parser.add_argument('--llms', default=['gpt2-xl','llama-2-7b-chat-hf'], nargs='+', type=str)
    # parser.add_argument('--llm_1', default=None, type=str)
    # parser.add_argument('--llm_2', default=None, type=str)
    parser.add_argument('--outer_obj', default="combined", type=str)
    parser.add_argument('--inner_obj', default="ce", type=str)
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--task_name', default="rte", type=str)
    # parser.add_argument('--num_use_samples_inner', default=1000, type=int)
    parser.add_argument('--num_use_samples_inner', default=[200000,200000], nargs='+', type=int)
    # parser.add_argument('--num_use_samples_inner_1', default=1000, type=int)
    # parser.add_argument('--num_use_samples_inner_2', default=1000, type=int)
    parser.add_argument('--num_use_samples_outer', default=1000, type=int)
    parser.add_argument('--init_label', default=10, type=int)
    parser.add_argument('--init_theta', default=1, type=float)
    parser.add_argument('--epoch_converge', default=20, type=int)
    parser.add_argument('--epoch_converge_fully_train', default=5, type=int)
    parser.add_argument('--check_ft_every', default=10, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument("--iterative", default=False, action="store_true")
    parser.add_argument("--mean_grad", default=False, action="store_true")
    parser.add_argument("--use_test", default=False, action="store_true")
    parser.add_argument("--shuffle_train", default=False, action="store_true")
    parser.add_argument("--use_sigmoid", default=False, action="store_true")
    parser.add_argument("--hard", default=False, action="store_true")
    parser.add_argument("--use_dev_outer", default=False, action="store_true")
    parser.add_argument('--clip_constant', default=3, type=float)
    parser.add_argument('--end_temp', default=0.1, type=float) # temperature
    parser.add_argument('--wandb', default=False, action="store_true")
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--subset_outer', action="store_true")
    parser.add_argument('--stochastic_outer', action="store_true")
    parser.add_argument('--disable_outer_scheduler', action="store_true")
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--temp_anneal', action="store_true") # temperature schedule

    parser.add_argument('--gpu', default=0, type=int, help='gpu device id')
    parser.add_argument("--small_model_name", type=str, default='distilbert-base-uncased', help="The small Transformer language model to use.")
    parser.add_argument('--mix', action="store_true")
    parser.add_argument('--query_percent', default=0.2, type=float, help='make some of the training samples non-trainable in the next iteraton according to query result')
    parser.add_argument('--drop_percent', default=0.2, type=float, help='make some of the training samples non-trainable in the next iteraton')
    
    parser.add_argument("--test_ratio", type=float, default=0.1, help="separate a ratio of syn-dataset for model importance estimation")
    parser.add_argument('--cross_check_every', default=1, type=int, help='query the other local small models for item evaluation')
    parser.add_argument('--BETA', default=0.9, type=float, help='weight adjust ratio')
    parser.add_argument("--kd_slm", default=1, type=int, help='whether the trained small models are used to fuse a ')
    parser.add_argument("--kd_aggregate_weight", default='same', type=str, help="which weight is used during the soft-label aggregation of kd label, ['Equal' (based on #sample), 'Model' (based on model_importance), 'EqualModel' (based on #sample*model_importance)]")
    parser.add_argument("--kd_alpha", default=0, type=float, help="weight in the kd loss for Loss(p_student, y) and (1-args.kd_alpha) for Loss(p_student, p_teacher)")
    parser.add_argument("--kd_temperature", default=1, type=float, help="temperature for kd distillation, 1 is ok for 3 party, <1 should be used for 6 party")
    parser.add_argument("--fuse_dataset", default=0, type=int, help='whether the dataset is fused together to train a new LLM')
    parser.add_argument("--weight_adjust_criterial", type=str, default='none', help="['none', 'error', 'loss']")
    parser.add_argument("--fuse_dataset_portion", type=str, default='imbalance', help="['imbalance', 'balance']")
    parser.add_argument("--fuse_dataset_sample_selection", type=str, default='random', help="['random', 'error', 'loss', 'all', 'halfTheta', 'largerTheta', 'increasedTheta']")
    parser.add_argument("--fuse_dataset_weight", type=str, default='new', help="['new'(new theta that are equal for all without adjust), 'inheritModelAndSample', 'inheritModel', 'inheritSample', 'Adjust', 'inheritModelAndSampleAdjust', 'inheritModelAdjust', 'inheritSampleAdjust']")

    parser.add_argument("--max_input_length", default=512, type=int, help='max input token length for Language Model')
    parser.add_argument("--steps", type=int, default=1, help="how much steps for constructing the total dataset from all LLMs")
    parser.add_argument('--num_use_samples_init', default=[2000,2000], nargs='+', type=int)
    parser.add_argument("--gen_few_shot_k", type=int, default=8, help="how much few shot samples are provided for each few shot prompt")
    parser.add_argument("--gen_few_shot_pool_size", type=int, default=40, help="how much candidate few shot samples are selected for each few shot prompt")
    parser.add_argument("--gen_few_shot_ambiguous_ratio", type=float, default=0.5, help="the ratio of ambiguous samples in the candidate pool")
    parser.add_argument("--gen_batch_size", type=int, default=4, help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--gen_max_length", type=int, default=40, help="The maximum output length for each generated text.")
    parser.add_argument("--gen_min_length", type=int, default=1, help="The minimum output length for each generated text.")
    parser.add_argument("--gen_sample_select", type=str, default='influenceCartography', help="['influenceCartography','influenceEasy','influenceAmbiguous','influence','Cartography','Easy','Ambiguous','random']") #,'influenceCartography'


    # print(f'here1, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

    args = parser.parse_args()
    if 'TestAll' in args.weight_adjust_criterial:
        args.fuse_dataset_sample_selection = 'all'
        print(args.fuse_dataset_sample_selection)
    if args.syn_data_path != None:
        SYN_DATA_PATH = args.syn_data_path 
    print(f"llm models: {args.llms}, inner samples: {args.num_use_samples_inner}")
    if args.mix:
        # args.llms = ['gpt2-xl__llama-2-7b-chat-hf']
        temp = args.llms[0]
        for _t in args.llms[1:]:
            temp += f'__{_t}'
        args.llms = [temp]
        # args.num_use_samples_inner = [20000]
        args.separate_num_use_samples_inner = args.num_use_samples_inner
        temp = int(args.num_use_samples_inner[0])
        for _t in args.num_use_samples_inner[1:]:
            temp += int(_t)
        args.num_use_samples_inner = [temp]
        # args.separate_num_use_samples_inner = ['10000_10000']
        temp = str(args.separate_num_use_samples_inner[0])
        for _t in args.separate_num_use_samples_inner[1:]:
            temp += f'_{str(_t)}'
        args.separate_num_use_samples_inner = [temp]
        print(args.llms, args.num_use_samples_inner, args.separate_num_use_samples_inner)
    args.len_LLM = len(args.llms)
    assert args.len_LLM == len(args.num_use_samples_inner), "Must specify the number of inner samples used for every LLM's generated data"
    args.samples_text = [[]]*args.len_LLM # store the original samples in string format

    if args.BETA == 0.0:
        args.adjustable_beta = True
        args.BETA = 1 + np.sqrt(2*np.log(args.num_use_samples_inner[0]/args.max_outer_iter))
        print(f"BETA change with #data, BETA={args.BETA}")
    else:
        args.adjustable_beta = False
    
    args.train_ratio = 1.0 - args.test_ratio
    if 'Easy' in args.gen_sample_select:
        args.gen_sample_select.replace('Easy','Cartography')
        args.gen_few_shot_ambiguous_ratio = 0.0
    elif 'Ambiguous' in args.gen_sample_select:
        args.gen_sample_select.replace('Ambiguous','Cartography')
        args.gen_few_shot_ambiguous_ratio = 0.0
    
    randnum = random.random()
    torch.cuda.set_device(args.gpu)  
    args.device = device
    torch.cuda.empty_cache()

    if args.wandb:
        os.system("wandb login --relogin xxxxxx")
        wandb.init(project=f"generative_data_training",config={'llms':args.llms, 'samples':args.num_use_samples_inner, 'epochs':args.max_outer_iter, 'train_batch_size':args.train_batch_size, 'backward_batch_size':args.backward_batch_size, 'outer_lr':args.outer_lr})

    set_seed(args.seed)

    print(f"learning rate: {args.inner_lr}")
    print(f"seed: {args.seed}")

    # prepare STM that does not depends on the dataset vocabulary here
    if 'bert' in args.small_model_name.lower():
        # print(f"small model is {args.small_model_name}")
        # init_model = BertForSequenceClassification.from_pretrained(MODEL_PATH[args.small_model_name], num_labels=args.num_classes)
        print(f"load tokenizer for bert first")
        args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[args.small_model_name])
        # print(f"{init_model.parameters()=}, {init_model.bert.parameters()=}")
        # # for param in init_model.base_model.parameters():
        # for param in init_model.bert.parameters():
        #     param.requires_grad = False
    # print(f'here2, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

    loss_func = nn.CrossEntropyLoss()
    # args.save_path=os.path.join(args.syn_data_path,f'best_thetas_inner{args.num_use_samples_inner}_outter{args.num_use_samples_outer}')
    args.model_name_sample = f'{args.task_name}/[mix]_{args.llms[0]}_{args.num_use_samples_inner[0]}' if args.mix else f'{args.task_name}/{args.llms[0]}_{args.num_use_samples_inner[0]}'
    for _model, num_samples_inner in zip(args.llms[1:], args.num_use_samples_inner[1:]):
        args.model_name_sample += f'__{_model}_{num_samples_inner}'
    args.save_path=os.path.join(f'results/few_shot_accumulate_{SYN_DATA_PATH[:-1]}_{args.gen_sample_select}/init{args.num_use_samples_init[0]}_steps{args.steps}/{args.small_model_name}/{args.train_ratio}_{SYN_DATA_PATH}', args.model_name_sample)
    args.save_path = f"{args.save_path}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.max_outer_iter}_{args.seed}"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        handlers=[logging.FileHandler('my_log_file.log'), logging.StreamHandler()])
    log_format = '%(asctime)s [%(levelname)s]: %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    if not os.path.exists(f'./logging/eval_on_real/few_shot_accumulate_{args.gen_sample_select}/{args.small_model_name}/{args.train_ratio}_{args.weight_adjust_criterial}_{args.fuse_dataset_weight}_{args.fuse_dataset_sample_selection}_{args.kd_aggregate_weight}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/{args.kd_alpha}_{args.kd_temperature}_init{args.num_use_samples_init[0]}_steps{args.steps}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.model_name_sample}/'):
        os.makedirs(f'./logging/eval_on_real/few_shot_accumulate_{args.gen_sample_select}/{args.small_model_name}/{args.train_ratio}_{args.weight_adjust_criterial}_{args.fuse_dataset_weight}_{args.fuse_dataset_sample_selection}_{args.kd_aggregate_weight}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/{args.kd_alpha}_{args.kd_temperature}_init{args.num_use_samples_init[0]}_steps{args.steps}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.model_name_sample}/')
    fh = logging.FileHandler(os.path.join(f'./logging/eval_on_real/few_shot_accumulate_{args.gen_sample_select}/{args.small_model_name}/{args.train_ratio}_{args.weight_adjust_criterial}_{args.fuse_dataset_weight}_{args.fuse_dataset_sample_selection}_{args.kd_aggregate_weight}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/{args.kd_alpha}_{args.kd_temperature}_init{args.num_use_samples_init[0]}_steps{args.steps}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.model_name_sample}/', f'log_{SYN_DATA_PATH[:-1]}_{args.BETA}_{args.seed}.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    args.result_file_path = f'./results/eval_on_real/few_shot_accumulate_{args.gen_sample_select}/{args.small_model_name}/{args.train_ratio}_{args.weight_adjust_criterial}_{args.fuse_dataset_weight}_{args.fuse_dataset_sample_selection}_{args.kd_aggregate_weight}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/{args.kd_alpha}_{args.kd_temperature}_init{args.num_use_samples_init[0]}_steps{args.steps}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.model_name_sample}/{args.seed}/'
    if not os.path.exists(args.result_file_path):
        os.makedirs(args.result_file_path)

    if args.steps == 0:
        args.sample_each_llm = args.num_use_samples_inner
        args.sample_each_llm = torch.tensor(args.sample_each_llm).to(args.device)

        print('num of use syn samples:{}'.format(args.num_use_samples_inner))
        if 'bert' in args.small_model_name.lower():
            train_iter, small_train_iter, small_valid_iter, train_iter_backward, dev_iter, test_iter, train_data, small_train_data, small_valid_data, dev_data_all = load_iters(args, args.train_batch_size, args.backward_batch_size, device, args.gold_data_path, SYN_DATA_PATH, vectors, False, args.num_use_samples_inner, args.num_use_samples_outer,args.shuffle_train)
            args.num_classes = len(torch.unique(train_data[0].label))
        else: # lstm
            train_iter, small_train_iter, small_valid_iter, train_iter_backward, dev_iter, test_iter, TEXT, LABEL, train_data, small_train_data, small_valid_data, dev_data_all = load_iters(args, args.train_batch_size, args.backward_batch_size, device, args.gold_data_path, SYN_DATA_PATH, vectors, False, args.num_use_samples_inner, args.num_use_samples_outer,args.shuffle_train)
            args.num_classes = len(LABEL.vocab.stoi)
        print(f'num of lable {args.num_classes}')
        print(f'here0-3, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
        if args.use_test:
            dev_iter = test_iter

        args.accumulate_sampels = [0]
        for _i in range(args.len_LLM):
            args.accumulate_sampels.append(args.accumulate_sampels[-1]+args.num_use_samples_inner[_i]) 
        args.accumulate_sampels = torch.tensor(args.accumulate_sampels, dtype=torch.long).to(args.device)

        # initialize all the 'local' and 'global' model as the same model parameters
        if args.small_model_name == 'LSTM':
            print(f"small model is LSTM RNN")
            init_model = RNN(len(TEXT.vocab), len(LABEL.vocab.stoi),
                        EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM,
                        TEXT.vocab.vectors, freeze)
        elif 'bert' in args.small_model_name.lower():
            print(f"small model is {args.small_model_name}")
            init_model = BertForSequenceClassification.from_pretrained(MODEL_PATH[args.small_model_name], num_labels=args.num_classes)
            # args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[args.small_model_name])
            print(f"{init_model.parameters()=}, {init_model.bert.parameters()=}")
        else:
            print(f"small model is {args.small_model_name}")
            init_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH[args.small_model_name])
        print(f'The model has {count_parameters(init_model):,} trainable parameters')
        # model = [copy.deepcopy(init_model).to(device) for _ in range(args.len_LLM)]
        # args.fused_model = copy.deepcopy(init_model).to(device)
        model = [copy.deepcopy(init_model) for _ in range(args.len_LLM)]
        args.fused_model = copy.deepcopy(init_model)
        print(f'here0-4, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

        if 'bert' in args.small_model_name.lower():
            print(f'use {[len(train_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
            print(f'use {[len(small_train_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
            print(f'use {[len(small_valid_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
            # print(f'use {[len(train_data[i].idx) for i in range(args.len_LLM)]} in train_data')
            # print(f'use {[len(small_train_data[i].idx) for i in range(args.len_LLM)]} in small_train_data')
            print(f'use {len(dev_iter.dataset.idx)} dev data...')
            print(f'use {len(test_iter.dataset.idx)} test data...')
        elif args.small_model_name == 'LSTM':
            print(f'use {[len(train_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
            print(f'use {[len(small_train_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
            print(f'use {[len(small_valid_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
            # print(f'use {[len(train_data[i].examples) for i in range(args.len_LLM)]} in train_data')
            # print(f'use {[len(small_train_data[i].examples) for i in range(args.len_LLM)]} in small_train_data')
            print(f'use {len(dev_iter.dataset.examples)} dev data...')
            print(f'use {len(test_iter.dataset.examples)} test data...')
        else:
            print(f"specital model, what is it ? {args.small_model_name}")
        print(f'save path is {args.save_path}')
        # best_theta = solve(model, train_iter, train_iter_backward, dev_iter, test_iter)
        best_theta = solve_with_local_cross_validation(args, model, train_data, small_train_data, small_valid_data, train_iter_backward, dev_iter, test_iter)
        # best_theta = solve_with_model_importance_estimation(args, model, train_data, test_iter)
        torch.save(best_theta, f"{args.save_path}/best_thetas.pth")
        print(f"best thetas saved to {args.save_path}/best_thetas.pth")
        print(f"best thetas {best_theta}")
    
    else: # args.steps >= 1
        print(f"{args.steps} steps are taken to construct the total dataset")
        # get the schedular for each sample
        # total_samples = sum(args.num_use_samples_inner)
        args.num_use_samples_each_step_extend = [(args.num_use_samples_inner[im]-args.num_use_samples_init[im])//args.steps for im in range(args.len_LLM)]
        args.init_sample_path = []
        args.working_sample_dir = []
        # args.working_sample_path = []
        args.working_prompt_dir = []
        # for im in range(args.len_LLM):
        #     if os.path.exists(args.working_sample_dir[im]):

        for im in range(args.len_LLM):
            args.init_sample_path.append(f'data_accumulate_start/{args.task_name}/{args.llms[im]}/{args.num_use_samples_inner[im]}_{args.num_use_samples_init[im]}/train.jsonl')
            args.working_sample_dir.append(f'{SYN_DATA_PATH}{args.gen_sample_select}/{args.model_name_sample}/{args.small_model_name}/{args.fuse_dataset_sample_selection}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.seed}/{args.llms[im]}/{args.num_use_samples_inner[im]}_{args.num_use_samples_init[im]}_{args.num_use_samples_each_step_extend[im]}/')
            if not os.path.exists(args.working_sample_dir[im]):
                os.makedirs(args.working_sample_dir[im])
            # args.working_sample_path.append(f'{args.working_sample_dir[im]}train.jsonl')
            args.working_prompt_dir.append(f'{SYN_DATA_PATH}{args.gen_sample_select}/{args.model_name_sample}/{args.small_model_name}/{args.fuse_dataset_sample_selection}_KD{args.kd_slm}_FuseDataset{args.fuse_dataset}/fewshotK{args.gen_few_shot_k}_{args.gen_few_shot_pool_size}_{args.gen_few_shot_ambiguous_ratio}/{args.seed}/prompt/{args.llms[im]}/{args.num_use_samples_inner[im]}_{args.num_use_samples_init[im]}_{args.num_use_samples_each_step_extend[im]}/')
            for sample_file_name in ['train_noflip', 'train']: # save 2 files, one for the original generated samples (train_noflip), another for samples after flip
                prepare_sample_file(args.init_sample_path[im], f'{args.working_sample_dir[im]}{sample_file_name}.jsonl', args.num_use_samples_init[im])
        
        args.num_use_samples_init = torch.tensor(args.num_use_samples_init,dtype=torch.long).to(args.device)
        args.num_use_samples_each_step_extend = torch.tensor(args.num_use_samples_each_step_extend,dtype=torch.long).to(args.device)
        for i_step in range(args.steps+1):
            args.i_step = i_step
            args.sample_each_llm = args.num_use_samples_init + i_step * args.num_use_samples_each_step_extend
            args.sample_each_llm = torch.tensor(args.sample_each_llm).to(args.device)
            print('num of use syn samples in total: {}'.format(sum(args.num_use_samples_inner)))
            print(f'num of current syn samples for step {i_step}: {args.sample_each_llm}')

            if 'bert' in args.small_model_name.lower():
                train_iter, small_train_iter, small_valid_iter, train_iter_backward, dev_iter, test_iter, train_data, small_train_data, small_valid_data, dev_data_all = load_iters(args, args.train_batch_size, args.backward_batch_size, device, args.gold_data_path, SYN_DATA_PATH, vectors, False, args.sample_each_llm, args.num_use_samples_outer,args.shuffle_train)
                args.num_classes = len(torch.unique(train_data[0].label))
            else: # lstm
                train_iter, small_train_iter, small_valid_iter, train_iter_backward, dev_iter, test_iter, TEXT, LABEL, train_data, small_train_data, small_valid_data, dev_data_all = load_iters(args, args.train_batch_size, args.backward_batch_size, device, args.gold_data_path, SYN_DATA_PATH, vectors, False, args.sample_each_llm, args.num_use_samples_outer,args.shuffle_train)
                args.num_classes = len(LABEL.vocab.stoi)
            print(f'num of lable {args.num_classes}')
            # print(f'here0-3, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
            if args.use_test:
                dev_iter = test_iter

            args.accumulate_sampels = [0]
            for _i in range(args.len_LLM):
                args.accumulate_sampels.append(args.accumulate_sampels[-1]+args.sample_each_llm[_i]) 
            args.accumulate_sampels = torch.tensor(args.accumulate_sampels, dtype=torch.long).to(args.device)

            # initialize all the 'local' and 'global' model as the same model parameters
            if args.small_model_name == 'LSTM':
                print(f"small model is LSTM RNN")
                init_model = RNN(len(TEXT.vocab), len(LABEL.vocab.stoi),
                            EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM,
                            TEXT.vocab.vectors, freeze)
            elif 'bert' in args.small_model_name.lower(): # models that does not depend on the dataset vocabulary have already been created
                print(f"small model is {args.small_model_name}")
                init_model = BertForSequenceClassification.from_pretrained(MODEL_PATH[args.small_model_name], num_labels=args.num_classes)
                # args.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH[args.small_model_name])
                print(f"{init_model.parameters()=}, {init_model.bert.parameters()=}")
            else:
                print(f"small model is {args.small_model_name}")
                init_model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH[args.small_model_name])
            print(f'The model has {count_parameters(init_model):,} trainable parameters')
            model = [copy.deepcopy(init_model) for _ in range(args.len_LLM)]
            args.fused_model = copy.deepcopy(init_model)
            # print(f'here0-4, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

            if 'bert' in args.small_model_name.lower():
                print(f'use {[len(train_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
                print(f'use {[len(small_train_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
                print(f'use {[len(small_valid_iter[i].dataset.idx) for i in range(args.len_LLM)]} train data...')
                # print(f'use {[len(train_data[i].idx) for i in range(args.len_LLM)]} in train_data')
                # print(f'use {[len(small_train_data[i].idx) for i in range(args.len_LLM)]} in small_train_data')
                print(f'use {len(dev_iter.dataset.idx)} dev data...')
                print(f'use {len(test_iter.dataset.idx)} test data...')
            elif args.small_model_name == 'LSTM':
                print(f'use {[len(train_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
                print(f'use {[len(small_train_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
                print(f'use {[len(small_valid_iter[i].dataset.examples) for i in range(args.len_LLM)]} train data...')
                # print(f'use {[len(train_data[i].examples) for i in range(args.len_LLM)]} in train_data')
                # print(f'use {[len(small_train_data[i].examples) for i in range(args.len_LLM)]} in small_train_data')
                print(f'use {len(dev_iter.dataset.examples)} dev data...')
                print(f'use {len(test_iter.dataset.examples)} test data...')
            else:
                print(f"specital model, what is it ? {args.small_model_name}")
            print(f'save path is {args.save_path}')
            # best_theta = solve(model, train_iter, train_iter_backward, dev_iter, test_iter)
            best_theta = solve_with_local_cross_validation(args, model, train_data, small_train_data, small_valid_data, train_iter_backward, dev_iter, test_iter, perform_few_shot_gen=(not i_step==args.steps))
            # best_theta = solve_with_model_importance_estimation(args, model, train_data, test_iter)
            torch.save(best_theta, f"{args.save_path}/best_thetas.pth")
            print(f"best thetas saved to {args.save_path}/best_thetas.pth")
            print(f"best thetas {best_theta}")