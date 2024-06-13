# -*- coding:utf8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains some utility functions.
"""

import json
import jsonlines
import logging
import os
import random
import sys
from typing import List, Dict
from tqdm import tqdm
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from bilevel_tools.meta import MetaSGD
from bilevel_tools.tbtools import AverageMeter
import torch.nn.functional as F

from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
# from torchtext.data import Iterator, BucketIterator
# from torchtext import data

from utils.bert_dataset import *

def init_logging(log_file, stdout=False):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_jsonl(entries: List[Dict], path: str):
    with open(path, 'w', encoding='utf8') as fh:
        for entry in entries:
            fh.write(f'{json.dumps(entry)}\n')


def save_jsonl_append(entries: List[Dict], path: str):
    with open(path, 'a', encoding='utf8') as fh:
        for entry in entries:
            fh.write(f'{json.dumps(entry)}\n')


def read_jsonl(path: str) -> List[Dict]:
    pairs = []
    with open(path, 'r', encoding='utf8') as fh:
        for line in fh:
            pairs.append(json.loads(line))
    return pairs


def modify_idx_in_json_file(gen_path: str, save_path: str):
    obj_list = read_jsonl(gen_path)
    for i, obj in enumerate(obj_list):
        obj_list[i]['idx'] = i
    with jsonlines.open(save_path, 'w') as writer:
        for obj in obj_list:
            writer.write(obj)


def save_flipped_samples(args, new_data, file_name):
    for im in range(args.len_LLM):
        file_path = f'{args.working_sample_dir[im]}{file_name}.jsonl'
        samples = []
        with jsonlines.open(file_path,'r') as reader:
            for json_obj in reader:
                samples.append(json_obj)
                print(json_obj)
        print(f"{len(samples)=}")
        if args.small_model_name.upper() == 'LSTM':
            print(f"{len(new_data[im].examples)=}")
        elif 'bert' in args.small_model_name.lower():
            print(f"{len(new_data[im].idx)=}")
        with jsonlines.open(file_path, 'w') as writer:
            for _i, json_obj in enumerate(samples):
                if args.small_model_name.upper() == 'LSTM':
                    json_obj['Y'] = new_data[im][_i].label
                elif 'bert' in args.small_model_name.lower():
                    json_obj['Y'] = int(new_data[im].label[_i].item())
                writer.write(json_obj)
        print(f"Finish writing {len(samples)} flipped samples into {file_path}")


def prepare_sample_file(from_path, to_path, sample_count):
    counter = 0
    with jsonlines.open(from_path, 'r') as reader, jsonlines.open(to_path, 'w') as writer:
        for json_obj in reader:
            # Modify the JSON object
            modified_json_obj = json_obj
            modified_json_obj['idx'] = counter
            counter += 1
            # Write the modified JSON object to the output file
            writer.write(modified_json_obj)
            if counter == sample_count:
                break
    assert counter == sample_count, f"Error: Insufficient initial samples in file '{from_path}', need {sample_count}, but contains only {counter}"
    print(f"Finish copying original {counter} samples, with target of copying {sample_count} samples")


def merge_all_dataset(args, datasets, max_sample_count_for_total=100):
    if max_sample_count_for_total != -1:
        max_sample_count_for_each = max_sample_count_for_total // args.len_LLM
    else:
        max_sample_count_for_each = -1
    # ############### prepare total_data ###############
    # accumulate_sampels = [0]
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in datasets:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            random.shuffle(_dataset_examples)
            if max_sample_count_for_each != -1:
                print(f"{len(_dataset_examples)=}, use {min(len(_dataset_examples),max_sample_count_for_each)} samples")
                total_data += copy.deepcopy(_dataset_examples[:min(len(_dataset_examples),max_sample_count_for_each)])
            else:
                print(f"{len(_dataset_examples)=}, use {len(_dataset_examples)} samples")
                total_data += copy.deepcopy(_dataset_examples[:])
            # accumulate_sampels.append(accumulate_sampels[-1]+len(_dataset_examples)) 
        for _i in range(len(total_data)):
            total_data[_i].idx = _i
        total_dataset = data.Dataset(total_data, datasets[0].fields)
    elif 'bert' in args.small_model_name.lower():
        _id = 0
        total_dataset = TokenizedDataset(
            file_path=(''),
        )
        total_dataset.text = [] # clear all the samples
        total_dataset.ids = [] # clear all the samples
        total_dataset.attention_mask = [] # clear all the samples
        total_dataset.label = [] # clear all the samples
        total_dataset.idx = [] # clear all the samples
        for row in range(args.len_LLM):
            # accumulate_sampels.append(accumulate_sampels[-1]+len(datasets[row].idx))
            idx_list = [_i for _i in range(len(datasets[row].idx))]
            random.shuffle(idx_list)
            if max_sample_count_for_each != -1:
                idx_list = idx_list[:min(len(datasets[row].idx),max_sample_count_for_each)]
                print(f"{len(datasets[row].idx)=}, use {min(len(datasets[row].idx),max_sample_count_for_each)} samples")
            else:
                idx_list = idx_list[:]
                print(f"{len(datasets[row].idx)=}, use {len(datasets[row].idx)} samples")
            for column in idx_list:
                total_dataset.text += [datasets[row].text[column]]
                total_dataset.ids += [datasets[row].ids[column]]
                total_dataset.attention_mask += [datasets[row].attention_mask[column]]
                total_dataset.label += [datasets[row].label[column]]
                total_dataset.idx += [_id]
                _id += 1
    # accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)
    # ############### prepare total_data ###############
    return total_dataset
     

def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None, limit=100000):
    TEXT = data.Field(lower=True, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(batch_first=True)
    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL)}

    train_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='train.jsonl',
        test='test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )
    train_data = data.Dataset(train_data.examples[:limit], train_data.fields)
    dev_data = test_data
    print(f'using {len(train_data)} train data...')


    if vectors is not None:
        TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data, max_size=50000)
    LABEL.build_vocab(train_data.label)
    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, TEXT, LABEL


def eval_get_loss(args, model, data, use_soft_label=False):
    if args.small_model_name.upper() == 'LSTM':
        data_iter, = BucketIterator.splits(
            (data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        data_iter = DataLoader(data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
    loss_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    correctness_per_sample = torch.zeros((len(data),), dtype=torch.bool).to(args.device)
    prediction_per_sample = torch.zeros((len(data),), dtype=torch.long).to(args.device)
    model.to(args.device)
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            if args.small_model_name == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                output = model(inputs, lens)
                labels = batch.label
                idx = batch.idx
                # type(idx)=='list'
            else:
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
                # print(f"{inputs.shape=}, {attention_mask.shape=}, {labels.shape=}, {idx.shape=}")
                # print(f"{inputs[:3]=}, {attention_mask[:3]=}, {labels[:3]=}, {idx[:3]=}")
                # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits

            all_labels.append(labels)
            predicts = output.argmax(-1).reshape(-1)

            soft_max_output = torch.softmax(output,dim=-1)
            for _i, index in enumerate(idx):
                error_per_sample[index] = abs(1-soft_max_output[_i,labels[_i]])

            # loss = loss_func(output, labels)
            if args.inner_obj == "ce":
                loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                # if not args.normalize:
                #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                # else:
                #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
            elif args.inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss_per_sample[idx] = loss_vec
                # loss = torch.mean(loss_vec*theta[idx])
                loss = torch.mean(loss_vec)

            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            correctness_per_sample[idx] = (predicts == labels)
            prediction_per_sample[idx] = predicts
            err_num += (predicts != labels).sum().item()

    acc = correct_num / (correct_num + err_num)
    tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
    all_labels = torch.cat(all_labels)
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    # return acc, total_loss/len(data_iter)
    model.to("cpu")
    return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample


def eval_get_pred(args, model, data, use_soft_label=False):
    if args.small_model_name.upper() == 'LSTM':
        data_iter, = BucketIterator.splits(
            (data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        data_iter = DataLoader(data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
    loss_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    correctness_per_sample = torch.zeros((len(data),), dtype=torch.bool).to(args.device)
    prediction_per_sample = torch.zeros((len(data),), dtype=torch.long).to(args.device)
    logits_per_sample = torch.zeros((len(data),args.num_classes), dtype=torch.float32).to(args.device)
    model.to(args.device)
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            if args.small_model_name == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                output = model(inputs, lens)
                labels = batch.label
                idx = batch.idx
            else:
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
                # print(f"{inputs.shape=}, {attention_mask.shape=}, {labels.shape=}, {idx.shape=}")
                # print(f"{inputs[:3]=}, {attention_mask[:3]=}, {labels[:3]=}, {idx[:3]=}")
                # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits

            all_labels.append(labels)
            predicts = output.argmax(-1).reshape(-1)

            soft_max_output = torch.softmax(output,dim=-1)
            for _i, index in enumerate(idx):
                error_per_sample[index] = abs(1-soft_max_output[_i,labels[_i]])
                logits_per_sample[index] = soft_max_output[_i]

            # loss = loss_func(output, labels)
            if args.inner_obj == "ce":
                loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                # if not args.normalize:
                #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                # else:
                #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
            elif args.inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss_per_sample[idx] = loss_vec
                # loss = torch.mean(loss_vec*theta[idx])
                loss = torch.mean(loss_vec)

            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            correctness_per_sample[idx] = (predicts == labels)
            prediction_per_sample[idx] = predicts
            err_num += (predicts != labels).sum().item()

    acc = correct_num / (correct_num + err_num)
    tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
    all_labels = torch.cat(all_labels)
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    model.to("cpu")
    # return acc, total_loss/len(data_iter)
    return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, logits_per_sample




def majority_voting_label(args, data_sample, mode_list, data_fields=None, origin_model_index=-1, use_weight=False):
    # pred_list = []
    # if args.small_model_name.upper() == 'LSTM':
    #     total_data = data.Dataset([data_sample], data_fields)
    #     assert len(total_data) == 1, f"{len(total_data)}, should be only 1 sample"
    #     data_iter, = BucketIterator.splits(
    #         (total_data,),
    #         batch_sizes=(len(total_data),),
    #         device=args.device,
    #         sort_key=lambda x: len(x.text),
    #         sort_within_batch=True,
    #         repeat=False,
    #         shuffle=False,
    #     )
    #     for _j, model in enumerate(mode_list):
    #         if _j == origin_model_index:
    #             continue
    #         model.to(args.device)
    #         model.eval()
    #         with torch.no_grad():
    #             for i, batch in enumerate(data_iter):
    #                 (inputs, lens), labels = batch.text, batch.label
    #                 output = model(inputs, lens)
    #                 # type(idx)=='list'
    #                 # all_labels.append(labels)
    #                 predicts = output.argmax(-1).reshape(-1)
    #                 # print("predicts", predicts)
    #                 pred_list.append(predicts[0].item())
    # elif 'bert' in args.small_model_name.lower():
    #     for _j, model in enumerate(mode_list):
    #         if _j == origin_model_index:
    #             continue
    #         model.to(args.device)
    #         model.eval()
    #         with torch.no_grad():
    #             # print(f"{data_sample=}")
    #             inputs, attention_mask, labels, idx = data_sample
    #             inputs = torch.unsqueeze(inputs, 0).to(args.device)
    #             attention_mask = torch.unsqueeze(attention_mask, 0).to(args.device)
    #             labels = torch.unsqueeze(labels, 0).to(args.device)
    #             idx = torch.unsqueeze(idx, 0).to(args.device)
    #             output = model(inputs, attention_mask=attention_mask, labels=labels).logits
    #             # all_labels.append(labels)
    #             predicts = output.argmax(-1).reshape(-1)
    #             # print("predicts", predicts)
    #             pred_list.append(predicts[0].item())
    # # print(pred_list)
    # majority_voting_result = max(pred_list, key=pred_list.count)
    # # print("majority_voting_result", majority_voting_result, labels, data_sample.label, type(data_sample.label))
    
    voting_tabel = [0.0] * args.num_classes
    if args.small_model_name.upper() == 'LSTM':
        total_data = data.Dataset([data_sample], data_fields)
        assert len(total_data) == 1, f"{len(total_data)}, should be only 1 sample"
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(len(total_data),),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
        for _j, model in enumerate(mode_list):
            if _j == origin_model_index:
                continue
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_iter):
                    (inputs, lens), labels = batch.text, batch.label
                    output = model(inputs, lens)
                    # type(idx)=='list'
                    # all_labels.append(labels)
                    predicts = output.argmax(-1).reshape(-1)
                    if sum(output[0])!=1.0:
                        logits = torch.nn.functional.softmax(output[0], dim=-1)
                    else:
                        logits = output[0]
                    print(f"model #{_j}, predicts", predicts, f"{logits=}", f"entropy={1-(-torch.sum(logits * torch.log2(logits)))}")
                    voting_tabel[predicts[0].item()] += 1-(-torch.sum(logits * torch.log2(logits)))
                    if _j == 0:
                        print(f"add additional: {labels[0].item()=}")
                        voting_tabel[labels[0].item()] += 1
    elif 'bert' in args.small_model_name.lower():
        for _j, model in enumerate(mode_list):
            if _j == origin_model_index:
                continue
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                # print(f"{data_sample=}")
                inputs, attention_mask, labels, idx = data_sample
                inputs = torch.unsqueeze(inputs, 0).to(args.device)
                attention_mask = torch.unsqueeze(attention_mask, 0).to(args.device)
                labels = torch.unsqueeze(labels, 0).to(args.device)
                idx = torch.unsqueeze(idx, 0).to(args.device)
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                logits = torch.nn.functional.softmax(output[0], dim=-1)
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                # print("predicts", predicts, f"{voting_tabel=}")
                # print("predicts", predicts[0].item(), f"{logits=}, entropy={(-torch.sum(logits * torch.log2(logits)))}")
                voting_tabel[predicts[0].item()] += 1-(-torch.sum(logits * torch.log2(logits)))
                if _j == 0:
                    print(f"add additional: {labels[0].item()=}")
                    voting_tabel[labels[0].item()] += 1
    # print(pred_list)
    # majority_voting_result = max(pred_list, key=pred_list.count)
    majority_voting_result = torch.argmax(torch.tensor(voting_tabel)).item()
    # print("majority_voting_result", majority_voting_result, labels, voting_tabel)
    
    model.to("cpu")
    return majority_voting_result


def minimax_voting(args, data_sample, mode_list, data_fields=None, origin_model_index=-1, use_weight=False):
    
    # ballots):
    ballots = [[]] * args.len_LLM
    if args.small_model_name.upper() == 'LSTM':
        total_data = data.Dataset([data_sample], data_fields)
        assert len(total_data) == 1, f"{len(total_data)}, should be only 1 sample"
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(len(total_data),),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
        for _j, model in enumerate(mode_list):
            if _j == origin_model_index:
                continue
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_iter):
                    (inputs, lens), labels = batch.text, batch.label
                    output = model(inputs, lens)
                    # type(idx)=='list'
                    # all_labels.append(labels)
                    predicts = output.argmax(-1).reshape(-1)
                    if sum(output[0])!=1.0:
                        logits = torch.nn.functional.softmax(output[0], dim=-1)
                    else:
                        logits = output[0]
                    print(f"model #{_j}, predicts", predicts, f"{logits=}", f"entropy={1-(-torch.sum(logits * torch.log2(logits)))}")
                    ballots[_j] = logits.detach().cpu().tolist()
                    # if _j == 0:
                    #     print(f"add additional: {labels[0].item()=}")
                    #     ballots[labels[0].item()] += 1
    elif 'bert' in args.small_model_name.lower():
        for _j, model in enumerate(mode_list):
            if _j == origin_model_index:
                continue
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                # print(f"{data_sample=}")
                inputs, attention_mask, labels, idx = data_sample
                inputs = torch.unsqueeze(inputs, 0).to(args.device)
                attention_mask = torch.unsqueeze(attention_mask, 0).to(args.device)
                labels = torch.unsqueeze(labels, 0).to(args.device)
                idx = torch.unsqueeze(idx, 0).to(args.device)
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                logits = torch.nn.functional.softmax(output[0], dim=-1)
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                # print("predicts", predicts, f"{voting_tabel=}")
                # print("predicts", predicts[0].item(), f"{logits=}, entropy={(-torch.sum(logits * torch.log2(logits)))}")
                ballots[_j] = logits.detach().cpu().tolist()
                # voting_tabel[predicts[0].item()] += 1-(-torch.sum(logits * torch.log2(logits)))
                # if _j == 0:
                #     print(f"add additional: {labels[0].item()=}")
                #     voting_tabel[labels[0].item()] += 1
    model.to("cpu")

    # Count the number of candidates
    num_candidates = len(ballots[0])
    
    # Initialize pairwise preference matrix
    pairwise_preferences = defaultdict(int)
    
    # Populate the pairwise preference matrix
    for ballot in ballots:
        entropy = (1-(-np.sum(ballot * np.log2(ballot))))
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                if ballot[i] > ballot[j]:
                    pairwise_preferences[(i, j)] += entropy
                else:
                    pairwise_preferences[(j, i)] += entropy
    
    # Calculate the maximum margin of defeat for each candidate
    max_defeats = [0] * num_candidates
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                defeat_margin = pairwise_preferences[(i, j)] - pairwise_preferences[(j, i)]
                if defeat_margin > max_defeats[i]:
                    max_defeats[i] = defeat_margin
    
    # The Minimax winner is the candidate with the smallest maximum margin of defeat
    minimax_winner = min(range(num_candidates), key=lambda x: max_defeats[x])
    return minimax_winner


def eval_get_predicts(args, model, data_iter, name, epoch=None, use_soft_label=False):
    loss_func = nn.CrossEntropyLoss()
    model.to(args.device)
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    all_predicts = []
    all_predicts_votes = []
    all_labels = []
    if args.small_model_name.upper() == 'LSTM':
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                (inputs, lens), labels = batch.text, batch.label
                output = model(inputs, lens)
                if torch.sum(output[0]).item() != 1.0:
                    output = torch.nn.functional.softmax(output)
                all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                one_hot_preditcs = torch.eye(args.num_classes)[predicts].to(args.device)
                predict_voiting_weights = (-torch.sum(output * torch.log2(output), dim=-1)).to(args.device)
                # print(f"{one_hot_preditcs.shape=}, {predict_voiting_weights.shape=}")
                predict_voiting_weights = (-args.num_classes * ((1/args.num_classes)*torch.log2(torch.tensor(1/args.num_classes)))) - predict_voiting_weights
                # print(f"{predict_voiting_weights.shape=}")
                all_predicts_votes.append(one_hot_preditcs*predict_voiting_weights.unsqueeze(1))
                loss = loss_func(output, labels)
                total_loss += loss.item()
                correct_num += (predicts == labels).sum().item()
                err_num += (predicts != labels).sum().item()
                all_predicts.append(predicts)
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
                # print(f"{inputs[:3]=}, {attention_mask[:3]=}, {labels[:3]=}, {idx[:3]=}")

                # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                if torch.sum(output[0]).item() != 1.0:
                    output = torch.nn.functional.softmax(output)
                all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)
                one_hot_preditcs = torch.eye(args.num_classes)[predicts].to(args.device)
                predict_voiting_weights = (-torch.sum(output * torch.log2(output), dim=-1)).to(args.device)
                predict_voiting_weights = (-args.num_classes * ((1/args.num_classes)*torch.log2(torch.tensor(1/args.num_classes)))) - predict_voiting_weights
                all_predicts_votes.append(one_hot_preditcs*predict_voiting_weights.unsqueeze(1))
                loss = loss_func(output, labels)
                total_loss += loss.item()
                correct_num += (predicts == labels).sum().item()
                err_num += (predicts != labels).sum().item()
                all_predicts.append(predicts)
    
    acc = correct_num / (correct_num + err_num)
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, acc, total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, acc, total_loss))
    all_labels = torch.cat(all_labels)
    all_predicts = torch.cat(all_predicts)
    all_predicts_votes = torch.cat(all_predicts_votes)
    print(f"{all_predicts.shape=}, {all_predicts_votes.shape=}, {all_predicts_votes[:5]=}")
    model.to("cpu")
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    return acc, total_loss/len(data_iter), all_predicts, all_predicts_votes, all_labels


def run_divergence_calculation(args, models, dataset, use_soft_label=False):
    use_soft_label = False
    # loss_func = nn.CrossEntropyLoss()
    if args.small_model_name.upper() == 'LSTM':
        data_iter = BucketIterator.splits(
            (dataset,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
        data_iter = list(data_iter)[0]
    elif 'bert' in args.small_model_name.lower():
        data_iter = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)


    loss_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.float32).to(args.device)
    correctness_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.bool).to(args.device)
    prediction_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.long).to(args.device)
    logits_per_sample = torch.zeros((len(models), len(dataset), args.num_classes), dtype=torch.float32).to(args.device)
    # labels_per_sample = torch.zeros((len(data),), dtype=torch.long).to(args.device)
    for im, model in enumerate(models):
        model.to(args.device)
        model.eval()
        correct_num = 0
        err_num = 0
        total_loss = 0
        all_labels = []
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                if args.small_model_name == 'LSTM':
                    (inputs, lens), labels = batch.text, batch.label
                    output = model(inputs, lens)
                    labels = batch.label
                    idx = batch.idx
                else:
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
                    # print(f"{inputs.shape=}, {attention_mask.shape=}, {labels.shape=}, {idx.shape=}")
                    # print(f"{inputs[:3]=}, {attention_mask[:3]=}, {labels[:3]=}, {idx[:3]=}")
                    output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits

                all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample[im][index] = abs(1-soft_max_output[_i,labels[_i]])
                    logits_per_sample[im][index] = soft_max_output[_i]

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample[im][idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample[im][idx] = loss_vec
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
                correct_num += (predicts == labels).sum().item()
                correctness_per_sample[im][idx] = (predicts == labels)
                prediction_per_sample[im][idx] = predicts
                err_num += (predicts != labels).sum().item()

        acc = correct_num / (correct_num + err_num)
        tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
        all_labels = torch.cat(all_labels)
        print(f"num of zeros: {torch.sum(all_labels == 0)}")
        print(f"num of ones: {torch.sum(all_labels == 1)}")
        model.to("cpu")
        # return acc, total_loss/len(data_iter)
        # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, logits_per_sample

    # logits_per_sample (#models, #samples, #classes) with (#samples, #classes) is the list of logits (model outputs) of all the samples
    # all_labels is the real label of all samples
    confidence_per_sample = logits_per_sample[:,torch.arange(all_labels.size(0)), all_labels]
    variability_per_sample = torch.std(confidence_per_sample, dim=0)
    confidence_per_sample = torch.mean(confidence_per_sample, dim=0)
    return list(confidence_per_sample.detach().cpu().numpy()), list(variability_per_sample.detach().cpu().numpy())


def importance_dynamic_selection(influence_score, confidence_score, variability_score, count):
    # print(f"{influence_score=}， {influence_score[0]=}")
    model_sample_count = [len(influence_score[i]) for i in range(len(influence_score))]
    accumulated_sample_count = [0]
    influence_score_flat = []
    confidence_score_flat = []
    variability_score_flat = []
    for i in range(len(influence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
        influence_score_flat += influence_score[i]
        confidence_score_flat += confidence_score[i]
        variability_score_flat += variability_score[i]
    influence_score_flat = np.array(influence_score_flat)
    confidence_score_flat = np.array(confidence_score_flat)
    variability_score_flat = np.array(variability_score_flat)

    ascending_indices = list(np.argsort(influence_score_flat))
    descending_indices = ascending_indices[::-1]
    # print(f"{ascending_indices=}, {descending_indices=}")
    print(f"{accumulated_sample_count=}")

    variability_score_max = np.max(variability_score_flat)
    variability_score_min = np.max(variability_score_flat)
    variability_score_up_threshold = variability_score_max * 0.96
    variability_score_down_threshold = min(variability_score_min * 100, 0.02)
    
    selected_sample_idx_list = []
    _counter = 0
    for ic in range(len(descending_indices)):
        if variability_score_flat[descending_indices[ic]] < variability_score_up_threshold:
            continue
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= descending_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = descending_indices[ic] - accumulated_sample_count[model_idx]
        # print(f"{descending_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        selected_sample_idx_list.append((int(model_idx), int(sample_idx)))
        _counter += 1
        if _counter == count:
            break
    for ic in range(len(descending_indices)):
        if variability_score_flat[descending_indices[ic]] > variability_score_down_threshold:
            continue
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= descending_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = descending_indices[ic] - accumulated_sample_count[model_idx]
        # print(f"{descending_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        selected_sample_idx_list.append((int(model_idx), int(sample_idx)))
        _counter += 1
        if _counter == count*2:
            break
    
    random.shuffle(selected_sample_idx_list)

    return selected_sample_idx_list


def sample_dynamic_selection(confidence_score, variability_score, count, pool_size=40, ambiguous_ratio=0.5, is_random='Cartography'):
    top_ambiguous_easy_to_learn_idx = [[] for _ in range(len(confidence_score))]

    model_sample_count = [len(confidence_score[i]) for i in range(len(confidence_score))]
    accumulated_sample_count = [0]
    confidence_score_flat = []
    variability_score_flat = []
    for i in range(len(confidence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
        confidence_score_flat += confidence_score[i]
        variability_score_flat += variability_score[i]
    confidence_score_flat = np.array(confidence_score_flat)
    variability_score_flat = np.array(variability_score_flat)

    if is_random == 'random' or is_random == '': # 'random'.reaplace('influence','') and 'influence'.reaplace('influence','')
        selected_indices = random.sample(range(accumulated_sample_count[-1]), (count*pool_size))
    else:
        # variability_score_max = np.max(variability_score_flat)
        # variability_score_min = np.max(variability_score_flat)
        # variability_score_up_threshold = variability_score_max * 0.96
        # variability_score_down_threshold = min(variability_score_min * 100, 0.02)

        ascending_indices = list(np.argsort(variability_score_flat)) # from easy/hard-to-learn to ambiguous

        # if len(ascending_indices)//2 >= count:
        #     selected_indices = ascending_indices[:count] + ascending_indices[-count:]
        # if len(ascending_indices)//2 >= (count*pool_size)//2:
        #     selected_indices = ascending_indices[:int((count*pool_size)//2)] + ascending_indices[-int((count*pool_size)//2):]
        if len(ascending_indices) >= (count*pool_size):
            # if is_random == 'Easy':
            #     ambiguous_count = 0
            # elif is_random == 'Ambiguous':
            #     ambiguous_count = int(count*pool_size)
            # else:
            #     assert is_random == 'Cartography', f"[ERROR] cross-model cartography selection should be in ['Easy','Ambiguous','Cartography'], but have {is_random}"
            ambiguous_count = int((count*pool_size)*ambiguous_ratio)
            easy_to_learn_count = int(count*pool_size) - ambiguous_count # easy-or-hard-to-learn, as we do not integrate the label information
            assert ambiguous_count + easy_to_learn_count == (count*pool_size), f"[[SOS]] Why should ambigous+easy_to_learn!=total, {ambiguous_count}+{easy_to_learn_count}!={(count*pool_size)}"
            if ambiguous_count == 0:
                selected_indices = ascending_indices[:easy_to_learn_count]
            elif easy_to_learn_count == 0:
                selected_indices = ascending_indices[-ambiguous_count:]
            else:
                selected_indices = ascending_indices[:easy_to_learn_count] + ascending_indices[-ambiguous_count:]
        else:
            selected_indices = ascending_indices

    for ic in range(len(selected_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= selected_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = selected_indices[ic] - accumulated_sample_count[model_idx]
        print(f"{selected_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        top_ambiguous_easy_to_learn_idx[model_idx].append(int(sample_idx))
    
    return top_ambiguous_easy_to_learn_idx


def sample_dynamic_selection_v2(confidence_score, variability_score, count, pool_size=40, ambiguous_ratio=0.5, random_guess=0.5):
    top_ambiguous_easy_to_learn_idx = [[] for _ in range(len(confidence_score))]

    model_sample_count = [len(confidence_score[i]) for i in range(len(confidence_score))]
    accumulated_sample_count = [0]
    confidence_score_flat = []
    variability_score_flat = []
    for i in range(len(confidence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
        confidence_score_flat += confidence_score[i]
        variability_score_flat += variability_score[i]
    confidence_score_flat = np.array(confidence_score_flat)
    variability_score_flat = np.array(variability_score_flat)

    # variability_score_max = np.max(variability_score_flat)
    # variability_score_min = np.max(variability_score_flat)
    # variability_score_up_threshold = variability_score_max * 0.96
    # variability_score_down_threshold = min(variability_score_min * 100, 0.02)

    confidence_score_near_random_flat = np.abs(confidence_score_flat - random_guess)
    nearest_to_random = list(np.argsort(confidence_score_near_random_flat)) # from random_guess to very confident samples

    ascending_indices = list(np.argsort(variability_score_flat)) # from easy/hard-to-learn to ambiguous

    # if len(ascending_indices)//2 >= count:
    #     selected_indices = ascending_indices[:count] + ascending_indices[-count:]
    # if len(ascending_indices)//2 >= (count*pool_size)//2:
    #     selected_indices = ascending_indices[:int((count*pool_size)//2)] + ascending_indices[-int((count*pool_size)//2):]
    if len(ascending_indices) > (count*pool_size):
        ambiguous_count = int((count*pool_size)*ambiguous_ratio) # target of ambiguous sample count, but may not satisfy at the beginning
        selected_indices = []
        ambiguous_get_counter = 0
        for _is in range(accumulated_sample_count[-1]-1, -1 -1): # add variablity>=0.395 to ambiguous list
            if variability_score_flat[ascending_indices[_is]] >= random_guess * 0.79:
                selected_indices.append(ascending_indices[_is])
                print(f"[debug] adding top disagreed, idx={ascending_indices[_is]}, confidence={confidence_score_flat[ascending_indices[_is]]}, variability={variability_score_flat[ascending_indices[_is]]} ")
                ambiguous_get_counter += 1
                if ambiguous_get_counter == ambiguous_count:
                    break
        if ambiguous_get_counter < ambiguous_count: # add near-random with large enough
            for _is in range(0, accumulated_sample_count[-1]):
                if nearest_to_random[_is] in selected_indices:
                    continue
                if variability_score_flat[nearest_to_random[_is]] >= random_guess * 0.5:
                    selected_indices.append(nearest_to_random[_is])
                    print(f"[debug] adding top random, idx={nearest_to_random[_is]}, confidence={confidence_score_flat[nearest_to_random[_is]]}, variability={variability_score_flat[nearest_to_random[_is]]} ")
                    ambiguous_get_counter += 1
                    if ambiguous_get_counter == ambiguous_count:
                        break
        assert ambiguous_get_counter == len(selected_indices), f"[ERROR] the counter for ambiguous is incorrect, have {len(selected_indices)} samples, but counter={ambiguous_get_counter}"
        easy_to_learn_count = int(count*pool_size) - len(selected_indices) # easy-or-hard-to-learn, as we do not integrate the label information
        print(f"{easy_to_learn_count=}, {ambiguous_get_counter=}")
        assert ambiguous_get_counter + easy_to_learn_count == (count*pool_size), f"[[SOS]] Why should ambigous+easy_to_learn!=total, {ambiguous_get_counter}+{easy_to_learn_count}!={(count*pool_size)}"
        easy_to_learn_get_counter = 0
        for _is in range(0, accumulated_sample_count[-1]):
            if ascending_indices[_is] in selected_indices:
                continue
            selected_indices.append(ascending_indices[_is])
            print(f"[debug] adding easy-to-learn, idx={ascending_indices[_is]}, confidence={confidence_score_flat[ascending_indices[_is]]}, variability={variability_score_flat[ascending_indices[_is]]} ")
            easy_to_learn_get_counter += 1
            if easy_to_learn_get_counter == easy_to_learn_count:
                break
        # selected_indices = ascending_indices[:easy_to_learn_count] + selected_indices
    else:
        selected_indices = ascending_indices

    for ic in range(len(selected_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= selected_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = selected_indices[ic] - accumulated_sample_count[model_idx]
        print(f"{selected_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        top_ambiguous_easy_to_learn_idx[model_idx].append(int(sample_idx))
    
    return top_ambiguous_easy_to_learn_idx    


def sample_dynamic_selection_self(confidence_score, variability_score, count, pool_size=40, ambiguous_ratio=0.5):
    top_ambiguous_easy_to_learn_idx = [[] for _ in range(len(confidence_score))]

    model_sample_count = [len(confidence_score[i]) for i in range(len(confidence_score))]
    accumulated_sample_count = [0]
    # confidence_score_flat = []
    # variability_score_flat = []
    for i in range(len(confidence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
        # confidence_score_flat += confidence_score[i]
        # variability_score_flat += variability_score[i]
    # confidence_score_flat = np.array(confidence_score_flat)
    # variability_score_flat = np.array(variability_score_flat)

    ascending_indices = [list(np.argsort(np.array(variability_score[_i]))) for _i in range(len(confidence_score))] # from easy/hard-to-learn to ambiguous
    print(f"1 {ascending_indices=}")
    ascending_indices = [[idx+accumulated_sample_count[_i] for idx in ascending_indices[_i]] for _i in range(len(ascending_indices))]
    print(f"2 {ascending_indices=}")

    selected_indices = []
    for i_model in range(len(confidence_score)):
        if len(ascending_indices[i_model]) > (count*pool_size):
            ambiguous_count = int((count*pool_size)*ambiguous_ratio)
            easy_to_learn_count = int(count*pool_size) - ambiguous_count # easy-or-hard-to-learn, as we do not integrate the label information
            assert ambiguous_count + easy_to_learn_count == (count*pool_size), f"[[SOS]] Why should ambigous+easy_to_learn!=total, {ambiguous_count}+{easy_to_learn_count}!={(count*pool_size)}"
            selected_indices = selected_indices + ascending_indices[i_model][:easy_to_learn_count] + ascending_indices[i_model][-ambiguous_count:]
        else:
            selected_indices = selected_indices + ascending_indices[i_model]

    for ic in range(len(selected_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= selected_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = selected_indices[ic] - accumulated_sample_count[model_idx]
        print(f"{selected_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        top_ambiguous_easy_to_learn_idx[model_idx].append(int(sample_idx))
    
    return top_ambiguous_easy_to_learn_idx


def sample_dynamic_selection_v2_self(confidence_score, variability_score, count, pool_size=40, ambiguous_ratio=0.5, random_guess=0.5):
    top_ambiguous_easy_to_learn_idx = [[] for _ in range(len(confidence_score))]

    model_sample_count = [len(confidence_score[i]) for i in range(len(confidence_score))]
    accumulated_sample_count = [0]
    # confidence_score_flat = []
    # variability_score_flat = []
    for i in range(len(confidence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
    #     confidence_score_flat += confidence_score[i]
    #     variability_score_flat += variability_score[i]
    # confidence_score_flat = np.array(confidence_score_flat)
    # variability_score_flat = np.array(variability_score_flat)

    # variability_score_max = np.max(variability_score_flat)
    # variability_score_min = np.max(variability_score_flat)
    # variability_score_up_threshold = variability_score_max * 0.96
    # variability_score_down_threshold = min(variability_score_min * 100, 0.02)

    confidence_score_near_random = [np.abs(np.array(confidence_score[_i]) - random_guess) for _i in range(len(confidence_score))]
    nearest_to_random = [list(np.argsort(confidence_score_near_random[_i])) for _i in range(len(confidence_score))] # from random_guess to very confident samples
    ascending_indices = [list(np.argsort(np.array(variability_score[_i]))) for _i in range(len(confidence_score))] # from easy/hard-to-learn to ambiguous

    selected_indices = []
    for i_model in range(len(confidence_score)):
        if len(ascending_indices[i_model]) > (count*pool_size):
            ambiguous_count = int((count*pool_size)*ambiguous_ratio) # target of ambiguous sample count, but may not satisfy at the beginning
            ambiguous_get_counter = 0
            for _is in range(model_sample_count[i_model]-1, -1 -1): # add variablity>=0.395 to ambiguous list
                if variability_score[i_model][ascending_indices[i_model][_is]] >= random_guess * 0.795:
                    selected_indices.append(ascending_indices[i_model][_is]+accumulated_sample_count[i_model])
                    print(f"[debug] adding top disagreed, {i_model=}, idx={ascending_indices[i_model][_is]+accumulated_sample_count[i_model]}, confidence={confidence_score[i_model][ascending_indices[i_model][_is]]}, variability={variability_score[i_model][ascending_indices[i_model][_is]]} ")
                    ambiguous_get_counter += 1
                    if ambiguous_get_counter == ambiguous_count:
                        break
            if ambiguous_get_counter < ambiguous_count: # add near-random with large enough
                for _is in range(0, model_sample_count[i_model]):
                    if nearest_to_random[i_model][_is]+accumulated_sample_count[i_model] in selected_indices:
                        continue
                    if variability_score[i_model][nearest_to_random[i_model][_is]] >= random_guess * 0.5:
                        selected_indices.append(nearest_to_random[i_model][_is]+accumulated_sample_count[i_model])
                        print(f"[debug] adding top random, {i_model=}, idx={nearest_to_random[i_model][_is]+accumulated_sample_count[i_model]}, confidence={confidence_score[i_model][nearest_to_random[i_model][_is]]}, variability={variability_score[i_model][nearest_to_random[i_model][_is]]} ")
                        ambiguous_get_counter += 1
                        if ambiguous_get_counter == ambiguous_count:
                            break
            assert ambiguous_get_counter == len(selected_indices)-(count*pool_size)*i_model, f"[ERROR] the counter for ambiguous is incorrect, have {len(selected_indices)} samples, but counter={ambiguous_get_counter}"
            easy_to_learn_count = int(count*pool_size) - (len(selected_indices)-(count*pool_size)*i_model) # easy-or-hard-to-learn, as we do not integrate the label information
            print(f"{easy_to_learn_count=}, {ambiguous_get_counter=}")
            assert ambiguous_get_counter + easy_to_learn_count == (count*pool_size), f"[[SOS]] Why should ambigous+easy_to_learn!=total, {ambiguous_get_counter}+{easy_to_learn_count}!={(count*pool_size)}"
            easy_to_learn_get_counter = 0
            for _is in range(0, model_sample_count[i_model]):
                if ascending_indices[i_model][_is]+accumulated_sample_count[i_model] in selected_indices:
                    continue
                selected_indices.append(ascending_indices[i_model][_is]+accumulated_sample_count[i_model])
                print(f"[debug] adding easy-to-learn, {i_model=}, idx={ascending_indices[i_model][_is]+accumulated_sample_count[i_model]}, confidence={confidence_score[i_model][ascending_indices[i_model][_is]]}, variability={variability_score[i_model][ascending_indices[i_model][_is]]} ")
                easy_to_learn_get_counter += 1
                if easy_to_learn_get_counter == easy_to_learn_count:
                    break
            # selected_indices = ascending_indices[:easy_to_learn_count] + selected_indices
        else:
            selected_indices = selected_indices + [idx+accumulated_sample_count[i_model] for idx in ascending_indices[i_model]]

    for ic in range(len(selected_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= selected_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = selected_indices[ic] - accumulated_sample_count[model_idx]
        print(f"{selected_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        top_ambiguous_easy_to_learn_idx[model_idx].append(int(sample_idx))
    
    return top_ambiguous_easy_to_learn_idx


def sample_dynamic_selection_random(confidence_score, variability_score, count, pool_size=40, ambiguous_ratio=0.5, random_guess=0.5):
    top_ambiguous_easy_to_learn_idx = [[] for _ in range(len(confidence_score))]

    model_sample_count = [len(confidence_score[i]) for i in range(len(confidence_score))]
    accumulated_sample_count = [0]
    for i in range(len(confidence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
    
    selected_indices = list(np.random.choice(range(accumulated_sample_count[-1]), size=int(count*pool_size), replace=False))

    for ic in range(len(selected_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= selected_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = selected_indices[ic] - accumulated_sample_count[model_idx]
        print(f"random selection, {selected_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {model_idx=}, {sample_idx=}")
        top_ambiguous_easy_to_learn_idx[model_idx].append(int(sample_idx))
    
    return top_ambiguous_easy_to_learn_idx    


def importance_selection_among_good_dynamic_samples(influence_score, top_ambiguous_easy_to_learn_idx, count):
    # print(f"{influence_score=}， {influence_score[0]=}")
    print(f"{len(influence_score)=}, {len(top_ambiguous_easy_to_learn_idx)=}")
    for im in range(len(influence_score)):
        print(f"{len(influence_score[im])=}, {len(top_ambiguous_easy_to_learn_idx[im])=}")
    
    model_sample_count = [len(influence_score[i]) for i in range(len(influence_score))]
    accumulated_sample_count = [0]
    influence_score_flat = []
    for i in range(len(influence_score)):
        accumulated_sample_count.append(accumulated_sample_count[-1]+model_sample_count[i])
        influence_score_flat += influence_score[i]
    influence_score_flat = np.array(influence_score_flat)

    ascending_indices = list(np.argsort(influence_score_flat))
    descending_indices = ascending_indices[::-1]
    # print(f"{ascending_indices=}, {descending_indices=}")
    print(f"{accumulated_sample_count=}")
    
    selected_sample_idx_list = []
    _counter = 0
    for ic in range(len(descending_indices)):
        for im in range(len(accumulated_sample_count)-1):
            if accumulated_sample_count[im] <= descending_indices[ic] < accumulated_sample_count[im+1]:
                model_idx = im
                break
        sample_idx = top_ambiguous_easy_to_learn_idx[model_idx][descending_indices[ic] - accumulated_sample_count[model_idx]]
        print(f"{top_ambiguous_easy_to_learn_idx=}")
        print(f"{descending_indices[ic]=}, [{accumulated_sample_count[model_idx]}, {accumulated_sample_count[model_idx+1]}], {descending_indices[ic] - accumulated_sample_count[model_idx]=}, {model_idx=}, {sample_idx=}")
        selected_sample_idx_list.append((int(model_idx), int(sample_idx)))
        _counter += 1
        if _counter == count:
            break

    random.shuffle(selected_sample_idx_list)

    return selected_sample_idx_list
