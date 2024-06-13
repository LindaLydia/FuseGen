from tqdm import tqdm
import copy

import numpy as np
import torch

from bilevel_tools.meta import MetaSGD
from bilevel_tools.tbtools import AverageMeter
import torch.nn.functional as F

from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
# from torchtext.data import Iterator, BucketIterator
# from torchtext import data

from utils.bert_dataset import *


def kd_label(args, mode_list, dataset_list, model_importance):
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in dataset_list:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
            total_data += copy.deepcopy(_dataset_examples)
        # print(type(total_data))
        # print(total_data[0])
        # print(len(total_data))
        for _i, _data_item in enumerate(total_data):
            _data_item.idx = _i
        # print(f'[debug] in total, #{len(total_data)} data samples for next step SLM training')
        total_data = data.Dataset(total_data, dataset_list[0].fields)
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
    elif 'bert' in args.small_model_name:
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
            for column in range(len(dataset_list[row].idx)):
                total_data.text += [dataset_list[row].text[column]]
                total_data.ids += [dataset_list[row].ids[column]]
                total_data.attention_mask += [dataset_list[row].attention_mask[column]]
                total_data.label += [dataset_list[row].label[column]]
                total_data.idx += [_id]
                _id += 1
        data_iter = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    logits_per_sample = torch.zeros((len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    # correctness_per_sample = torch.zeros((len(total_data),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(total_data),), dtype=torch.long).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample[index] += abs(1-soft_max_output[_i,labels[_i]]) * model_importance[_j]
                    logits_per_sample[index] += soft_max_output[_i] * model_importance[_j]

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample[idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # # correctness_per_sample[idx] = (predicts == labels)
                # # prediction_per_sample[idx] = predicts
                # err_num += (predicts != labels).sum().item()

        # acc = correct_num / (correct_num + err_num)
        # tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
        # all_labels = torch.cat(all_labels)
        # print(f"num of zeros: {torch.sum(all_labels == 0)}")
        # print(f"num of ones: {torch.sum(all_labels == 1)}")
    # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample
    # for _i, data_item in enumerate(total_data):
    #     data_item.label = logits_per_sample[_i]
    #     # print(data_item.label, type(data_item.label))
    model.to("cpu")
    return total_data, logits_per_sample, loss_per_sample, error_per_sample


def kd_label_entropy(args, mode_list, dataset_list, model_importance):
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in dataset_list:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
            total_data += copy.deepcopy(_dataset_examples)
        # print(type(total_data))
        # print(total_data[0])
        # print(len(total_data))
        for _i, _data_item in enumerate(total_data):
            _data_item.idx = _i
        # print(f'[debug] in total, #{len(total_data)} data samples for next step SLM training')
        total_data = data.Dataset(total_data, dataset_list[0].fields)
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
    elif 'bert' in args.small_model_name:
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
            for column in range(len(dataset_list[row].idx)):
                total_data.text += [dataset_list[row].text[column]]
                total_data.ids += [dataset_list[row].ids[column]]
                total_data.attention_mask += [dataset_list[row].attention_mask[column]]
                total_data.label += [dataset_list[row].label[column]]
                total_data.idx += [_id]
                _id += 1
        data_iter = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    logits_per_sample_model = torch.zeros((len(mode_list), len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    entropy_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    # correctness_per_sample_model = torch.zeros((len(total_data),), dtype=torch.bool).to(args.device)
    # prediction_per_sample_model = torch.zeros((len(total_data),), dtype=torch.long).to(args.device)
    logits_per_sample = torch.zeros((len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    # entropy_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample_model[_j][index] += abs(1-soft_max_output[_i,labels[_i]])
                    logits_per_sample_model[_j][index] += soft_max_output[_i]
                    entropy_per_sample_model[_j][index] += (-torch.sum(soft_max_output[_i] * torch.log2(soft_max_output[_i])))

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample_model[_j][idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample_model[_j][idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # # correctness_per_sample_model[idx] = (predicts == labels)
                # # prediction_per_sample_model[idx] = predicts
                # err_num += (predicts != labels).sum().item()
    model.to("cpu")

        # acc = correct_num / (correct_num + err_num)
        # tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
        # all_labels = torch.cat(all_labels)
        # print(f"num of zeros: {torch.sum(all_labels == 0)}")
        # print(f"num of ones: {torch.sum(all_labels == 1)}")
    # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample
    # for _i, data_item in enumerate(total_data):
    #     data_item.label = logits_per_sample[_i]
    #     # print(data_item.label, type(data_item.label))

    # for _i in range(len(total_data)):
    #     _sum = torch.sum(entropy_per_sample_model[:,_i])
    #     entropy_per_sample_model[:,_i] = _sum - 
    _sum = torch.sum(entropy_per_sample_model,dim=0)
    entropy_per_sample_model = _sum - entropy_per_sample_model
    entropy_per_sample_model = torch.softmax(entropy_per_sample_model, dim=0)
    print("entropy", entropy_per_sample_model)
    print("logits", logits_per_sample_model)

    if args.small_model_name.upper() == 'LSTM':
        print(total_data[0].text, total_data[0].label)
        print(total_data[1].text, total_data[1].label)
        print(total_data[2].text, total_data[2].label)
        print(total_data[-3].text, total_data[-3].label)
        print(total_data[-2].text, total_data[-2].label)
        print(total_data[-1].text, total_data[-1].label)
    elif 'bert' in args.small_model_name.lower():
        print(total_data.text[0], total_data.label[0])
        print(total_data.text[1], total_data.label[1])
        print(total_data.text[2], total_data.label[2])
        print(total_data.text[-3], total_data.label[-3])
        print(total_data.text[-2], total_data.label[-2])
        print(total_data.text[-1], total_data.label[-1])
    for _j in range(len(mode_list)):
        # print(logits_per_sample.shape, logits_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)).shape)
        logits_per_sample += logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)
        # print(loss_per_sample.shape, loss_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (loss_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        loss_per_sample += loss_per_sample_model[_j]*entropy_per_sample_model[_j]
        # print(error_per_sample.shape, error_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (error_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        error_per_sample += error_per_sample_model[_j]*entropy_per_sample_model[_j]

    return total_data, logits_per_sample, loss_per_sample, error_per_sample


def kd_label_aware(args, mode_list, dataset_list, model_importance, high_row, high_column, low_row, low_colum):
    return None, None, None, None
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in dataset_list:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
            total_data += copy.deepcopy(_dataset_examples)
        # print(type(total_data))
        # print(total_data[0])
        # print(len(total_data))
        for _i, _data_item in enumerate(total_data):
            _data_item.idx = _i
        # print(f'[debug] in total, #{len(total_data)} data samples for next step SLM training')
        total_data = data.Dataset(total_data, dataset_list[0].fields)
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
    elif 'bert' in args.small_model_name:
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
            for column in range(len(dataset_list[row].idx)):
                total_data.text += [dataset_list[row].text[column]]
                total_data.ids += [dataset_list[row].ids[column]]
                total_data.attention_mask += [dataset_list[row].attention_mask[column]]
                total_data.label += [dataset_list[row].label[column]]
                total_data.idx += [_id]
                _id += 1
        data_iter = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    logits_per_sample = torch.zeros((len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    # correctness_per_sample = torch.zeros((len(total_data),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(total_data),), dtype=torch.long).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample[index] += abs(1-soft_max_output[_i,labels[_i]]) * model_importance[_j]
                    logits_per_sample[index] += soft_max_output[_i] * model_importance[_j]

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample[idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
    model.to("cpu")
    return total_data, logits_per_sample, loss_per_sample, error_per_sample


def kd_label_entropy_aware(args, mode_list, dataset_list, model_importance, high_row, high_column, low_row, low_colum):
    accumulate_sampels = [0]
    if args.small_model_name.upper() == 'LSTM':
        total_data = []
        for _dataset in dataset_list:
            _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
            print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
            total_data += copy.deepcopy(_dataset_examples)
            accumulate_sampels.append(accumulate_sampels[-1]+len(_dataset_examples)) 
        accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)
        # print(type(total_data))
        # print(total_data[0])
        # print(len(total_data))
        for _i, _data_item in enumerate(total_data):
            _data_item.idx = _i
        # print(f'[debug] in total, #{len(total_data)} data samples for next step SLM training')
        total_data = data.Dataset(total_data, dataset_list[0].fields)
        data_iter, = BucketIterator.splits(
            (total_data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
    elif 'bert' in args.small_model_name:
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
            accumulate_sampels.append(accumulate_sampels[-1]+len(dataset_list[row].idx)) 
            for column in range(len(dataset_list[row].idx)):
                total_data.text += [dataset_list[row].text[column]]
                total_data.ids += [dataset_list[row].ids[column]]
                total_data.attention_mask += [dataset_list[row].attention_mask[column]]
                total_data.label += [dataset_list[row].label[column]]
                total_data.idx += [_id]
                _id += 1
        accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)
        data_iter = DataLoader(total_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
    
    low_row = torch.tensor(low_row, dtype=torch.long).to(args.device)
    low_colum = torch.tensor(low_colum, dtype=torch.long).to(args.device)

    logits_per_sample_model = torch.zeros((len(mode_list), len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    entropy_per_sample_model = torch.zeros((len(mode_list), len(total_data),), dtype=torch.float32).to(args.device)
    # correctness_per_sample_model = torch.zeros((len(total_data),), dtype=torch.bool).to(args.device)
    # prediction_per_sample_model = torch.zeros((len(total_data),), dtype=torch.long).to(args.device)
    logits_per_sample = torch.zeros((len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    # entropy_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    logits_per_sample_ns = torch.zeros((len(total_data), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample_ns = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    error_per_sample_ns = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    # entropy_per_sample = torch.zeros((len(total_data),), dtype=torch.float32).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample_model[_j][index] += abs(1-soft_max_output[_i,labels[_i]])
                    logits_per_sample_model[_j][index] += soft_max_output[_i]
                    entropy_per_sample_model[_j][index] += (-torch.sum(soft_max_output[_i] * torch.log2(soft_max_output[_i])))

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample_model[_j][idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample_model[_j][idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
    model.to("cpu")
    _entropy_per_sample_model_save = copy.deepcopy(entropy_per_sample_model)

    logits_per_sample_s = logits_per_sample_model[low_row, accumulate_sampels[low_row]+low_colum]
    loss_per_sample_s = loss_per_sample_model[low_row, accumulate_sampels[low_row]+low_colum]
    error_per_sample_s = error_per_sample_model[low_row, accumulate_sampels[low_row]+low_colum]

    _sum = torch.sum(entropy_per_sample_model,dim=0)
    entropy_per_sample_model = _sum - entropy_per_sample_model
    entropy_per_sample_model = torch.softmax(entropy_per_sample_model, dim=0)
    print("entropy", entropy_per_sample_model)
    print("logits", logits_per_sample_model)
    if args.small_model_name.upper() == 'LSTM':
        print(total_data[0].text, total_data[0].label)
        print(total_data[1].text, total_data[1].label)
        print(total_data[2].text, total_data[2].label)
        print(total_data[-3].text, total_data[-3].label)
        print(total_data[-2].text, total_data[-2].label)
        print(total_data[-1].text, total_data[-1].label)
    elif 'bert' in args.small_model_name.lower():
        print(total_data.text[0], total_data.label[0])
        print(total_data.text[1], total_data.label[1])
        print(total_data.text[2], total_data.label[2])
        print(total_data.text[-3], total_data.label[-3])
        print(total_data.text[-2], total_data.label[-2])
        print(total_data.text[-1], total_data.label[-1])
    for _j in range(len(mode_list)):
        # print(logits_per_sample.shape, logits_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)).shape)
        logits_per_sample += logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)
        # print(loss_per_sample.shape, loss_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (loss_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        loss_per_sample += loss_per_sample_model[_j]*entropy_per_sample_model[_j]
        # print(error_per_sample.shape, error_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (error_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        error_per_sample += error_per_sample_model[_j]*entropy_per_sample_model[_j]

    entropy_per_sample_model = copy.deepcopy(_entropy_per_sample_model_save)
    for r,c in zip(low_row, low_colum):
        entropy_per_sample_model[r][accumulate_sampels[r]+c] = 0.0 # do not contribute to sum
    _sum = torch.sum(entropy_per_sample_model,dim=0)
    entropy_per_sample_model = _sum - entropy_per_sample_model
    for r,c in zip(low_row, low_colum):
        entropy_per_sample_model[r][accumulate_sampels[r]+c] = -1e30 # after softmax, it is zero
    entropy_per_sample_model = torch.nn.functional.softmax(entropy_per_sample_model, dim=0)
    entropy_per_sample_model = torch.where(torch.isnan(entropy_per_sample_model), torch.full_like(entropy_per_sample_model, 0), entropy_per_sample_model)
    for _j in range(len(mode_list)):
        # print(logits_per_sample.shape, logits_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)).shape)
        logits_per_sample_ns += logits_per_sample_model[_j]*entropy_per_sample_model[_j].view(-1,1)
        # print(loss_per_sample.shape, loss_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (loss_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        loss_per_sample_ns += loss_per_sample_model[_j]*entropy_per_sample_model[_j]
        # print(error_per_sample.shape, error_per_sample_model[_j].shape, entropy_per_sample_model[_j].view(-1,1).shape, (error_per_sample_model[_j]*entropy_per_sample_model[_j]).shape)
        error_per_sample_ns += error_per_sample_model[_j]*entropy_per_sample_model[_j]
    # print(f"{low_row=}, {low_colum=}")
    # print(f"{logits_per_sample.shape=}, {logits_per_sample_ns.shape=}, {logits_per_sample_s.shape=}")
    for i, (r,c) in enumerate(zip(low_row, low_colum)):
        # print(f"{logits_per_sample_model[r][accumulate_sampels[r]+c]=}")
        # for _r in range(len(mode_list)):
        #         print(f"{_r=}, {logits_per_sample_model[_r][accumulate_sampels[_r]+c]=}, {entropy_per_sample_model[_r][accumulate_sampels[_r]+c]}")
        print(f"total kd: {logits_per_sample[accumulate_sampels[r]+c]}, none-seen kd: {logits_per_sample_ns[accumulate_sampels[r]+c]}, seen kd: {logits_per_sample_s[i]},")
        if args.small_model_name.upper() == 'LSTM':
            print(f"related text and label:", total_data[accumulate_sampels[r]+c].text, total_data[accumulate_sampels[r]+c].label)
        elif 'bert' in args.small_model_name.lower():
            print(f"related text and label:", total_data.text[accumulate_sampels[r]+c], total_data.label[accumulate_sampels[r]+c])
        if i == 16:
            break
    print(f"{logits_per_sample=}")
    print(f"{logits_per_sample_ns=}")
    print(f"{logits_per_sample_s=}")
    return total_data, logits_per_sample, loss_per_sample, error_per_sample, logits_per_sample_ns, loss_per_sample_ns, error_per_sample_ns, logits_per_sample_s, loss_per_sample_s, error_per_sample_s


def kd_label_dataset(args, mode_list, dataset, model_importance):
    print(f"[debug] in kd_label_iter, len(dataset)={len(dataset)}")
    if args.small_model_name.upper() == 'LSTM':
        data_iter, = BucketIterator.splits(
            (dataset,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=False,
        )
        print(len(data_iter), len(dataset))
    elif 'bert' in args.small_model_name:
        data_iter = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    logits_per_sample = torch.zeros((len(dataset), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(dataset),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(dataset),), dtype=torch.float32).to(args.device)
    # correctness_per_sample = torch.zeros((len(dataset),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(dataset),), dtype=torch.long).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample[index] += abs(1-soft_max_output[_i,labels[_i]]) * model_importance[_j]
                    logits_per_sample[index] += soft_max_output[_i] * model_importance[_j]

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample[idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # # correctness_per_sample[idx] = (predicts == labels)
                # # prediction_per_sample[idx] = predicts
                # err_num += (predicts != labels).sum().item()

        # acc = correct_num / (correct_num + err_num)
        # tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
        # all_labels = torch.cat(all_labels)
        # print(f"num of zeros: {torch.sum(all_labels == 0)}")
        # print(f"num of ones: {torch.sum(all_labels == 1)}")
    # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample
    # for _i, data_item in enumerate(total_data):
    #     data_item.label = logits_per_sample[_i]
    #     # print(data_item.label, type(data_item.label))
    model.to("cpu")
    return data_iter, logits_per_sample, loss_per_sample, error_per_sample

def kd_label_iter(args, mode_list, data_iter, model_importance):
    print(f"[debug] in kd_label_iter, len(data_iter)={len(data_iter)}")
    logits_per_sample = torch.zeros((len(data_iter), args.num_classes), dtype=torch.float32).to(args.device)
    loss_per_sample = torch.zeros((len(data_iter),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(data_iter),), dtype=torch.float32).to(args.device)
    # correctness_per_sample = torch.zeros((len(data_iter),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(data_iter),), dtype=torch.long).to(args.device)
    for _j, model in enumerate(mode_list):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
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
                    output = model(inputs, attention_mask=attention_mask, labels=labels).logits
                # all_labels.append(labels)
                predicts = output.argmax(-1).reshape(-1)

                soft_max_output = torch.softmax(output,dim=-1)
                for _i, index in enumerate(idx):
                    error_per_sample[index] += abs(1-soft_max_output[_i,labels[_i]]) * model_importance[_j]
                    logits_per_sample[index] += soft_max_output[_i] * model_importance[_j]

                # loss = loss_func(output, labels)
                if args.inner_obj == "ce":
                    loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
                    # if not args.normalize:
                    #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
                    # else:
                    #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
                elif args.inner_obj=='kl':
                    one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                    loss_per_sample[idx] += loss_vec * model_importance[_j]
                    # loss = torch.mean(loss_vec*theta[idx])
                    loss = torch.mean(loss_vec)

                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # # correctness_per_sample[idx] = (predicts == labels)
                # # prediction_per_sample[idx] = predicts
                # err_num += (predicts != labels).sum().item()

        # acc = correct_num / (correct_num + err_num)
        # tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (acc, total_loss))
        # all_labels = torch.cat(all_labels)
        # print(f"num of zeros: {torch.sum(all_labels == 0)}")
        # print(f"num of ones: {torch.sum(all_labels == 1)}")
    # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample
    # for _i, data_item in enumerate(total_data):
    #     data_item.label = logits_per_sample[_i]
    #     # print(data_item.label, type(data_item.label))
    model.to("cpu")
    return data_iter, logits_per_sample, loss_per_sample, error_per_sample