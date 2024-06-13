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


def find_qa_predict_results(args, contexts, start_logits, end_logits, offset_mapping, sample_mapping, start_positions, answer_text, idx): #, features, examples
    n_best = 20
    max_answer_length = 30

    # example_to_features = defaultdict(list)
    # for idx, feature in enumerate(features):
    #     example_to_features[feature["example_id"]].append(idx)

    # print("\n")
    # print("\n")
    # print(f"{contexts=}")
    # print(f"{start_logits=}")
    # print(f"{end_logits=}")
    # print(f"{offset_mapping=}, {offset_mapping.shape=}")
    # print(f"{sample_mapping=}, {sample_mapping.shape=}")
    # print(f"{start_positions=}")
    # print(f"{answer_text=}")
    # print(f"{idx=}")

    predicted_answers = []
    sample_feature_count = [len(_map) for _map in sample_mapping]
    sample_feature_accumualte = [0]
    for _i in range(len(sample_feature_count)):
        sample_feature_accumualte.append(sample_feature_accumualte[-1]+sample_feature_count[_i])

    # # for _i, (context, )
    # for _i, (context, _start_logit, _end_logit, _offsets, samples_map, real_start_pos, real_text, sample_idx) in\
    #       enumerate(zip(contexts, start_logits, end_logits, offset_mapping[0], sample_mapping, start_positions, answer_text, idx)):
    #     print(f"{context=}")
    #     print(f"{_start_logit=}")
    #     print(f"{_end_logit=}")
    #     print(f"{_offsets=}, {_offsets.shape=}")
    #     print(f"{samples_map=}, {samples_map.shape=}")
    #     print(f"{real_start_pos=}")
    #     print(f"{real_text=}")
    #     print(f"{sample_idx=}")
    #     example_id = sample_idx
    #     answers = []

    context = contexts[0]
    real_start_pos = start_positions[0].item()
    real_text = str(answer_text[0])
    sample_idx = idx[0].item()
    # Loop through all features associated with that example
    for feature_index in range(len(sample_mapping[0])):
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        # offsets = features[feature_index]["offset_mapping"]
        offsets = offset_mapping[0][feature_index]

        # print("\n")
        # print(f"{context=}")
        # print(f"{start_logit=}")
        # print(f"{end_logit=}")
        # print(f"{offsets=}, {offsets.shape=}")
        # # print(f"{samples_map=}, {samples_map.shape=}")
        # print(f"{real_start_pos=}")
        # print(f"{real_text=}")
        # print(f"{sample_idx=}")
        example_id = sample_idx

        answers = []

        start_indexes = np.argsort(start_logit.detach().cpu()).numpy()[::-1][:n_best].tolist()
        end_indexes = np.argsort(end_logit.detach().cpu()).numpy()[::-1][:n_best].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answer = {
                    "text": context[offsets[start_index][0] : offsets[end_index][1]],
                    "logit_score": start_logit[start_index] + end_logit[end_index],
                }
                answers.append(answer)

    # Select the answer with the best score
    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append(
            {"id": example_id, "prediction_text": best_answer["text"]}
        )
    else:
        predicted_answers.append({"id": example_id, "prediction_text": ""})

    # theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    theoretical_answers = [{"id": example_id, "answers": {"answer_start": [real_start_pos], "text": [real_text]}}]
    
    print(f"{predicted_answers=}")
    print(f"{theoretical_answers=}")
    return args.metric.compute(predictions=predicted_answers, references=theoretical_answers)


def eval_get_loss_qa(args, model, data, use_soft_label=False):
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


def eval_get_pred_qa(args, model, data, use_soft_label=False):
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
        data_iter = DataLoader(data, batch_size=1, shuffle=False)
    loss_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(data),), dtype=torch.float32).to(args.device)
    correctness_per_sample = torch.zeros((len(data),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(data),), dtype=torch.long).to(args.device)
    # logits_per_sample = torch.zeros((len(data),args.num_classes), dtype=torch.float32).to(args.device)
    model.to(args.device)
    model.eval()
    # correct_num = 0
    # err_num = 0
    total_loss = 0
    # all_labels = []
    f1_final = AverageMeter("F1", ":.6f")
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            if args.small_model_name == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                output = model(inputs, lens)
                labels = batch.label
                idx = batch.idx
            else:
                contexts, inputs, attention_mask, offset_mapping, sample_mapping, labels, idx = batch
                # print(f"eval {inputs=}\n{attention_mask=}\n{offset_mapping=}\n{sample_mapping=}\n{idx=}")
                inputs = inputs.squeeze(dim=0).to(args.device)
                attention_mask = attention_mask.squeeze(dim=0).to(args.device)
                # print(inputs.shape, attention_mask.shape)
                offset_mapping = offset_mapping.to(args.device)
                sample_mapping = sample_mapping.to(args.device)
                # print(f"eval {offset_mapping.shape=}, {sample_mapping.shape=}")
                # labels = labels.to(args.device)
                start_positions, end_positions, answer_text = [], [], []
                # print(f"eval {labels=}")
                for _start_idx, _text in zip(labels["answer_start"][0], list(labels["text"][0])):
                    for _ in range(len(inputs)):
                        start_positions.append(_start_idx.item())
                        end_positions.append(_start_idx.item()+len(_text))
                        answer_text.append(_text)
                # print(f"eval {start_positions=}, {end_positions=}")
                start_positions = torch.tensor(start_positions).long().to(args.device)
                end_positions = torch.tensor(end_positions).long().to(args.device)
                # print(f"eval {start_positions=}, {end_positions=}")
                idx = idx.to(args.device)
                # print(f"eval {inputs=}\n{attention_mask=}\n{offset_mapping=}\n{sample_mapping=}\n{idx=}")
                
                # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                output = model(inputs, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                # print(f"eval {output=}")
                start_logits = output.start_logits
                end_logits = output.end_logits
                # print(f"eval {start_logits=}\n{end_logits=}")
                loss = output.loss
                # # all_labels.append(labels)
                # predicts = output.argmax(-1).reshape(-1)
                # loss = loss_func(output, labels)
                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # err_num += (predicts != labels).sum().item()

                eval_metrics = find_qa_predict_results(args, contexts, start_logits, end_logits, offset_mapping, sample_mapping, start_positions, answer_text, idx)
                f1 = eval_metrics['f1']
                # print(f"eval {idx=}, {f1=}")
                f1_final.update(f1, labels["answer_start"][0].size(0))
                loss_per_sample[idx[0].item()] = loss
                soft_max_start_logits = torch.softmax(start_logits,dim=-1)
                soft_max_end_logits = torch.softmax(end_logits,dim=-1)
                print(f"{soft_max_start_logits=}, {soft_max_start_logits.shape=}")
                error_per_sample[idx[0].item()] = abs(1-soft_max_start_logits[0,start_positions[0]]) + abs(1-soft_max_end_logits[0,end_positions[0]])
                print(f"[debug] idx={idx[0].item()}, {loss=}, {f1=}")


            # all_labels.append(labels)
            # predicts = output.argmax(-1).reshape(-1)

            # soft_max_output = torch.softmax(output,dim=-1)
            # for _i, index in enumerate(idx):
            #     error_per_sample[index] = abs(1-soft_max_output[_i,labels[_i]])
            #     logits_per_sample[index] = soft_max_output[_i]

            # # loss = loss_func(output, labels)
            # if args.inner_obj == "ce":
            #     loss_per_sample[idx] = F.cross_entropy(output, labels, reduction='none').flatten()
            #     # if not args.normalize:
            #     #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])
            #     # else:
            #     #     loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[idx])/torch.sum(theta[idx])
            #     loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten())
            # elif args.inner_obj=='kl':
            #     one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
            #     one_hot = F.softmax(one_hot, dim=1)
            #     loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
            #     loss_per_sample[idx] = loss_vec
            #     # loss = torch.mean(loss_vec*theta[idx])
            #     loss = torch.mean(loss_vec)

            total_loss += loss.item()
            correctness_per_sample[idx] = (f1 == 1.0)

    tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (f1_final.avg, total_loss))
    model.to("cpu")
    return loss_per_sample, error_per_sample, correctness_per_sample



def merge_all_dataset_qa(args, datasets, max_sample_count_for_total=100):
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
        total_dataset = TokenizedQADataset(
            file_path=(''),
        )
        total_dataset.context = [] # clear all the samples
        total_dataset.question = [] # clear all the samples
        total_dataset.ids = [] # clear all the samples
        total_dataset.attention_mask = [] # clear all the samples
        total_dataset.offset_mapping = [] # clear all the samples
        total_dataset.sample_mapping = [] # clear all the samples
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
                total_dataset.context += [datasets[row].context[column]]
                total_dataset.question += [datasets[row].question[column]]
                total_dataset.ids += [datasets[row].ids[column]]
                total_dataset.attention_mask += [datasets[row].attention_mask[column]]
                total_dataset.offset_mapping += [datasets[row].offset_mapping[column]]
                total_dataset.sample_mapping += [datasets[row].sample_mapping[column]]
                total_dataset.label += [datasets[row].label[column]]
                total_dataset.idx += [_id]
                _id += 1
    # accumulate_sampels = torch.tensor(accumulate_sampels, dtype=torch.long).to(args.device)
    # ############### prepare total_data ###############
    return total_dataset


def run_divergence_calculation_qa(args, models, dataset, use_soft_label=False):
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
        data_iter = DataLoader(dataset, batch_size=1, shuffle=False)


    loss_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.float32).to(args.device)
    error_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.float32).to(args.device)
    # correctness_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.bool).to(args.device)
    # prediction_per_sample = torch.zeros((len(models), len(dataset),), dtype=torch.long).to(args.device)
    logits_per_sample = torch.zeros((len(models), len(dataset)), dtype=torch.float32).to(args.device)
    # labels_per_sample = torch.zeros((len(data),), dtype=torch.long).to(args.device)
    for im, model in enumerate(models):
        model.to(args.device)
        model.eval()
        # correct_num = 0
        # err_num = 0
        total_loss = 0
        # all_labels = []
        f1_final = AverageMeter("F1", ":.6f")
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                if args.small_model_name == 'LSTM':
                    (inputs, lens), labels = batch.text, batch.label
                    output = model(inputs, lens)
                    labels = batch.label
                    idx = batch.idx
                else:
                    contexts, inputs, attention_mask, offset_mapping, sample_mapping, labels, idx = batch
                    # print(f"eval {inputs=}\n{attention_mask=}\n{offset_mapping=}\n{sample_mapping=}\n{idx=}")
                    inputs = inputs.squeeze(dim=0).to(args.device)
                    attention_mask = attention_mask.squeeze(dim=0).to(args.device)
                    # print(inputs.shape, attention_mask.shape)
                    offset_mapping = offset_mapping.to(args.device)
                    sample_mapping = sample_mapping.to(args.device)
                    # print(f"eval {offset_mapping.shape=}, {sample_mapping.shape=}")
                    # labels = labels.to(args.device)
                    start_positions, end_positions, answer_text = [], [], []
                    # print(f"eval {labels=}")
                    for _start_idx, _text in zip(labels["answer_start"][0], list(labels["text"][0])):
                        for _ in range(len(inputs)):
                            start_positions.append(_start_idx.item())
                            end_positions.append(_start_idx.item()+len(_text))
                            answer_text.append(_text)
                    # print(f"eval {start_positions=}, {end_positions=}")
                    start_positions = torch.tensor(start_positions).long().to(args.device)
                    end_positions = torch.tensor(end_positions).long().to(args.device)
                    # print(f"eval {start_positions=}, {end_positions=}")
                    idx = idx.to(args.device)
                    # print(f"eval {inputs=}\n{attention_mask=}\n{offset_mapping=}\n{sample_mapping=}\n{idx=}")
                    
                    # output = model(inputs, attention_mask=attention_mask, labels=eval_labels).logits
                    output = model(inputs, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                    # print(f"eval {output=}")
                    start_logits = output.start_logits
                    end_logits = output.end_logits
                    # print(f"eval {start_logits=}\n{end_logits=}")
                    loss = output.loss
                    # # all_labels.append(labels)
                    # predicts = output.argmax(-1).reshape(-1)
                    # loss = loss_func(output, labels)
                    total_loss += loss.item()
                    # correct_num += (predicts == labels).sum().item()
                    # err_num += (predicts != labels).sum().item()

                    eval_metrics = find_qa_predict_results(args, contexts, start_logits, end_logits, offset_mapping, sample_mapping, start_positions, answer_text, idx)
                    f1 = eval_metrics['f1']
                    # print(f"eval {idx=}, {f1=}")
                    f1_final.update(f1, labels["answer_start"][0].size(0))
                    loss_per_sample[im][idx[0].item()] = loss
                    soft_max_start_logits = torch.softmax(start_logits,dim=-1)
                    soft_max_end_logits = torch.softmax(end_logits,dim=-1)
                    # print(f"{soft_max_start_logits=}, {soft_max_start_logits.shape=}")
                    error_per_sample[im][idx[0].item()] = abs(1-soft_max_start_logits[0,start_positions[0]]) + abs(1-soft_max_end_logits[0,end_positions[0]])
                    logits_per_sample[im][idx[0].item()] = soft_max_start_logits[0,start_positions[0]]*soft_max_end_logits[0,end_positions[0]]

                # all_labels.append(labels)
                # predicts = output.argmax(-1).reshape(-1)

                # soft_max_output = torch.softmax(output,dim=-1)
                # for _i, index in enumerate(idx):
                #     error_per_sample[im][index] = abs(1-soft_max_output[_i,labels[_i]])
                #     logits_per_sample[im][index] = soft_max_output[_i]

                total_loss += loss.item()
                # correct_num += (predicts == labels).sum().item()
                # correctness_per_sample[im][idx] = (f1 == 1.0)
                # prediction_per_sample[im][idx] = predicts
                # err_num += (predicts != labels).sum().item()

        tqdm.write("cross validation: Acc: %.3f, Loss %.3f" % (f1_final.avg, total_loss))
        model.to("cpu")
        # return acc, total_loss/len(data_iter)
        # return loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, logits_per_sample

    # logits_per_sample (#models, #samples, #classes) with (#samples, #classes) is the list of logits (model outputs) of all the samples
    # all_labels is the real label of all samples
    confidence_per_sample = logits_per_sample[:,torch.arange(len(dataset))]
    variability_per_sample = torch.std(confidence_per_sample, dim=0)
    confidence_per_sample = torch.mean(confidence_per_sample, dim=0)
    return list(confidence_per_sample.detach().cpu().numpy()), list(variability_per_sample.detach().cpu().numpy())

