import numpy as np
import copy
import wandb
import math
from tqdm import tqdm
import logging
import random

import torch
import torch.nn.functional as F
from torch.optim import Adam,SGD
from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
from bilevel_tools.meta import MetaSGD
from bilevel_tools.tbtools import AverageMeter
import bilevel_tools.loss_utils as loss_utils

from utils.basic_utils import set_seed
from utils.bert_dataset import *

import time

loss_func = torch.nn.CrossEntropyLoss()


def construct_outer_subloader(args, train_data, indices = None, idx_to_order=None):
    if indices is None:
        if args.small_model_name.upper() == 'LSTM':
            num_use_samples_inner=len(train_data.examples)
        elif 'bert' in args.small_model_name.lower():
            num_use_samples_inner=len(train_data.idx)
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
        return subset_iter[0]
    elif 'bert' in args.small_model_name.lower():
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
        dev_data.text = [train_data.text[ix] for ix in indices]
        dev_data.ids = [train_data.ids[ix] for ix in indices]
        dev_data.attention_mask = [train_data.attention_mask[ix] for ix in indices]
        dev_data.label = [train_data.label[ix] for ix in indices]
        dev_data.idx = [train_data.idx[ix] for ix in indices]
        dev_iter = DataLoader(dev_data, batch_size=args.backward_batch_size, shuffle=True)
        return dev_iter


def construct_outer_loader(args, train_data, indices = None, idx_to_order=None):
    if args.subset_outer:
        if indices is None:
            if args.small_model_name.upper() == 'LSTM':
                num_use_samples_inner=len(train_data.examples)
            elif 'bert' in args.small_model_name.lower():
                num_use_samples_inner=len(train_data.idx)
            indices = np.random.choice(list(range(num_use_samples_inner)), args.num_use_samples_outer, replace=False)
        else:
            indices = [idx_to_order[idx] for idx in indices]
        if args.small_model_name.upper() == 'LSTM':
            dev_data = data.Dataset([train_data.examples[ix] for ix in indices], train_data.fields)
        elif 'bert' in args.small_model_name.lower():
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
            dev_data.text = [train_data.text[ix] for ix in indices]
            dev_data.ids = [train_data.ids[ix] for ix in indices]
            dev_data.attention_mask = [train_data.attention_mask[ix] for ix in indices]
            dev_data.label = [train_data.label[ix] for ix in indices]
            dev_data.idx = [train_data.idx[ix] for ix in indices]
    else:
        dev_data = copy.deepcopy(train_data)
    if args.small_model_name.upper() == 'LSTM':
        subset_iter= BucketIterator.splits(
            (dev_data,),
            batch_sizes=(args.backward_batch_size,),
            device='cuda',
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=True,
        )
        return subset_iter[0]
    elif 'bert' in args.small_model_name.lower():
        dev_iter = DataLoader(dev_data, batch_size=args.backward_batch_size, shuffle=True)
        return dev_iter

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def get_grad_weights_on_valid(args, model, val_iter, reweight_theta, soft_label=None):
    grad_weights_on_full_train = []
    losses = AverageMeter("OuterLoss", ":.3f")
    top1 = AverageMeter("OuterAcc@1", ":6.2f")
    model.to(args.device)
    for batch_idx, batch in enumerate(val_iter):
        if args.small_model_name.upper() == 'LSTM':
            reweight_theta_batch = reweight_theta[batch.idx]
            (inputs, lens), labels = batch.text, batch.label
            idx = batch.idx
        elif 'bert' in args.small_model_name.lower():
            inputs, attention_mask, labels, idx = batch
            inputs = inputs.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            idx = idx.to(args.device)
            reweight_theta_batch = reweight_theta[idx]

        # if soft_label == None:
        #     if args.small_model_name.upper() == 'LSTM':
        #         labels = batch.label
        #     elif 'bert' in args.small_model_name.lower():
        #         target = copy.deepcopy(torch.unsqueeze(labels, 1))
        #         eval_labels = torch.zeros(target.size(0), args.num_classes, device=args.device)
        #         eval_labels.scatter_(1, target, 1)
        # else:
        #     labels = soft_label[idx]
            # assert args.outer_obj != 'kl', "KL divergence loss does not support soft label currently"
            # assert args.outer_obj != 'mae', "MAE loss does not support soft label currently"
        # print("[debug] in get_grad_weights_on_valid, batch.idx", batch.idx)
        # print("[debug] in get_grad_weights_on_valid, reweight_theta_batch", reweight_theta_batch)
        # print("[debug] in get_grad_weights_on_valid, labels", labels)
        if args.small_model_name.upper() == 'LSTM':
            output = model(inputs, lens)
        elif 'bert' in args.small_model_name.lower():
            output = model(inputs, attention_mask=attention_mask, labels=labels).logits
        # print("[debug] in get_grad_weights_on_valid, output", output)
        # print("[debug] in get_grad_weights_on_valid, reweight_theta", reweight_theta)
        if args.use_dev_outer:
            loss = loss_func(output, labels)
        else:
            if args.hard:
                val, idx = torch.topk(reweight_theta, int(args.threshold * len(reweight_theta)))
                subnet = (reweight_theta_batch >= val[-1]).float()
                selection = torch.nonzero(subnet.squeeze()).flatten()
            if args.outer_obj == "entropy":
                loss = - torch.mul(F.softmax(output), F.log_softmax(output))
            elif args.outer_obj == "mae":
                outputvar = output[:, labels]
                loss = (1. - outputvar)
            elif args.outer_obj == "kl":
                if soft_label == None:
                    one_hot = torch.zeros(len(labels), args.num_classes).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot))
                else:
                    loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(labels))
            else:
                if soft_label == None:
                    one_hot = torch.zeros(len(labels), args.num_classes).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                    one_hot = F.softmax(one_hot, dim=1)
                    loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)) - torch.mul(F.softmax(output, dim=1), F.log_softmax(output, dim=1))
                else:
                    loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(labels)) - torch.mul(F.softmax(output, dim=1), F.log_softmax(output, dim=1))
            if args.hard:
                loss = torch.mean(loss[selection])
            else:
                loss = torch.mean(reweight_theta_batch.detach().view(-1,1)*loss)
        # print("[debug] in get_grad_weights_on_valid, loss", loss)

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

        losses.update(loss.item(), labels.size(0))
        top1.update(acc, labels.size(0))
        grad_weights_on_full_train_batch = torch.autograd.grad(loss, model.parameters())
        if batch_idx > 0:
            grad_weights_on_full_train = [wb+w for wb, w in zip(grad_weights_on_full_train_batch, grad_weights_on_full_train)]
        else:
            grad_weights_on_full_train = grad_weights_on_full_train_batch
    if args.mean_grad:
        grad_weights_on_full_train = [g/len(val_iter) for g in grad_weights_on_full_train]
    model.to("cpu")
    return grad_weights_on_full_train,  top1.avg, losses.avg


def repass_backward(args, model, model_checkpoints, opt_checkpoints, outer_grads_w, train_data, reweight_theta_mapped, reweight_theta, soft_label=None):
    # accumulate gradients backwards to leverage hessian-vector product
    reweight_theta_grads = [torch.zeros_like(reweight_theta)]
    old_params = model_checkpoints[0]
    old_opt = opt_checkpoints[0]
    model_copy = copy.deepcopy(model)
    model_copy.load_state_dict(old_params)
    model_copy.to(args.device)
    reweight_theta_sum = reweight_theta_mapped.detach().sum()
    
    if args.small_model_name.upper() == 'LSTM':
        train_iter, = BucketIterator.splits(
            (train_data,),
            batch_sizes=(args.backward_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        train_iter = DataLoader(train_data, batch_size=args.backward_batch_size, shuffle=args.shuffle_train)
    with torch.backends.cudnn.flags(enabled=False):
        for batch_idx, batch in enumerate(train_iter):
            if args.small_model_name.upper() == 'LSTM':
                (inputs, lens), labels = batch.text, batch.label
                idx = batch.idx
            elif 'bert' in args.small_model_name.lower():
                inputs, attention_mask, labels, idx = batch
                inputs = inputs.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                idx = idx.to(args.device)

            if soft_label == None:
                target = copy.deepcopy(torch.unsqueeze(labels, 1))
                eval_labels = torch.zeros(target.size(0), args.num_classes, device=args.device)
                eval_labels.scatter_(1, target, 1)
            else:
                labels = soft_label[idx]
            if args.small_model_name.upper() == 'LSTM':
                old_params_, w_mapped = pseudo_updated_params(args, model_copy, old_params, old_opt, inputs, lens, labels, reweight_theta_mapped[idx], reweight_theta_sum)
            elif 'bert' in args.small_model_name.lower():
                old_params_, w_mapped = pseudo_updated_params_bert(args, model_copy, old_params, old_opt, inputs, attention_mask, labels, reweight_theta_mapped[idx], reweight_theta_sum)
            # print(f"[debug] w_mapped={w_mapped}")
            grad_batch = torch.autograd.grad(w_mapped, reweight_theta, grad_outputs=outer_grads_w, retain_graph=True)
            reweight_theta_grads = [g + b for g, b in zip(reweight_theta_grads, grad_batch)]
    model_copy.to("cpu")
    return reweight_theta_grads[0]


def pseudo_updated_params(args, pseudo_net, model_checkpoint, opt_checkpoint, inputs, lens, labels, reweight_theta, reweight_theta_sum):
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=args.inner_lr)
    w_old = [p for p in pseudo_net.parameters()]
    output = pseudo_net(inputs, lens)
    pseudo_loss_vector = F.cross_entropy(output, labels, reduction='none').flatten()
    pseudo_loss_vector *= reweight_theta
    pseudo_loss = torch.sum(pseudo_loss_vector/reweight_theta_sum)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=opt_checkpoint['param_groups'][0]['lr'])
    return w_old, w_mapped

def pseudo_updated_params_bert(args, pseudo_net, model_checkpoint, opt_checkpoint, inputs, attention_mask, labels, reweight_theta, reweight_theta_sum):
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=args.inner_lr)
    w_old = [p for p in pseudo_net.parameters()]
    pseudo_net.to(args.device)
    output = pseudo_net(inputs, attention_mask=attention_mask, labels=labels).logits
    pseudo_loss_vector = F.cross_entropy(output, labels, reduction='none').flatten()
    pseudo_loss_vector *= reweight_theta
    pseudo_loss = torch.sum(pseudo_loss_vector/reweight_theta_sum)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=opt_checkpoint['param_groups'][0]['lr'])
    # pseudo_net.to("cpu")
    return w_old, w_mapped


def eval_for_reweight(args, model, data_iter, name, epoch=None):
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
                # if use_soft_label == True:
                #     target = copy.deepcopy(torch.unsqueeze(labels, 1))
                #     eval_labels = torch.zeros(target.size(0), args.num_classes, device=args.device)
                #     eval_labels.scatter_(1, target, 1)
                # else:
                #     eval_labels = labels
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
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    model.to("cpu")
    return acc, total_loss/len(data_iter)


def train_to_converge_reweight(args, model, train_data, reweight_theta, epoch_converge, inner_obj, soft_label=None):
    
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
        train_iter, = BucketIterator.splits(
            (train_data,),
            batch_sizes=(args.train_batch_size,),
            device=args.device,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            repeat=False,
            shuffle=args.shuffle_train,
        )
    elif 'bert' in args.small_model_name.lower():
        train_iter = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)

    model_copy.to(args.device)
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
            # model_copy.zero_grad() # this is equal to optimizer.zero_grad() when optimizer contains only this model's whole parameters
            optimizer.zero_grad()
            if args.small_model_name.upper() == 'LSTM':
                output = model_copy(inputs, lens)
            elif 'bert' in args.small_model_name.lower():
                output = model_copy(inputs, attention_mask=attention_mask, labels=labels).logits

            if inner_obj == "ce":
                if not args.normalize:
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*reweight_theta[idx])
                else:
                    loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*reweight_theta[idx])/torch.sum(reweight_theta[idx])
            elif inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(args.num_classes)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss = torch.mean(loss_vec*reweight_theta[idx])
            
            if soft_label != None:
                temperatured_output = F.softmax(output/args.kd_temperature)
                temperatured_labels = F.softmax(soft_label[idx]/args.kd_temperature)
                if not args.normalize:
                    loss = args.kd_alpha * loss + (1-args.kd_alpha) * torch.mean(F.cross_entropy(temperatured_output, temperatured_labels, reduction='none').flatten()*reweight_theta[idx])
                else:
                    loss = args.kd_alpha * loss + (1-args.kd_alpha) * torch.sum(F.cross_entropy(temperatured_output, temperatured_labels, reduction='none').flatten()*reweight_theta[idx])/torch.sum(reweight_theta[idx])
            
            if soft_label == None:
                # print(f"{labels=}, {output=}, {labels.shape=}, {output.shape=}")
                acc = loss_utils.accuracy(output, labels)
            else:
                acc = loss_utils.accuracy(output, torch.argmax(soft_label[idx],dim=-1))
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


def solve_reweight(args, model, train_data, theta, selected_sample_indexs, train_loader_backward, valid_loader, test_loader, soft_label=None):
    print("[info] in solve_reweight")
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]
    # print(selected_sample_rows, selected_sample_columns)
    # if soft_label != None:
    #     soft_label, backward_soft_label, valid_soft_label = soft_label

    if selected_sample_rows != None and selected_sample_columns != None:
        selected_train_dataset = []
        _id = 0
        for row, column in zip(selected_sample_rows,selected_sample_columns):
            selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
            selected_train_dataset[_id].idx = _id
            _id += 1
        selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
        # print(selected_train_data.fields)
        # selected_train_data = torch.tensor(train_data)[selected_sample_rows,selected_sample_columns]
        theta = torch.stack(theta)[selected_sample_rows,selected_sample_columns]
        init_theta = copy.deepcopy(theta)
        # print(theta)
        # theta = torch.stack(theta)
        # print(type(theta), theta.shape)
        # if args.use_sigmoid:
        #     reweight_theta = torch.full([len(selected_train_data)], 0, dtype=torch.float, requires_grad=True, device="cuda")
        # else:
        #     reweight_theta = torch.full([len(selected_train_data)], args.init_reweight_theta, dtype=torch.float, requires_grad=True, device="cuda")
        reweight_theta = copy.deepcopy(theta)
        reweight_theta.requires_grad_(True)
    else:
        selected_train_data = train_data[0]
        reweight_theta = copy.deepcopy(theta)
        reweight_theta.requires_grad_(True)
        # print(reweight_theta)

    set_seed(args.seed)

    # train_iter, = BucketIterator.splits(
    #     (selected_train_data,),
    #     batch_sizes=(args.train_batch_size,),
    #     device=args.device,
    #     sort_key=lambda x: len(x.text),
    #     sort_within_batch=True,
    #     repeat=False,
    #     shuffle=args.shuffle_train,
    # )

    if args.optim =='Adam':
        reweight_theta_opt = Adam([reweight_theta], lr=args.outer_lr)
    elif args.optim =='SGD':
        reweight_theta_opt = SGD([reweight_theta], lr=args.outer_lr, momentum=0.9)
    reweight_theta.grad = torch.zeros_like(reweight_theta)
    best_reweight_theta = reweight_theta
    for outer_iter in range(args.max_reweight_outer_iter):
        if args.temp_anneal:
            temp = args.end_temp + (args.max_reweight_outer_iter - outer_iter)/args.max_reweight_outer_iter * (1-args.end_temp)
            print(temp, args.end_temp, args.max_reweight_outer_iter, outer_iter)
        else:
            temp = 1
        if args.use_sigmoid:
            reweight_theta_mapped = F.sigmoid(reweight_theta/temp)
        else:
            reweight_theta_mapped = reweight_theta
        print(reweight_theta_mapped)
        if not args.disable_outer_scheduler:
            assign_learning_rate(reweight_theta_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_reweight_outer_iter)) * args.outer_lr)
        diverged = True # diverged==True means loss==nan, which means the training failed
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge_reweight(args, model, selected_train_data, reweight_theta_mapped.detach(),args.epoch_converge,args.inner_obj, soft_label=soft_label)
            print(f"diverged {diverged} loss {loss}")
            print(f'train acc{train_acc}')
            if outer_iter % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge_reweight(args, model, selected_train_data, reweight_theta_mapped.detach(), args.epoch_converge_fully_train,args.inner_obj, soft_label=soft_label)
        
        start_time = time.time()
        # if args.stochastic_outer and args.subset_outer: # False and True for default setting (given by the config file)
        #     if args.use_dev_outer:
        #         valid_loader = construct_outer_subloader(args, dev_data_all)
        #     else:
        #         valid_loader = construct_outer_subloader(args, train_data)
        valid_loader = construct_outer_loader(args, selected_train_data)
        valid_soft_label = soft_label
        grad_weights_on_full_train, top1_outer, loss_outer = get_grad_weights_on_valid(args, model_copy_converged, valid_loader, reweight_theta_mapped.detach(), soft_label=valid_soft_label)
        print(f"outer acc {top1_outer}, loss_outer {loss_outer}")
        grad_reweight_theta = repass_backward(args, model, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, selected_train_data, reweight_theta_mapped, reweight_theta, soft_label=soft_label)
        reweight_theta_opt.zero_grad()
        print(f"sum grads {sum([g for g in grad_reweight_theta])}")
        with torch.no_grad():
            reweight_theta.grad += grad_reweight_theta.data
        torch.nn.utils.clip_grad_norm_(reweight_theta, args.clip_constant)
        reweight_theta_opt.step()
        if not args.use_sigmoid:
            with torch.no_grad():
                reweight_theta.data.clamp_(min=0, max=args.theta_upper_lim)
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"[timing] Reweight cost {end_time-start_time}s")
        if outer_iter % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval_for_reweight(args, model_copy_converged_ft, test_loader, name="test")
            print(f"reweight train, train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
            logging.info(f"reweight train: #iter={outer_iter}, train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
            if args.wandb:
                wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})
        reweight_theta_score=copy.deepcopy(reweight_theta)
        # print(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
        # logging.info(f"reweight train: #iter={outer_iter}, train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
        # if args.wandb:
        #     wandb.log({"train_loss": loss, "loss_outer": loss_outer, "temp":temp})
        best_reweight_theta=reweight_theta_score

    # betst_reweight_theta, fused_trained_model, fused_train_loss, fused_train_acc
    print("++++++++++++++++finished solving reweight++++++++++++++++++++")
    return best_reweight_theta, model_copy_converged_ft, 0.0, 0.0




def solve_reweight_v2(args, model, train_data, theta, selected_sample_indexs, train_loader_backward, valid_loader, test_loader, print_info, reweight_epoch, soft_label=None):
    print("[info] in solve_reweight_v2")
    selected_sample_rows, selected_sample_columns = selected_sample_indexs[0], selected_sample_indexs[1]

    for _theta in theta:
        _theta = _theta.detach()
    _saved_theta = copy.deepcopy(theta)
    accumulate_samples = [0]
    _id = 0
    if args.small_model_name.upper() == 'LSTM':
        if selected_sample_rows != None and selected_sample_columns != None:
            selected_train_dataset = []
            for row, column in zip(selected_sample_rows,selected_sample_columns):
                selected_train_dataset.append(copy.deepcopy(train_data[row][column]))
                selected_train_dataset[_id].idx = _id
                _id += 1
            selected_train_data = data.Dataset(selected_train_dataset, train_data[0].fields)
            # print(selected_train_data.fields)
            # selected_train_data = torch.tensor(train_data)[selected_sample_rows,selected_sample_columns]
            theta = torch.stack(_saved_theta)[selected_sample_rows,selected_sample_columns]
            init_theta = copy.deepcopy(theta)
            # print(theta)
            # theta = torch.stack(theta)
            # print(type(theta), theta.shape)
            # if args.use_sigmoid:
            #     reweight_theta = torch.full([len(selected_train_data)], 0, dtype=torch.float, requires_grad=True, device="cuda")
            # else:
            #     reweight_theta = torch.full([len(selected_train_data)], args.init_reweight_theta, dtype=torch.float, requires_grad=True, device="cuda")
            reweight_theta = copy.deepcopy(theta)
            reweight_theta.requires_grad_(True)
            if soft_label != None:
                for _i in range(len(train_data)):
                    accumulate_samples.append(accumulate_samples[-1]+len(train_data[_i].examples))
                accumulate_samples = torch.tensor(accumulate_samples, dtype=torch.long).to(args.device)
                _selected_sample_rows = torch.tensor(selected_sample_rows, dtype=torch.long).to(args.device)
                _selected_sample_columns = torch.tensor(selected_sample_columns, dtype=torch.long).to(args.device)
                soft_label = soft_label[accumulate_samples[_selected_sample_rows]+_selected_sample_columns]
        else:
            selected_train_data = []
            for _dataset in train_data:
                _dataset_examples = _dataset.examples[:-1] + [_dataset.examples[-1]]
                # print(f"type of _dataset_examples={type(_dataset_examples)}, len(_dataset_examples)={len(_dataset_examples)}")
                selected_train_data += copy.deepcopy(_dataset_examples)
                accumulate_samples.append(accumulate_samples[-1]+len(_dataset_examples)) 
            for _i in range(len(selected_train_data)):
                selected_train_data[_i].idx = _i
                theta = torch.stack(_saved_theta).view(-1)
            selected_train_data = data.Dataset(selected_train_data, train_data[0].fields)
            init_theta = copy.deepcopy(theta)
            reweight_theta = copy.deepcopy(theta)
            reweight_theta.requires_grad_(True)
            # print(reweight_theta)
    elif 'bert' in args.small_model_name.lower():
        if selected_sample_rows != None and selected_sample_columns != None:
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
            theta = torch.tensor([_saved_theta[row][col] for row,col in zip(selected_sample_rows,selected_sample_columns)]).to(args.device)
            # train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
            init_theta = copy.deepcopy(theta)
            reweight_theta = copy.deepcopy(theta)
            reweight_theta.requires_grad_(True)
            if soft_label != None:
                for _i in range(len(train_data)):
                    accumulate_samples.append(accumulate_samples[-1]+len(train_data[_i].idx))
                accumulate_samples = torch.tensor(accumulate_samples, dtype=torch.long).to(args.device)
                _selected_sample_rows = torch.tensor(selected_sample_rows, dtype=torch.long).to(args.device)
                _selected_sample_columns = torch.tensor(selected_sample_columns, dtype=torch.long).to(args.device)
                soft_label = soft_label[accumulate_samples[_selected_sample_rows]+_selected_sample_columns]
        else:
            _id = 0
            selected_train_data = TokenizedDataset(
                file_path=(''),
            )
            selected_train_data.text = [] # clear all the samples
            selected_train_data.ids = [] # clear all the samples
            selected_train_data.attention_mask = [] # clear all the samples
            selected_train_data.label = [] # clear all the samples
            selected_train_data.idx = [] # clear all the samples
            for row in range(args.len_LLM):
                accumulate_samples.append(accumulate_samples[-1]+len(train_data[row].idx)) 
                for column in range(len(train_data[row].idx)):
                    selected_train_data.text += [train_data[row].text[column]]
                    selected_train_data.ids += [train_data[row].ids[column]]
                    selected_train_data.attention_mask += [train_data[row].attention_mask[column]]
                    selected_train_data.label += [train_data[row].label[column]]
                    selected_train_data.idx += [_id]
                    _id += 1
            theta = torch.stack(_saved_theta).view(-1)
            # train_iter = DataLoader(selected_train_data, batch_size=args.train_batch_size, shuffle=args.shuffle_train)
            init_theta = copy.deepcopy(theta)
            reweight_theta = copy.deepcopy(theta)
            reweight_theta.requires_grad_(True)
            # print(reweight_theta)

    set_seed(args.seed)

    if args.optim =='Adam':
        reweight_theta_opt = Adam([reweight_theta], lr=args.outer_lr)
    elif args.optim =='SGD':
        reweight_theta_opt = SGD([reweight_theta], lr=args.outer_lr, momentum=0.9)
    reweight_theta.grad = torch.zeros_like(reweight_theta)
    best_reweight_theta = reweight_theta
    for outer_iter in range(reweight_epoch):
        if args.temp_anneal:
            temp = args.end_temp + (args.max_reweight_outer_iter - outer_iter)/args.max_reweight_outer_iter * (1-args.end_temp)
            print(temp, args.end_temp, args.max_reweight_outer_iter, outer_iter)
        else:
            temp = 1
        if args.use_sigmoid:
            reweight_theta_mapped = F.sigmoid(reweight_theta/temp)
        else:
            reweight_theta_mapped = reweight_theta
        # print(reweight_theta_mapped)
        if not args.disable_outer_scheduler:
            assign_learning_rate(reweight_theta_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_reweight_outer_iter)) * args.outer_lr)
        diverged = True # diverged==True means loss==nan, which means the training failed
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge_reweight(args, model, selected_train_data, reweight_theta_mapped.detach(),args.epoch_converge,args.inner_obj, soft_label=soft_label)
            print(f"diverged {diverged} loss {loss}")
            print(f'train acc{train_acc}')
            if outer_iter % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge_reweight(args, model, selected_train_data, reweight_theta_mapped.detach(), args.epoch_converge_fully_train,args.inner_obj, soft_label=soft_label)
        # if args.stochastic_outer and args.subset_outer: # False and True for default setting (given by the config file)
        #     if args.use_dev_outer:
        #         valid_loader = construct_outer_subloader(args, dev_data_all)
        #     else:
        #         valid_loader = construct_outer_subloader(args, train_data)
        start_time = time.time()
        valid_loader = construct_outer_loader(args, selected_train_data)
        valid_soft_label = soft_label
        grad_weights_on_full_train, top1_outer, loss_outer = get_grad_weights_on_valid(args, model_copy_converged, valid_loader, reweight_theta_mapped.detach(), soft_label=valid_soft_label)
        print(f"outer acc {top1_outer}, loss_outer {loss_outer}")
        grad_reweight_theta = repass_backward(args, model, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, selected_train_data, reweight_theta_mapped, reweight_theta, soft_label=soft_label)
        reweight_theta_opt.zero_grad()
        print(f"sum grads {sum([g for g in grad_reweight_theta])}")
        with torch.no_grad():
            reweight_theta.grad += grad_reweight_theta.data
        torch.nn.utils.clip_grad_norm_(reweight_theta, args.clip_constant)
        reweight_theta_opt.step()
        if not args.use_sigmoid:
            with torch.no_grad():
                reweight_theta.data.clamp_(min=0, max=args.theta_upper_lim)
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f"[timing] Reweight cost {end_time-start_time}s")
        if outer_iter % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval_for_reweight(args, model_copy_converged_ft, test_loader, name="test")
            print(f"{print_info[0]}: #{print_info[1]}={outer_iter}, beta({args.BETA}), {print_info[2]}={loss_ft}, {print_info[3]}={train_acc_ft}, {print_info[4]}={test_acc1_ft}, {print_info[5]}={test_loss_ft}")
            logging.info(f"{print_info[0]}: #{print_info[1]}={outer_iter}, beta({args.BETA}), {print_info[2]}={loss_ft}, {print_info[3]}={train_acc_ft}, {print_info[4]}={test_acc1_ft}, {print_info[5]}={test_loss_ft}")
            if args.wandb:
                wandb.log({f"{print_info[2]}": loss_ft,f"{print_info[3]}":train_acc_ft,f"{print_info[4]}": test_acc1_ft, f"{print_info[5]}":test_loss_ft})
        reweight_theta_score=copy.deepcopy(reweight_theta)
        # print(f"train_loss={loss}, loss_outer={loss_outer}, temp={temp}")
        # logging.info(f"reweight train: #iter={outer_iter}, train_loss_ft={loss_ft}, train_acc_ft={train_acc_ft}, test_acc_ft={test_acc1_ft}, test_loss_ft={test_loss_ft}")
        # if args.wandb:
        #     wandb.log({"train_loss": loss, "loss_outer": loss_outer, "temp":temp})
        best_reweight_theta=reweight_theta_score

    # betst_reweight_theta, fused_trained_model, fused_train_loss, fused_train_acc
    print("++++++++++++++++finished solving reweight++++++++++++++++++++")
    return best_reweight_theta, model_copy_converged_ft, loss_ft, train_acc_ft

