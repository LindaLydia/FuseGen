# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import Dict, List, Union, Optional, Tuple, Iterator, Any
import torch.nn.functional as F
import resource
import psutil
import gc

usage = resource.getrusage(resource.RUSAGE_SELF)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def convert_ids_to_string(
        tokenizer: PreTrainedTokenizer,
        ids: torch.LongTensor) -> str:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(tokens)


def get_loss_with_weight_decay(
        args,
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs,
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        obj: str = "rce"
) -> float:
    model.train()
    # for k, v in inputs.items():
    #     inputs[k] = v.to(device)

    batch = inputs
    if args.small_model_name.upper() == "LSTM":
        (_inputs, lens), labels, idx = batch.text, batch.label, batch.idx
        outputs = model(_inputs, lens)
        logits = outputs
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(outputs, labels)
    elif 'bert' in args.small_model_name.lower():
        if not 'squad' in args.task_name:
            _inputs, attention_mask, labels, idx = batch
            _inputs = _inputs.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            idx = idx.to(args.device)
            outputs = model(_inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        else:
            contexts, _inputs, attention_mask, offset_mapping, sample_mapping, labels, idx = batch
            _inputs = _inputs.squeeze(dim=0).to(args.device)
            attention_mask = attention_mask.squeeze(dim=0).to(args.device)
            offset_mapping = offset_mapping.to(args.device)
            sample_mapping = sample_mapping.to(args.device)
            start_positions, end_positions, answer_text = [], [], []
            for _start_idx, _text in zip(labels["answer_start"], labels["text"]):
                for _ in range(len(_inputs)):
                    start_positions.append(_start_idx[0].item())
                    end_positions.append(_start_idx[0].item()+len(_text[0]))
                    answer_text.append(_text[0])
            start_positions = torch.tensor(start_positions).long().to(args.device)
            end_positions = torch.tensor(end_positions).long().to(args.device)
            idx = idx.to(args.device)
            # labels={'answer_start': [tensor([162])], 'text': [('Palawan',)]}
            
            print(f"{_inputs=}, {_inputs.shape=}")
            print(f"{attention_mask=}, {attention_mask.shape=}")
            print(f"{start_positions=}, {start_positions.shape=}")
            print(f"{end_positions=}, {end_positions.shape=}")
            print(f"{idx=}, {idx.shape=}")
            print(f"{offset_mapping=}, {offset_mapping.shape=}")
            print(f"{sample_mapping=}, {sample_mapping.shape=}")
            print(f"{labels=}")

            output = model(_inputs, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            start_logits = output.start_logits
            end_logits = output.end_logits
            print(f"[debug] {start_logits=}")
            print(f"[debug] {end_logits=}")
            print(f"[debug] {start_positions=}")
            print(f"[debug] {end_positions=}")
            print(f"[debug] {start_logits.shape=}, {end_logits.shape=}, {start_positions.shape=}, {end_positions.shape=}")
            loss = output.loss
            s = F.softmax(start_logits,dim=-1)[torch.arange(start_logits.size(0)),start_positions].reshape(start_logits.size(0),1)
            e = F.softmax(end_logits,dim=-1)[torch.arange(end_logits.size(0)),end_positions].reshape(end_logits.size(0),1)
            log_s = F.log_softmax(start_logits,dim=-1)[torch.arange(start_logits.size(0)),start_positions].reshape(start_logits.size(0),1)
            log_e = F.log_softmax(end_logits,dim=-1)[torch.arange(end_logits.size(0)),end_positions].reshape(end_logits.size(0),1)
            logits = torch.cat([s,e],dim=-1).to(args.device)
            log_logits = torch.cat([log_s,log_e],dim=-1).to(args.device)


    if obj == "rce" or obj == "sce":
        if not 'squad' in args.task_name:
            one_hot = labels.new_zeros(len(labels), args.num_classes, dtype=float).scatter_(1, labels.view(-1, 1), 1)
            one_hot = F.softmax(one_hot, dim=1)
            rce_loss = F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) - torch.log(one_hot)) - torch.mul(
                        F.softmax(logits, dim=1), F.log_softmax(logits, dim=1))
        else:
            one_hot = torch.ones(logits.shape[0], 2, dtype=float).to(args.device)
            one_hot = F.softmax(one_hot, dim=1)
            rce_loss = logits * (log_logits - torch.log(one_hot)) - torch.mul(logits, log_logits)
        rce_loss = rce_loss.sum(-1).mean()
        if obj == "sce":
            loss = rce_loss + 0.1 * loss
        else:
            loss = rce_loss

    if n_gpu > 1:
        # mean() to average on multi-gpu parallel training
        loss = loss.mean()

    # In PyTorch, weight-decay loss and gradients are calculated in
    # optimizers rather in nn.Module, so we have to manually specify
    # this for the loss here.
    if weight_decay is not None:
        no_decay = (
            weight_decay_ignores
            if weight_decay_ignores
               is not None else [])

        weight_decay_loss = torch.cat([
            p.square().view(-1)
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]).sum() * weight_decay
        loss = loss + weight_decay_loss

    return loss


def compute_gradients(
        args,
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs,
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        obj: str = 'ce'
) -> List[torch.FloatTensor]:
    if params_filter is None:
        params_filter = []

    # an example of psutil.virtual_memory(): svmem(total=10367352832, available=6472179712, percent=37.6, used=8186245120, free=2181107712, active=4748992512, inactive=2758115328, buffers=790724608, cached=3500347392, shared=787554304, slab=199348224)
    # print(f'here5-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    model.zero_grad()
    # if args.small_model_name.upper() == "LSTM":
    #     (inputs, lens), labels, idx = batch.text, batch.label, batch.idx
    #     labels = batch.label
    #     output = model(inputs, lens)   
    # elif 'bert' in args.small_model_name.lower():
    #     inputs, attention_mask, labels, idx = batch
    #     inputs = inputs.to(args.device)
    #     attention_mask = attention_mask.to(args.device)
    #     labels = labels.to(args.device)
    #     idx = idx.to(args.device)
    #     output = model(inputs, attention_mask=attention_mask, labels=labels).logits

    if args.small_model_name.upper() == "LSTM":
        batch = inputs
        (_inputs, lens), labels, idx = batch.text, batch.label, batch.idx
        assert len(labels) == 1, f"Error, in GenDataFuse code, len(labels) should be 1, but now {len(labels)} with {labels=}"
        # output = model(inputs, lens)   
    elif 'bert' in args.small_model_name.lower():
        if 'squad' not in args.task_name:
            batch = inputs
            _inputs, attention_mask, labels, idx = batch
            # inputs = inputs.to(args.device)
            # attention_mask = attention_mask.to(args.device)
            # labels = labels.to(args.device)
            # idx = idx.to(args.device)
            assert len(labels) == 1, f"Error, in GenDataFuse code, len(labels) should be 1, but now {len(labels)} with {labels=}"
            # output = model(inputs, attention_mask=attention_mask, labels=labels).logits
        else:
            batch = inputs

    loss = get_loss_with_weight_decay(
        args=args,
        device=device, n_gpu=n_gpu,
        model=model, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        obj=obj
    )

    grad = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)

    # print(f'here5-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    del loss
    torch.cuda.empty_cache()
    return [a.detach() for a in grad]
    
    if isinstance(inputs, torch.utils.data.DataLoader):
        grad = None
        for _inputs in inputs:
            _grad = compute_gradients(
                args=args,
                model=model,
                n_gpu=n_gpu,
                device=device,
                inputs=_inputs,
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores,
                obj=obj)
            if grad is None:
                grad = _grad
            else:
                grad = [a+b for a, b in zip(grad, _grad)]
        grad = [a / len(inputs) for a in grad]  # average over all instances, note here assume using mean loss
    else:
        loss = get_loss_with_weight_decay(
            args=args,
            device=device, n_gpu=n_gpu,
            model=model, inputs=inputs,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            obj=obj
        )

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=[
                param for name, param
                in model.named_parameters()
                if name not in params_filter],
            create_graph=True)

    return [a.detach() for a in grad]


def compute_hessian_vector_products(
        args,
        device: torch.device,
        n_gpu: int,
        model: torch.nn.Module,
        inputs,
        vectors: torch.FloatTensor,
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]]
) -> List[torch.FloatTensor]:
    if params_filter is None:
        params_filter = []

    # print(f'here6-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    model.zero_grad()
    loss = get_loss_with_weight_decay(
        args=args,
        model=model, n_gpu=n_gpu,
        device=device, inputs=inputs,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores)

    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        create_graph=True)
    
    del loss
    gc.collect()
    torch.cuda.empty_cache()

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[
            param for name, param
            in model.named_parameters()
            if name not in params_filter],
        grad_outputs=vectors,
        only_inputs=True
    )
    # print(f'here6-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    del grad_tuple
    gc.collect()
    torch.cuda.empty_cache()
    return [a.detach() for a in grad_grad_tuple]


def compute_s_test(
        args,
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs,
        train_data_loaders, # of a list
        params_filter: Optional[List[str]],
        weight_decay: Optional[float],
        weight_decay_ignores: Optional[List[str]],
        damp: float,
        scale: float,
        s_test_iterations: int = 1,
        num_samples: Optional[int] = None,
        verbose: bool = True,
        s_test_obj: str = "rce"
) -> List[torch.FloatTensor]:
    # print(f'here4-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    v = compute_gradients(
        args=args,
        model=model,
        n_gpu=n_gpu,
        device=device,
        inputs=test_inputs,
        params_filter=params_filter,
        weight_decay=weight_decay,
        weight_decay_ignores=weight_decay_ignores,
        obj=s_test_obj)
    if verbose is True:
        print("init v norm: ", v[0].norm().item())
    inverse_hvp = None
    # print(f'here4-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    for _ in range(s_test_iterations):
        # Technically, it's hv^-1
        last_estimate = list(v).copy()
        with tqdm(total=num_samples) as pbar:
            for data_loader in train_data_loaders:
                for i, batch in enumerate(data_loader):
                    this_estimate = compute_hessian_vector_products(
                        args=args,
                        model=model,
                        n_gpu=n_gpu,
                        device=device,
                        vectors=last_estimate,
                        inputs=batch, # of a batch, not single
                        params_filter=params_filter,
                        weight_decay=weight_decay,
                        weight_decay_ignores=weight_decay_ignores)
                    # Recursively caclulate h_estimate
                    # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
                    gc.collect()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        new_estimate = [
                            a + (1 - damp) * b - c / scale
                            for a, b, c in zip(v, last_estimate, this_estimate)
                        ]

                    pbar.update(1)
                    if verbose is True:
                        new_estimate_norm = new_estimate[0].norm().item()
                        last_estimate_norm = last_estimate[0].norm().item()
                        estimate_norm_diff = new_estimate_norm - last_estimate_norm
                        pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")

                    last_estimate = new_estimate
                    if num_samples is not None and i > num_samples:
                        break

        if inverse_hvp is None:
            inverse_hvp = [X / scale for X in last_estimate]
        else:
            inverse_hvp = [
                pre + X / scale for pre, X in zip(inverse_hvp, last_estimate)
            ]
    # print(f'here4-3, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

    # average across multiple runs
    inverse_hvp = [i / s_test_iterations for i in inverse_hvp]
    del this_estimate
    del last_estimate
    del v
    gc.collect()
    torch.cuda.empty_cache()
    # print(f"{inverse_hvp=}, {len(inverse_hvp)=}")
    print(f"{len(inverse_hvp)=}")
    return inverse_hvp


def compute_influences(
        args,
        n_gpu: int,
        device: torch.device,
        model: torch.nn.Module,
        test_inputs, # of a single test sample
        batch_train_data_loader,
        instance_train_data_loader,
        params_filter: Optional[List[str]] = None,
        weight_decay: Optional[float] = None,
        weight_decay_ignores: Optional[List[str]] = None,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        s_test_obj: str = "rce",
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int]]] = None,
        verbose: bool = True,):
    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if weight_decay_ignores is None:
        # https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/trainer.py#L325
        weight_decay_ignores = [
            "bias",
            "LayerNorm.weight"]

    # print(f'here3-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = compute_s_test(
            args=args,
            n_gpu=n_gpu,
            device=device,
            model=model,
            test_inputs=test_inputs, # single test sample
            train_data_loaders=[batch_train_data_loader],
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            damp=s_test_damp,
            scale=s_test_scale,
            s_test_iterations=s_test_iterations,
            num_samples=s_test_num_samples,
            verbose=verbose,
            s_test_obj=s_test_obj
        )

    # print(f'here3-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    influences = {}
    # for index, train_inputs in enumerate(instance_train_data_loader):
    for index, batch in enumerate(instance_train_data_loader):

        # Skip indices when a subset is specified to be included, 
        # should have train_indices_to_include==None in GenDataFuse code
        if (train_indices_to_include is not None) and (
                index not in train_indices_to_include):
            continue

        if (precomputed_grad_train_dict is not None and index not in precomputed_grad_train_dict) or \
                precomputed_grad_train_dict is None:
            grad_z = compute_gradients(
                args=args,
                n_gpu=n_gpu,
                device=device,
                model=model,
                inputs=batch,
                params_filter=params_filter,
                weight_decay=weight_decay,
                weight_decay_ignores=weight_decay_ignores)
            if precomputed_grad_train_dict is not None:
                precomputed_grad_train_dict[index] = [i.cpu() for i in grad_z]
        else:
            grad_z = [i.to(s_test[0].device) for i in precomputed_grad_train_dict[index]]

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)]

        influences[index] = sum(influence).item()

    # print(f'here3-3, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    print(f"in file <influence_nn_utils>, {influences=}")

    influences_list = [0.0] * len(list(influences.keys()))
    for k, v in influences.items():
        influences_list[k] = v
    
    gc.collect()
    torch.cuda.empty_cache()
    return influences_list, s_test
    # return influences, s_test
