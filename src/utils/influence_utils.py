import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import torch
from typing import *
from datasets import Dataset
from tqdm import tqdm
# from contexttimer import Timer
from transformers import default_data_collator, AutoTokenizer, DataCollatorWithPadding
import logging
import resource
import psutil

from utils import influence_nn_utils, influence_misc_utils #, influence_faiss_utils

RECALL_KS = [10, 100, 1000]
NUM_NEIGHBORS = [10, 100, 1000, 10000]
RECALL_NAMES = ["Most Helpful",
                "Most Harmful",
                "Most Influencetial",
                "Least Influential"]

NUMS = [1, 10, 100, 1000, 5000, 10000]
PERCENT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

usage = resource.getrusage(resource.RUSAGE_SELF)

def compute_influences_single(
        args,
        model: torch.nn.Module,
        test_inputs,
        train_dataset: Dataset,
        batch_size: int = 1,
        data_collator=default_data_collator,
        s_test_damp: float = 5e-3,
        s_test_scale: float = 1e4,
        s_test_num_samples: int = 1000,
        s_test_iterations: int = 1,
        s_test_obj: str = "rce",
        k: int = None,
        faiss_index = None,
        faiss_index_use_mean_features_as_query: bool = False,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        weight_decay: Optional[float] = None
) -> [Dict[int, float], List[List[torch.FloatTensor]]]:
    """calculate influence score for all train instance over single test input, when test_input is
        a dataset, then the influence score is on whole val dataset."""
    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
                               "bias",
                               "LayerNorm.weight"] + [
                               n for n, p in model.named_parameters()
                               if not p.requires_grad]

    if faiss_index is not None: # always faiss_index==None in GenDataFuse code
        features = influence_misc_utils.compute_BERT_CLS_feature(model, test_inputs)
        features = features.cpu().detach().numpy()

        if faiss_index_use_mean_features_as_query is True:
            # We use the mean embedding as the final query here
            features = features.mean(axis=0, keepdims=True)

        KNN_distances, KNN_indices = faiss_index.search(
            k=k, queries=features)
    else:
        KNN_indices = None

    # if args.small_model_name.upper() == "LSTM":
    #     val_batch_size = (1,)
    #     batch_size = (2,)
    #     val_dataset = (val_dataset,)
    batch_train_data_loader = influence_misc_utils.get_dataloader(args=args, dataset=train_dataset, batch_size=batch_size, shuffle=True)
    instance_train_data_loader = influence_misc_utils.get_dataloader(args=args, dataset=train_dataset, batch_size=1, shuffle=False)
    # if isinstance(test_inputs, Dataset):
    #     test_inputs = influence_misc_utils.get_dataloader(args=args, test_inputs, batch_size=128, shuffle=False)

    # print(f'here2-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    with torch.backends.cudnn.flags(enabled=False):
        influences, s_test = influence_nn_utils.compute_influences(
            args=args,
            n_gpu=1,
            device=args.device,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            model=model,
            test_inputs=test_inputs,
            params_filter=params_filter,
            weight_decay=weight_decay,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            s_test_iterations=s_test_iterations,
            s_test_obj=s_test_obj,
            train_indices_to_include=KNN_indices,
            precomputed_s_test=precomputed_s_test,
            precomputed_grad_train_dict=precomputed_grad_train_dict)
    # print(f'here2-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')

    return influences, s_test


def compute_influences_multiple(
        args, model, val_dataset, train_dataset,
        batch_size: int = 1,
        data_collator=default_data_collator,
        s_test_damp: float = 5e-3,
        s_test_scale: float = 1e4,
        s_test_num_samples: int = 1000,
        k: int = None,
        faiss_index = None,
        faiss_index_use_mean_features_as_query: bool = False,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        precomputed_grad_train_dict: Optional[Dict[int, List[torch.FloatTensor]]] = None,
        weight_decay: Optional[float] = None):
    """calculate influence score for each train-val pair"""

    eval_instance_data_loader = influence_misc_utils.get_dataloader(args=args, dataset=val_dataset, batch_size=1, random=False)

    output_collection = {}
    for test_index, val_input in enumerate(tqdm(eval_instance_data_loader)):
        precomputed_s_test = precomputed_s_test_dict[test_index] \
            if (precomputed_s_test_dict is not None and test_index in precomputed_s_test_dict) else None

        influences, s_test = compute_influences_single(
            args=args,
            k=k,
            faiss_index=faiss_index,
            model=model,
            test_inputs=val_input,
            train_dataset=train_dataset,
            batch_size=batch_size,
            data_collator=data_collator,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            device_ids=device_ids,
            weight_decay=weight_decay,
            precomputed_s_test=precomputed_s_test,
            precomputed_grad_train_dict=precomputed_grad_train_dict,
            faiss_index_use_mean_features_as_query=faiss_index_use_mean_features_as_query
        )

        if precomputed_s_test_dict is not None:
            precomputed_s_test_dict[test_index] = s_test

        outputs = {
            "influences": influences
        }
        output_collection[test_index] = outputs
    return output_collection



def run_full_influence_functions(args, model, val_dataset, train_dataset, num_examples_to_test, s_test_num_samples=1000,
        mode: str = "all"
) -> Dict[int, Dict[str, Any]]:
    if mode not in ["all", "only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")

    val_batch_size = 1
    # if args.small_model_name.upper() == "LSTM":
    #     val_batch_size = (1,)
    #     val_dataset = (val_dataset,)
    eval_instance_data_loader = influence_misc_utils.get_dataloader(args=args, dataset=val_dataset, batch_size=val_batch_size, shuffle=True)

    num_examples_tested = 0
    len_val_dataset = len(val_dataset.examples) if args.small_model_name.upper() == 'LSTM' else len(val_dataset.idx)
    outputs_collections = [0.0] * len_val_dataset
    print(f"{len(outputs_collections)=}")
    precomputed_grad_train_dict = {}
    batch_size = min(max(1, len(train_dataset) // s_test_num_samples), 128)

    # print(f'here1-0, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    print(f"using batch_sise = {batch_size} to calculate s_test")
    model.to(args.device)
    torch.cuda.empty_cache()
    # print(f'here1-1, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    if args.small_model_name.upper() == 'LSTM':
        for _index, batch in enumerate(eval_instance_data_loader):
            print(f"in enumerate valid dataset, {_index=}")
            if num_examples_tested >= num_examples_to_test:
                break
            (inputs, lens), labels, idx = batch.text, batch.label, batch.idx
            labels = batch.label
            # print(f"{inputs=}, {lens=}, {labels=}, {idx=}")
            output = model(inputs, lens)
            
            prediction_is_correct = (output.argmax(-1).reshape(-1)==labels)[0]
            if mode == "only-correct" and prediction_is_correct is False:
                continue
            if mode == "only-incorrect" and prediction_is_correct is True:
                continue    
            
            influences, s_test = compute_influences_single(
                args=args,
                model=model,
                test_inputs=batch,
                train_dataset=train_dataset,
                batch_size=batch_size,
                # data_collator=DataCollatorWithPadding(
                #     tokenizer=tokenizer),
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,
                s_test_iterations=1 if batch_size == 1 else 4,
                weight_decay=0.005,
                precomputed_grad_train_dict=precomputed_grad_train_dict
            )
            # outputs = {
            #     "test_index": _index,
            #     "test_inputs": inputs,
            #     "influences": influences,
            #     "time": -1.0,
            #     "correct": prediction_is_correct,
            # }
            num_examples_tested += 1
            # outputs_collections[_index] = outputs
            # for k, v in influences.items():
            #     outputs_collections[k] = v
            outputs_collections[_index] = influences
            print(f"Status: #{_index} | {num_examples_tested} / {num_examples_to_test}")
    
    elif 'bert' in args.small_model_name.lower():
        for _index, batch in enumerate(eval_instance_data_loader):
            print(f"in enumerate valid dataset, {_index=}")
            if num_examples_tested >= num_examples_to_test:
                break
            # inputs, attention_mask, labels, idx = batch
            # print(f"{inputs=}, {attention_mask=}, {labels=}, {idx=}")
            # inputs = inputs.to(args.device)
            # attention_mask = attention_mask.to(args.device)
            # labels = labels.to(args.device)
            # idx = idx.to(args.device)
            # output = model(inputs, attention_mask=attention_mask, labels=labels).logits
            
            # prediction_is_correct = (output.argmax(-1).reshape(-1)==labels)[0]
            # if mode == "only-correct" and prediction_is_correct is False:
            #     continue
            # if mode == "only-incorrect" and prediction_is_correct is True:
            #     continue

            influences, s_test = compute_influences_single(
                args=args,
                model=model,
                test_inputs=batch,
                train_dataset=train_dataset,
                batch_size=batch_size,
                # data_collator=DataCollatorWithPadding(
                #     tokenizer=tokenizer),
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,
                s_test_iterations=1 if batch_size == 1 else 4,
                weight_decay=0.005,
                precomputed_grad_train_dict=precomputed_grad_train_dict
            )
            # outputs = {
            #     "test_index": _index,
            #     "test_inputs": inputs,
            #     "influences": influences,
            #     "time": -1.0,
            #     "correct": prediction_is_correct,
            # }
            num_examples_tested += 1
            # outputs_collections[_index] = outputs
            # for k, v in influences.items():
            #     outputs_collections[k] = v
            outputs_collections[_index] = influences
            print(f"Status: #{_index} | {num_examples_tested} / {num_examples_to_test}")
    
    del precomputed_grad_train_dict
    model.to("cpu")
    torch.cuda.empty_cache()
    # print(f'here1-2, {psutil.virtual_memory().available/1024/1024=}M, {psutil.virtual_memory().used/1024/1024=}M, {usage.ru_maxrss/1024=}M, {torch.cuda.memory_reserved()/1024/1024=}M, {torch.cuda.memory_allocated()/1024/1024=}M')
    outputs_collections = np.asarray(outputs_collections)
    outputs_collections = list(np.mean(outputs_collections, axis=0))
    return outputs_collections


# def run_full_influence_functions_dataset(
#         args,
#         model: torch.nn.Module,
#         val_dataset: Dataset,
#         train_dataset: Dataset,
#         tokenizer: AutoTokenizer,
#         s_test_num_samples: int = 1000,
#         s_test_obj: str = "rce",
#         weight_decay: Optional[float] = None
# ) -> Dict[int, Dict[str, Any]]:
#     batch_size = min(max(1, len(train_dataset) // s_test_num_samples), 128)
#     print(f"using batch_sise = {batch_size} to calculate s_test")

#     # with Timer() as timer:
#     #     influences, s_test = compute_influences_single(
#     #         args=args,
#     #         model=model,
#     #         test_inputs=val_dataset,
#     #         train_dataset=train_dataset,
#     #         batch_size=batch_size,
#     #         data_collator=DataCollatorWithPadding(
#     #             tokenizer=tokenizer),
#     #         s_test_damp=5e-3,
#     #         s_test_scale=1e4,
#     #         s_test_num_samples=s_test_num_samples,
#     #         s_test_iterations=1,
#     #         s_test_obj=s_test_obj,
#     #         weight_decay=weight_decay
#     #     )

#     #     outputs = {
#     #         -1: {"influences": influences,
#     #              "time": timer.elapsed}
#     #     }
#     influences, s_test = compute_influences_single(
#         args=args,
#         model=model,
#         test_inputs=val_dataset,
#         train_dataset=train_dataset,
#         batch_size=batch_size,
#         data_collator=DataCollatorWithPadding(
#             tokenizer=tokenizer),
#         s_test_damp=5e-3,
#         s_test_scale=1e4,
#         s_test_num_samples=s_test_num_samples,
#         s_test_iterations=1,
#         s_test_obj=s_test_obj,
#         weight_decay=weight_decay
#     )

#     outputs = {
#         -1: {"influences": influences,
#                 "time": -1.0}
#     }
#     return outputs
