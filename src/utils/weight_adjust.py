import os, sys
import numpy as np
import torch
from tqdm import tqdm

import time

from utils.basic_utils import *
from utils.qa_utils import *

def weight_decay(args, current_outer_iter_trained_model, train_data, theta_mapped, beta, _type="none", single_dataset=False, use_soft_label=False):
    
    num_models = args.len_LLM
    if single_dataset:
        num_models = 1
        _type =  _type.replace('NoSelf','')
        _type =  _type.replace('OnlySelf','')
        _type =  _type.replace('TestAll','')

    original_theta_sum = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    for j in range(num_models):   
        original_theta_sum[j] = torch.sum(theta_mapped[j])
    
    model_total_acc = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    prediction_for_model_data = [[torch.zeros((len(train_data[_j]),),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    logits_for_model_data = [[torch.zeros((len(train_data[_j]),args.num_classes),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    
    sum_of_time_for_weight_adjust = 0.0
    for i in range(num_models): # the i^{th} local small model 
        # record the loss, error, correctness of the i^{th} local small model on each syn dataset
        loss_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        error_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        correctness_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.bool).to(args.device) for _j in range(num_models)]
        # get the loss fo current data with small local models trained with syn dataset of other LLMs

        if num_models == 1:
            _type =  _type.replace('NoSelf','')
            _type =  _type.replace('OnlySelf','')
            _type =  _type.replace('TestAll','')
        for j in range(num_models): # test with j^{th} synthetic dataset
            if args.adjustable_beta:
                beta = 1 + np.sqrt(max(1,2*np.log(len(train_data[j])/args.max_outer_iter)))
                print(f"BETA change with #data, current beta={beta}")
            else:
                print(f"fix BETA, current beta={args.BETA}")
            if (not 'TestAll' in _type) and 'NoSelf' in _type:
                if i == j:
                    continue
            if (not 'TestAll' in _type) and 'OnlySelf' in _type:
                if i != j:
                    continue
            loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample, logtis_per_sample = eval_get_pred(args, current_outer_iter_trained_model[i], train_data[j], use_soft_label)
            loss_for_data[j] = loss_per_sample
            error_for_data[j] = error_per_sample
            correctness_for_data[j] = correctness_per_sample
            logits_for_model_data[i][j] = logtis_per_sample
            print(f"{logtis_per_sample=}")
            print(f"{logits_for_model_data[i][j]=}")
            # print(f"loss_per_sample={loss_per_sample}")
            # print(f"error_per_sample={error_per_sample}")
            # print(f"correctness_per_sample={correctness_per_sample}")
            ######### change theta #########
            if 'NoSelfTestAll' in _type:
                if i == j:
                    continue
            if 'OnlySelfTestAll' in _type:
                if i != j:
                    continue
            start_time = time.time()
            for s in range(len(theta_mapped[j])):
                if 'errorOnlyWrong' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (1 if correctness_for_data[j][s] else beta**((error_for_data[j][s])*(-1)))
                elif 'error' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (beta**((error_for_data[j][s])*(1 if correctness_for_data[j][s] else -1)))
                elif 'lossOnlyWrong' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (1 if correctness_for_data[j][s] else beta**((loss_for_data[j][s])*(-1)))
                elif 'loss' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (beta**((loss_for_data[j][s])*(1 if correctness_for_data[j][s] else -1)))
                else:
                    assert _type=="none", f"None supported weight adjustment type: {_type}"
                    # print(f'[info] No weight adjustment.')
            end_time = time.time()
            print(f"[timing] WeightAdjust cost for model {i} {end_time-start_time}s")
            sum_of_time_for_weight_adjust += (end_time-start_time)
            ######### change theta #########
        correct, total = 0, 0
        for j in range(len(correctness_for_data)):
            correct += torch.sum(correctness_for_data[j])
            total += correctness_for_data[j].numel()
        model_total_acc[i] = correct/total
    print(f"[debug] total ACC = {model_total_acc}")

    start_time = time.time()
    ######### normalize theta #########
    for j in range(num_models):   
        current_theta_sum = torch.sum(theta_mapped[j])
        # print(f"original_theta_sum[j]={original_theta_sum[j]}, current_theta_sum={current_theta_sum}")
        # print(f"new theta_mapped[j]={theta_mapped[j]}")
        theta_mapped[j] = theta_mapped[j] * original_theta_sum[j] / current_theta_sum
        # theta[j] = copy.deepcopy(theta_mapped[j])
        # theta_score = copy.deepcopy(theta[j])
        # # print(f"new theta[j] = {theta[j]}")
        # best_theta[j]=theta_score
    ######### normalize theta #########
    end_time = time.time()
    print(f"[timing] WeightAdjust cost for normalization {end_time-start_time}s")
    sum_of_time_for_weight_adjust += (end_time-start_time)
    print(f"[timing] WeightAdjust {sum_of_time_for_weight_adjust}s")

    logits_for_model_data = torch.stack([torch.stack(_c) for _c in logits_for_model_data])
    torch.save(logits_for_model_data, f"{args.result_file_path}/logits_withoutlabel_of_data.pth")

    return theta_mapped, model_total_acc


def weight_decay_qa(args, current_outer_iter_trained_model, train_data, theta_mapped, beta, _type="none", single_dataset=False, use_soft_label=False):
    
    num_models = args.len_LLM
    if single_dataset:
        num_models = 1
        _type =  _type.replace('NoSelf','')
        _type =  _type.replace('OnlySelf','')
        _type =  _type.replace('TestAll','')

    original_theta_sum = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    for j in range(num_models):   
        original_theta_sum[j] = torch.sum(theta_mapped[j])
    
    model_total_acc = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    # prediction_for_model_data = [[torch.zeros((len(train_data[_j]),),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    # logits_for_model_data = [[torch.zeros((len(train_data[_j]),args.num_classes),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    for i in range(num_models): # the i^{th} local small model 
        # record the loss, error, correctness of the i^{th} local small model on each syn dataset
        loss_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        error_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        correctness_for_data = [torch.zeros((len(train_data[_j]),),dtype=torch.bool).to(args.device) for _j in range(num_models)]
        # get the loss fo current data with small local models trained with syn dataset of other LLMs

        if num_models == 1:
            _type =  _type.replace('NoSelf','')
            _type =  _type.replace('OnlySelf','')
            _type =  _type.replace('TestAll','')
        for j in range(num_models): # test with j^{th} synthetic dataset
            if args.adjustable_beta:
                beta = 1 + np.sqrt(max(1,2*np.log(len(train_data[j])/args.max_outer_iter)))
                print(f"BETA change with #data, current beta={beta}")
            else:
                print(f"fix BETA, current beta={args.BETA}")
            if (not 'TestAll' in _type) and 'NoSelf' in _type:
                if i == j:
                    continue
            if (not 'TestAll' in _type) and 'OnlySelf' in _type:
                if i != j:
                    continue
            loss_per_sample, error_per_sample, correctness_per_sample = eval_get_pred_qa(args, current_outer_iter_trained_model[i], train_data[j], use_soft_label=False)
            loss_for_data[j] = loss_per_sample
            error_for_data[j] = error_per_sample
            correctness_for_data[j] = correctness_per_sample
            # logits_for_model_data[i][j] = logtis_per_sample
            # print(f"{logtis_per_sample=}")
            # print(f"{logits_for_model_data[i][j]=}")

            print(f"correctness_per_sample={correctness_per_sample}")
            ######### change theta #########
            if 'NoSelfTestAll' in _type:
                if i == j:
                    continue
            if 'OnlySelfTestAll' in _type:
                if i != j:
                    continue
            for s in range(len(theta_mapped[j])):
                if 'errorOnlyWrong' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (1 if correctness_for_data[j][s] else beta**((error_for_data[j][s])*(-1)))
                elif 'error' in _type:
                    theta_mapped[j][s] = theta_mapped[j][s] * (beta**((error_for_data[j][s])*(1 if correctness_for_data[j][s] else -1)))
                # elif 'lossOnlyWrong' in _type:
                #     theta_mapped[j][s] = theta_mapped[j][s] * (1 if correctness_for_data[j][s] else beta**((loss_for_data[j][s])*(-1)))
                # elif 'loss' in _type:
                #     theta_mapped[j][s] = theta_mapped[j][s] * (beta**((loss_for_data[j][s])*(1 if correctness_for_data[j][s] else -1)))
                else:
                    assert _type=="none", f"None supported weight adjustment type: {_type}"
                    # print(f'[info] No weight adjustment.')
            ######### change theta #########
        # correct, total = 0, 0
        # for j in range(len(correctness_for_data)):
        #     correct += torch.sum(correctness_for_data[j])
        #     total += correctness_for_data[j].numel()
        # model_total_acc[i] = correct/total
        model_total_acc = None
    # print(f"[debug] total ACC = {model_total_acc}")
    ######### normalize theta #########
    for j in range(num_models):   
        current_theta_sum = torch.sum(theta_mapped[j])
        # print(f"original_theta_sum[j]={original_theta_sum[j]}, current_theta_sum={current_theta_sum}")
        # print(f"new theta_mapped[j]={theta_mapped[j]}")
        theta_mapped[j] = theta_mapped[j] * original_theta_sum[j] / current_theta_sum
        # theta[j] = copy.deepcopy(theta_mapped[j])
        # theta_score = copy.deepcopy(theta[j])
        # # print(f"new theta[j] = {theta[j]}")
        # best_theta[j]=theta_score
    ######### normalize theta #########

    # logits_for_model_data = torch.stack([torch.stack(_c) for _c in logits_for_model_data])
    # torch.save(logits_for_model_data, f"{args.result_file_path}/logits_withoutlabel_of_data.pth")

    return theta_mapped, model_total_acc



def model_importance_estimation(args, current_outer_iter_trained_model, valid_data, _type="none", single_dataset=False):
    num_models = args.len_LLM
    if single_dataset:
        num_models = 1

    # original_theta_sum = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    # for j in range(num_models):   
    #     original_theta_sum[j] = torch.sum(theta_mapped[j])
    
    model_total_acc = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    prediction_for_model_data = [[torch.zeros((len(valid_data[_j]),),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    for i in range(num_models): # the i^{th} local small model 
        # record the loss, error, correctness of the i^{th} local small model on each syn dataset
        loss_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        error_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        correctness_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.bool).to(args.device) for _j in range(num_models)]
        # get the loss fo current data with small local models trained with syn dataset of other LLMs

        for j in range(num_models): # test with j^{th} synthetic dataset
            loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample = eval_get_loss(args, current_outer_iter_trained_model[i], valid_data[j])
            loss_for_data[j] = loss_per_sample
            error_for_data[j] = error_per_sample
            correctness_for_data[j] = correctness_per_sample
            # print(f"loss_per_sample={loss_per_sample}")
            # print(f"error_per_sample={error_per_sample}")
            # print(f"correctness_per_sample={correctness_per_sample}")

        correct, total = 0, 0
        for j in range(len(correctness_for_data)):
            # print(f"correct for {j}th data is {torch.sum(correctness_for_data[j])}, with total={correctness_for_data[j].numel()}")
            correct += torch.sum(correctness_for_data[j])
            total += correctness_for_data[j].numel()
        # print("correct=", correct, "total=", total)
        model_total_acc[i] = correct/total
    print(f"[debug] total ACC on small_valid_dataset = {model_total_acc}")


    return model_total_acc


def model_importance_estimation_qa(args, current_outer_iter_trained_model, valid_data, _type="none", single_dataset=False):
    num_models = args.len_LLM
    if single_dataset:
        num_models = 1

    # original_theta_sum = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    # for j in range(num_models):   
    #     original_theta_sum[j] = torch.sum(theta_mapped[j])
    
    model_total_acc = torch.zeros((num_models,),dtype=torch.float32).to(args.device)
    prediction_for_model_data = [[torch.zeros((len(valid_data[_j]),),dtype=torch.long).to(args.device) for _j in range(num_models)] for _i in range(num_models)]
    for i in range(num_models): # the i^{th} local small model 
        # record the loss, error, correctness of the i^{th} local small model on each syn dataset
        loss_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        error_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.float32).to(args.device) for _j in range(num_models)]
        correctness_for_data = [torch.zeros((len(valid_data[_j]),),dtype=torch.bool).to(args.device) for _j in range(num_models)]
        # get the loss fo current data with small local models trained with syn dataset of other LLMs

        for j in range(num_models): # test with j^{th} synthetic dataset
            loss_per_sample, error_per_sample, correctness_per_sample, prediction_per_sample = eval_get_loss_qa(args, current_outer_iter_trained_model[i], valid_data[j])
            loss_for_data[j] = loss_per_sample
            error_for_data[j] = error_per_sample
            correctness_for_data[j] = correctness_per_sample
            # print(f"loss_per_sample={loss_per_sample}")
            # print(f"error_per_sample={error_per_sample}")
            # print(f"correctness_per_sample={correctness_per_sample}")

        correct, total = 0, 0
        for j in range(len(correctness_for_data)):
            # print(f"correct for {j}th data is {torch.sum(correctness_for_data[j])}, with total={correctness_for_data[j].numel()}")
            correct += torch.sum(correctness_for_data[j])
            total += correctness_for_data[j].numel()
        # print("correct=", correct, "total=", total)
        model_total_acc[i] = correct/total
    print(f"[debug] total ACC on small_valid_dataset = {model_total_acc}")


    return model_total_acc