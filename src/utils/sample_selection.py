import torch
import random

NUM_BINS = 40

def choose_data_sample(args, theta, num_kept=-1):
    # args.len_LLM

    # sorted_theta_indices = torch.argsort(theta, descending=True)
    # sorted_theta_elements = theta[sorted_theta_indices]

    theta_values = torch.stack(theta).view(-1)
    sorted_theta_indices = torch.argsort(theta_values, descending=True)
    # sorted_theta_elements = theta_values[sorted_theta_indices]
    num_cols = len(theta[0])
    original_row_indices = (sorted_theta_indices // num_cols).long()
    original_col_indices = (sorted_theta_indices % num_cols).long()
    if num_kept <= 0:
        num_kept = args.num_use_samples_inner[0]
    return original_row_indices[:num_kept], original_col_indices[:num_kept]


def choose_data_sample_global(args, theta, model_importance, num_kept=-1):
    # args.len_LLM

    # sorted_theta_indices = torch.argsort(theta, descending=True)
    # sorted_theta_elements = theta[sorted_theta_indices]

    # ############### equal samples ###############
    # theta_values = model_importance.reshape(-1,1) * torch.stack(theta)
    # theta_values = theta_values.view(-1)
    # sorted_theta_indices = torch.argsort(theta_values, descending=True)
    # # sorted_theta_elements = theta_values[sorted_theta_indices]
    # num_cols = len(theta[0])
    # num_rows = len(theta)
    # original_row_indices = (sorted_theta_indices // num_cols).long()
    # original_col_indices = (sorted_theta_indices % num_cols).long()
    # if num_kept <= 0:
    #     if args.fuse_dataset_sample_selection == 'all':
    #         num_kept = sum(args.num_use_samples_inner)
    #         print(f'[debug] num_kept = {num_kept}')
    #     else:
    #         num_kept = args.num_use_samples_inner[0]
    # for model_i in range(num_rows):
    #     print(f"count of sample from model {model_i} = {(original_row_indices[:num_kept] == model_i).sum().item()}")
    # return original_row_indices[:num_kept], original_col_indices[:num_kept]
    # ############### equal samples ###############

    # ############### unequal samples ###############
    theta_values = [model_importance[k]*theta[k] for k in range(len(model_importance))]
    theta_values = torch.cat(theta_values,dim=-1)
    sorted_theta_indices = torch.argsort(theta_values, descending=True)
    # sorted_theta_elements = theta_values[sorted_theta_indices]
    num_cols = [0]
    for k in range(len(theta)):
        num_cols.append(num_cols[-1]+len(theta[k]))
        # [sample[0], sample[0]+sample[1], sample[0]+sample[1]+sample[2], ..., total]
    num_rows = len(theta)
    original_row_indices, original_col_indices = [], []
    for indice in sorted_theta_indices:
        model_index = 0
        for k in range(len(theta)-1, -1, -1):
            if indice >= num_cols[k]:
                model_index = k
                break
        original_row_indices.append(model_index)
        original_col_indices.append(indice-num_cols[model_index])
    original_row_indices = torch.tensor(original_row_indices).long()
    original_col_indices = torch.tensor(original_col_indices).long()
    # print(f"num_cols={num_cols}")
    # for i in [32, 10,100,400,1309]:
    #     print(f"sample {i}, sorted_theta_indices={sorted_theta_indices[i]}, original_row_indices={original_row_indices[i]}, original_col_indices={original_col_indices[i]}")
    if num_kept <= 0:
        if 'halfTheta' in args.fuse_dataset_sample_selection:
            max_theta = theta_values[sorted_theta_indices[0]]
            threshold_theta = max_theta * 0.5
            print(f"numbers of elements larger than threshold_theta_value={threshold_theta}=0.5*max_theta={max_theta} is {torch.sum(theta_values >= threshold_theta).item()/theta_values.numel()}")
            num_kept = torch.sum(theta_values >= threshold_theta).item()
        if 'largerTheta' in args.fuse_dataset_sample_selection:
            max_theta = theta_values[sorted_theta_indices[0]]
            min_theta = theta_values[sorted_theta_indices[-1]]
            num_bins = NUM_BINS # Define the number of bins and the range for bins
            bin_edges = torch.linspace(max_theta, min_theta, num_bins+1) # Compute bin edges
            threshold_theta = max_theta * 0.5
            histogram = torch.histc(theta_values.float(), bins=num_bins, min=min_theta, max=max_theta)# Create histogram using torch.histc
            print(f"bin_edges={bin_edges}")
            print(f"histogram={histogram}")
            for _j in range(len(histogram)):
                if histogram[_j] < 5:
                    print(f"found a bin with less than 5 samples, [{bin_edges[_j]},{bin_edges[_j+1]}]")
                    threshold_theta = bin_edges[_j+1]
                    break
            print(f"numbers of elements larger than threshold_theta_value={threshold_theta} is {torch.sum(theta_values >= threshold_theta).item()/theta_values.numel()}")
            num_kept = torch.sum(theta_values >= threshold_theta).item()
        # elif args.fuse_dataset_sample_selection == 'increasedTheta':
        #     # first collect all the samples, and then choose from them 
        elif args.fuse_dataset_sample_selection == 'all' or ('increasedTheta' in args.fuse_dataset_sample_selection):
            num_kept = sum(args.num_use_samples_inner)
            print(f'[debug] num_kept = {num_kept}')
        else:
            num_kept = args.num_use_samples_inner[0]
    # for model_i in range(num_rows):
    #     print(f"count of sample from model {model_i} = {(original_row_indices[:num_kept] == model_i).sum().item()}")
    return original_row_indices[:num_kept], original_col_indices[:num_kept]
    # ############### unequal samples ###############
    

def choose_data_sample_model_importance_weighting(args, model_importance, sample_weight, portion="imbalance", selection="random"):
    print(f'[debug] model_importance={model_importance}')
    num_kept = args.num_use_samples_inner[0]
    if portion == "imbalance":
        model_num_kept = args.num_use_samples_inner[0] * model_importance
        model_num_kept = model_num_kept.long()
        k = 0
        while torch.sum(model_num_kept) > num_kept:
            model_num_kept[k] -= 1
            k = (k+1) % args.len_LLM
        while torch.sum(model_num_kept) < num_kept:
            model_num_kept[k] += 1
            k = (k+1) % args.len_LLM
    else:
        assert portion=="balance", f"None supported portion type: {portion}"
        model_num_kept = torch.full([args.len_LLM], num_kept//args.len_LLM, dtype=torch.long, device=args.device)
        model_num_kept = model_num_kept.long()
        k = 0
        while torch.sum(model_num_kept) != num_kept:
            model_num_kept[k] += 1
            k = (k+1) % args.len_LLM
    
    original_row_indices, original_col_indices = [], []
    if selection == "random":
        for k in range(args.len_LLM): # k^th local small model
            random_sample_indexs = random.sample(range(0, args.num_use_samples_inner[k]), model_num_kept[k])
            original_col_indices += random_sample_indexs
            original_row_indices += [k] * model_num_kept[k]
    elif selection == 'all':
        num_kept = 0
        for k in range(args.len_LLM): # k^th local small model
            num_kept += args.num_use_samples_inner[k]
            original_col_indices += [_i for _i in range(args.num_use_samples_inner[k])]
            original_row_indices += [k] * args.num_use_samples_inner[k]
    else:
        pass
        assert selection == "random", f"None supported portion type: {portion}"
    return original_row_indices, original_col_indices


def add_data_sample_v2(args, theta, loss_for_data, error_for_data, correctness_for_data, based_llm):
    # theta is not used in this version
    selected_position = torch.zeros((args.num_use_samples_inner[based_llm],2),dtype=torch.long).to(args.device)
    
    _is = args.num_use_samples_inner[based_llm]-1
    # add all the syn data generated by other LLMs that are miss classified
    for j in range(args.len_LLM):
        if j != based_llm:
            for k in range(len(correctness_for_data[j])):
                if not correctness_for_data[j][k] and _is >= 0:
                    selected_position[_is] = torch.tensor([j,k],dtype=torch.long)
                    _is -= 1
    # add the data generated by the current LLM with larger errors
    error_for_self_data = error_for_data[based_llm]
    sorted_error_indices = torch.argsort(error_for_self_data, descending=False)
    # sorted_error_indices = torch.argsort(error_for_self_data, descending=True)
    sorted_error_indices = sorted_error_indices[:_is+1]
    while _is >= 0:
        selected_position[_is] = torch.tensor([based_llm,sorted_error_indices[_is]],dtype=torch.long)
        _is -= 1
    
    transposed_selected_position = selected_position.transpose(1,0)
    assert transposed_selected_position.shape == (2,args.num_use_samples_inner[based_llm]), f'the exact shape of transposed_selected_position is {transposed_selected_position.shape}, but should be (2,{args.num_use_samples_inner[based_llm]})'
    return transposed_selected_position[0], transposed_selected_position[1]

    # # theta is not used in this version, add all the samples of the same syn dataset into the fused dataset
    # selected_position = torch.zeros((args.num_use_samples_inner[based_llm],2),dtype=torch.long).to(args.device)
    
    # for k in range(len(correctness_for_data[based_llm])):
    #     selected_position[k] = torch.tensor([based_llm,k],dtype=torch.long)
    
    # transposed_selected_position = selected_position.transpose(1,0)
    # assert transposed_selected_position.shape == (2,args.num_use_samples_inner[based_llm]), f'the exact shape of transposed_selected_position is {transposed_selected_position.shape}, but should be (2,{args.num_use_samples_inner[based_llm]})'
    # return transposed_selected_position[0], transposed_selected_position[1]
