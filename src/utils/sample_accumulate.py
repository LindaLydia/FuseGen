import numpy as np
import torch

def calculate_num_sample_next(current_num_sample_each_llm, each_step_accumulated, num_LLM, LLM_importance, i_step):

    LLM_importance = LLM_importance / torch.sum(LLM_importance)
    total_delta = each_step_accumulated[i_step+1] - each_step_accumulated[i_step]
    each_llm_delta = (total_delta * LLM_importance).long()
    print(f"step {i_step}, total_delta={total_delta}, LLM_importance={LLM_importance}, each_llm_delta={each_llm_delta}")
    if torch.sum(each_llm_delta).item() != total_delta:
        assert total_delta-num_LLM < torch.sum(each_llm_delta).item() < total_delta, f"total_delta-num_LLM={total_delta-num_LLM} < {torch.sum(each_llm_delta).item()} < total_delta={total_delta} does not hold"
        additional = total_delta - torch.sum(each_llm_delta)
        sorted_most_important = torch.argsort(LLM_importance, descending=True)[:additional]
        for ia in sorted_most_important:
            each_llm_delta[ia] += 1
        assert torch.sum(each_llm_delta).item() == total_delta, f"adding more data with bug"
    
    delta_num_sample_each_llm = each_llm_delta
    new_num_sample_each_llm = current_num_sample_each_llm + each_llm_delta
    return new_num_sample_each_llm, delta_num_sample_each_llm