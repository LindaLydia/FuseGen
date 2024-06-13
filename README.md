# FuseGen
This is the code for paper FuseGen.

We implement our code for data generation basing on the code provided in [this repo](https://github.com/jiacheng-ye/ZeroGen).

### Key Components:
1. Cross-model Data Quality Evaluation in Cross-model Data Generation:
    1. Cross-model Variability: Please see function `sample_dynamic_selection` in file `./src/utils/basic_utils.py`.
    2. Sample Influence: Please see function `run_full_influence_functions` in file `./src/utils/influence_utils.py`.
2. Cross-model Data Quality Improvement:
    1. Self-boosting Weight Adjustment: Please see function `weight_decay` in `./src/utils/weight_adjust.py`.
