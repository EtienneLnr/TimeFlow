#!/bin/bash

dataset_name=Electricity
epochs=40000
draw_ratio=0.1
version=0
test_inner_steps=3

python3 -u inference_imputation.py --dataset_name $dataset_name --epochs $epochs --draw_ratio $draw_ratio --version $version --test_inner_steps $test_inner_steps 