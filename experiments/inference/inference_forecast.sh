#!/bin/bash

dataset_name=Electricity
epochs=40000
horizon=96
version=0
test_inner_steps=3

python3 -u inference_forecast.py --dataset_name $dataset_name --epochs $epochs --horizon $horizon --version $version --test_inner_steps $test_inner_steps 