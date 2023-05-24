#!/bin/bash

hidden_dim=256
latent_dim=128
depth=5
lr_inr=5e-4
inner_steps=3
test_inner_steps=3
lr_code=0.01
batch_size=64
epochs=40000
dataset_name=Electricity
num_frequencies=64
max_frequencies=10
base_frequency=2
log_sampling=True
length_of_interest=4000
look_back_window=512
passed_ratio=0.80
horizon_ratio=0.30
version=1
horizon=96

python3 -u inr_forecast.py "data.look_back_window=${look_back_window}" "data.horizon=${horizon}" "data.length_of_interest=${length_of_interest}" "data.dataset_name=${dataset_name}" "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "inr.depth=${depth}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "optim.lr_inr=${lr_inr}" "optim.lr_code=${lr_code}" "optim.inner_steps=${inner_steps}" "data.version=${version}" "inr.passed_ratio=${passed_ratio}" "inr.horizon_ratio=${horizon_ratio}" "inr.log_sampling=${log_sampling}" "inr.num_frequencies=${num_frequencies}" "inr.base_frequency=${base_frequency}" "inr.max_frequencies=${max_frequencies}"
