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
length_of_interest=2000
sample_ratio_batch=0.6
version=0
draw_ratio=0.10

python3 -u inr_imputation.py "optim.sample_ratio_batch=${sample_ratio_batch}" "data.length_of_interest=${length_of_interest}" "data.dataset_name=${dataset_name}" "data.draw_ratio=${draw_ratio}" "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "inr.depth=${depth}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "optim.lr_inr=${lr_inr}"  "optim.lr_code=${lr_code}" "optim.inner_steps=${inner_steps}" "optim.test_inner_steps=${test_inner_steps}" "data.version=${version}"