import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from omegaconf import OmegaConf
import argparse

from src.metalearning.metalearning_imputation import outer_step
from src.network import  ModulatedFourierFeatures
from src.utils import (
    DatasetSamples,
    set_seed,
    fixed_subsampling_series_imputations,
    fixed_sampling_series_imputations,
    z_normalize,
    z_normalize_out,
    z_denormalize_out,
    convert_error,
)

from sklearn.metrics import mean_absolute_error as mae

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Electricity')
parser.add_argument('--epochs', type=int, default=40000)
parser.add_argument('--draw_ratio', type=float, default=0.1)
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--test_inner_steps', type=int, default=3)
args = parser.parse_args()


dataset_name = args.dataset_name
epochs = args.epochs
draw_ratio = args.draw_ratio
version = args.version
test_inner_steps = args.test_inner_steps


RESULTS_DIR = str(Path(__file__).parents[2]) + '/save_models/'


inr_training_input = torch.load(f"{RESULTS_DIR}/models_imputation_{dataset_name}_{draw_ratio}_{epochs}_{version}.pt",
                                                map_location=torch.device('cpu'))       


data_cfg = OmegaConf.create(inr_training_input["data"])
inr_cfg = OmegaConf.create(inr_training_input["cfg_inr"])  
small_grid = inr_training_input["coords"]
permutations = inr_training_input["permutations"]

output_dim = 1
num_frequencies = inr_cfg.num_frequencies
frequency_embedding = inr_cfg.frequency_embedding
max_frequencies = inr_cfg.max_frequencies
include_input = inr_cfg.include_input
scale = inr_cfg.scale 
base_frequency = inr_cfg.base_frequency

model_type = inr_cfg.model_type
latent_dim = inr_cfg.latent_dim
depth = inr_cfg.depth
hidden_dim = inr_cfg.hidden_dim
modulate_scale = inr_cfg.modulate_scale
modulate_shift = inr_cfg.modulate_shift
last_activation = None  
loss_type = "mse"
alpha = 0.01
            

input_dim=1
                
inr = ModulatedFourierFeatures(
        input_dim=input_dim,
        output_dim=output_dim,
        num_frequencies=num_frequencies,
        latent_dim=latent_dim,
        width=hidden_dim,
        depth=depth,
        modulate_scale=modulate_scale,
        modulate_shift=modulate_shift,
        frequency_embedding=frequency_embedding,
        include_input=include_input,
        scale=scale,
        max_frequencies=max_frequencies,
        base_frequency=base_frequency,
        )


inr.load_state_dict(inr_training_input["inr"])


X_tr, grid = fixed_sampling_series_imputations(dataset_name,  
                                               version=version,
                                               setting='classic',
                                               train_or_test='train')

modulations_train = torch.zeros(X_tr.shape[0], latent_dim) 

small_X_tr = [X_tr[ii,permutations[ii],:] for ii in range(X_tr.shape[0])]
small_X_tr = torch.stack(small_X_tr)
small_X_tr_norm, small_X_tr_mean, small_X_tr_std = z_normalize_out(small_X_tr)

inr.eval()

n_samples = small_X_tr.shape[0]

            
outputs = outer_step(
            inr,
            small_grid,
            small_X_tr_norm,
            test_inner_steps,
            alpha,
            is_train=False,
            gradient_checkpointing=False,
            loss_type=loss_type,
            modulations=torch.zeros_like(modulations_train),
            )

loss = outputs["loss"] 
modulations_train = outputs['modulations'].detach().cpu()


fit_train = inr.modulated_forward(small_grid, modulations_train)
fit_train_denorm = z_denormalize_out(fit_train, small_X_tr_mean, small_X_tr_std)

interpo = inr.modulated_forward(grid, modulations_train)
interpo_denorm = z_denormalize_out(interpo, small_X_tr_mean, small_X_tr_std)

mae_error_small_train = mae(small_X_tr[:,:,0].detach().numpy(), fit_train_denorm[:,:,0].detach().numpy())
error_our_interpo = mae(X_tr[:,:,0].detach().numpy(), interpo_denorm[:,:,0].detach().numpy())
error_on_unknown_grid = convert_error(draw_ratio, mae_error_small_train, error_our_interpo)

print('Dataset', dataset_name)
print('Version', version)
print('Draw ratio', draw_ratio)
print('Mae score on known grid', mae_error_small_train)
print('Mae score on global grid', error_our_interpo)
print('Mae score on the missing grid', error_on_unknown_grid)