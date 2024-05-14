import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from omegaconf import DictConfig, OmegaConf
import argparse 
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler


#Custom imports
from src.metalearning.metalearning_forecasting import outer_step
from src.network import  ModulatedFourierFeatures
from src.utils import (
    DatasetSamplesForecasting,
    fixed_sampling_series_forecasting,
)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Electricity')
parser.add_argument('--epochs', type=int, default=40000)
parser.add_argument('--horizon', type=int, default=96)
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--test_inner_steps', type=int, default=3)
args = parser.parse_args()


dataset_name = args.dataset_name
epochs = args.epochs
horizon = args.horizon
version = args.version
test_inner_steps = args.test_inner_steps


RESULTS_DIR = str(Path(__file__).parents[2]) + '/save_models/'
look_back_window = 512
alpha = 0.01


inr_training_input = torch.load(f"{RESULTS_DIR}/models_forecasting_{dataset_name}_{horizon}_{epochs}_{version}.pt",
                                map_location=torch.device('cpu'))                       


data_cfg = OmegaConf.create(inr_training_input["data"])
inr_cfg = OmegaConf.create(inr_training_input["cfg_inr"])  #inr


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
        log_sampling=True,
        max_frequencies=max_frequencies,
        base_frequency=base_frequency,
        )


inr.load_state_dict(inr_training_input["inr"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inr = inr.to(device)
inr.eval()

series_passed, series_target, grid = fixed_sampling_series_forecasting(
                                            dataset_name, 
                                            horizon, 
                                            version=version,
                                            setting='classic',
                                            train_or_test='train'
                                            )

series_p = series_target[:,:look_back_window,:]
grid_p = grid[:,:look_back_window,:]
                
series_h = series_target[:,look_back_window : look_back_window + horizon,:]
grid_h = grid[:,look_back_window : look_back_window + horizon,:]
                

                
modulations = torch.zeros(series_p.shape[0], latent_dim)
n_samples = series_target.shape[0]

grid_p = grid_p.to(device)
grid_h = grid_h.to(device)
scaler = StandardScaler()
scaler.fit(series_p[:,:,0].transpose(1,0))
series_p = scaler.transform(series_p[:,:,0].transpose(1,0))
series_p = torch.tensor(series_p).float().transpose(1,0).unsqueeze(-1).to(device)
series_h = series_h.to(device)
                    
outputs = outer_step(
                inr,
                grid_p,
                grid_h,
                series_p,
                series_h,
                test_inner_steps,
                alpha,
                look_back_window,
                horizon,
                w_passed=0.5,
                w_futur=0.5,
                is_train=False,
                gradient_checkpointing=False,
                loss_type=loss_type,
                modulations=torch.zeros_like(modulations).to(device),
                )

loss = outputs["loss"] 
modulations = outputs['modulations'].detach()

fit = inr.modulated_forward(grid_p, modulations)
mae_loss_fit = mae(fit[:,:,0].cpu().detach().numpy(), series_p[:,:,0].cpu().detach().numpy())

forecast_train = inr.modulated_forward(grid_h, modulations)
forecast_train = torch.tensor(scaler.inverse_transform(forecast_train[..., 0].transpose(1,0).cpu().numpy())).transpose(1,0)
mae_error_forecast = mae(series_h[:,:,0].cpu().detach().numpy(), forecast_train[:,:,0].cpu().detach().numpy())

print('Dataset', dataset_name)
print('Version', version)
print('Horizon', horizon)
print('Mae score on passed', mae_loss_fit)
print('Mae score forecasting', mae_error_forecast)