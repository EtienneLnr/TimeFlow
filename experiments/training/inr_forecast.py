import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append(str(Path(__file__).parents[2]))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler

#Custom imports
from src.metalearning.metalearning_forecasting import outer_step
from src.network import  ModulatedFourierFeatures
from src.utils import (
    DatasetSamplesForecasting,
    fixed_sampling_series_forecasting,
)

import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="../config/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    
    # save path
    RESULTS_DIR = str(Path(__file__).parents[2]) + '/save_models/'

    # data
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    look_back_window = cfg.data.look_back_window
    horizon = cfg.data.horizon
    version = cfg.data.version

    # optim
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    inner_steps = cfg.optim.inner_steps
    epochs = cfg.optim.epochs
    weight_decay = cfg.optim.weight_decay

    # inr

    latent_dim = cfg.inr.latent_dim
    output_dim = 1
    w_passed = cfg.inr.w_passed
    w_futur = cfg.inr.w_futur
    passed_ratio = cfg.inr.passed_ratio
    horizon_ratio = cfg.inr.horizon_ratio


    series_passed, _, small_grid = fixed_sampling_series_forecasting(
                                      dataset_name, 
                                      horizon, 
                                      version=version,
                                      setting='classic',
                                      train_or_test='train'
                                      )

    trainset = DatasetSamplesForecasting(series_passed, 
                                         small_grid, 
                                         latent_dim, 
                                         look_back_window,
                                         horizon,
                                         length_of_interest=4000,
                                         passed_ratio=passed_ratio,
                                         horizon_ratio=horizon_ratio)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    ntrain = series_passed.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    input_dim = 1

    inr = ModulatedFourierFeatures(
                input_dim=input_dim,
                output_dim=output_dim,
                num_frequencies=cfg.inr.num_frequencies,
                latent_dim=cfg.inr.latent_dim,
                width=cfg.inr.hidden_dim,
                depth=cfg.inr.depth,
                modulate_scale=cfg.inr.modulate_scale,
                modulate_shift=cfg.inr.modulate_shift,
                frequency_embedding=cfg.inr.frequency_embedding,
                include_input=cfg.inr.include_input,
                scale=cfg.inr.scale,
                log_sampling=cfg.inr.log_sampling,
                min_frequencies=cfg.inr.min_frequencies,
                max_frequencies=cfg.inr.max_frequencies,
                base_frequency=cfg.inr.base_frequency,
            )
    
    inr = inr.to(device)
    
    alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))

    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": lr_inr, "weight_decay": weight_decay},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)
    best_loss = np.inf

    for step in range(epochs):

        fit_train_samples = 0

        for substep, (series_p, series_h, modulations, coords_p, coords_h, idx) in enumerate(train_loader):
            inr.train()
            scaler = StandardScaler()
            scaler.fit(series_p[:,:,0].transpose(1,0))
            series_p = scaler.transform(series_p[:,:,0].transpose(1,0))
            series_h = scaler.transform(series_h[:,:,0].transpose(1,0))
            series_p = torch.tensor(series_p).float().transpose(1,0).unsqueeze(-1).to(device)
            series_h = torch.tensor(series_h).float().transpose(1,0).unsqueeze(-1).to(device)
            modulations = modulations.to(device)
            coords_p = coords_p.to(device)
            coords_h = coords_h.to(device)
            n_samples = series_p.shape[0]

            outputs = outer_step(
                inr,
                coords_p,
                coords_h,
                series_p,
                series_h,
                inner_steps,
                alpha,
                look_back_window,
                horizon,
                w_passed=0.5,
                w_futur=0.5,
                is_train=True,
                gradient_checkpointing=False,
                loss_type="mse",
                modulations=torch.zeros_like(modulations),
            )

            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            optimizer.step()
            loss = outputs["loss"].cpu().detach()

            with torch.set_grad_enabled(False):
                loss_samples = loss
                fit_train_samples += loss_samples.item() * n_samples

        train_samples_loss = fit_train_samples / (ntrain)

        if step % 100 == 0:
            print('epoch :', step)
            print('train loss :', train_samples_loss)

        scheduler.step(train_samples_loss)

        if train_samples_loss < best_loss:
            best_loss = train_samples_loss

            torch.save(
                {
                    "data": cfg.data,
                    "cfg_inr": cfg.inr,
                    "epoch": step,
                    "horizon": horizon,
                    "look_back_window": look_back_window,
                    "coords": small_grid,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "train_loss": train_samples_loss,
                    "alpha": alpha,
                },
                f"{RESULTS_DIR}/models_forecasting_{dataset_name}_{horizon}_{epochs}_{version}.pt",
            )

    return train_samples_loss


if __name__ == "__main__":
    main()
