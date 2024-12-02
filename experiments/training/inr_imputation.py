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

from src.metalearning.metalearning_imputation import outer_step
from src.network import  ModulatedFourierFeatures
from src.utils import (
    DatasetSamples,
    set_seed,
    fixed_subsampling_series_imputations,
    z_normalize
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
    draw_ratio = cfg.data.draw_ratio
    version = cfg.data.version

    # optim
    batch_size = cfg.optim.batch_size
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    weight_decay = cfg.optim.weight_decay
    sample_ratio_batch = cfg.optim.sample_ratio_batch

    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim
    depth = cfg.inr.depth
    hidden_dim = cfg.inr.hidden_dim
    modulate_scale = cfg.inr.modulate_scale
    modulate_shift = cfg.inr.modulate_shift
    length_of_interest = cfg.data.length_of_interest
    output_dim = 1


    small_data, small_grid, permutations = fixed_subsampling_series_imputations(
                                            dataset_name, 
                                            draw_ratio, 
                                            version=version,
                                            setting='classic',
                                            train_or_test='train'
                                            )

    trainset = DatasetSamples(small_data, small_grid, latent_dim, sample_ratio_batch)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    ntrain = small_data.shape[0]
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
                max_frequencies=cfg.inr.max_frequencies,
                base_frequency=cfg.inr.base_frequency,
            )
    
    inr = inr.to(device)
    
    alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))
    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code

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
        rel_train_mse = 0
        fit_train_mae = 0
        fit_train_samples = 0
        use_rel_loss = step % 10 == 0

        for substep, (series, modulations, coords, idx) in enumerate(train_loader):
            inr.train() 
            series = z_normalize(series)
            series = series.to(device)
            modulations = modulations.to(device)
            coords = coords.to(device)
            n_samples = series.shape[0]

            outputs = outer_step(
                inr,
                coords,
                series,
                inner_steps,
                alpha,
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
            print('loss :', train_samples_loss)

        scheduler.step(train_samples_loss)

        if train_samples_loss < best_loss:
            best_loss = train_samples_loss

            torch.save(
                {
                    "data": cfg.data,
                    "cfg_inr": cfg.inr,
                    "epoch": step,
                    "draw_ratio": draw_ratio,
                    "coords": small_grid,
                    "permutations": permutations,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "train_loss": train_samples_loss,
                    "alpha": alpha,
                },
                f"{RESULTS_DIR}/models_imputation_{dataset_name}_{draw_ratio}_{epochs}_{version}.pt",
            )

    return train_samples_loss 


if __name__ == "__main__":
    main()
