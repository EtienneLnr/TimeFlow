from functools import partial
import sys
from pathlib import Path

#Custom imports
import src.metalearning.losses as losses
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import autograd
from torch.nn.parallel import DistributedDataParallel as DDP


def inner_loop(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    lambda_fft=0,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.

    Args:
        func_rep (models.ModulatedSiren):
        modulations (torch.Tensor): Shape (batch_size, latent_dim).
        coordinates (torch.Tensor): Coordinates at which function representation
            should be evaluated. Shape (batch_size, *, coordinate_dim).
        features (torch.Tensor): Target features for model to match. Shape
            (batch_size, *, feature_dim).
        inner_steps (int): Number of inner loop steps to take.
        inner_lr (float): Learning rate for inner loop.
        is_train (bool):
        gradient_checkpointing (bool): If True uses gradient checkpointing. This
            can massively reduce memory consumption.
    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        if gradient_checkpointing:
            fitted_modulations = cp.checkpoint(
                inner_loop_step,
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                torch.as_tensor(inner_lr),
                torch.as_tensor(is_train),
                torch.as_tensor(gradient_checkpointing),
                loss_type,
            )
        else:
            fitted_modulations = inner_loop_step(
                func_rep,
                fitted_modulations,
                coordinates,
                features,
                inner_lr,
                is_train,
                gradient_checkpointing,
                loss_type,
                lambda_fft,
            )
    return fitted_modulations


def inner_loop_step(
    func_rep,
    modulations,
    coordinates,
    features,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    lambda_fft=0,
):
    """Performs a single inner loop step."""
    detach = not torch.is_grad_enabled() and gradient_checkpointing
    batch_size = len(features)
    if loss_type == "mse":
        element_loss_fn = losses.per_element_mse_fn
    elif loss_type == "bce":
        element_loss_fn = losses.per_element_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        element_loss_fn = partial(
            losses.per_element_multi_scale_fn,
            loss_name=loss_name,
            last_element=False,
        )
    elif loss_type == 'fft':
        element_loss_fn = losses.Batch_fft_loss(lambda_fft)

    N, C = features.shape[0], features.shape[-1]

    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        loss = element_loss_fn(features_recon, features).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]
        # if clip_grad_value is not None:
        #    nn.utils.clip_grad_value_(grad, clip_grad_value)
    # Perform single gradient descent step
    return modulations - inner_lr * grad


def outer_step(
    func_rep,
    coordinates,
    features,
    inner_steps,
    inner_lr,
    is_train=False,
    gradient_checkpointing=False,
    loss_type="mse",
    lambda_fft=0,
    modulations=0,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    if loss_type == "mse":
        loss_fn = losses.batch_mse_fn
    elif loss_type == "bce":
        loss_fn = losses.batch_nll_fn
    elif "multiscale" in loss_type:
        loss_name = loss_type.split("-")[1]
        loss_fn = partial(losses.batch_multi_scale_fn, loss_name=loss_name)
    elif loss_type == 'fft':
        loss_fn = losses.Batch_fft_loss(lambda_fft)

    func_rep.zero_grad()
    batch_size = len(coordinates)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = modulations.requires_grad_()

    feat = features.clone()
    coords = coordinates.clone()

    # Run inner loop

    modulations = inner_loop(
        func_rep,
        modulations,
        coords,
        feat,
        inner_steps,
        inner_lr,
        is_train,
        gradient_checkpointing,
        loss_type,
    )

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coordinates, modulations)
        per_example_loss = loss_fn(features_recon, features)  # features
        loss = per_example_loss.mean()

    outputs = {
        "loss": loss,
        "psnr": losses.mse2psnr(per_example_loss).mean().item(),
        "modulations": modulations,
    }

    return outputs
