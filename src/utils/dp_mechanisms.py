"""
Differential Privacy Mechanisms for Federated Learning

This module provides utility functions for implementing differential privacy
mechanisms in federated learning, including noise addition and gradient clipping.
"""

from typing import List, Union, Dict, Tuple, Optional

import torch
from torch.nn.utils import clip_grad_norm_


def add_gaussian_noise(
    tensor: torch.Tensor, 
    sigma: float,
    device: torch.device = None
) -> torch.Tensor:
    """Add Gaussian noise to a tensor for differential privacy.
    
    Args:
        tensor: Input tensor to add noise to
        sigma: Noise standard deviation
        device: Device to generate noise on (default: same as input tensor)
        
    Returns:
        Tensor with added Gaussian noise
    """
    if device is None:
        device = tensor.device
    
    noise = torch.normal(
        mean=0.0, 
        std=sigma, 
        size=tensor.shape, 
        device=device
    )
    
    return tensor + noise


def clip_gradients(
    parameters: Union[torch.Tensor, List[torch.Tensor]], 
    clip_norm: float
) -> float:
    """Clip gradients by norm for differential privacy.
    
    Args:
        parameters: Model parameters or list of parameters
        clip_norm: Maximum norm for gradient clipping
        
    Returns:
        Total norm of gradients before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    elif hasattr(parameters, 'parameters'):
        parameters = list(parameters.parameters())
    
    total_norm = clip_grad_norm_(parameters, max_norm=clip_norm)
    return float(total_norm)


def compute_per_sample_grads(model, inputs, targets, criterion, cached_params=None, cached_buffers=None):
    """
    Compute per-sample gradients using torch.func.grad_and_value and vmap.

    Args:
        model: The neural network model
        inputs: Input batch tensor [batch_size, ...]
        targets: Target batch tensor [batch_size, ...]
        criterion: Loss function
        cached_params: Optional pre-computed parameters dict to avoid repeated extraction
        cached_buffers: Optional pre-computed buffers dict to avoid repeated extraction

    Returns:
        tuple: (per_sample_grads, per_sample_losses) where:
            - per_sample_grads: dict {param_name: per_sample_grad_tensor} with shape [batch_size, *param_shape]
            - per_sample_losses: tensor with shape [batch_size] containing per-sample loss values
    """
    def compute_loss_for_sample(params, buffers, sample_input, sample_target):
        predictions = torch.func.functional_call(model, (params, buffers), sample_input.unsqueeze(0))
        return criterion(predictions, sample_target.unsqueeze(0))

    # Use cached parameters if available, otherwise extract them
    if cached_params is not None and cached_buffers is not None:
        params = cached_params
        buffers = cached_buffers
    else:
        params = {name: param for name, param in model.named_parameters()}
        buffers = {name: buffer for name, buffer in model.named_buffers()}

    # Use vmap to vectorize over the batch dimension with grad_and_value for efficiency
    per_sample_grads, per_sample_losses = torch.vmap(
        torch.func.grad_and_value(compute_loss_for_sample, argnums=0),
        in_dims=(None, None, 0, 0)
    )(params, buffers, inputs, targets)

    return per_sample_grads, per_sample_losses


def compute_per_sample_norms(per_sample_grads):
    """
    Compute L2 norms of per-sample gradients.

    Args:
        per_sample_grads: dict of {param_name: per_sample_grad_tensor}

    Returns:
        torch.Tensor: Per-sample gradient norms [batch_size]
    """
    batch_size = None
    per_param_norms = []

    for param_name, grad_tensor in per_sample_grads.items():
        if batch_size is None:
            batch_size = grad_tensor.shape[0]

        # Reshape to [batch_size, -1] and compute norm for each sample
        param_norms = grad_tensor.reshape(batch_size, -1).norm(2, dim=-1)
        per_param_norms.append(param_norms)

    if not per_param_norms:
        return torch.zeros(0)

    # Stack parameter norms and compute total norm per sample
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    return per_sample_norms
