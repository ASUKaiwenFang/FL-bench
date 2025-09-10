"""
Differential Privacy Mechanisms for Federated Learning

This module provides utility functions for implementing differential privacy
mechanisms in federated learning, including noise addition and gradient clipping.
"""

from typing import List, Union

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
