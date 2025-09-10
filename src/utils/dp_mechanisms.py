"""
Differential Privacy Mechanisms for Federated Learning

This module provides utility functions for implementing differential privacy
mechanisms in federated learning, including noise addition, gradient clipping,
and privacy accounting.
"""

import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_


def add_gaussian_noise(
    tensor: torch.Tensor, 
    noise_multiplier: float, 
    sensitivity: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """Add calibrated Gaussian noise to a tensor for differential privacy.
    
    Args:
        tensor: Input tensor to add noise to
        noise_multiplier: Noise multiplier (σ/S where σ is noise std, S is sensitivity)
        sensitivity: Sensitivity of the function (default: 1.0)
        device: Device to generate noise on (default: same as input tensor)
        
    Returns:
        Tensor with added Gaussian noise
    """
    if device is None:
        device = tensor.device
    
    noise_std = noise_multiplier * sensitivity
    noise = torch.normal(
        mean=0.0, 
        std=noise_std, 
        size=tensor.shape, 
        device=device
    )
    
    return tensor + noise


def add_laplace_noise(
    tensor: torch.Tensor, 
    epsilon: float, 
    sensitivity: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """Add calibrated Laplace noise to a tensor for differential privacy.
    
    Args:
        tensor: Input tensor to add noise to
        epsilon: Privacy budget parameter
        sensitivity: Sensitivity of the function (default: 1.0)
        device: Device to generate noise on (default: same as input tensor)
        
    Returns:
        Tensor with added Laplace noise
    """
    if device is None:
        device = tensor.device
    
    scale = sensitivity / epsilon
    noise = torch.tensor(
        np.random.laplace(loc=0.0, scale=scale, size=tensor.shape),
        dtype=tensor.dtype,
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


def compute_noise_multiplier(
    epsilon: float, 
    delta: float, 
    lot_size: int, 
    steps: int
) -> float:
    """Compute noise multiplier for given privacy parameters using RDP accounting.
    
    This is a simplified implementation. For production use, consider using
    specialized DP libraries like Opacus or TensorFlow Privacy.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter for (ε,δ)-DP
        lot_size: Logical batch size (lot size)
        steps: Number of training steps
        
    Returns:
        Noise multiplier σ for Gaussian mechanism
    """
    # Simplified RDP-based computation
    # For more accurate computation, use proper RDP accounting
    q = lot_size / 10000  # Assuming dataset size of 10000, adjust as needed
    
    if steps * q >= 1:
        # When sampling probability is high, use conservative bound
        noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    else:
        # RDP-based approximation for small sampling probability
        # This is a simplified version - use proper RDP libraries for production
        rdp_budget = epsilon / (2 * math.sqrt(2 * steps * q * math.log(1 / delta)))
        noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / rdp_budget
    
    return max(noise_multiplier, 0.1)  # Minimum noise for numerical stability


def privacy_accountant(
    noise_multiplier: float, 
    lot_size: int, 
    steps: int, 
    dataset_size: int = 10000
) -> Tuple[float, float]:
    """Compute privacy cost using RDP accounting.
    
    This is a simplified implementation. For production use, consider using
    specialized DP libraries.
    
    Args:
        noise_multiplier: Noise multiplier used
        lot_size: Logical batch size
        steps: Number of training steps completed
        dataset_size: Total dataset size
        
    Returns:
        Tuple of (epsilon, delta) privacy cost
    """
    q = lot_size / dataset_size
    
    if noise_multiplier == 0 or steps == 0:
        return float('inf'), 1.0
    
    # Simplified RDP computation
    # For α = 2 (Gaussian RDP)
    alpha = 2.0
    rdp_epsilon = alpha * q * steps / (2 * noise_multiplier ** 2)
    
    # Convert RDP to (ε,δ)-DP
    delta = 1e-5  # Standard choice
    epsilon = rdp_epsilon + math.log(1 / delta) / (alpha - 1)
    
    return epsilon, delta


class PrivacyEngine:
    """Simple privacy engine for tracking privacy consumption."""
    
    def __init__(self, epsilon: float, delta: float, lot_size: int):
        self.target_epsilon = epsilon
        self.target_delta = delta
        self.lot_size = lot_size
        self.steps = 0
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
    
    def step(self, noise_multiplier: float, dataset_size: int = 10000):
        """Update privacy consumption for one training step."""
        self.steps += 1
        epsilon, delta = privacy_accountant(
            noise_multiplier, self.lot_size, self.steps, dataset_size
        )
        self.consumed_epsilon = epsilon
        self.consumed_delta = delta
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy consumption."""
        return self.consumed_epsilon, self.consumed_delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        remaining_eps = max(0, self.target_epsilon - self.consumed_epsilon)
        remaining_delta = max(0, self.target_delta - self.consumed_delta)
        return remaining_eps, remaining_delta
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return (self.consumed_epsilon >= self.target_epsilon or 
                self.consumed_delta >= self.target_delta)
