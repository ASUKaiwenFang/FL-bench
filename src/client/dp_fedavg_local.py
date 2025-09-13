from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.optimizers.optimizer import _generate_noise

from src.client.fedavg import FedAvgClient


class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with Opacus Per-Sample Gradients.
    
    This client implements local differential privacy using Opacus GradSampleModule
    for efficient and compatible per-sample gradient computation. The GradSampleModule
    automatically computes per-sample gradients during the backward pass, providing
    excellent compatibility with complex models including BatchNorm layers.
    Each sample's gradients are computed and clipped independently,
    then averaged and noised before parameter updates.
    
    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step (current implementation)
    - last_noise: Add noise to parameter differences after training completion
    """
    
    # Define algorithm variant constants
    ALGORITHM_VARIANTS = {
        'last_noise': 1,
        'step_noise': 2
    }
    
    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader = None
        
        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        
        # Support string or numeric configuration
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
        
    
    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.iter_trainloader = iter(self.trainloader)
        # Wrap model with GradSampleModule for per-sample gradient computation
        if not isinstance(self.model, GradSampleModule):
            original_device = self.model.device
            self.model = GradSampleModule(self.model)
            # Preserve device attribute for compatibility
            self.model.device = original_device
            # Update parameter names after wrapping
            self.regular_params_name = list(key for key, _ in self.model.named_parameters())
            
            # Update regular_model_params with new parameter names for return_diff mode
            if self.return_diff:
                from collections import OrderedDict
                model_params = self.model.state_dict()
                self.regular_model_params = OrderedDict(
                    (key, model_params[key].clone().cpu())
                    for key in self.regular_params_name
                )
    
    def fit(self):
        """Train the model with local differential privacy using per-sample gradients.
        
        Supports two algorithm variants:
        - step_noise: Add noise to gradients at each training step
        - last_noise: Add noise to parameter differences after training completion
        """
        if self.algorithm_variant == 1:  # last_noise
            return self._last_noise_training()
        elif self.algorithm_variant == 2:  # step_noise
            return self._step_noise_training()
        else:
            raise ValueError(f"Unknown algorithm variant: {self.algorithm_variant}")
    
    def _step_noise_training(self):
        """Gradient-level noise addition (original implementation).
        
        This method implements the per-sample DP-SGD algorithm using Opacus:
        1. Forward pass (GradSampleModule automatically computes per-sample gradients)
        2. Backward pass (per-sample gradients stored in param.grad_sample)
        3. Clip each sample's gradients independently
        4. Average the clipped gradients and add noise
        5. Apply noisy gradients
        """
        self.model.train()
        self.dataset.train()
        
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # Standard Opacus forward+backward pass
            self.optimizer.zero_grad()
            
            # Clear any existing grad_sample from previous iterations
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None
            
            # Forward and backward pass - GradSampleModule automatically computes per-sample gradients
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            
            # Apply DP processing: clip and add noise
            self._clip_and_add_noise_opacus()
            
            # Optimizer step
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def _last_noise_training(self):
        """Parameter-level noise addition.
        
        Train without noise, then add noise to parameter differences.
        Uses noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        """
        self.model.train()
        self.dataset.train()
        
        # Store initial parameters for difference calculation
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        # Standard training without noise
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # Standard forward+backward pass without DP noise
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        # Add noise to parameter differences
        self._add_parameter_level_noise(initial_params)
    
    def _add_parameter_level_noise(self, initial_params):
        """Add noise to parameter differences for last_noise variant.
        
        Uses noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        where C = clip_norm, K = local_epoch, η_l = learning_rate, σ_g = sigma, b = batch_size
        """
        # Get batch size from last training batch (approximate)
        try:
            x, y = self.get_data_batch()
            batch_size = len(x)
        except:
            # Fallback to a reasonable default
            batch_size = 32
        
        # Calculate noise standard deviation for parameter-level noise
        # σ_DP = C * K * η_l * σ_g / b
        learning_rate = self.args.optimizer.lr
        K = self.local_epoch
        sigma_dp = self.clip_norm * K * learning_rate * self.sigma / batch_size
        
        # Add noise to parameter differences
        for name, param in self.model.named_parameters():
            if name in initial_params:
                # Calculate parameter difference
                param_diff = param.data - initial_params[name]
                
                # Add Gaussian noise to the difference
                noise = _generate_noise(
                    std=sigma_dp,
                    reference=param_diff,
                    generator=None,
                    secure_mode=False
                )
                
                # Apply noisy difference: param = initial + (diff + noise)
                param.data = initial_params[name] + param_diff + noise
    
    def get_data_batch(self):
        # Initialize iterator if not already done
        if self.iter_trainloader is None:
            self.iter_trainloader = iter(self.trainloader)
            
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
    
    
    
    def _compute_per_sample_norms_opacus(self):
        """
        Compute per-sample gradient norms using Opacus approach.
        Reads directly from param.grad_sample attributes.
                            
        Returns:
            per_sample_norms: Tensor of shape [batch_size] containing L2 norms
        """
        per_param_norms = []
        batch_size = None
        
        for param in self.model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                if batch_size is None:
                    batch_size = param.grad_sample.shape[0]
                
                # Reshape to [batch_size, -1] and compute norm for each sample
                param_norms = param.grad_sample.reshape(batch_size, -1).norm(2, dim=-1)
                per_param_norms.append(param_norms)
        
        if not per_param_norms:
            return torch.zeros(0, device=self.device)
        
        # Stack parameter norms and compute total norm per sample
        # Shape: [batch_size, num_params] -> [batch_size]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        
        return per_sample_norms
    
    def _clip_and_add_noise_opacus(self):
        """
        Clip per-sample gradients and add noise using Opacus approach.
        Operates directly on param.grad_sample and sets param.grad.
        Noise standard deviation follows the formula: σ_DP = C * σ_g / b
        """
        # Compute per-sample norms
        per_sample_norms = self._compute_per_sample_norms_opacus()
        
        if len(per_sample_norms) == 0:
            return
        
        # Compute clipping factors using Opacus formula
        per_sample_clip_factor = (
            self.clip_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)
        # Calculate DP noise standard deviation: σ_DP = C * σ_g / b
        batch_size = per_sample_norms.size(0)
        sigma_dp = self.clip_norm * self.sigma / batch_size
        # Apply clipping and noise to each parameter
        for param in self.model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                # Clip per-sample gradients using torch.einsum
                # per_sample_clip_factor: [batch_size]
                # param.grad_sample: [batch_size, *param_shape]
                clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, param.grad_sample)
                
                # Add Gaussian noise using Opacus _generate_noise
                noisy_grad = clipped_grad + _generate_noise(
                    std=sigma_dp,
                    reference=clipped_grad,
                    generator=None,
                    secure_mode=False
                )
                
                # Set the parameter's gradient
                param.grad = noisy_grad
    
    
    def package(self):
        """Package client data including DP parameters."""
        client_package = super().package()
        
        # Fix parameter names for GradSampleModule compatibility
        # Remove '_module.' prefix from parameter names to match server expectations
        if "regular_model_params" in client_package:
            fixed_params = {}
            for key, value in client_package["regular_model_params"].items():
                new_key = key.replace("_module.", "") if key.startswith("_module.") else key
                fixed_params[new_key] = value
            client_package["regular_model_params"] = fixed_params
        
        if "model_params_diff" in client_package:
            fixed_diff = {}
            for key, value in client_package["model_params_diff"].items():
                new_key = key.replace("_module.", "") if key.startswith("_module.") else key
                fixed_diff[new_key] = value
            client_package["model_params_diff"] = fixed_diff
        
        # Add DP parameters for server tracking
        client_package["dp_parameters"] = {
            "clip_norm": self.clip_norm,
            "sigma": self.sigma
        }
        
        return client_package
