from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule

from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import (
    add_gaussian_noise,
    clip_gradients
)


class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with Opacus Per-Sample Gradients.
    
    This client implements local differential privacy using Opacus GradSampleModule
    for efficient and compatible per-sample gradient computation. The GradSampleModule
    automatically computes per-sample gradients during the backward pass, providing
    excellent compatibility with complex models including BatchNorm layers.
    Each sample's gradients are computed and clipped independently,
    then averaged and noised before parameter updates.
    """
    
    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader = None
        
        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        
    
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
        
        This method implements the per-sample DP-SGD algorithm using Opacus:
        1. Compute per-sample gradients using Opacus GradSampleModule
        2. Clip each sample's gradients independently
        3. Average the clipped gradients
        4. Add calibrated Gaussian noise
        5. Apply noisy gradients
        """
        self.model.train()
        self.dataset.train()
        
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # Compute per-sample gradients using Opacus
            per_sample_grads = self._compute_per_sample_gradients_opacus(x, y)
            
            # Clip per-sample gradients
            clipped_grads = self._clip_per_sample_gradients_opacus(per_sample_grads)
            
            # Apply averaged gradients with noise
            self._apply_gradients_with_noise_opacus(clipped_grads)
            
            # Optimizer step
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
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
    
    def _compute_per_sample_gradients_opacus(self, x, y):
        """Compute per-sample gradients using Opacus GradSampleModule.
        
        This method uses Opacus's GradSampleModule to automatically compute
        per-sample gradients during the backward pass. This approach is more
        stable than torch.func and handles BatchNorm layers correctly.
        
        Args:
            x: Input batch tensor [batch_size, ...]
            y: Target batch tensor [batch_size, ...]
            
        Returns:
            dict: Dictionary mapping parameter names to per-sample gradient tensors
                 Each gradient tensor has shape [batch_size, *param_shape]
        """
        self.optimizer.zero_grad()
        
        # Clear any existing grad_sample from previous iterations
        for param in self.model.parameters():
            if hasattr(param, 'grad_sample'):
                param.grad_sample = None
        
        # Forward and backward pass
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        
        # Extract per-sample gradients from GradSampleModule
        per_sample_grads = {}
        for name, param in self.model.named_parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                per_sample_grads[name] = param.grad_sample
        
        return per_sample_grads
    
    def _clip_per_sample_gradients_opacus(self, per_sample_grads):
        """Clip per-sample gradients using Opacus format.
        
        Args:
            per_sample_grads: Dictionary of per-sample gradients from Opacus
                            Each value has shape [batch_size, *param_shape]
                            
        Returns:
            dict: Dictionary of parameter names to clipped per-sample gradients
        """
        if not per_sample_grads:
            return {}
            
        batch_size = next(iter(per_sample_grads.values())).shape[0]
        
        # Compute per-sample gradient norms across all parameters
        per_sample_norms = torch.zeros(batch_size, device=self.device)
        
        for param_grads in per_sample_grads.values():
            # Flatten each sample's gradients and compute norm
            for i in range(batch_size):
                sample_grad = param_grads[i].flatten()
                per_sample_norms[i] += sample_grad.norm().pow(2)
        
        per_sample_norms = per_sample_norms.sqrt()
        
        # Compute clipping factors
        clipping_factors = torch.clamp(self.clip_norm / per_sample_norms, max=1.0)
        
        # Apply clipping
        clipped_grads = {}
        for param_name, param_grads in per_sample_grads.items():
            clipped_param_grads = []
            for i in range(batch_size):
                clipped_grad = param_grads[i] * clipping_factors[i]
                clipped_param_grads.append(clipped_grad)
            clipped_grads[param_name] = torch.stack(clipped_param_grads, dim=0)
        
        return clipped_grads
    
    def _apply_gradients_with_noise_opacus(self, clipped_grads):
        """Apply averaged gradients with noise using Opacus format.
        
        Args:
            clipped_grads: Dictionary of clipped per-sample gradients
        """
        if not clipped_grads:
            return
        
        # Average across batch and add noise, then apply to model parameters
        for param_name, clipped_param_grads in clipped_grads.items():
            # Average across batch dimension
            averaged_grad = clipped_param_grads.mean(dim=0)
            
            # Add Gaussian noise
            noisy_grad = add_gaussian_noise(
                averaged_grad,
                sigma=self.sigma,
                device=averaged_grad.device
            )
            
            # Apply to corresponding parameter
            for name, param in self.model.named_parameters():
                if name == param_name and param.requires_grad:
                    param.grad = noisy_grad
                    break
    
    
    
    

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
