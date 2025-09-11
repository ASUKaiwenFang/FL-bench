from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import (
    add_gaussian_noise,
    clip_gradients
)


class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client.
    
    This client implements local differential privacy by adding noise to gradients
    during local training. Each client's gradients are clipped and noised before
    being used for parameter updates.
    """
    
    def __init__(self, **commons):
        super().__init__(**commons)
        
        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        self.noise_mode = self.args.dp_fedavg_local.noise_mode
        
        # Set noise handler based on mode for efficiency
        if self.noise_mode == "gradient":
            self._noise_handler = self._add_noise_to_gradients
        else:  # parameter mode
            self._noise_handler = self._no_op
    
    def fit(self):
        """Train the model with local differential privacy.
        
        This method implements the DP-SGD algorithm:
        1. Compute gradients on minibatch
        2. Clip gradients by norm
        3. Add calibrated Gaussian noise
        4. Apply noisy gradients
        """
        self.model.train()
        self.dataset.train()
        
        for epoch in range(self.local_epoch):
            for x, y in self.trainloader:
                # Skip small batches to avoid BatchNorm issues
                if len(x) <= 1:
                    continue
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logit = self.model(x)
                loss = self.criterion(logit, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for DP
                total_norm = clip_gradients(self.model.parameters(), self.clip_norm)
                
                # Add noise based on mode (gradient or no-op for parameter mode)
                self._noise_handler()
                
                # Optimizer step
                self.optimizer.step()
                
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def _no_op(self):
        """No operation for parameter noise mode."""
        pass
    
    def _add_noise_to_gradients(self):
        """Add calibrated Gaussian noise to model gradients."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Add Gaussian noise to gradients
                param.grad = add_gaussian_noise(
                    param.grad,
                    sigma=self.sigma,
                    device=param.device
                )
    
    def _add_noise_to_parameters(self, client_package):
        """Add calibrated Gaussian noise to model parameters."""
        # Determine which parameter set to add noise to
        if self.return_diff:
            param_key = "model_params_diff"
        else:
            param_key = "regular_model_params"
        
        # Check if the key exists in the package
        if param_key in client_package:
            for key, param in client_package[param_key].items():
                client_package[param_key][key] = add_gaussian_noise(
                    param,
                    sigma=self.sigma,
                    device=param.device
                )
    
    def package(self):
        """Package client data including DP parameters."""
        client_package = super().package()
        
        # Add noise to parameters if in parameter mode
        if self.noise_mode == "parameter":
            self._add_noise_to_parameters(client_package)
        
        # Add DP parameters for server tracking
        client_package["dp_parameters"] = {
            "clip_norm": self.clip_norm,
            "sigma": self.sigma,
            "noise_mode": self.noise_mode
        }
        
        return client_package
