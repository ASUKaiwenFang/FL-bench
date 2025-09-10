from typing import Any

import torch

from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import (
    PrivacyEngine,
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
        self.epsilon = self.args.dp_fedavg_local.epsilon
        self.delta = self.args.dp_fedavg_local.delta
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.noise_multiplier = self.args.dp_fedavg_local.noise_multiplier
        self.lot_size = self.args.dp_fedavg_local.lot_size
        self.accounting_mode = self.args.dp_fedavg_local.accounting_mode
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine(
            epsilon=self.epsilon,
            delta=self.delta,
            lot_size=self.lot_size
        )
        
        # Track privacy consumption
        self.privacy_consumed = {"epsilon": 0.0, "delta": 0.0}
        self.training_steps = 0
    
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
                
                # Add noise to gradients
                self._add_noise_to_gradients()
                
                # Optimizer step
                self.optimizer.step()
                
                # Update privacy accounting
                self.training_steps += 1
                self.privacy_engine.step(self.noise_multiplier, len(self.trainset))
                
                # Update privacy consumption tracking
                eps, delta = self.privacy_engine.get_privacy_spent()
                self.privacy_consumed["epsilon"] = eps
                self.privacy_consumed["delta"] = delta
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
    
    def _add_noise_to_gradients(self):
        """Add calibrated Gaussian noise to model gradients."""
        for param in self.model.parameters():
            if param.grad is not None:
                # Add Gaussian noise to gradients
                param.grad = add_gaussian_noise(
                    param.grad,
                    noise_multiplier=self.noise_multiplier,
                    sensitivity=self.clip_norm,  # After clipping, sensitivity is clip_norm
                    device=param.device
                )
    
    def package(self):
        """Package client data including privacy information."""
        client_package = super().package()
        
        # Add privacy consumption information
        client_package["privacy_consumed"] = self.privacy_consumed.copy()
        client_package["training_steps"] = self.training_steps
        client_package["privacy_budget_exhausted"] = self.privacy_engine.is_budget_exhausted()
        
        # Add DP parameters for server tracking
        client_package["dp_parameters"] = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "clip_norm": self.clip_norm
        }
        
        return client_package
