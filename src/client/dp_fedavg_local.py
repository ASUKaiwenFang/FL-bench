from typing import Any, Iterator

import torch
import torch.func
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import (
    add_gaussian_noise,
    clip_gradients
)


class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with torch.func Per-Sample Gradients.
    
    This client implements local differential privacy using modern torch.func
    for efficient per-sample gradient computation. Uses vmap + grad for vectorized
    gradient computation with significant performance improvements over manual loops.
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
    
    def fit(self):
        """Train the model with local differential privacy using per-sample gradients.
        
        This method implements the per-sample DP-SGD algorithm using torch.func:
        1. Compute per-sample gradients using torch.func (vmap + grad)
        2. Clip each sample's gradients independently
        3. Average the clipped gradients
        4. Add calibrated Gaussian noise
        5. Apply noisy gradients
        """
        self.model.train()
        self.dataset.train()
        
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()
            
            # Compute per-sample gradients using torch.func
            per_sample_grads = self._compute_per_sample_gradients_torch_func(x, y)
            
            # Clip gradients and add noise
            final_grads = self._clip_and_noise_gradients(per_sample_grads)
            
            # Clear existing gradients and apply final noisy gradients
            self.optimizer.zero_grad()
            
            # Apply gradients to model parameters
            for param_name, grad in final_grads.items():
                for name, param in self.model.named_parameters():
                    if name == param_name and param.requires_grad:
                        param.grad = grad
                        break
            
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
    
    def _extract_model_params_and_buffers(self):
        """Extract and detach model parameters and buffers for torch.func.
        
        Returns:
            tuple: (params_dict, buffers_dict) where both are detached parameter dictionaries
        """
        params = {k: v.detach() for k, v in self.model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        return params, buffers
    
    def _compute_single_sample_loss(self, params, buffers, sample, target):
        """Compute loss for a single sample using functional_call.
        
        Args:
            params: Parameter dictionary for the model
            buffers: Buffer dictionary for the model  
            sample: Single input sample tensor
            target: Single target tensor
            
        Returns:
            Scalar loss tensor for the sample
        """
        sample_batch = sample.unsqueeze(0)
        target_batch = target.unsqueeze(0)
        
        # Use functional_call with strict=False to avoid issues with buffers that 
        # have in-place operations like BatchNorm's num_batches_tracked
        predictions = functional_call(self.model, (params, buffers), (sample_batch,), strict=False)
        loss = self.criterion(predictions, target_batch)
        return loss
    
    def _compute_per_sample_gradients_torch_func(self, x, y):
        """Compute per-sample gradients using torch.func (vmap + grad).
        
        This method uses the modern torch.func approach with vmap for vectorized
        gradient computation, providing significant performance improvements over
        manual loops while maintaining mathematical correctness.
        
        Args:
            x: Input batch tensor [batch_size, ...]
            y: Target batch tensor [batch_size, ...]
            
        Returns:
            dict: Dictionary mapping parameter names to per-sample gradient tensors
                 Each gradient tensor has shape [batch_size, *param_shape]
        """
        params, buffers = self._extract_model_params_and_buffers()
        
        # Create gradient computation function for single sample
        grad_fn = grad(self._compute_single_sample_loss)
        
        # Vectorize across batch using vmap
        # in_dims: (None, None, 0, 0) - params and buffers are not batched, x and y are batched along dim 0
        vmap_grad_fn = vmap(grad_fn, in_dims=(None, None, 0, 0))
        
        # Compute per-sample gradients for entire batch
        per_sample_grads = vmap_grad_fn(params, buffers, x, y)
        
        return per_sample_grads
    
    def _clip_and_noise_gradients(self, per_sample_grads):
        """Clip per-sample gradients and add noise using torch.func format.
        
        Args:
            per_sample_grads: Dictionary of per-sample gradients from torch.func
                            Each value has shape [batch_size, *param_shape]
                            
        Returns:
            dict: Dictionary of parameter names to final noisy gradients ready for optimizer
        """
        batch_size = next(iter(per_sample_grads.values())).shape[0]
        
        # Clip each sample's gradients independently
        clipped_grads = {}
        for param_name, param_grads in per_sample_grads.items():
            clipped_param_grads = []
            
            for i in range(batch_size):
                sample_grad = param_grads[i]
                
                # Compute L2 norm for this sample's gradient for this parameter
                grad_norm = sample_grad.detach().norm().item()
                
                # Clip if necessary
                if grad_norm > self.clip_norm:
                    scaling_factor = self.clip_norm / grad_norm
                    clipped_sample_grad = sample_grad * scaling_factor
                else:
                    clipped_sample_grad = sample_grad
                    
                clipped_param_grads.append(clipped_sample_grad)
            
            # Stack clipped gradients back into batch format
            clipped_grads[param_name] = torch.stack(clipped_param_grads, dim=0)
        
        # Average across batch and add noise
        final_grads = {}
        for param_name, clipped_param_grads in clipped_grads.items():
            # Average across batch dimension
            averaged_grad = clipped_param_grads.mean(dim=0)
            
            # Add Gaussian noise
            if param_name in self.model.state_dict():
                param_device = self.model.state_dict()[param_name].device
            else:
                param_device = self.device
                
            noisy_grad = add_gaussian_noise(
                averaged_grad,
                sigma=self.sigma,
                device=param_device
            )
            
            final_grads[param_name] = noisy_grad
        
        return final_grads

    def package(self):
        """Package client data including DP parameters."""
        client_package = super().package()
        
        # Add DP parameters for server tracking
        client_package["dp_parameters"] = {
            "clip_norm": self.clip_norm,
            "sigma": self.sigma
        }
        
        return client_package
