from typing import Any, Iterator
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.optimizers.optimizer import _generate_noise
from collections import OrderedDict
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
        self.sigma_dp = None
        self.dp_processed_diff = None

        # Support string or numeric configuration
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
        
    
    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.iter_trainloader = iter(self.trainloader)

        # Reset DP processed difference for new training round
        self.dp_processed_diff = None

        # Wrap model with GradSampleModule for per-sample gradient computation
        if not isinstance(self.model, GradSampleModule):
            original_device = self.model.device
            self.model = GradSampleModule(self.model)
            # Preserve device attribute for compatibility
            self.model.device = original_device
            # Update parameter names after wrapping
            self.regular_params_name = list(key for key, _ in self.model.named_parameters())

            # Force re-save parameters with wrapped parameter names for consistency
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
            self._last_noise_training()
        elif self.algorithm_variant == 2:  # step_noise
            self._step_noise_training()
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
            
            self.optimizer.zero_grad()
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self._clip_and_add_noise_opacus()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._step_noise_post_processing()
    
    def _last_noise_training(self):
        """Parameter-level noise addition.

        Train without noise, then add noise to parameter differences.
        Uses noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        """
        self.model.train()
        self.dataset.train()


        # Standard training without noise
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._last_noise_post_processing()

    @torch.no_grad()
    def _step_noise_post_processing(self):
        """Post-processing for step_noise variant: calculate DP-processed parameter differences."""
        self.dp_processed_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                clean_name = self._get_clean_param_name(name)
                self.dp_processed_diff[clean_name] = param_diff.clone().cpu()

    @torch.no_grad()
    def _last_noise_post_processing(self):
        """Post-processing for last_noise variant: integrated parameter difference calculation and noise addition."""
        # Get the batch size from last training iteration
        try:
            x, y = self.get_data_batch()
            batch_size = len(x)
        except:
            batch_size = self.args.common.batch_size

        # σ_DP = C * K * η_l * σ_g / b
        sigma_dp = self.clip_norm * self.local_epoch * self.args.optimizer.lr * self.sigma / batch_size
        self.sigma_dp = sigma_dp

        # Calculate noisy parameter differences and store them
        self.dp_processed_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                noise = _generate_noise(
                    std=sigma_dp,
                    reference=param_diff,
                    generator=None,
                    secure_mode=False
                )
                noisy_diff = param_diff + noise
                clean_name = self._get_clean_param_name(name)
                self.dp_processed_diff[clean_name] = noisy_diff.clone().cpu()

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
    
    def _clip_gradients(self):
        """Clip per-sample gradients without adding noise."""
        per_sample_norms = self._compute_per_sample_norms_opacus()
        if len(per_sample_norms) == 0:
            return
        per_sample_clip_factor = (
            self.clip_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)
        for param in self.model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, param.grad_sample)
                param.grad = clipped_grad
                
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
        self.sigma_dp = sigma_dp
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
        """Package client data including DP parameters.

        Optimized implementation that avoids redundant calculations
        based on the algorithm variant.
        """

        # Common package components
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            model_params_diff=self.dp_processed_diff,
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {} if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
            sigma_dp=self.sigma_dp
        )

        return client_package
    
    
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

    def _get_clean_param_name(self, name: str) -> str:
        """Remove _module. prefix from parameter names for compatibility."""
        return name.replace("_module.", "") if name.startswith("_module.") else name
