from typing import Any
from copy import deepcopy
from enum import Enum
import torch
from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import compute_per_sample_grads, compute_per_sample_norms


class AlgorithmVariant(Enum):
    """Algorithm variants for DP-FedAvg Local implementation."""
    LAST_NOISE = 1
    STEP_NOISE = 2



class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with Per-Sample Gradients.

    This client implements local differential privacy using torch.func.grad_and_value and vmap
    for efficient per-sample gradient computation. This approach provides
    excellent performance and compatibility with PyTorch 2.0+ compilation features.
    Each sample's gradients are computed and clipped independently,
    then averaged and noised before parameter updates.

    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step (current implementation)
    - last_noise: Add noise to parameter differences after training completion
    """

    # Configuration constants
    numerical_epsilon = 1e-6

    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader = None

        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        self.sigma_dp = None
        self.model_params_diff = None

        # Support string or numeric configuration with enum
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = getattr(AlgorithmVariant, variant_config.upper())
        else:
            # Legacy numeric support
            self.algorithm_variant = AlgorithmVariant(variant_config)


    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.iter_trainloader = iter(self.trainloader)
        self.model_params_diff = None

    
    def fit(self):
        """Train the model with local differential privacy using per-sample gradients.

        Supports two algorithm variants:
        - step_noise: Add noise to gradients at each training step
        - last_noise: Add noise to parameter differences after training completion
        """
        if self.algorithm_variant == AlgorithmVariant.LAST_NOISE:
            self._last_noise_training()
        elif self.algorithm_variant == AlgorithmVariant.STEP_NOISE:
            self._step_noise_training()
        else:
            raise ValueError(f"Unknown algorithm variant: {self.algorithm_variant}")
    
    def _step_noise_training(self):
        """Gradient-level noise addition using per-sample gradients.

        This method implements the per-sample DP-SGD algorithm using torch.func:
        1. Compute per-sample gradients using torch.func.grad + vmap
        2. Clip each sample's gradients independently
        3. Average the clipped gradients and add noise
        4. Apply noisy gradients to parameters
        """
        self.model.train()
        self.dataset.train()

        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            self._clip_and_add_noise(x, y)
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
            self._clip_without_noise(x, y)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._last_noise_post_processing()
        

    @torch.no_grad()
    def _step_noise_post_processing(self):
        """Post-processing for step_noise variant: calculate DP-processed parameter differences."""
        self.model_params_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                self.model_params_diff[name] = param_diff.clone().cpu()

    @torch.no_grad()
    def _last_noise_post_processing(self):
        """Post-processing for last_noise variant: integrated parameter difference calculation and noise addition."""
        # Use configured batch size for noise calculation
        batch_size = self.args.common.batch_size

        # σ_DP = C * K * η_l * σ_g / b
        sigma_dp = self.clip_norm * self.local_epoch * self.args.optimizer.lr * self.sigma / batch_size
        self.sigma_dp = sigma_dp

        # Calculate noisy parameter differences and store them
        self.model_params_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                # Generate Gaussian noise (inlined for efficiency)
                noise = torch.randn_like(param_diff, device=param.device) * sigma_dp
                noisy_diff = param_diff + noise
                self.model_params_diff[name] = noisy_diff.clone().cpu()

    def _clip_without_noise(self, inputs, targets):
        """Clip per-sample gradients and add noise using DP-SGD algorithm.

        This method implements the core DP-SGD algorithm with clip→mean→add_noise order:
        - Computes per-sample gradients
        - Clips gradients based on L2 norm
        - Averages clipped gradients across batch
        - Sets final gradients to model parameters

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
        """

        # Compute per-sample gradients and losses
        per_sample_grads, per_sample_losses = compute_per_sample_grads(
            self.model, inputs, targets, self.criterion
        )

        # Compute per-sample gradient norms
        per_sample_norms = compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Calculate DP noise standard deviation: σ_DP = C * σ_g
        batch_size = per_sample_norms.size(0)
        sigma_dp = self.clip_norm * self.sigma
        self.sigma_dp = sigma_dp

        # Calculate per-sample clipping factors
        per_sample_clip_factor = (self.clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0)

        # Create model parameters dictionary for efficient access
        model_params = {name: param for name, param in self.model.named_parameters()}

        # Process gradients: clip → mean → add_noise
        for param_name, per_sample_grad in per_sample_grads.items():
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [per_sample_clip_factor.size(0)] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            model_params[param_name].grad = clipped_grad.mean(dim=0)

    
    
                
    def _clip_and_add_noise(self, inputs, targets):
        """Clip per-sample gradients and add noise using DP-SGD algorithm.

        This method implements the core DP-SGD algorithm with clip→mean→add_noise order:
        - Computes per-sample gradients
        - Clips gradients based on L2 norm
        - Averages clipped gradients across batch
        - Adds calibrated Gaussian noise to averaged gradient
        - Sets final gradients to model parameters

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
        """

        # Compute per-sample gradients and losses
        per_sample_grads, per_sample_losses = compute_per_sample_grads(
            self.model, inputs, targets, self.criterion
        )

        # Compute per-sample gradient norms
        per_sample_norms = compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Calculate DP noise standard deviation: σ_DP = C * σ_g
        batch_size = per_sample_norms.size(0)
        sigma_dp = self.clip_norm * self.sigma
        self.sigma_dp = sigma_dp

        # Calculate per-sample clipping factors
        per_sample_clip_factor = (self.clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0)

        # Create model parameters dictionary for efficient access
        model_params = {name: param for name, param in self.model.named_parameters()}

        # Process gradients: clip → mean → add_noise
        for param_name, per_sample_grad in per_sample_grads.items():
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [per_sample_clip_factor.size(0)] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)

            # Add Gaussian noise to averaged gradient
            noise = torch.randn_like(mean_clipped_grad, device=self.device) * sigma_dp

            # Set final gradient to model parameter
            model_params[param_name].grad = mean_clipped_grad + noise
    
    
    def package(self):
        """Package client data including DP parameters.

        Optimized implementation that avoids redundant calculations
        based on the algorithm variant.
        """

        # Optimized package components with conditional copying
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            model_params_diff=self.model_params_diff,
            sigma_dp=self.sigma_dp
        )

        # Only copy personal parameters if they exist
        if self.personal_params_name:
            model_params = self.model.state_dict()
            client_package['personal_model_params'] = {
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            }
        else:
            client_package['personal_model_params'] = {}

        # Conditional optimizer state copying based on reset setting
        if not self.args.common.reset_optimizer_on_global_epoch:
            client_package['optimizer_state'] = deepcopy(self.optimizer.state_dict())
        else:
            client_package['optimizer_state'] = {}

        # Conditional scheduler state copying
        if self.lr_scheduler is not None and not self.args.common.reset_optimizer_on_global_epoch:
            client_package['lr_scheduler_state'] = deepcopy(self.lr_scheduler.state_dict())
        else:
            client_package['lr_scheduler_state'] = {}

        return client_package
    
    
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
