from typing import Any
from copy import deepcopy
from enum import Enum
import random
import torch
import numpy as np
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
    - step_noise: Add noise to gradients at each training step
    - last_noise: Add noise to parameter differences after training completion

    Performance Optimizations:
    - Unified gradient processing method eliminates code duplication
    - Pre-computed sigma_dp values reduce redundant calculations
    - Cached model parameters dictionary avoids repeated creation
    - Optimized clip shape computation reduces loop overhead
    """

    # Configuration constants
    numerical_epsilon = 1e-6

    def _get_dp_config_value(self, key: str, default=None):
        """Dynamic DP configuration value access with fallback strategy.

        Priority order:
        1. Method-specific config (e.g., dp_scaffold.clip_norm)
        2. dp_fedavg_local config (backward compatibility)
        3. Provided default value
        4. Hardcoded class defaults

        Args:
            key: Configuration key to access
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # Get current method name from args
        method_name = self.args.method

        # Try method-specific config first (e.g., dp_scaffold)
        if hasattr(self.args, method_name):
            method_config = getattr(self.args, method_name)
            if hasattr(method_config, key):
                return getattr(method_config, key)

        # Fall back to dp_fedavg_local config for backward compatibility
        if hasattr(self.args, 'dp_fedavg_local'):
            dp_config = getattr(self.args, 'dp_fedavg_local')
            if hasattr(dp_config, key):
                return getattr(dp_config, key)

        # Use provided default
        if default is not None:
            return default

        # Hardcoded defaults as last resort
        defaults = {
            'clip_norm': 1.0,
            'sigma': 0.1,
            'algorithm_variant': 'step_noise',
            'clip_method': 'global'
        }

        if key in defaults:
            return defaults[key]

        # If nothing found, raise a helpful error
        raise AttributeError(
            f"Configuration key '{key}' not found. "
            f"Please add '{key}' to your {method_name} or dp_fedavg_local configuration section."
        )

    def __init__(self, **commons):
        super().__init__(**commons)

        # Initialize DP parameters using dynamic config access
        self.clip_norm = self._get_dp_config_value('clip_norm', 1.0)
        self.sigma = self._get_dp_config_value('sigma', 0.1)
        self.data_sample_ratio = self._get_dp_config_value('data_sample_ratio', None)
        self.clip_method = self._get_dp_config_value('clip_method', 'global')
        self.sigma_dp = None
        self.model_params_diff = None

        # Support string or numeric configuration with enum
        variant_config = self._get_dp_config_value('algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = getattr(AlgorithmVariant, variant_config.upper())
        else:
            # Legacy numeric support
            self.algorithm_variant = AlgorithmVariant(variant_config)

        # Cache for model parameters dictionary
        self._cached_model_params = None
        self._cached_model_buffers = None

        # Numerical stability epsilon
        self.numerical_epsilon = 1e-6


    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.model_params_diff = {}

        # Cache model parameters dictionary for efficient access
        self._cached_model_params = {name: param for name, param in self.model.named_parameters()}
        self._cached_model_buffers = {name: buffer for name, buffer in self.model.named_buffers()}

        # Recalculate effective batch size if data_sample_ratio is set
        if self.data_sample_ratio is not None:
            self.args.common.batch_size = round(self.data_sample_ratio * len(self.trainset))

    
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
            self._compute_clipped_gradients_dispatch(x, y, add_noise=True)
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
            self._compute_clipped_gradients_dispatch(x, y, add_noise=False)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._last_noise_post_processing()
        

    @torch.no_grad()
    def _step_noise_post_processing(self):
        """Post-processing for step_noise variant: calculate DP-processed parameter differences."""
        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                self.model_params_diff[name] = param_diff.clone().cpu()

    @torch.no_grad()
    def _last_noise_post_processing(self):
        """Post-processing for last_noise variant: integrated parameter difference calculation and noise addition."""

        # σ_DP = C * K * η_l * σ_g / b_actual
        # Note: For last_noise variant, we use configured batch_size as this represents
        # the expected batch size used throughout training
        self.sigma_dp = (2 * self.clip_norm * self.sigma / self.args.common.batch_size) #* self.local_epoch * self.args.optimizer.lr
        # Calculate noisy parameter differences and store them

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                # Generate Gaussian noise (inlined for efficiency)
                noise = torch.randn_like(param_diff, device=param.device) * self.sigma_dp
                noisy_diff = param_diff + noise
                self.model_params_diff[name] = noisy_diff.clone().cpu()

    def _compute_clipped_gradients(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with optional noise addition.

        This method implements the core DP-SGD algorithm with clip→mean→add_noise order:
        - Computes per-sample gradients
        - Clips gradients based on L2 norm
        - Averages clipped gradients across batch
        - Optionally adds calibrated Gaussian noise to averaged gradient
        - Sets final gradients to model parameters

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """

        # Compute per-sample gradients
        per_sample_grads = compute_per_sample_grads(
            self.model, inputs, targets, self.criterion,
            cached_params=self._cached_model_params,
            cached_buffers=self._cached_model_buffers
        )

        # Compute per-sample gradient norms
        per_sample_norms = compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Calculate DP noise standard deviation: σ_DP = C * σ_g / b_actual
        actual_batch_size = per_sample_norms.size(0)
        self.sigma_dp = 2 * self.clip_norm * self.sigma / actual_batch_size
        # Calculate per-sample clipping factors
        per_sample_clip_factor = (self.clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0)

        # Process gradients: clip → mean → add_noise
        for param_name, per_sample_grad in per_sample_grads.items():
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)

            if add_noise:
                # Add Gaussian noise to averaged gradient
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * self.sigma_dp
                self._cached_model_params[param_name].grad = mean_clipped_grad + noise
            else:
                self._cached_model_params[param_name].grad = mean_clipped_grad

    def _compute_clipped_gradients_heuristic(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with heuristic median-based clipping.

        This method uses a heuristic approach where each parameter layer computes its own
        dynamic clipping threshold (max_norm) based on the median of per-sample gradient norms
        for that layer. This allows adaptive clipping that responds to the actual gradient
        distribution of each layer independently.

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        # Compute per-sample gradients using parent class method
        per_sample_grads = compute_per_sample_grads(
            self.model, inputs, targets, self.criterion,
            cached_params=self._cached_model_params,
            cached_buffers=self._cached_model_buffers
        )

        if len(per_sample_grads) == 0:
            return

        # Get actual batch size
        actual_batch_size = next(iter(per_sample_grads.values())).size(0)
        # Process each parameter layer independently with its own heuristic max_norm
        for param_name, per_sample_grad in per_sample_grads.items():
            # Compute per-sample gradient norms for this layer
            # per_sample_grad shape: [batch_size, *param_shape]
            per_sample_norms_layer = per_sample_grad.reshape(actual_batch_size, -1).norm(2, dim=1)

            # Heuristic: use median of per-sample norms as max_norm for this layer
            max_norm = torch.median(per_sample_norms_layer).item()
            # print(f"max_norm: {max_norm}")
            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (max_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)

            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)
            if add_noise:
                # Calculate DP noise standard deviation: σ_DP = 2 * max_norm * σ_g / b_actual
                sigma_dp_layer = 2 * max_norm * self.sigma / actual_batch_size
                # print(f"sigma_dp_layer: {sigma_dp_layer}")
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * sigma_dp_layer
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Set the final gradient
            self._cached_model_params[param_name].grad = noisy_grad

    def _compute_clipped_gradients_per_layer(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with per-layer processing.

        This method processes each parameter layer independently while using
        a uniform clipping threshold (clip_norm) across all layers. This allows
        per-layer gradient processing with consistent privacy guarantees.

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        # Compute per-sample gradients using parent class method
        per_sample_grads = compute_per_sample_grads(
            self.model, inputs, targets, self.criterion,
            cached_params=self._cached_model_params,
            cached_buffers=self._cached_model_buffers
        )

        if len(per_sample_grads) == 0:
            return

        # Get actual batch size
        actual_batch_size = next(iter(per_sample_grads.values())).size(0)
        self.sigma_dp = 2 * self.clip_norm * self.sigma / actual_batch_size
        # Process each parameter layer independently
        for param_name, per_sample_grad in per_sample_grads.items():
            # Compute per-sample gradient norms for this layer
            # per_sample_grad shape: [batch_size, *param_shape]
            per_sample_norms_layer = per_sample_grad.reshape(actual_batch_size, -1).norm(2, dim=1)

            max_norm = self.clip_norm
            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (max_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)

            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)
            if add_noise:
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * self.sigma_dp
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Set the final gradient
            self._cached_model_params[param_name].grad = noisy_grad

    def _compute_clipped_gradients_dispatch(self, inputs, targets, add_noise=True):
        """Dispatch to appropriate gradient clipping method based on clip_method.

        This method routes to the appropriate clipping strategy:
        - 'global': Global clipping with fixed clip_norm
        - 'heuristic': Adaptive per-layer clipping using median
        - 'per_layer': Per-layer clipping with fixed clip_norm

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        if self.clip_method == 'global':
            return self._compute_clipped_gradients(inputs, targets, add_noise)
        elif self.clip_method == 'heuristic':
            return self._compute_clipped_gradients_heuristic(inputs, targets, add_noise)
        elif self.clip_method == 'per_layer':
            return self._compute_clipped_gradients_per_layer(inputs, targets, add_noise)
        else:
            raise ValueError(
                f"Unknown clip_method: {self.clip_method}. "
                f"Valid options are: 'global', 'heuristic', 'per_layer'"
            )

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
        """Get a batch of data by random sampling from trainset.

        This method samples data randomly from the full training set each time,
        ensuring true randomness across steps when data_sample_ratio is used.

        Returns:
            Tuple of (x, y) tensors on the appropriate device
        """
        full_indices = list(self.trainset.indices)
        batch_size = self.args.common.batch_size

        # Handle edge case: batch_size might exceed trainset size
        actual_sample_size = min(batch_size, len(full_indices))

        # Ensure batch size > 1 to avoid batchnorm issues
        if actual_sample_size <= 1:
            actual_sample_size = min(2, len(full_indices))

        # Random sampling from full training set
        sampled_indices = random.sample(full_indices, actual_sample_size)

        # Fetch sampled data
        sampled_data = [self.dataset[idx] for idx in sampled_indices]
        x = torch.stack([item[0] for item in sampled_data])
        y = torch.tensor([item[1] for item in sampled_data])

        return x.to(self.device), y.to(self.device)
