from typing import Any
from enum import Enum
import torch
from collections import OrderedDict
from src.client.dp_scaffold import DPScaffoldClient
from src.utils.jse_utils import JSEProcessor
from src.utils.dp_mechanisms import compute_per_sample_grads, compute_per_sample_norms


class ScaffSteinAlgorithmVariant(Enum):
    """Algorithm variants for DP-ScaffStein implementation."""
    LAST_NOISE_SERVER_JSE = 1
    STEP_NOISE_STEP_JSE = 2
    STEP_NOISE_FINAL_JSE = 3


class DPScaffSteinClient(DPScaffoldClient):
    """DP-ScaffStein Client combining Differential Privacy, SCAFFOLD control variates, and JSE.

    This client extends DPScaffoldClient to add James-Stein Estimator (JSE) functionality
    across three algorithm variants:
    1. last_noise_server_jse: DP noise at last step, JSE at server
    2. step_noise_step_jse: DP noise and JSE at each step
    3. step_noise_final_jse: DP noise at each step, JSE at final step

    Inherits SCAFFOLD and DP functionality from DPScaffoldClient while adding JSE processing.
    """

    def __init__(self, **commons):
        # Temporarily override algorithm_variant config to prevent parent class conflicts
        original_args = commons['args']
        temp_variant_config = None

        # Check if we have ScaffStein-specific algorithm variant
        if hasattr(original_args, 'dp_scaffstein') and hasattr(original_args.dp_scaffstein, 'algorithm_variant'):
            temp_variant_config = original_args.dp_scaffstein.algorithm_variant
            # Map ScaffStein variants to parent class compatible variants
            if temp_variant_config in ['last_noise_server_jse', 'LAST_NOISE_SERVER_JSE']:
                original_args.dp_scaffstein.algorithm_variant = 'last_noise'
            elif temp_variant_config in ['step_noise_step_jse', 'STEP_NOISE_STEP_JSE', 'step_noise_final_jse', 'STEP_NOISE_FINAL_JSE']:
                original_args.dp_scaffstein.algorithm_variant = 'step_noise'

        super().__init__(**commons)

        # Restore and set ScaffStein-specific algorithm variant
        if temp_variant_config:
            original_args.dp_scaffstein.algorithm_variant = temp_variant_config
            if isinstance(temp_variant_config, str):
                self.scaffstein_algorithm_variant = getattr(ScaffSteinAlgorithmVariant, temp_variant_config.upper())
            else:
                self.scaffstein_algorithm_variant = ScaffSteinAlgorithmVariant(temp_variant_config)
        else:
            # Default variant
            self.scaffstein_algorithm_variant = ScaffSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE

        # Read data_sample_ratio from dp_scaffstein config if not already set by parent
        if not hasattr(self, 'data_sample_ratio') or self.data_sample_ratio is None:
            if hasattr(original_args, 'dp_scaffstein') and hasattr(original_args.dp_scaffstein, 'data_sample_ratio'):
                self.data_sample_ratio = original_args.dp_scaffstein.data_sample_ratio

        # Initialize shrinkage factor tracking
        self.shrinkage_factors = []
        self.last_local_epoch_shrinkage = None
        self.client_shrinkage_factor = None

        # Initialize JSE sign tracking for variant 2
        self.jse_sign_tracker = {'positive': 0, 'negative': 0, 'zero': 0}

    def fit(self):
        """Train the model with DP-ScaffStein algorithm.

        Routes to the appropriate JSE algorithm variant implementation:
        - Variant 1: last_noise_server_jse - DP noise at last step, JSE at server
        - Variant 2: step_noise_step_jse - DP noise and JSE at each step
        - Variant 3: step_noise_final_jse - DP noise at each step, JSE at final step
        """
        if self.scaffstein_algorithm_variant == ScaffSteinAlgorithmVariant.LAST_NOISE_SERVER_JSE:
            self._fit_variant_1_last_noise_server_jse()
        elif self.scaffstein_algorithm_variant == ScaffSteinAlgorithmVariant.STEP_NOISE_STEP_JSE:
            self._fit_variant_2_step_noise_step_jse()
        elif self.scaffstein_algorithm_variant == ScaffSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE:
            self._fit_variant_3_step_noise_final_jse()
        else:
            raise ValueError(f"Unknown ScaffStein algorithm variant: {self.scaffstein_algorithm_variant}")

    def _fit_variant_1_last_noise_server_jse(self):
        """Algorithm Variant 1: Last noise with server-side JSE.

        Reuses parent's _last_noise_training but removes client-side JSE
        since JSE will be applied at the server.
        """
        # Use parent class last_noise training
        self._last_noise_training()

    def _compute_clipped_gradients_with_step_jse(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with step-wise JSE application.

        This method implements the complete gradient computation flow with custom order:
        1. Compute per-sample gradients and clip
        2. Average clipped gradients
        3. Add noise
        4. Apply global JSE
        5. Add SCAFFOLD control variate

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

        # Compute per-sample gradient norms
        per_sample_norms = compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Calculate DP noise standard deviation: σ_DP = C * σ_g / b_actual
        actual_batch_size = per_sample_norms.size(0)
        self.sigma_dp = 2 * self.clip_norm * self.sigma / actual_batch_size

        # Calculate per-sample clipping factors
        per_sample_clip_factor = (self.clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0)

        # Process gradients: clip → mean → add_noise (without SCAFFOLD correction yet)
        for param_name, per_sample_grad in per_sample_grads.items():
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)

            if add_noise:
                # Add Gaussian noise to averaged gradient
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * self.sigma_dp
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Set the gradient (without control variate yet)
            self._cached_model_params[param_name].grad = noisy_grad

        # Apply global JSE to gradients
        if add_noise:
            shrinkage_factor = JSEProcessor.apply_global_jse_to_gradients(
                list(self.model.parameters()), self.sigma_dp**2
            )
            # Store shrinkage factor
            self.shrinkage_factors.append(shrinkage_factor)

            # Track sign of shrinkage factor
            if shrinkage_factor > 0:
                self.jse_sign_tracker['positive'] += 1
            elif shrinkage_factor < 0:
                self.jse_sign_tracker['negative'] += 1
            else:
                self.jse_sign_tracker['zero'] += 1

        # Add SCAFFOLD control variate correction after JSE
        for param_name in per_sample_grads.keys():
            c_global = self.c_global[param_name]
            c_local = self.c_local[param_name]
            self._cached_model_params[param_name].grad += (c_global - c_local).to(self.device)

    def _compute_clipped_gradients_per_layer_with_step_jse(self, inputs, targets, add_noise=True):
        """Compute per-layer clipped gradients with step-wise global JSE and SCAFFOLD correction.

        This method implements per-layer gradient clipping followed by global JSE and SCAFFOLD:
        1. Compute per-sample gradients
        2. Clip each layer independently using uniform clip_norm
        3. Average clipped gradients across batch
        4. Add noise
        5. Apply global JSE across all parameters
        6. Add SCAFFOLD control variate correction

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

            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (self.clip_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)

            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)

            if add_noise:
                # Add Gaussian noise to averaged gradient
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * self.sigma_dp
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Set the gradient (without control variate yet)
            self._cached_model_params[param_name].grad = noisy_grad

        # Apply global JSE to gradients (across all parameters)
        if add_noise:
            shrinkage_factor = JSEProcessor.apply_global_jse_to_gradients(
                list(self.model.parameters()), self.sigma_dp**2
            )
            # Store shrinkage factor
            self.shrinkage_factors.append(shrinkage_factor)

            # Track sign of shrinkage factor
            if shrinkage_factor > 0:
                self.jse_sign_tracker['positive'] += 1
            elif shrinkage_factor < 0:
                self.jse_sign_tracker['negative'] += 1
            else:
                self.jse_sign_tracker['zero'] += 1

        # Add SCAFFOLD control variate correction after JSE
        for param_name in per_sample_grads.keys():
            c_global = self.c_global[param_name]
            c_local = self.c_local[param_name]
            self._cached_model_params[param_name].grad += (c_global - c_local).to(self.device)

    def _compute_clipped_gradients_dispatch_with_step_jse(self, inputs, targets, add_noise=True):
        """Dispatch to appropriate gradient clipping method with step-wise JSE and SCAFFOLD.

        Routes to the appropriate clipping strategy based on clip_method:
        - 'global': Global clipping with fixed clip_norm
        - 'per_layer': Per-layer clipping with fixed clip_norm

        Note: 'heuristic' method is not supported for JSE variants.

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        if self.clip_method == 'global':
            return self._compute_clipped_gradients_with_step_jse(inputs, targets, add_noise)
        elif self.clip_method == 'per_layer':
            return self._compute_clipped_gradients_per_layer_with_step_jse(inputs, targets, add_noise)
        else:
            raise ValueError(
                f"Unsupported clip_method for JSE: {self.clip_method}. "
                f"JSE variants only support 'global' and 'per_layer'. "
                f"'heuristic' is not compatible with JSE processing."
            )

    def _fit_variant_2_step_noise_step_jse(self):
        """Algorithm Variant 2: Step-wise DP training with per-step JSE.

        Executes training with per-step DP processing and JSE using the new
        torch.func based gradient computation.
        """
        self.model.train()
        self.dataset.train()

        # Clear shrinkage factors for new round
        self.shrinkage_factors = []

        # Clear JSE sign tracker for new round
        self.jse_sign_tracker = {'positive': 0, 'negative': 0, 'zero': 0}

        # Local training loop with per-step DP processing and JSE
        for epoch_idx in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            # Use gradient computation with step-wise JSE (supports clip_method dispatch)
            self._compute_clipped_gradients_dispatch_with_step_jse(x, y, add_noise=True)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Record shrinkage factor from last local epoch
            if epoch_idx == self.local_epoch - 1 and len(self.shrinkage_factors) > 0:
                self.last_local_epoch_shrinkage = self.shrinkage_factors[-1]

        self._compute_param_diff_and_control_variates(add_noise=False)

    def _fit_variant_3_step_noise_final_jse(self):
        """Algorithm Variant 3: Gradient-level noise + Final global JSE on parameter differences.

        Training flow:
        1. Standard step-wise DP training (reuse parent class logic)
        2. Apply global JSE to final parameter differences with accumulated noise variance

        Global JSE processes all parameter differences simultaneously using unified
        shrinkage based on the combined norm of all parameters, providing consistent
        and mathematically principled shrinkage across the entire model.
        """
        # Execute standard step-wise DP training using parent class
        self._step_noise_training()

        # Apply global JSE to final parameter differences with K factor
        # Following Algorithm 3: shrinkage = (d-2)K·σ²_DP / ||A||², where K = local_epoch
        shrinkage_factor = JSEProcessor.apply_global_jse_to_parameter_diff(
            self.model_params_diff, self.sigma_dp**2, k_factor=self.local_epoch
        )

        # Store shrinkage factor
        self.client_shrinkage_factor = shrinkage_factor



    def package(self):
        """Package client data including DP parameters, SCAFFOLD control variates, and JSE information."""
        client_package = super().package()

        # Add sigma_dp for server-side JSE processing (variant 1)
        if hasattr(self, 'sigma_dp'):
            client_package["sigma_dp"] = self.sigma_dp

        # Add shrinkage factor for variant 2 and 3
        if self.scaffstein_algorithm_variant == ScaffSteinAlgorithmVariant.STEP_NOISE_STEP_JSE:
            # Variant 2: send last local epoch shrinkage
            client_package["shrinkage_factor"] = self.last_local_epoch_shrinkage if self.last_local_epoch_shrinkage is not None else 1.0
            # Add JSE sign statistics
            total_count = sum(self.jse_sign_tracker.values())
            positive_ratio = self.jse_sign_tracker['positive'] / total_count if total_count > 0 else 0.0
            client_package["jse_sign_stats"] = self.jse_sign_tracker.copy()
            client_package["jse_positive_ratio"] = positive_ratio
        elif self.scaffstein_algorithm_variant == ScaffSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE:
            # Variant 3: send client shrinkage factor
            client_package["shrinkage_factor"] = self.client_shrinkage_factor if self.client_shrinkage_factor is not None else 1.0

        return client_package