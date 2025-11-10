from typing import Any
from enum import Enum
import torch
from src.client.dp_fedavg_local import DPFedAvgLocalClient, AlgorithmVariant
from src.utils.jse_utils import JSEProcessor
from src.utils.dp_mechanisms import compute_per_sample_grads, compute_per_sample_norms


class FedSteinAlgorithmVariant(Enum):
    """Algorithm variants for DP-FedStein implementation."""
    LAST_NOISE_SERVER_JSE = 1
    STEP_NOISE_STEP_JSE = 2
    STEP_NOISE_FINAL_JSE = 3


class DPFedSteinClient(DPFedAvgLocalClient):

    def __init__(self, **commons):
        # Temporarily override algorithm_variant config to prevent parent class conflicts
        original_args = commons['args']
        temp_variant_config = None

        # Check if we have FedStein-specific algorithm variant
        if hasattr(original_args, 'dp_fed_stein') and hasattr(original_args.dp_fed_stein, 'algorithm_variant'):
            temp_variant_config = original_args.dp_fed_stein.algorithm_variant
            # Map FedStein variants to parent class compatible variants
            if temp_variant_config in ['last_noise_server_jse', 'LAST_NOISE_SERVER_JSE']:
                original_args.dp_fed_stein.algorithm_variant = 'last_noise'
            elif temp_variant_config in ['step_noise_step_jse', 'STEP_NOISE_STEP_JSE', 'step_noise_final_jse', 'STEP_NOISE_FINAL_JSE']:
                original_args.dp_fed_stein.algorithm_variant = 'step_noise'

        super().__init__(**commons)

        # Restore and set FedStein-specific algorithm variant
        if temp_variant_config:
            original_args.dp_fed_stein.algorithm_variant = temp_variant_config
            if isinstance(temp_variant_config, str):
                self.fed_stein_algorithm_variant = getattr(FedSteinAlgorithmVariant, temp_variant_config.upper())
            else:
                self.fed_stein_algorithm_variant = FedSteinAlgorithmVariant(temp_variant_config)
        else:
            # Default variant
            self.fed_stein_algorithm_variant = FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE

        # Read data_sample_ratio from dp_fed_stein config if not already set by parent
        if not hasattr(self, 'data_sample_ratio') or self.data_sample_ratio is None:
            if hasattr(original_args, 'dp_fed_stein') and hasattr(original_args.dp_fed_stein, 'data_sample_ratio'):
                self.data_sample_ratio = original_args.dp_fed_stein.data_sample_ratio

        # Initialize shrinkage factor tracking
        self.shrinkage_factors = []
        self.last_local_epoch_shrinkage = None
        self.client_shrinkage_factor = None

        # Initialize JSE sign tracking for variant 2
        self.jse_sign_tracker = {'positive': 0, 'negative': 0, 'zero': 0}


    def fit(self):
        """Train the model with local differential privacy and JSE enhancement.

        Routes to appropriate algorithm variant implementation based on FedStein variant.
        """
        if self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.LAST_NOISE_SERVER_JSE:
            self._fit_variant_1_last_noise_server_jse()
        elif self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE:
            self._fit_variant_2_step_noise_step_jse()
        elif self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE:
            self._fit_variant_3_step_noise_final_jse()
        else:
            raise ValueError(f"Unknown FedStein algorithm variant: {self.fed_stein_algorithm_variant}")

    def _fit_variant_1_last_noise_server_jse(self):
        """Algorithm Variant 1: Training with parameter-level noise, server-side JSE.

        Executes training with clipping but no noise addition.
        Noise is added at parameter level after training completion.
        JSE is applied to parameter differences after noise addition.
        """
        # Use parent class last_noise training
        self._last_noise_training()


    def _compute_clipped_gradients_with_step_jse(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with step-wise JSE application.

        This method implements the complete gradient computation flow:
        1. Compute per-sample gradients and clip
        2. Average clipped gradients
        3. Add noise
        4. Apply global JSE

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
        self.sigma_dp = self.clip_norm * self.sigma / actual_batch_size

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
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Set the gradient
            self._cached_model_params[param_name].grad = noisy_grad

        # Apply global JSE to gradients
        if add_noise:
            # shrinkage_factor = JSEProcessor.apply_global_jse_to_gradients(
            #     list(self.model.parameters()), self.sigma_dp**2
            # )
            shrinkage_factor = 1.0
            JSEProcessor.apply_layerwise_jse_to_gradients(
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

    def _compute_clipped_gradients_per_layer_with_step_jse(self, inputs, targets, add_noise=True):
        """Compute per-layer clipped gradients with step-wise global JSE application.

        This method implements per-layer gradient clipping followed by global JSE:
        1. Compute per-sample gradients
        2. Clip each layer independently using uniform clip_norm
        3. Average clipped gradients across batch
        4. Add noise
        5. Apply global JSE across all parameters

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
        self.sigma_dp = self.clip_norm * self.sigma / actual_batch_size

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

            # Set the gradient
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

    def _compute_clipped_gradients_heuristic_with_step_jse(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with heuristic median-based clipping and step-wise JSE.

        This method uses a heuristic approach where each parameter layer computes its own
        dynamic clipping threshold (max_norm) based on the median of per-sample gradient norms
        for that layer. Each layer uses its own sigma_dp_layer for noise addition and JSE processing.

        Implementation flow:
        1. Compute per-sample gradients
        2. For each layer independently:
           - Compute median of per-sample norms as max_norm
           - Clip gradients using layer-specific max_norm
           - Average clipped gradients
           - Add noise with layer-specific sigma_dp_layer = 2 * max_norm * σ_g / b_actual
           - Apply JSE with layer-specific noise_variance = sigma_dp_layer²

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

        # Process each parameter layer independently with its own heuristic max_norm
        for param_name, per_sample_grad in per_sample_grads.items():
            # Compute per-sample gradient norms for this layer
            # per_sample_grad shape: [batch_size, *param_shape]
            per_sample_norms_layer = per_sample_grad.reshape(actual_batch_size, -1).norm(2, dim=1)

            # Heuristic: use median of per-sample norms as max_norm for this layer
            max_norm = torch.median(per_sample_norms_layer).item()
            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (max_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)
            
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)


            if add_noise:
                # Calculate layer-specific DP noise standard deviation: σ_DP = 2 * max_norm * σ_g / b_actual
                sigma_dp_layer = 2 * max_norm * self.sigma / actual_batch_size
                # Add Gaussian noise to averaged gradient
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * sigma_dp_layer
                noisy_grad = mean_clipped_grad + noise
                # Apply JSE shrinkage with layer-specific noise variance
                # Original: apply JSE to entire tensor
                jse_grad, shrinkage_factor = JSEProcessor.apply_jse_shrinkage(noisy_grad, sigma_dp_layer**2)

                # Store shrinkage factor
                self.shrinkage_factors.append(shrinkage_factor)

                # Track sign of shrinkage factor
                if shrinkage_factor > 0:
                    self.jse_sign_tracker['positive'] += 1
                elif shrinkage_factor < 0:
                    self.jse_sign_tracker['negative'] += 1
                else:
                    self.jse_sign_tracker['zero'] += 1

                # Set the final gradient
                self._cached_model_params[param_name].grad = jse_grad
            else:   
                # Set the gradient without noise
                self._cached_model_params[param_name].grad = mean_clipped_grad

    def _compute_clipped_gradients_dispatch_with_step_jse(self, inputs, targets, add_noise=True):
        """Dispatch to appropriate gradient clipping method with step-wise JSE.

        Routes to the appropriate clipping strategy based on clip_method:
        - 'global': Global clipping with fixed clip_norm
        - 'per_layer': Per-layer clipping with fixed clip_norm
        - 'heuristic': Adaptive per-layer clipping using median

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        if self.clip_method == 'global':
            return self._compute_clipped_gradients_with_step_jse(inputs, targets, add_noise)
        elif self.clip_method == 'per_layer':
            return self._compute_clipped_gradients_per_layer_with_step_jse(inputs, targets, add_noise)
        elif self.clip_method == 'heuristic':
            return self._compute_clipped_gradients_heuristic_with_step_jse(inputs, targets, add_noise)
        else:
            raise ValueError(
                f"Unsupported clip_method for JSE: {self.clip_method}. "
                f"JSE variants support 'global', 'per_layer', and 'heuristic'."
            )

    def _fit_variant_2_step_noise_step_jse(self):
        """Algorithm Variant 2: Step-wise DP training with per-step JSE.

        Executes training with per-step DP processing and JSE using the new
        torch.func based gradient computation.
        """

        # def init_weights(m):
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         torch.nn.init.zeros_(m.bias)

        # self.model.apply(init_weights) # Apply the initialization function to all modules

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

        self._step_noise_post_processing()

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
        """Package client data including DP parameters and JSE information."""
        client_package = super().package()

        # Add sigma_dp for server-side JSE processing (variant 1)
        if hasattr(self, 'sigma_dp'):
            client_package["sigma_dp"] = self.sigma_dp

        # Add shrinkage factor for variant 2 and 3
        if self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE:
            # Variant 2: send last local epoch shrinkage
            client_package["shrinkage_factor"] = self.last_local_epoch_shrinkage if self.last_local_epoch_shrinkage is not None else 1.0
            # Add JSE sign statistics
            total_count = sum(self.jse_sign_tracker.values())
            positive_ratio = self.jse_sign_tracker['positive'] / total_count if total_count > 0 else 0.0
            client_package["jse_sign_stats"] = self.jse_sign_tracker.copy()
            client_package["jse_positive_ratio"] = positive_ratio
        elif self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE:
            # Variant 3: send client shrinkage factor
            client_package["shrinkage_factor"] = self.client_shrinkage_factor if self.client_shrinkage_factor is not None else 1.0

        return client_package


