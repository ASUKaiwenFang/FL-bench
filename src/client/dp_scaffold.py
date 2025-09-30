from typing import Any
from copy import deepcopy
import torch
import numpy as np
from collections import OrderedDict
from src.client.dp_fedavg_local import DPFedAvgLocalClient, AlgorithmVariant
from src.utils.dp_mechanisms import compute_per_sample_grads, compute_per_sample_norms


class DPScaffoldClient(DPFedAvgLocalClient):
    """DP-SCAFFOLD Client combining Differential Privacy with SCAFFOLD control variates.

    This client implements SCAFFOLD algorithm with local differential privacy using
    torch.func for efficient per-sample gradient computation. It combines
    the control variate mechanism from SCAFFOLD with the differential privacy
    protection from DP-FedAvg Local.

    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step
    - last_noise: Add noise to parameter differences after training completion

    Performance optimizations:
    - Unified parameter processing method eliminates code duplication
    - Direct iteration over regular_model_params improves efficiency
    - Optimized data transfers reduce CPU/GPU memory overhead
    """


    def __init__(self, **commons):
        super().__init__(**commons)

        # Override DP parameters from dp_scaffold config if different from parent
        if hasattr(self.args, 'dp_scaffold'):
            self.clip_norm = self.args.dp_scaffold.clip_norm
            self.sigma = self.args.dp_scaffold.sigma

            # Override algorithm variant configuration using AlgorithmVariant enum
            variant_config = getattr(self.args.dp_scaffold, 'algorithm_variant', 'step_noise')
            if isinstance(variant_config, str):
                self.algorithm_variant = getattr(AlgorithmVariant, variant_config.upper())
            else:
                # Legacy numeric support
                self.algorithm_variant = AlgorithmVariant(variant_config)

        # Read data_sample_ratio from dp_scaffold config if not already set by parent
        if not hasattr(self, 'data_sample_ratio') or self.data_sample_ratio is None:
            if hasattr(self.args, 'dp_scaffold') and hasattr(self.args.dp_scaffold, 'data_sample_ratio'):
                self.data_sample_ratio = self.args.dp_scaffold.data_sample_ratio

        # Initialize SCAFFOLD control variates using OrderedDict format
        self.c_local: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.c_global: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.c_delta: OrderedDict[str, torch.Tensor] = OrderedDict()

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

        # Set SCAFFOLD control variates from server package
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]

    def _compute_clipped_gradients(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with SCAFFOLD control variate correction.

        This method extends the parent class gradient computation by adding
        SCAFFOLD control variate corrections after gradient computation.

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

        # Process gradients: clip → mean → add_noise → add SCAFFOLD correction
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

            # Apply SCAFFOLD control variate correction
            c_global = self.c_global[param_name]
            c_local = self.c_local[param_name]
            noisy_grad += (c_global - c_local).to(self.device)

            # Set the final gradient
            self._cached_model_params[param_name].grad = noisy_grad

    def _compute_clipped_gradients_heuristic(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with heuristic median-based clipping and SCAFFOLD control variate correction.

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
        # print(f"client id: {self.client_id}, actual_batch_size: {actual_batch_size}")
        # Process each parameter layer independently with its own heuristic max_norm
        for param_name, per_sample_grad in per_sample_grads.items():
            # Compute per-sample gradient norms for this layer
            # per_sample_grad shape: [batch_size, *param_shape]
            per_sample_norms_layer = per_sample_grad.reshape(actual_batch_size, -1).norm(2, dim=1)

            # Heuristic: use median of per-sample norms as max_norm for this layer
            max_norm = float(np.median(per_sample_norms_layer.detach().cpu().numpy()))
            # if self.client_id == 14:
            #     print(f"client id: {self.client_id}, param_name: {param_name}, max_norm: {max_norm}")
            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (max_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)

            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)
            if self.client_id == 14:
                print(f"client id: {self.client_id}, param_name: {param_name}, max_norm: {max_norm}, mean_clipped_grad: {mean_clipped_grad}")
            if add_noise:
                # Calculate DP noise standard deviation: σ_DP = 2 * max_norm * σ_g / b_actual
                sigma_dp_layer = 2 * max_norm * self.sigma / actual_batch_size
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * sigma_dp_layer
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Apply SCAFFOLD control variate correction
            c_global = self.c_global[param_name]
            c_local = self.c_local[param_name]
            noisy_grad += (c_global - c_local).to(self.device)

            # Set the final gradient
            self._cached_model_params[param_name].grad = noisy_grad
            
    def _compute_clipped_gradients_per_layer(self, inputs, targets, add_noise=True):
        """Compute clipped per-sample gradients with heuristic median-based clipping and SCAFFOLD control variate correction.

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
        self.sigma_dp = 2 * self.clip_norm * self.sigma / actual_batch_size
        # print(f"client id: {self.client_id}, actual_batch_size: {actual_batch_size}")
        # Process each parameter layer independently with its own heuristic max_norm
        for param_name, per_sample_grad in per_sample_grads.items():
            # Compute per-sample gradient norms for this layer
            # per_sample_grad shape: [batch_size, *param_shape]
            per_sample_norms_layer = per_sample_grad.reshape(actual_batch_size, -1).norm(2, dim=1)

            # Heuristic: use median of per-sample norms as max_norm for this layer
            max_norm = self.clip_norm
            # if self.client_id == 14:
            #     print(f"client id: {self.client_id}, param_name: {param_name}, max_norm: {max_norm}")
            # Calculate per-sample clipping factors for this layer
            per_sample_clip_factor = (max_norm / (per_sample_norms_layer + self.numerical_epsilon)).clamp(max=1.0)

            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [actual_batch_size] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Average clipped gradients across batch
            mean_clipped_grad = clipped_grad.mean(dim=0)
            if self.client_id == 14:
                print(f"client id: {self.client_id}, param_name: {param_name}, max_norm: {max_norm}, mean_clipped_grad: {mean_clipped_grad}")
            if add_noise:
                # Calculate DP noise standard deviation: σ_DP = 2 * max_norm * σ_g / b_actual
                # sigma_dp_layer = 2 * max_norm * self.sigma / actual_batch_size
                noise = torch.randn_like(mean_clipped_grad, device=self.device) * self.sigma_dp
                noisy_grad = mean_clipped_grad + noise
            else:
                noisy_grad = mean_clipped_grad

            # Apply SCAFFOLD control variate correction
            c_global = self.c_global[param_name]
            c_local = self.c_local[param_name]
            noisy_grad += (c_global - c_local).to(self.device)

            # Set the final gradient
            self._cached_model_params[param_name].grad = noisy_grad

    def _step_noise_training(self):
        """Gradient-level noise addition with SCAFFOLD control variates.

        This method implements the per-sample DP-SGD algorithm with SCAFFOLD:
        1. Compute per-sample gradients using torch.func
        2. Clip each sample's gradients independently
        3. Average the clipped gradients and add noise
        4. Add SCAFFOLD control variate correction to gradients
        5. Apply noisy gradients
        """
        self.model.train()
        self.dataset.train()

        for _ in range(self.local_epoch):

            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            # Use new torch.func based gradient computation with SCAFFOLD integration
            # self._compute_clipped_gradients_heuristic(x, y, add_noise=True)
            # self._compute_clipped_gradients(x, y, add_noise=True)
            self._compute_clipped_gradients_per_layer(x, y, add_noise=True)
            self.optimizer.step()

            if self.lr_scheduler is not None:   
                self.lr_scheduler.step()

        self._compute_param_diff_and_control_variates(add_noise=False)

    def _last_noise_training(self):
        """Parameter-level noise addition with SCAFFOLD control variates.

        Train with SCAFFOLD control variates, then add noise to parameter differences.
        Uses noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        """
        self.model.train()
        self.dataset.train()

        # Standard training with SCAFFOLD control variates
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            # Use new torch.func based gradient computation with SCAFFOLD integration (no noise)
            self._compute_clipped_gradients(x, y, add_noise=False)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Add noise to parameter differences and update control variates simultaneously
        self._compute_param_diff_and_control_variates(add_noise=True)


    def package(self):
        """Package client data including DP parameters and SCAFFOLD control variates."""
        client_package = super().package()

        client_package["c_delta"] = self.c_delta

        return client_package

    @torch.no_grad()
    def _compute_param_diff_and_control_variates(self, add_noise=False):
        """Unified method for computing parameter differences and updating control variates.

        Args:
            add_noise: Whether to add Gaussian noise to parameter differences

        Returns:
            None (updates self.model_params_diff, self.c_delta, and self.c_local)
        """
        # Initialize storage
        self.model_params_diff = {}
        self.c_delta = OrderedDict()
        c_plus = OrderedDict()

        coef = 1 / (self.local_epoch * self.args.optimizer.lr)

        # Optimized loop: iterate directly over regular_model_params
        for name in self.regular_model_params:
            # Calculate parameter difference using cached model params
            current_param = self._cached_model_params[name].data
            param_diff = current_param - self.regular_model_params[name].to(current_param.device)

            # DP processing: optionally add noise to parameter difference
            if add_noise:
                # σ_DP = C * K * η_l * σ_g / b_actual (for parameter-level noise)
                self.sigma_dp = (self.clip_norm * self.sigma / self.args.common.batch_size) * self.local_epoch * self.args.optimizer.lr
                noise = torch.randn_like(param_diff, device=param_diff.device) * self.sigma_dp
                noisy_diff = param_diff + noise
                self.model_params_diff[name] = noisy_diff.cpu()
            else:
                # Store parameter difference (already noisy from step_noise training)
                self.model_params_diff[name] = param_diff.cpu()

            # SCAFFOLD control variate processing
            c_global = self.c_global[name]
            c_local = self.c_local[name]

            # Use param_diff for control variate calculation (clean version without noise)
            c_plus[name] = c_local - c_global - coef * self.model_params_diff[name]
            self.c_delta[name] = c_plus[name] - c_local

        # Update local control variates
        self.c_local = c_plus