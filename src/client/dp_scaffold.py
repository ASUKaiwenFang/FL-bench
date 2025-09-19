from typing import Any, Iterator
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.optimizers.optimizer import _generate_noise
from collections import OrderedDict
from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.client.fedavg import FedAvgClient


class DPScaffoldClient(DPFedAvgLocalClient):
    """DP-SCAFFOLD Client combining Differential Privacy with SCAFFOLD control variates.

    This client implements SCAFFOLD algorithm with local differential privacy using
    Opacus GradSampleModule for efficient per-sample gradient computation. It combines
    the control variate mechanism from SCAFFOLD with the differential privacy
    protection from DP-FedAvg Local.

    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step
    - last_noise: Add noise to parameter differences after training completion
    """

    def __init__(self, **commons):
        # Skip DPFedAvgLocalClient.__init__ and call FedAvgClient.__init__ directly
        FedAvgClient.__init__(self, **commons)

        # Initialize iter_trainloader (from DPFedAvgLocalClient)
        self.iter_trainloader = None

        # Initialize DP parameters from dp_scaffold config
        self.clip_norm = self.args.dp_scaffold.clip_norm
        self.sigma = self.args.dp_scaffold.sigma
        self.sigma_dp = None
        self.model_params_diff = None

        # Support string or numeric configuration for algorithm variant
        ALGORITHM_VARIANTS = {
            "last_noise": 1,
            "step_noise": 2,
        }
        variant_config = getattr(self.args.dp_scaffold, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config

        # Initialize SCAFFOLD control variates using OrderedDict format
        self.c_local: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.c_global: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.c_delta: OrderedDict[str, torch.Tensor] = OrderedDict()

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)

        # Set SCAFFOLD control variates from server package
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]



    def _step_noise_training(self):
        """Gradient-level noise addition with SCAFFOLD control variates.

        This method implements the per-sample DP-SGD algorithm with SCAFFOLD:
        1. Forward pass (GradSampleModule automatically computes per-sample gradients)
        2. Backward pass (per-sample gradients stored in param.grad_sample)
        3. Add SCAFFOLD control variate correction to gradients
        4. Clip each sample's gradients independently
        5. Average the clipped gradients and add noise
        6. Apply noisy gradients
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

            # Apply DP clipping and noise with integrated SCAFFOLD control variate correction
            self._clip_and_add_noise_with_scaffold()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        self._step_noise_post_processing_with_integrated_control_variates()

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
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            
            # Clip gradients with integrated SCAFFOLD control variate correction
            self._clip_gradients_with_scaffold()

            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Add noise to parameter differences and update control variates simultaneously
        self._last_noise_post_processing_with_integrated_control_variates()


    @torch.no_grad()
    def _last_noise_post_processing_with_integrated_control_variates(self):
        """Integrated processing for last_noise variant: DP noise addition and control variate updates.

        This method combines:
        1. Parameter difference calculation for DP noise addition
        2. Control variate updates (c_delta, c_local)
        All in a single loop for optimal performance.
        """
        # Calculate DP noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        sigma_dp = self.clip_norm * self.local_epoch * self.args.optimizer.lr * self.sigma / self.args.common.batch_size
        self.sigma_dp = sigma_dp

        # Initialize storage
        self.model_params_diff = {}
        self.c_delta = OrderedDict()
        c_plus = OrderedDict()

        model_params = self.model.state_dict()
        coef = 1 / (self.local_epoch * self.args.optimizer.lr)

        # Integrated processing loop
        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                # Calculate parameter difference once
                param_diff = param.data - self.regular_model_params[name].to(param.device)

                # DP processing: add noise to parameter difference
                noise = _generate_noise(
                    std=sigma_dp,
                    reference=param_diff,
                    generator=None,
                    secure_mode=False
                )
                noisy_diff = param_diff + noise
                clean_name = self._get_clean_param_name(name)
                self.model_params_diff[clean_name] = noisy_diff.clone().cpu()

                # SCAFFOLD control variate processing: use clean parameter difference
                # Try both original name and clean name for compatibility
                clean_name = self._get_clean_param_name(name)
                control_key = name if name in self.c_global else clean_name

                if control_key in self.c_global and control_key in self.c_local:

                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]

                    # Use clean param_diff for control variate calculation
                    param_diff_cpu = param_diff.cpu()
                    c_plus[control_key] = c_local - c_global - coef * param_diff_cpu
                    self.c_delta[control_key] = c_plus[control_key] - c_local

        # Update local control variates
        self.c_local = c_plus

    @torch.no_grad()
    def _step_noise_post_processing_with_integrated_control_variates(self):
        """Integrated post-processing for step_noise variant: DP difference calculation and control variate updates.

        This method combines:
        1. Parameter difference calculation for DP-processed differences
        2. Control variate updates (c_delta, c_local)
        All in a single loop for optimal performance.
        """
        # Initialize storage
        self.model_params_diff = {}
        self.c_delta = OrderedDict()
        c_plus = OrderedDict()

        coef = 1 / (self.local_epoch * self.args.optimizer.lr)

        # Integrated processing loop
        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                # Calculate parameter difference once (already includes DP noise from training)
                param_diff = param.data - self.regular_model_params[name].to(param.device)

                # DP processing: store parameter difference (already noisy from step_noise training)
                clean_name = self._get_clean_param_name(name)
                self.model_params_diff[clean_name] = param_diff.clone().cpu()

                # SCAFFOLD control variate processing: use parameter difference
                # Try both original name and clean name for compatibility
                control_key = name if name in self.c_global else clean_name

                if control_key in self.c_global and control_key in self.c_local:
                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]

                    # Use param_diff for control variate calculation
                    param_diff_cpu = param_diff.cpu()
                    c_plus[control_key] = c_local - c_global - coef * param_diff_cpu
                    self.c_delta[control_key] = c_plus[control_key] - c_local

        # Update local control variates
        self.c_local = c_plus

    def _clip_gradients_with_scaffold(self):
        """Clip per-sample gradients and apply SCAFFOLD control variate correction.

        This method integrates the gradient clipping from DP-FedAvg Local with
        SCAFFOLD control variate correction for optimal performance.
        """
        # Compute per-sample norms (copied from parent class)
        per_sample_norms = self._compute_per_sample_norms_opacus()
        if len(per_sample_norms) == 0:
            return

        per_sample_clip_factor = (
            self.clip_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        # Process each parameter with integrated clipping and control variate correction
        for name, param in self.model.named_parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                # Apply DP clipping
                clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, param.grad_sample)

                # Apply SCAFFOLD control variate correction
                clean_name = self._get_clean_param_name(name)
                control_key = name if name in self.c_global else clean_name
                if control_key in self.c_global and control_key in self.c_local:
                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]
                    clipped_grad += (c_global - c_local).to(self.device)

                # Set the final gradient
                param.grad = clipped_grad

    def _clip_and_add_noise_with_scaffold(self):
        """Clip per-sample gradients, add noise, and apply SCAFFOLD control variate correction.

        This method integrates the DP processing from DP-FedAvg Local with
        SCAFFOLD control variate correction for optimal performance.
        """
        # Compute per-sample norms (copied from parent class)
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

        # Process each parameter with integrated clipping, noise, and control variate correction
        for name, param in self.model.named_parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                # Apply DP clipping
                clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, param.grad_sample)

                # Add Gaussian noise
                noisy_grad = clipped_grad + _generate_noise(
                    std=sigma_dp,
                    reference=clipped_grad,
                    generator=None,
                    secure_mode=False
                )

                # Apply SCAFFOLD control variate correction
                clean_name = self._get_clean_param_name(name)
                control_key = name if name in self.c_global else clean_name
                if control_key in self.c_global and control_key in self.c_local:
                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]
                    noisy_grad += (c_global - c_local).to(self.device)

                # Set the final gradient
                param.grad = noisy_grad

    def package(self):
        """Package client data including DP parameters and SCAFFOLD control variates."""
        client_package = super().package()

        client_package["c_delta"] = self.c_delta

        return client_package