from typing import Any
import logging

import torch
from torch.utils.data import DataLoader

from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.utils.jse_utils import JSEProcessor


class DPFedSteinClient(DPFedAvgLocalClient):

    # Numerical stability constants
    EPSILON = 1e-6
    DEFAULT_BATCH_SIZE = 32

    # Define algorithm variant constants for DPFedStein
    ALGORITHM_VARIANTS = {
        'last_noise_server_jse': 1,
        'step_noise_step_jse': 2,
        'step_noise_final_jse': 3
    }

    def __init__(self, **commons):
        dp_fed_stein_config = commons['args'].dp_fed_stein

        # Setup dp_fedavg_local config for parent class compatibility
        if not hasattr(commons['args'], 'dp_fedavg_local'):
            from omegaconf import DictConfig
            dp_fedavg_local_config = DictConfig({
                'clip_norm': dp_fed_stein_config.clip_norm,
                'sigma': dp_fed_stein_config.sigma,
                'algorithm_variant': 2  # step_noise variant (numeric value for parent class)
            })
            commons['args'].dp_fedavg_local = dp_fedavg_local_config

        # Initialize parent class
        super().__init__(**commons)

        # Override DP parameters from dp_fed_stein config
        self.clip_norm = self.args.dp_fed_stein.clip_norm
        self.sigma = self.args.dp_fed_stein.sigma

        # Override algorithm variants to support JSE variants
        variant_config = getattr(self.args.dp_fed_stein, 'algorithm_variant', 'step_noise_step_jse')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config


    def fit(self):

        # Route to appropriate algorithm variant implementation
        if self.algorithm_variant == 1:  # last_noise_server_jse
            self._fit_variant_1_last_noise_server_jse()
        elif self.algorithm_variant == 2:  # step_noise_step_jse
            self._fit_variant_2_step_noise_step_jse()
        elif self.algorithm_variant == 3:  # step_noise_final_jse
            self._fit_variant_3_step_noise_final_jse()
        else:
            raise ValueError(f"Unknown algorithm variant: {self.algorithm_variant}")

    def _fit_variant_1_last_noise_server_jse(self):
        """Algorithm Variant 1: Training with parameter-level noise, server-side JSE.

        Executes training with clipping but no noise addition.
        Noise is added at parameter level after training completion.
        """

        self._last_noise_training()


    def _fit_variant_2_step_noise_step_jse(self):
        """Algorithm Variant 2: Step-wise DP training with per-step JSE.

        Executes training with per-step DP processing and JSE.
        """

        self.model.train()
        self.dataset.train()

        # Local training loop with per-step DP processing and JSE
        for _ in range(self.local_epoch):
            # Get training batch for this step
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self._clip_and_add_noise_opacus()
            JSEProcessor.apply_global_jse_to_gradients(
                list(self.model.parameters()), self.sigma_dp
            )
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _fit_variant_3_step_noise_final_jse(self):
        """Algorithm Variant 3: Gradient-level noise + Final global JSE on parameter differences.

        Training flow:
        1. Standard step-wise DP training (reuse parent class logic)
        2. Apply global JSE to final parameter differences with accumulated noise variance

        Global JSE processes all parameter differences simultaneously using unified
        shrinkage based on the combined norm of all parameters, providing consistent
        and mathematically principled shrinkage across the entire model.

        Returns:
            Parameter differences after final global JSE processing
        """
        # Execute standard step-wise DP training (JSE-compatible version)
        self._step_noise_training()

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                clean_name = self._get_clean_param_name(name)
                self.dp_processed_diff[clean_name] = param_diff.clone().cpu()
                
        # Apply final JSE to aggregated parameter differences
        JSEProcessor.apply_global_jse_to_parameter_diff(
            self.dp_processed_diff, self.sigma_dp
        )


    def package_for_algorithm_variant(self, client_package: dict):

        # Algorithm-specific data packaging
        if self.algorithm_variant == 2:  # step_noise_step_jse
            client_package["model_params_diff"] = self._compute_clean_diff()
        else:  # last_noise_server_jse, step_noise_final_jse
            client_package["model_params_diff"] = self.dp_processed_diff
            

