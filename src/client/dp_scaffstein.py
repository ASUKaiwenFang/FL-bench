from typing import Any
from enum import Enum
import torch
from collections import OrderedDict
from src.client.dp_scaffold import DPScaffoldClient
from src.utils.jse_utils import JSEProcessor


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

        This method extends the parent class gradient computation by adding
        step-wise JSE processing after gradient computation.

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
            add_noise: Whether to add Gaussian noise to gradients
        """
        # Use parent class gradient computation
        self._compute_clipped_gradients(inputs, targets, add_noise)
        # self._compute_clipped_gradients_heuristic(inputs, targets, add_noise)
        # self._compute_clipped_gradients_per_layer(inputs, targets, add_noise)

        # Apply step-wise JSE to gradients
        if add_noise:
            JSEProcessor.apply_global_jse_to_gradients(
                list(self.model.parameters()), self.sigma_dp**2
            )

    def _fit_variant_2_step_noise_step_jse(self):
        """Algorithm Variant 2: Step-wise DP training with per-step JSE.

        Executes training with per-step DP processing and JSE using the new
        torch.func based gradient computation.
        """
        self.model.train()
        self.dataset.train()

        # Local training loop with per-step DP processing and JSE
        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            # Use new gradient computation with step-wise JSE
            self._compute_clipped_gradients_with_step_jse(x, y, add_noise=True)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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
        JSEProcessor.apply_global_jse_to_parameter_diff(
            self.model_params_diff, self.sigma_dp**2, k_factor=self.local_epoch
        )



    def package(self):
        """Package client data including DP parameters, SCAFFOLD control variates, and JSE information."""
        client_package = super().package()

        # Add sigma_dp for server-side JSE processing (variant 1)
        if hasattr(self, 'sigma_dp'):
            client_package["sigma_dp"] = self.sigma_dp

        return client_package