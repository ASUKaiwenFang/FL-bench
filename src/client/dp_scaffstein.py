from typing import Any
import torch
from src.client.dp_scaffold import DPScaffoldClient
from src.utils.jse_utils import JSEProcessor
from opacus.optimizers.optimizer import _generate_noise

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
        # Initialize with parent class logic first
        super().__init__(**commons)

        # Handle configuration compatibility between dp_scaffold and dp_scaffstein
        if hasattr(self.args, 'dp_scaffstein'):
            # Use dp_scaffstein config if available
            config = self.args.dp_scaffstein
            self.clip_norm = config.clip_norm
            self.sigma = config.sigma
            variant_config = getattr(config, 'algorithm_variant', 'step_noise_final_jse')
        else:
            # Fallback to dp_scaffold config (map to nearest JSE equivalent)
            config = self.args.dp_scaffold
            variant_config = getattr(config, 'algorithm_variant', 'step_noise_final_jse')

        # DP-ScaffStein specific algorithm variants (JSE variants only)
        ALGORITHM_VARIANTS = {
            "last_noise_server_jse": 1,
            "step_noise_step_jse": 2,
            "step_noise_final_jse": 3
        }

        if isinstance(variant_config, str):
            self.algorithm_variant = ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config

    def fit(self):
        """Train the model with DP-ScaffStein algorithm.

        Routes to the appropriate JSE algorithm variant implementation:
        - Variant 1: last_noise_server_jse - DP noise at last step, JSE at server
        - Variant 2: step_noise_step_jse - DP noise and JSE at each step
        - Variant 3: step_noise_final_jse - DP noise at each step, JSE at final step
        """
        if self.algorithm_variant == 1:  # last_noise_server_jse
            self._fit_variant_1_last_noise_server_jse()
        elif self.algorithm_variant == 2:  # step_noise_step_jse
            self._fit_variant_2_step_noise_step_jse()
        elif self.algorithm_variant == 3:  # step_noise_final_jse
            self._fit_variant_3_step_noise_final_jse()
        else:
            raise ValueError(f"Unknown algorithm variant: {self.algorithm_variant}")

    def _fit_variant_1__last_noise_server_jse(self):
        """Algorithm Variant 3: Last noise with server-side JSE.

        Reuses parent's _last_noise_training but removes client-side JSE
        since JSE will be applied at the server.
        """
        # Delegate to parent class implementation
        # Server will apply JSE to aggregated parameter differences
        self._last_noise_training()

    def _fit_variant_2_step_noise_step_jse(self):
        """Algorithm Variant 4: Step-wise DP training with per-step JSE.

        Extends parent's _step_noise_training to add JSE at each step.
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

            # Apply DP clipping, noise, and JSE with SCAFFOLD control variate correction
            self._clip_and_add_noise_opacus()
            JSEProcessor.apply_global_jse_to_gradients(
                list(self.model.parameters()), self.sigma_dp**2
            )
            for name, param in self.model.named_parameters():
                clean_name = self._get_clean_param_name(name)
                control_key = name if name in self.c_global else clean_name

                if control_key in self.c_global and control_key in self.c_local:
                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]
                    param.grad += (c_global - c_local).to(self.device)
                    
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Use parent's post-processing for SCAFFOLD control variates
        self._step_noise_post_processing_with_integrated_control_variates()

    def _fit_variant_3_step_noise_final_jse(self):
        """Algorithm Variant 5: Step-wise DP with final JSE.

        Uses parent's step_noise training, then applies JSE to final parameter differences.
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

            # Use parent's DP processing without JSE
            self._clip_and_add_noise_with_scaffold()
                    
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Apply final JSE to parameter differences before SCAFFOLD post-processing
        self._step_noise_final_jse_post_processing_with_integrated_control_variates()


    @torch.no_grad()
    def _step_noise_final_jse_post_processing_with_integrated_control_variates(self):
        """Post-processing for step_noise_final_jse variant.

        Applies JSE to final parameter differences, then updates SCAFFOLD control variates.
        """
        # Initialize storage
        self.dp_processed_diff = {}
        self.c_delta = self.OrderedDict() if hasattr(self, 'OrderedDict') else {}
        self.c_delta = OrderedDict()
        c_plus = OrderedDict()

        coef = 1 / (self.local_epoch * self.args.optimizer.lr)

        # First, collect parameter differences for JSE processing
        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                clean_name = self._get_clean_param_name(name)
                self.dp_processed_diff[clean_name] = param_diff.clone().cpu()

        # Apply global JSE to parameter differences with k_factor for accumulated noise
        JSEProcessor.apply_global_jse_to_parameter_diff(
            self.dp_processed_diff, self.sigma_dp ** 2, k_factor=self.local_epoch
        )

        # Then, update SCAFFOLD control variates using original parameter differences
        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)

                # SCAFFOLD control variate processing
                clean_name = self._get_clean_param_name(name)
                control_key = name if name in self.c_global else clean_name

                if control_key in self.c_global and control_key in self.c_local:
                    c_global = self.c_global[control_key]
                    c_local = self.c_local[control_key]

                    param_diff_cpu = param_diff.cpu()
                    c_plus[control_key] = c_local - c_global - coef * param_diff_cpu
                    self.c_delta[control_key] = c_plus[control_key] - c_local

        # Update local control variates
        self.c_local = c_plus

    def package(self):
        """Package client data including DP parameters, SCAFFOLD control variates, and JSE information."""
        client_package = super().package()

        # Add sigma_dp for server-side JSE processing (variant 1)
        if hasattr(self, 'sigma_dp'):
            client_package["sigma_dp"] = self.sigma_dp

        return client_package