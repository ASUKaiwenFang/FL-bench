from typing import Dict, List, Any
from collections import OrderedDict
from argparse import ArgumentParser, Namespace
import torch
from src.server.dp_scaffold import DPScaffoldServer
from src.client.dp_scaffstein import DPScaffSteinClient
from src.utils.jse_utils import JSEProcessor


class DPScaffSteinServer(DPScaffoldServer):
    """DP-ScaffStein Server combining Differential Privacy, SCAFFOLD control variates, and JSE.

    This server implements DP-ScaffStein algorithms with three variants:
    1. last_noise_server_jse: DP noise at last step, JSE at server
    2. step_noise_step_jse: DP noise and JSE at each step (JSE handled at client)
    3. step_noise_final_jse: DP noise at each step, JSE at final step (JSE handled at client)

    Only variant 1 requires server-side JSE processing.
    """
    algorithm_name: str = "DP-SCAFFSTEIN"
    client_cls = DPScaffSteinClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for DP-ScaffStein."""
        parser = ArgumentParser()

        # DP parameters (from DP-SCAFFOLD)
        parser.add_argument("--global_lr", type=float, default=1.0,
                           help="Global learning rate for parameter aggregation")
        parser.add_argument("--clip_norm", type=float, default=1.0,
                           help="Gradient clipping norm")
        parser.add_argument("--sigma", type=float, default=1.0,
                           help="Noise standard deviation")
        parser.add_argument("--algorithm_variant", type=str,
                           choices=["last_noise_server_jse", "step_noise_step_jse", "step_noise_final_jse"],
                           default="step_noise_final_jse",
                           help="Algorithm variant: JSE variants for DP-ScaffStein")

        return parser.parse_args(args_list)

    def __init__(self, **commons):
        super().__init__(**commons)

        # Get algorithm variant from config
        variant_config = getattr(self.args.dp_scaffstein, 'algorithm_variant', 'step_noise_final_jse')
        if isinstance(variant_config, str):
            variant_map = {
                'last_noise_server_jse': 1,
                'step_noise_step_jse': 2,
                'step_noise_final_jse': 3
            }
            self.algorithm_variant = variant_map[variant_config]
        else:
            self.algorithm_variant = variant_config

        # Store sigma for server-side JSE processing
        self.sigma = getattr(self.args.dp_scaffstein, 'sigma', 1.0)

        # Initialize shrinkage history tracking
        self.shrinkage_history = {}

        # Setup tensorboard custom layout for shrinkage factors
        if self.args.common.monitor == "tensorboard":
            layout = {
                "JSE Shrinkage Factors": {
                    "Statistics": ["Multiline", [
                        "ShrinkageFactor/Client-Min",
                        "ShrinkageFactor/Client-Max",
                        "ShrinkageFactor/Client-Average"
                    ]],
                    "Server": ["Multiline", ["ShrinkageFactor/Server"]]
                }
            }
            self.tensorboard.add_custom_scalars(layout)

    def aggregate_client_updates(self, client_packages: List[Dict[str, Any]]) -> None:
        """Aggregate client updates with DP-ScaffStein processing.

        For variant 1 (last_noise_server_jse): Apply server-side JSE to aggregated parameter differences
        For variants 2 and 3: Use standard SCAFFOLD aggregation (JSE handled at client)
        """
        # Collect client shrinkage factors for variant 2 and 3 before aggregation
        if self.algorithm_variant in [2, 3]:  # step_noise_step_jse or step_noise_final_jse
            self._record_client_shrinkage_factors(client_packages)

        # First, perform standard SCAFFOLD aggregation
        super().aggregate_client_updates(client_packages)

        # Apply server-side JSE for variant 1 only
        if self.algorithm_variant == 1:  # last_noise_server_jse
            self._apply_server_jse(client_packages)

        # Log shrinkage factors to tensorboard
        self._log_shrinkage_to_tensorboard(client_packages)

    def _apply_server_jse(self, client_packages) -> None:
        """Apply server-side JSE to aggregated parameter differences.

        This method is only used for algorithm variant 1 (last_noise_server_jse).
        It applies global JSE shrinkage to the aggregated parameter differences.

        Args:
            client_packages: OrderedDict of client data packages
        """
        if not client_packages:
            return

        # Extract sigma_dp from first client package (should be consistent across clients)
        first_client_id = list(client_packages.keys())[0]
        first_package = client_packages[first_client_id]

        # Get sigma_dp from client package, fall back to server configuration if not present
        sigma_dp = first_package.get('sigma_dp', self.sigma)
        sigma_dp_squared = sigma_dp ** 2

        if sigma_dp_squared <= 0:
            return  # No JSE processing if no DP noise

        # Apply global JSE to the aggregated public model parameters
        # Note: For server-side JSE, we apply JSE to the aggregated parameters directly
        shrinkage_factor = JSEProcessor.apply_global_jse_to_parameter_diff(
            self.public_model_params, sigma_dp_squared, k_factor=1  # k_factor=1 for server-side JSE
        )

        # Record shrinkage factor for variant 1
        self.shrinkage_history[self.current_epoch] = {"server": shrinkage_factor}

    def _record_client_shrinkage_factors(self, client_packages: Dict[int, Dict[str, Any]]) -> None:
        """Record shrinkage factors from client packages.

        Extracts shrinkage factors from each client and stores them for the current epoch.

        Args:
            client_packages: Dictionary mapping client_id to client package data
        """
        current_epoch_data = {}

        for client_id, package in client_packages.items():
            if "shrinkage_factor" in package:
                current_epoch_data[f"client_{client_id}"] = package["shrinkage_factor"]

        if current_epoch_data:
            self.shrinkage_history[self.current_epoch] = current_epoch_data

    def _log_shrinkage_to_tensorboard(self, client_packages: Dict[int, Dict[str, Any]]) -> None:
        """Log JSE shrinkage factors to tensorboard.

        For Algorithm 1: Logs server shrinkage factor
        For Algorithm 2/3: Logs client statistics (min, max, average)

        Args:
            client_packages: Dictionary mapping client_id to client package data
        """
        if self.args.common.monitor != "tensorboard":
            return

        if self.algorithm_variant == 1:  # Algorithm 1: last_noise_server_jse
            # Log server shrinkage factor
            if self.current_epoch in self.shrinkage_history:
                shrinkage = self.shrinkage_history[self.current_epoch].get("server", None)
                if shrinkage is not None:
                    self.tensorboard.add_scalar(
                        "ShrinkageFactor/Server",
                        shrinkage,
                        self.current_epoch,
                        new_style=True
                    )

        elif self.algorithm_variant in [2, 3]:  # Algorithm 2/3
            # Extract shrinkage factors from client packages
            shrinkage_factors = [
                pkg["shrinkage_factor"]
                for pkg in client_packages.values()
                if "shrinkage_factor" in pkg
            ]

            if shrinkage_factors:
                import numpy as np

                # Log statistics
                self.tensorboard.add_scalar(
                    "ShrinkageFactor/Client-Min",
                    float(np.min(shrinkage_factors)),
                    self.current_epoch,
                    new_style=True
                )
                self.tensorboard.add_scalar(
                    "ShrinkageFactor/Client-Max",
                    float(np.max(shrinkage_factors)),
                    self.current_epoch,
                    new_style=True
                )
                self.tensorboard.add_scalar(
                    "ShrinkageFactor/Client-Average",
                    float(np.mean(shrinkage_factors)),
                    self.current_epoch,
                    new_style=True
                )

    def save_shrinkage_factors(self) -> None:
        """Save shrinkage factors to JSON file in output directory."""
        import json
        from pathlib import Path

        if not self.shrinkage_history:
            return  # Nothing to save

        # Determine algorithm variant name
        variant_names = {
            1: "last_noise_server_jse",
            2: "step_noise_step_jse",
            3: "step_noise_final_jse"
        }
        variant_name = variant_names.get(self.algorithm_variant, "unknown")

        # Prepare data structure
        data = {
            "algorithm_variant": variant_name,
            "data": {str(k): v for k, v in self.shrinkage_history.items()}
        }

        # Save to JSON file
        output_path = Path(self.output_dir) / "shrinkage_factors.json"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Shrinkage factors saved to: {output_path}")

    def run_experiment(self):
        """Override parent method to save shrinkage factors after training."""
        # Call parent run_experiment
        super().run_experiment()

        # Save shrinkage factors after training completes
        self.save_shrinkage_factors()

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Get hyperparameters for DP-ScaffStein method."""
        from argparse import ArgumentParser, Namespace
        parser = ArgumentParser()
        parser.add_argument("--clip_norm", type=float, default=1.0)
        parser.add_argument("--sigma", type=float, default=1.0)
        parser.add_argument("--algorithm_variant", type=str, default="step_noise_final_jse")
        parser.add_argument("--global_lr", type=float, default=1.0)
        return parser.parse_args(args_list)
    
    
# Create an alias for main.py's naming convention compatibility
# main.py expects class name to match "method_name + server" pattern
Dp_scaffsteinServer = DPScaffSteinServer