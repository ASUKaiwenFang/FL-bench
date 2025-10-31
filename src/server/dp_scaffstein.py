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

        # Initialize JSE sign statistics history
        self.jse_sign_history = {}

        # Setup tensorboard custom layout for shrinkage factors
        if self.args.common.monitor == "tensorboard":
            layout = {
                "JSE Shrinkage Factors": {
                    "Statistics": ["Multiline", [
                        "ShrinkageFactor/Client-Min",
                        "ShrinkageFactor/Client-Max",
                        "ShrinkageFactor/Client-Average",
                        "ShrinkageFactor/JSE_PositiveRatio"
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

        # For variant 1, we need custom aggregation with post-aggregation JSE
        if self.algorithm_variant == 1:  # last_noise_server_jse
            self._aggregate_with_post_jse_variant_1(client_packages)
        else:
            # For other variants, use standard aggregation
            super().aggregate_client_updates(client_packages)

        # Log shrinkage factors to tensorboard
        self._log_shrinkage_to_tensorboard(client_packages)

        # Record and log JSE sign statistics for variant 2
        if self.algorithm_variant == 2:  # step_noise_step_jse
            self._record_jse_sign_statistics(client_packages)

    def _aggregate_with_post_jse_variant_1(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates and then apply server-side JSE for variant 1.

        For the last_noise_server_jse variant, this method:
        1. Aggregates client parameter differences using weighted averaging
        2. Applies global JSE to the aggregated result
        3. Updates the global model parameters

        Args:
            client_packages: OrderedDict of client packages containing noisy parameter updates
        """

        # Step 1: Extract weights and compute normalized weights
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)

        # Step 2: Aggregate parameter differences
        aggregated_diff = {}
        for name, global_param in self.public_model_params.items():
            diffs = torch.stack(
                [package["model_params_diff"][name] for package in client_packages.values()],
                dim=-1,
            )
            aggregated_diff[name] = torch.sum(diffs * weights, dim=-1)

        # Step 3: Apply global JSE to the aggregated differences
        # Extract noise variance from first client package (should be same for all)
        noise_variance = list(client_packages.values())[0]["sigma_dp"]**2
        k_factor = 1/int(self.client_num * self.args.common.join_ratio)
        global_lr = self.args.dp_scaffstein.global_lr

        # Apply global JSE to the aggregated parameter differences
        # shrinkage_factor = JSEProcessor.apply_global_jse_to_parameter_diff(
        #     aggregated_diff, noise_variance, k_factor
        # )
        shrinkage_factor = 1.0
        JSEProcessor.apply_layerwise_jse_to_parameter_diff(
            aggregated_diff, noise_variance, k_factor
        )

        # Record shrinkage factor for variant 1
        self.shrinkage_history[self.current_epoch] = {"server": shrinkage_factor}

        # Step 4: Update global model parameters with JSE-processed differences
        for name, global_param in self.public_model_params.items():
            self.public_model_params[name].data += global_lr * aggregated_diff[name]

        # Step 5: Load updated parameters into model
        self.model.load_state_dict(self.public_model_params, strict=False)

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

    def _record_jse_sign_statistics(self, client_packages: Dict[int, Dict[str, Any]]) -> None:
        """Record JSE sign statistics from client packages for variant 2.

        Extracts JSE positive ratio from each client and computes simple average.

        Args:
            client_packages: Dictionary mapping client_id to client package data
        """
        client_positive_ratios = []

        for client_id, package in client_packages.items():
            if "jse_positive_ratio" in package:
                client_positive_ratios.append(package["jse_positive_ratio"])

        if client_positive_ratios:
            # Simple average of client positive ratios
            avg_positive_ratio = sum(client_positive_ratios) / len(client_positive_ratios)

            # Store in history
            self.jse_sign_history[self.current_epoch] = {
                "avg_positive_ratio": avg_positive_ratio,
                "client_ratios": client_positive_ratios
            }

            # Log to tensorboard
            if self.args.common.monitor == "tensorboard":
                self.tensorboard.add_scalar(
                    "ShrinkageFactor/JSE_PositiveRatio",
                    avg_positive_ratio,
                    self.current_epoch,
                    new_style=True
                )

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
        """Save shrinkage factors and JSE sign statistics to JSON file in output directory."""
        import json
        from pathlib import Path

        if not self.shrinkage_history and not self.jse_sign_history:
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

        # Add JSE sign statistics if available
        if self.jse_sign_history:
            data["jse_sign_statistics"] = {
                str(k): v for k, v in self.jse_sign_history.items()
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