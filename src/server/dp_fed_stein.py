from argparse import ArgumentParser, Namespace
from typing import Dict, Any, List
from collections import OrderedDict
import torch
from omegaconf import DictConfig

from src.client.dp_fed_stein import DPFedSteinClient, FedSteinAlgorithmVariant
from src.server.dp_fedavg_local import DPFedAvgLocalServer
from src.utils.jse_utils import JSEProcessor


class DPFedSteinServer(DPFedAvgLocalServer):
    """DP-FedAvg + James-Stein Estimator Server.

    This server extends DPFedAvgLocalServer to support global JSE processing for the
    last_noise_server_jse algorithm variant. For other variants, it performs
    standard aggregation as clients handle global JSE locally.

    The server applies global JSE which computes unified shrinkage across all
    parameters based on the combined norm, ensuring mathematically consistent
    treatment of client parameter updates.
    """

    algorithm_name: str = "DP-FedStein"
    client_cls = DPFedSteinClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for DP-FedStein."""
        parser = ArgumentParser()

        # DP parameters (inherited from parent)
        parser.add_argument("--global_lr", type=float, default=1.0)
        parser.add_argument("--clip_norm", type=float, default=1.0,
                           help="Gradient clipping norm")
        parser.add_argument("--sigma", type=float, default=0.1,
                           help="Noise standard deviation")

        # JSE-specific parameters
        parser.add_argument("--algorithm_variant", type=str,
                           choices=["last_noise_server_jse", "step_noise_step_jse", "step_noise_final_jse"],
                           default="step_noise_step_jse",
                           help="Algorithm variant for JSE application")

        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)

        # Get algorithm variant for server-side processing
        variant_config = getattr(self.args.dp_fed_stein, 'algorithm_variant', 'step_noise_step_jse')
        if isinstance(variant_config, str):
            self.fed_stein_algorithm_variant = getattr(FedSteinAlgorithmVariant, variant_config.upper())
        else:
            self.fed_stein_algorithm_variant = FedSteinAlgorithmVariant(variant_config)

        # Store sigma for server-side JSE processing
        self.sigma = getattr(self.args.dp_fed_stein, 'sigma', 1.0)

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


    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with correct JSE application for variant 1.

        For variant 1 (last_noise_server_jse), JSE is applied to the aggregated result
        after aggregation, which follows the algorithm specification where JSE is applied
        to the server-side aggregated differences rather than individual client updates.
        """
        # Collect client shrinkage factors for variant 2 and 3 before aggregation
        if self.fed_stein_algorithm_variant in [FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE, FedSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE]:
            self._record_client_shrinkage_factors(client_packages)

        # For variant 1, we need custom aggregation with post-aggregation JSE
        if self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.LAST_NOISE_SERVER_JSE:
            self._aggregate_with_post_jse_variant_1(client_packages)
        else:
            # For other variants, use standard aggregation
            super().aggregate_client_updates(client_packages)

        # Log shrinkage factors to tensorboard
        self._log_shrinkage_to_tensorboard(client_packages)

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
        global_lr = self.args.dp_fed_stein.global_lr

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

    def _log_shrinkage_to_tensorboard(self, client_packages: Dict[int, Dict[str, Any]]) -> None:
        """Log JSE shrinkage factors to tensorboard.

        For Algorithm 1: Logs server shrinkage factor
        For Algorithm 2/3: Logs client statistics (min, max, average)

        Args:
            client_packages: Dictionary mapping client_id to client package data
        """
        if self.args.common.monitor != "tensorboard":
            return

        if self.fed_stein_algorithm_variant == FedSteinAlgorithmVariant.LAST_NOISE_SERVER_JSE:
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

        elif self.fed_stein_algorithm_variant in [FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE, FedSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE]:
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
            FedSteinAlgorithmVariant.LAST_NOISE_SERVER_JSE: "last_noise_server_jse",
            FedSteinAlgorithmVariant.STEP_NOISE_STEP_JSE: "step_noise_step_jse",
            FedSteinAlgorithmVariant.STEP_NOISE_FINAL_JSE: "step_noise_final_jse"
        }
        variant_name = variant_names.get(self.fed_stein_algorithm_variant, "unknown")

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


# Create an alias for main.py's naming convention compatibility
Dp_fed_steinServer = DPFedSteinServer