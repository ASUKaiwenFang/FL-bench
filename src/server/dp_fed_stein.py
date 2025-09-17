from argparse import ArgumentParser, Namespace
from typing import Dict, Any, List
from collections import OrderedDict
import torch
from omegaconf import DictConfig

from src.client.dp_fed_stein import DPFedSteinClient
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
            self.algorithm_variant = DPFedSteinClient.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config


    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with correct JSE application for variant 1.

        For variant 1 (last_noise_server_jse), JSE is applied to the aggregated result
        after aggregation, which follows the algorithm specification where JSE is applied
        to the server-side aggregated differences rather than individual client updates.
        """

        # For variant 1, we need custom aggregation with post-aggregation JSE
        if self.algorithm_variant == 1:
            self._aggregate_with_post_jse_variant_1(client_packages)
        else:
            # For other variants, use standard aggregation
            super().aggregate_client_updates(client_packages)

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
        JSEProcessor.apply_global_jse_to_parameter_diff(
            aggregated_diff, noise_variance, k_factor
        )

        # Step 4: Update global model parameters with JSE-processed differences
        for name, global_param in self.public_model_params.items():
            self.public_model_params[name].data += global_lr * aggregated_diff[name]

        # Step 5: Load updated parameters into model
        self.model.load_state_dict(self.public_model_params, strict=False)


# Create an alias for main.py's naming convention compatibility
Dp_fed_steinServer = DPFedSteinServer