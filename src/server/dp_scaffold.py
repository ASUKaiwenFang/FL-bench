from argparse import ArgumentParser, Namespace
from typing import Dict, Any
from collections import OrderedDict
from copy import deepcopy

from omegaconf import DictConfig
import torch

from src.client.dp_scaffold import DPScaffoldClient
from src.server.dp_fedavg_local import DPFedAvgLocalServer


class DPScaffoldServer(DPFedAvgLocalServer):
    """DP-SCAFFOLD Server combining Differential Privacy with SCAFFOLD control variates.

    This server coordinates federated learning with local differential privacy and
    SCAFFOLD control variates. Clients add noise to their gradients locally and use
    control variates to reduce client drift. The server performs aggregation on the
    noisy updates and maintains global control variates.

    Only supports return_diff=True mode, similar to both parent algorithms.
    """

    algorithm_name: str = "DP-SCAFFOLD"
    all_model_params_personalized = False
    return_diff = True
    client_cls = DPScaffoldClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for DP-SCAFFOLD."""
        parser = ArgumentParser()

        # DP parameters (from DP-FedAvg Local)
        parser.add_argument("--global_lr", type=float, default=1.0,
                           help="Global learning rate for parameter aggregation")
        parser.add_argument("--clip_norm", type=float, default=1.0,
                           help="Gradient clipping norm")
        parser.add_argument("--sigma", type=float, default=0.1,
                           help="Noise standard deviation")
        parser.add_argument("--algorithm_variant", type=str,
                           choices=["last_noise", "step_noise"], default="step_noise",
                           help="Algorithm variant: last_noise (parameter-level) or step_noise (gradient-level)")

        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)

        # Initialize SCAFFOLD control variates using OrderedDict format
        self.c_global = OrderedDict([
            (name, torch.zeros_like(param))
            for name, param in self.public_model_params.items()
        ])
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]

    def _get_global_lr(self):
        """Get global_lr parameter with backward compatibility.

        Checks for method-specific config first, then falls back to dp_scaffold config.
        """
        # Try to get method-specific config first (e.g., dp_scaffold.global_lr)
        method_name = self.args.method
        if hasattr(self.args, method_name):
            method_config = getattr(self.args, method_name)
            if hasattr(method_config, 'global_lr'):
                return method_config.global_lr

        # Fall back to dp_fedavg_local config for compatibility
        if hasattr(self.args, 'dp_fedavg_local') and hasattr(self.args.dp_fedavg_local, 'global_lr'):
            return self.args.dp_fedavg_local.global_lr

        # Default fallback
        return 1.0

    def package(self, client_id: int):
        """Package server data including DP parameters and SCAFFOLD control variates."""
        server_package = super().package(client_id)

        # Add SCAFFOLD control variates
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]

        return server_package

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with DP protection and SCAFFOLD control variates."""

        # Aggregate parameter updates using DP-FedAvg Local method with weights
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)

        # Aggregate noisy parameter differences
        for name, global_param in self.public_model_params.items():
            diffs = torch.stack(
                [
                    package["model_params_diff"][name]
                    for package in client_packages.values()
                ],
                dim=-1,
            )
            aggregated = torch.sum(diffs * weights, dim=-1)
            self.public_model_params[name].data += self._get_global_lr() * aggregated

        # Update SCAFFOLD global control variates
        c_delta_list = [package["c_delta"] for package in client_packages.values()]

        # Update global control variates using SCAFFOLD aggregation (unweighted)
        for name in self.c_global.keys():
            # Collect c_delta values for this parameter
            c_delta_values = [c_delta[name] for c_delta in c_delta_list if name in c_delta]

            # Skip if no values found (empty list would cause stack error)
            if len(c_delta_values) == 0:
                continue

            c_deltas = torch.stack(c_delta_values, dim=-1)
            if c_deltas.numel() > 0:
                self.c_global[name].data += c_deltas.sum(dim=-1) / self.client_num

        # Load updated parameters into model
        self.model.load_state_dict(self.public_model_params, strict=False)


# Create an alias for main.py's naming convention compatibility
# main.py expects class name to match "method_name + server" pattern
Dp_scaffoldServer = DPScaffoldServer