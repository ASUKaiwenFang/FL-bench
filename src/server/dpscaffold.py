from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict
import logging

import torch

from src.client.dpscaffold import DPSCAFFOLDClient
from src.server.dpfedavg import DPFedAvgServer
from src.utils.dp_manager import DPManager


class DPSCAFFOLDServer(DPFedAvgServer):
    algorithm_name: str = "DP-SCAFFOLD"
    all_model_params_personalized = False
    return_diff = True
    client_cls = DPSCAFFOLDClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        # Inherit DP parameters from parent
        parser.add_argument("--epsilon", type=float, default=1.0,
                          help="Privacy budget parameter (smaller = more private)")
        parser.add_argument("--delta", type=float, default=1e-5,
                          help="Privacy parameter (should be < 1/dataset_size)")
        parser.add_argument("--max_grad_norm", type=float, default=1.0,
                          help="Maximum norm for gradient clipping")
        parser.add_argument("--noise_multiplier", type=float, default=None,
                          help="Noise multiplier (if None, will be auto-calculated)")
        parser.add_argument("--privacy_accountant", type=str, default="rdp", choices=["rdp", "gdp", "prv"],
                          help="Privacy accounting method")
        parser.add_argument("--auto_noise_multiplier", type=bool, default=True,
                          help="Whether to automatically calculate noise multiplier")
        parser.add_argument("--sample_rate", type=float, default=-1.0,
                          help="Poisson sampling rate (if negative, calculated from batch_size/dataset_size)")

        # SCAFFOLD-specific parameter
        parser.add_argument("--global_lr", type=float, default=1.0,
                          help="Server-side learning rate for model updates")

        return parser.parse_args(args_list)

    def __init__(self, args, init_trainer=True, init_model=True):
        # Initialize DP manager first with dpscaffold config
        # Create a modified args object where dpfedavg points to dpscaffold config
        class DPConfigWrapper:
            def __init__(self, original_args):
                self._original_args = original_args
                # Make dpfedavg config point to dpscaffold config for DPManager
                self.dpfedavg = original_args.dpscaffold

            def __getattr__(self, name):
                return getattr(self._original_args, name)

        wrapped_args = DPConfigWrapper(args)
        self.dp_manager = DPManager(wrapped_args)

        # Call parent initialization (FedAvgServer, skip DPFedAvgServer's __init__)
        # This avoids double initialization of DPManager
        from src.server.fedavg import FedAvgServer
        FedAvgServer.__init__(self, args, init_trainer=False, init_model=init_model)

        # Initialize trainer after DP setup
        if init_trainer:
            self.init_trainer()

        # Initialize SCAFFOLD control variates
        # c_global: global control variate (one per parameter)
        self.c_global = [
            torch.zeros_like(param) for param in self.public_model_params.values()
        ]

        # c_local: local control variates (one set per training client)
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]

        logging.info(f"DP-SCAFFOLD initialized with {len(self.c_global)} control variates")

    def init_model(self, model=None, preprocess_func=None, postprocess_func=None):
        """Initialize model with DP preparation."""
        # First, initialize the model normally
        from src.server.fedavg import FedAvgServer
        FedAvgServer.init_model(self, model, preprocess_func, postprocess_func)

        # Then prepare it for DP
        self.model = self.dp_manager.prepare_model_for_dp(self.model)

        # Use DP model parameters directly
        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        from collections import OrderedDict
        self.public_model_params = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

    def package(self, client_id: int):
        """Package parameters for client including DP config and control variates."""
        # Get base package from FedAvgServer
        from src.server.fedavg import FedAvgServer
        server_package = FedAvgServer.package(self, client_id)

        # Add DP configuration
        client_dataset_size = len(self.client_data_indices[client_id])
        batch_size = self.args.common.batch_size
        local_epoch = self.args.common.local_epoch

        server_package["dp_config"] = self.dp_manager.get_dp_config_for_client(
            batch_size=batch_size,
            dataset_size=client_dataset_size,
            local_epoch=local_epoch,
            trainloader_length=1  # Poisson sampling: 1 batch per epoch
        )

        # Add SCAFFOLD control variates
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]

        return server_package

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with SCAFFOLD control variate mechanism."""
        # Extract control variate deltas and model deltas
        c_delta_list = [package["c_delta"] for package in client_packages.values()]
        y_delta_list = [package["y_delta"] for package in client_packages.values()]

        # Calculate aggregation weights (uniform for now)
        weights = torch.ones(len(y_delta_list)) / len(y_delta_list)

        # Update global model parameters using y_delta
        for param, y_delta in zip(
            self.public_model_params.values(), zip(*y_delta_list)
        ):
            param.data += self.args.dpscaffold.global_lr * torch.sum(
                torch.stack(y_delta, dim=-1) * weights, dim=-1
            )

        # Update global control variate
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_list)):
            c_global.data += torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num

        # Update privacy statistics (from DPFedAvgServer)
        for package in client_packages.values():
            if "privacy_stats" in package:
                self.dp_manager.update_privacy_stats(package["privacy_stats"])

        # Check privacy budget
        budget_ok, budget_msg = self.dp_manager.validate_privacy_budget()
        if not budget_ok:
            logging.warning(f"Privacy budget warning: {budget_msg}")

        # Update the DP model with aggregated parameters
        self.model.load_state_dict(self.public_model_params, strict=False)

        logging.debug(f"Aggregated updates from {len(client_packages)} clients with DP-SCAFFOLD")
