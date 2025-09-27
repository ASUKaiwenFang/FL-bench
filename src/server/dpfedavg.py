from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict
import logging

from src.client.dpfedavg import DPFedAvgClient
from src.server.fedavg import FedAvgServer
from src.utils.dp_manager import DPManager


class DPFedAvgServer(FedAvgServer):
    algorithm_name: str = "DP-FedAvg"
    all_model_params_personalized = False
    return_diff = False
    client_cls = DPFedAvgClient

    def __init__(self, args, init_trainer=True, init_model=True):
        # Initialize DP manager first
        self.dp_manager = DPManager(args)

        # Call parent initialization
        super().__init__(args, init_trainer=False, init_model=init_model)

        # Initialize trainer after DP setup
        if init_trainer:
            self.init_trainer()

    def init_model(self, model=None, preprocess_func=None, postprocess_func=None):
        """Initialize model with DP preparation."""
        # First, initialize the model normally
        super().init_model(model, preprocess_func, postprocess_func)

        # Then prepare it for DP
        self.model = self.dp_manager.prepare_model_for_dp(self.model)

        # Use DP model parameters directly
        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

    def package(self, client_id: int):
        """Package parameters for client with DP configuration."""
        base_package = super().package(client_id)

        # Get client dataset size and training parameters for DP calculation
        client_dataset_size = len(self.client_data_indices[client_id])
        batch_size = self.args.common.batch_size
        local_epoch = self.args.common.local_epoch

        base_package["dp_config"] = self.dp_manager.get_dp_config_for_client(
            batch_size=batch_size,
            dataset_size=client_dataset_size,
            local_epoch=local_epoch,
            trainloader_length=1  # Poisson sampling: 1 batch per epoch
        )
        return base_package

    def get_client_model_params(self, client_id: int):
        """Get client model parameters."""
        # Get the model parameters (already in DP format)
        regular_params = self.public_model_params.copy()
        personal_params = self.clients_personal_model_params[client_id]

        return dict(
            regular_model_params=regular_params,
            personal_model_params=personal_params
        )

    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Aggregate client updates with DP parameter name handling."""
        # Update privacy statistics
        for package in client_packages.values():
            if "privacy_stats" in package:
                self.dp_manager.update_privacy_stats(package["privacy_stats"])

        # Check privacy budget
        budget_ok, budget_msg = self.dp_manager.validate_privacy_budget()
        if not budget_ok:
            logging.warning(f"Privacy budget warning: {budget_msg}")


        # Perform standard FedAvg aggregation directly
        super().aggregate_client_updates(client_packages)

        # Update the DP model with aggregated parameters (already in DP format)
        self.model.load_state_dict(self.public_model_params, strict=False)

    def display_metrics(self):
        """Display metrics including privacy information."""
        super().display_metrics()

        # Display privacy statistics
        if self.verbose:
            privacy_report = self.dp_manager.get_privacy_report()
            self.logger.log(
                f"Privacy Status: ε={privacy_report['epsilon_spent']:.4f}/{privacy_report['target_epsilon']:.1f}, "
                f"remaining={privacy_report['epsilon_remaining']:.4f}, "
                f"steps={privacy_report['total_steps']}"
            )

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
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
        return parser.parse_args(args_list)