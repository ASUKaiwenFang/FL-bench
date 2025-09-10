from argparse import ArgumentParser, Namespace
from typing import Dict, Any
from collections import OrderedDict

import torch
from omegaconf import DictConfig

from src.client.dp_fedavg_central import DPFedAvgCentralClient
from src.server.fedavg import FedAvgServer
from src.utils.dp_mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    privacy_accountant
)


class DPFedAvgCentralServer(FedAvgServer):
    """Central Differential Privacy FedAvg Server.
    
    This server implements central differential privacy by adding noise to
    the aggregated model parameters before sending them back to clients.
    Clients perform standard training without local noise.
    """
    
    algorithm_name: str = "DP-FedAvg-Central"
    all_model_params_personalized = False
    return_diff = False
    client_cls = DPFedAvgCentralClient
    
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for Central DP-FedAvg."""
        parser = ArgumentParser()
        
        # Privacy parameters
        parser.add_argument("--epsilon", type=float, default=8.0,
                           help="Privacy budget parameter")
        parser.add_argument("--delta", type=float, default=1e-5,
                           help="Privacy parameter for (ε,δ)-DP")
        
        # Server-side noise parameters
        parser.add_argument("--server_noise_multiplier", type=float, default=0.8,
                           help="Server-side noise multiplier")
        parser.add_argument("--noise_type", type=str, default="gaussian",
                           help="Noise type [gaussian, laplace]")
        
        # Sensitivity and clipping
        parser.add_argument("--sensitivity", type=float, default=1.0,
                           help="Sensitivity parameter for noise calibration")
        parser.add_argument("--clipping_mode", type=str, default="automatic",
                           help="Clipping mode [manual, automatic, adaptive]")
        
        # Aggregation parameters
        parser.add_argument("--aggregation_weights", type=str, default="uniform",
                           help="Aggregation weights [uniform, adaptive]")
        
        return parser.parse_args(args_list)
    
    def __init__(self, args: DictConfig):
        super().__init__(args)
        
        # Initialize central DP parameters
        self.epsilon = self.args.dp_fedavg_central.epsilon
        self.delta = self.args.dp_fedavg_central.delta
        self.server_noise_multiplier = self.args.dp_fedavg_central.server_noise_multiplier
        self.noise_type = self.args.dp_fedavg_central.noise_type
        self.sensitivity = self.args.dp_fedavg_central.sensitivity
        
        # Privacy accounting for central DP
        self.global_privacy_consumed = {"epsilon": 0.0, "delta": 0.0}
        self.aggregation_rounds = 0
        self.total_client_contributions = 0
        
        # Track client participation for sensitivity computation
        self.client_participation_history = []
    
    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Aggregate client updates and add central differential privacy noise."""
        
        # Perform standard FedAvg aggregation first
        super().aggregate_client_updates(client_packages)
        
        # Add noise to aggregated parameters for central DP
        self._add_central_dp_noise()
        
        # Update privacy accounting
        self._update_central_privacy_accounting(client_packages)
        
        # Track client participation
        self.client_participation_history.append(len(client_packages))
        self.aggregation_rounds += 1
    
    def _add_central_dp_noise(self):
        """Add differential privacy noise to the global model parameters."""
        for name, param in self.public_model_params.items():
            if self.noise_type == "gaussian":
                # Add Gaussian noise
                param.data = add_gaussian_noise(
                    param.data,
                    noise_multiplier=self.server_noise_multiplier,
                    sensitivity=self.sensitivity,
                    device=param.device
                )
            elif self.noise_type == "laplace":
                # Add Laplace noise
                param.data = add_laplace_noise(
                    param.data,
                    epsilon=self.epsilon,
                    sensitivity=self.sensitivity,
                    device=param.device
                )
            else:
                raise ValueError(f"Unsupported noise type: {self.noise_type}")
    
    def _update_central_privacy_accounting(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Update privacy consumption for central DP."""
        
        # Count total samples contributed in this round
        round_contributions = sum(
            pkg.get("dp_tracking", {}).get("client_contribution", 0)
            for pkg in client_packages.values()
        )
        self.total_client_contributions += round_contributions
        
        # Simplified privacy accounting for central DP
        # In practice, this should use more sophisticated accounting methods
        if self.noise_type == "gaussian":
            # For Gaussian mechanism, privacy cost depends on noise multiplier
            round_epsilon, round_delta = privacy_accountant(
                noise_multiplier=self.server_noise_multiplier,
                lot_size=round_contributions,
                steps=1,  # One aggregation step
                dataset_size=self.total_client_contributions
            )
        elif self.noise_type == "laplace":
            # For Laplace mechanism, delta = 0 and epsilon accumulates linearly
            round_epsilon = self.sensitivity / self.server_noise_multiplier
            round_delta = 0.0
        else:
            round_epsilon, round_delta = 0.0, 0.0
        
        # Update cumulative privacy consumption
        self.global_privacy_consumed["epsilon"] += round_epsilon
        self.global_privacy_consumed["delta"] += round_delta
        
        # Log privacy consumption
        if self.aggregation_rounds % 10 == 0:  # Log every 10 rounds
            self.logger.log(
                f"Central DP Privacy consumed: "
                f"ε={self.global_privacy_consumed['epsilon']:.3f}, "
                f"δ={self.global_privacy_consumed['delta']:.6f} "
                f"(Round {self.aggregation_rounds})"
            )
    
    def package(self, client_id: int):
        """Package parameters with privacy budget information."""
        server_package = super().package(client_id)
        
        # Add central privacy information
        server_package["central_dp_info"] = {
            "global_privacy_consumed": self.global_privacy_consumed.copy(),
            "aggregation_rounds": self.aggregation_rounds,
            "privacy_budget_remaining": {
                "epsilon": max(0, self.epsilon - self.global_privacy_consumed["epsilon"]),
                "delta": max(0, self.delta - self.global_privacy_consumed["delta"])
            },
            "server_noise_multiplier": self.server_noise_multiplier,
            "noise_type": self.noise_type
        }
        
        return server_package
    
    def is_privacy_budget_exhausted(self) -> bool:
        """Check if the global privacy budget is exhausted."""
        return (self.global_privacy_consumed["epsilon"] >= self.epsilon or
                self.global_privacy_consumed["delta"] >= self.delta)
    
    def get_central_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of central differential privacy consumption."""
        return {
            "algorithm": self.algorithm_name,
            "total_rounds": self.aggregation_rounds,
            "privacy_consumed": self.global_privacy_consumed.copy(),
            "privacy_budget": {
                "epsilon": self.epsilon,
                "delta": self.delta
            },
            "privacy_remaining": {
                "epsilon": max(0, self.epsilon - self.global_privacy_consumed["epsilon"]),
                "delta": max(0, self.delta - self.global_privacy_consumed["delta"])
            },
            "budget_exhausted": self.is_privacy_budget_exhausted(),
            "noise_parameters": {
                "noise_type": self.noise_type,
                "server_noise_multiplier": self.server_noise_multiplier,
                "sensitivity": self.sensitivity
            },
            "client_participation": {
                "average_clients_per_round": (
                    sum(self.client_participation_history) / len(self.client_participation_history)
                    if self.client_participation_history else 0
                ),
                "total_client_contributions": self.total_client_contributions
            }
        }
