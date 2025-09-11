from argparse import ArgumentParser, Namespace
from typing import Dict, Any
from collections import OrderedDict

import torch
from omegaconf import DictConfig

from src.client.dp_fedavg_central import DPFedAvgCentralClient
from src.server.fedavg import FedAvgServer
from src.utils.dp_mechanisms import (
    add_gaussian_noise
)


class DPFedAvgCentralServer(FedAvgServer):
    """Central Differential Privacy FedAvg Server.
    
    This server implements central differential privacy by adding noise to
    the aggregated model parameters before sending them back to clients.
    Clients perform standard training without local noise.
    """
    
    algorithm_name: str = "DP-FedAvg-Central"
    all_model_params_personalized = False
    return_diff = True
    client_cls = DPFedAvgCentralClient
    
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for Central DP-FedAvg."""
        parser = ArgumentParser()
        
        # DP parameters
        parser.add_argument("--sigma", type=float, default=0.1,
                           help="Noise standard deviation")
        
        return parser.parse_args(args_list)
    
    def __init__(self, args: DictConfig):
        super().__init__(args)
        
        # Initialize central DP parameters
        self.sigma = self.args.dp_fedavg_central.sigma
    
    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Aggregate client updates and add central differential privacy noise to aggregated diffs."""
        
        # Extract client weights and normalize
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        
        # Aggregate parameter differences with noise
        for name, global_param in self.public_model_params.items():
            # Stack parameter differences from all clients
            diffs = torch.stack(
                [
                    package["model_params_diff"][name]
                    for package in client_packages.values()
                ],
                dim=-1,
            )
            
            # Compute weighted aggregation of differences
            aggregated_diff = torch.sum(diffs * weights, dim=-1)
            
            # Add Gaussian noise to the aggregated difference
            noisy_aggregated_diff = add_gaussian_noise(
                aggregated_diff,
                sigma=self.sigma,
                device=global_param.device
            )
            
            # Update global parameters by subtracting noisy aggregated difference
            self.public_model_params[name].data -= noisy_aggregated_diff
        
        # Update model state dict
        self.model.load_state_dict(self.public_model_params, strict=False)
    
    
    def package(self, client_id: int):
        """Package parameters for client training."""
        server_package = super().package(client_id)
        
        # Add central DP information
        server_package["central_dp_info"] = {
            "sigma": self.sigma
        }
        
        return server_package
    
    
