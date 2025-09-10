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
    return_diff = False
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
        """Aggregate client updates and add central differential privacy noise."""
        
        # Perform standard FedAvg aggregation first
        super().aggregate_client_updates(client_packages)
        
        # Add noise to aggregated parameters for central DP
        self._add_central_dp_noise()
    
    def _add_central_dp_noise(self):
        """Add differential privacy noise to the global model parameters."""
        for name, param in self.public_model_params.items():
            # Add Gaussian noise
            param.data = add_gaussian_noise(
                param.data,
                sigma=10*self.sigma,
                device=param.device
            )
    
    
    def package(self, client_id: int):
        """Package parameters for client training."""
        server_package = super().package(client_id)
        
        # Add central DP information
        server_package["central_dp_info"] = {
            "sigma": self.sigma
        }
        
        return server_package
    
    
