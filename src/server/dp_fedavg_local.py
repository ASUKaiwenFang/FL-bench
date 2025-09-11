from argparse import ArgumentParser, Namespace
from typing import Dict, Any
from collections import OrderedDict

from omegaconf import DictConfig

from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.server.fedavg import FedAvgServer


class DPFedAvgLocalServer(FedAvgServer):
    """Local Differential Privacy FedAvg Server.
    
    This server coordinates federated learning with local differential privacy.
    Clients add noise to their gradients locally before sending updates to the server.
    The server performs standard FedAvg aggregation on the noisy updates.
    """
    
    algorithm_name: str = "DP-FedAvg-Local"
    all_model_params_personalized = False
    return_diff = True
    client_cls = DPFedAvgLocalClient
    
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for Local DP-FedAvg."""
        parser = ArgumentParser()
        
        # DP parameters
        parser.add_argument("--clip_norm", type=float, default=1.0,
                           help="Gradient clipping norm")
        parser.add_argument("--sigma", type=float, default=0.1,
                           help="Noise standard deviation")
        parser.add_argument("--noise_mode", type=str, 
                           choices=["gradient", "parameter"], default="gradient",
                           help="Where to add DP noise: gradient (during training) or parameter (before return)")
        
        return parser.parse_args(args_list)
    
    def __init__(self, args: DictConfig):
        super().__init__(args)
    
    def train_one_round(self):
        """Train one round."""
        client_packages = self.trainer.train()
        
        # Perform standard aggregation (clients already added noise locally)
        self.aggregate_client_updates(client_packages)
    
    
    
    def package(self, client_id: int):
        """Package parameters for client training."""
        return super().package(client_id)
    
