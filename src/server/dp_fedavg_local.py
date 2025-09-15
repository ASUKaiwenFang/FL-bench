from argparse import ArgumentParser, Namespace
from typing import Dict, Any
from collections import OrderedDict

from omegaconf import DictConfig
import torch

from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.server.fedavg import FedAvgServer


class DPFedAvgLocalServer(FedAvgServer):
    """Local Differential Privacy FedAvg Server.

    This server coordinates federated learning with local differential privacy.
    Clients add noise to their gradients locally before sending updates to the server.
    The server performs standard FedAvg aggregation on the noisy updates.

    Only supports return_diff=True mode, similar to SCAFFOLD.
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
        parser.add_argument("--global_lr", type=float, default=1.0)
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
        
    def aggregate_client_updates(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        for name, global_param in self.public_model_params.items():
            diffs = torch.stack(
                [
                    package["model_params_diff"][name]
                    for package in client_packages.values()
                ],
                dim=-1,
            )

            self.public_model_params[name].data += self.args.dp_fedavg_local.global_lr * aggregated
        self.model.load_state_dict(self.public_model_params, strict=False)
    
    
    


# Create an alias for main.py's naming convention compatibility
# main.py expects class name to match "method_name + server" pattern
Dp_fedavg_localServer = DPFedAvgLocalServer
    
