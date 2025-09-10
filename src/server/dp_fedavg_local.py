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
    return_diff = False
    client_cls = DPFedAvgLocalClient
    
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        """Define hyperparameters for Local DP-FedAvg."""
        parser = ArgumentParser()
        
        # Privacy parameters
        parser.add_argument("--epsilon", type=float, default=8.0,
                           help="Privacy budget parameter")
        parser.add_argument("--delta", type=float, default=1e-5,
                           help="Privacy parameter for (ε,δ)-DP")
        
        # Noise parameters
        parser.add_argument("--clip_norm", type=float, default=1.0,
                           help="Gradient clipping norm")
        parser.add_argument("--noise_multiplier", type=float, default=1.1,
                           help="Noise multiplier for calibrated noise")
        
        # Training parameters
        parser.add_argument("--lot_size", type=int, default=64,
                           help="Logical batch size for privacy accounting")
        
        # Privacy accounting
        parser.add_argument("--accounting_mode", type=str, default="rdp",
                           help="Privacy accounting method [rdp, gdp, ma]")
        
        # Advanced options
        parser.add_argument("--adaptive_clipping", type=bool, default=False,
                           help="Whether to use adaptive gradient clipping")
        parser.add_argument("--clip_percentile", type=int, default=50,
                           help="Percentile for adaptive clipping")
        
        return parser.parse_args(args_list)
    
    def __init__(self, args: DictConfig):
        super().__init__(args)
        
        # Initialize privacy tracking for the server
        self.client_privacy_consumption = {}
        self.global_privacy_tracking = {
            "total_clients": len(self.train_clients),
            "privacy_budgets": {},
            "exhausted_clients": set()
        }
        
        # Initialize client privacy budgets
        for client_id in self.train_clients:
            self.global_privacy_tracking["privacy_budgets"][client_id] = {
                "epsilon": self.args.dp_fedavg_local.epsilon,
                "delta": self.args.dp_fedavg_local.delta,
                "consumed_epsilon": 0.0,
                "consumed_delta": 0.0
            }
    
    def train_one_round(self):
        """Train one round with privacy tracking."""
        client_packages = self.trainer.train()
        
        # Update privacy tracking
        self._update_privacy_tracking(client_packages)
        
        # Check for exhausted privacy budgets
        self._check_exhausted_budgets(client_packages)
        
        # Perform standard aggregation (clients already added noise locally)
        self.aggregate_client_updates(client_packages)
    
    def _update_privacy_tracking(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Update privacy consumption tracking for all clients."""
        for client_id, package in client_packages.items():
            if "privacy_consumed" in package:
                privacy_info = package["privacy_consumed"]
                
                # Update global tracking
                self.global_privacy_tracking["privacy_budgets"][client_id].update({
                    "consumed_epsilon": privacy_info["epsilon"],
                    "consumed_delta": privacy_info["delta"]
                })
                
                # Track individual client consumption
                self.client_privacy_consumption[client_id] = {
                    "consumed_epsilon": privacy_info["epsilon"],
                    "consumed_delta": privacy_info["delta"],
                    "training_steps": package.get("training_steps", 0),
                    "dp_parameters": package.get("dp_parameters", {})
                }
    
    def _check_exhausted_budgets(self, client_packages: OrderedDict[int, Dict[str, Any]]):
        """Check and track clients with exhausted privacy budgets."""
        for client_id, package in client_packages.items():
            if package.get("privacy_budget_exhausted", False):
                self.global_privacy_tracking["exhausted_clients"].add(client_id)
                
                # Log warning about exhausted budget
                self.logger.log(
                    f"WARNING: Client {client_id} has exhausted privacy budget "
                    f"(ε={package['privacy_consumed']['epsilon']:.3f}, "
                    f"δ={package['privacy_consumed']['delta']:.6f})"
                )
    
    def package(self, client_id: int):
        """Package parameters for client training."""
        server_package = super().package(client_id)
        
        # Add privacy budget status
        if client_id in self.global_privacy_tracking["privacy_budgets"]:
            budget_info = self.global_privacy_tracking["privacy_budgets"][client_id]
            server_package["privacy_budget_status"] = {
                "remaining_epsilon": max(0, budget_info["epsilon"] - budget_info["consumed_epsilon"]),
                "remaining_delta": max(0, budget_info["delta"] - budget_info["consumed_delta"]),
                "budget_exhausted": client_id in self.global_privacy_tracking["exhausted_clients"]
            }
        
        return server_package
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get summary of privacy consumption across all clients."""
        summary = {
            "total_clients": self.global_privacy_tracking["total_clients"],
            "exhausted_clients": len(self.global_privacy_tracking["exhausted_clients"]),
            "active_clients": (self.global_privacy_tracking["total_clients"] - 
                              len(self.global_privacy_tracking["exhausted_clients"])),
            "privacy_consumption": {}
        }
        
        # Calculate aggregate privacy statistics
        total_epsilon = 0.0
        total_delta = 0.0
        max_epsilon = 0.0
        max_delta = 0.0
        
        for client_id, budget in self.global_privacy_tracking["privacy_budgets"].items():
            consumed_eps = budget["consumed_epsilon"]
            consumed_delta = budget["consumed_delta"]
            
            total_epsilon += consumed_eps
            total_delta += consumed_delta
            max_epsilon = max(max_epsilon, consumed_eps)
            max_delta = max(max_delta, consumed_delta)
        
        summary["privacy_consumption"] = {
            "average_epsilon": total_epsilon / self.global_privacy_tracking["total_clients"],
            "average_delta": total_delta / self.global_privacy_tracking["total_clients"],
            "max_epsilon": max_epsilon,
            "max_delta": max_delta,
            "total_epsilon": total_epsilon,
            "total_delta": total_delta
        }
        
        return summary
