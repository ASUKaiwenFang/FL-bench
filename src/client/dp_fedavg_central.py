from typing import Any

from src.client.fedavg import FedAvgClient


class DPFedAvgCentralClient(FedAvgClient):
    """Central Differential Privacy FedAvg Client.
    
    This client works with central differential privacy where the server
    adds noise during aggregation. The client performs standard local training
    and sends clean updates to the server.
    """
    
    def __init__(self, **commons):
        super().__init__(**commons)
        
        # Store DP parameters
        self.sigma = self.args.dp_fedavg_central.sigma
    
    def train(self, server_package: dict[str, Any]) -> dict:
        """Train with standard FedAvg (no client-side noise)."""
        # Call parent training method (standard FedAvg)
        return super().train(server_package)
    
    def package(self):
        """Package client data for central DP."""
        client_package = super().package()
        
        # Add DP configuration for server reference
        client_package["dp_config"] = {
            "sigma": self.sigma
        }
        
        return client_package
