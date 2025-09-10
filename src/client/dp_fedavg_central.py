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
        
        # Store DP parameters for tracking and reporting
        self.epsilon = self.args.dp_fedavg_central.epsilon
        self.delta = self.args.dp_fedavg_central.delta
        self.server_noise_multiplier = self.args.dp_fedavg_central.server_noise_multiplier
        self.noise_type = self.args.dp_fedavg_central.noise_type
        
        # Track client statistics for server-side privacy accounting
        self.training_rounds = 0
        self.total_samples_processed = 0
    
    def train(self, server_package: dict[str, Any]) -> dict:
        """Train with standard FedAvg (no client-side noise)."""
        # Increment training round counter
        self.training_rounds += 1
        
        # Track samples processed (for privacy accounting on server side)
        self.total_samples_processed += len(self.trainset)
        
        # Call parent training method (standard FedAvg)
        client_package = super().train(server_package)
        
        return client_package
    
    def package(self):
        """Package client data with tracking information for central DP."""
        client_package = super().package()
        
        # Add tracking information for server-side privacy accounting
        client_package["dp_tracking"] = {
            "training_rounds": self.training_rounds,
            "samples_processed": self.total_samples_processed,
            "client_contribution": len(self.trainset)
        }
        
        # Add DP configuration for server reference
        client_package["dp_config"] = {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "server_noise_multiplier": self.server_noise_multiplier,
            "noise_type": self.noise_type
        }
        
        return client_package
