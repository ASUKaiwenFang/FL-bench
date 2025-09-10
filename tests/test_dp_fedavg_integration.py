"""
Integration tests for DP-FedAvg implementations.
"""

import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.client.dp_fedavg_central import DPFedAvgCentralClient
from src.server.dp_fedavg_local import DPFedAvgLocalServer
from src.server.dp_fedavg_central import DPFedAvgCentralServer


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.targets = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def train(self):
        pass
    
    def eval(self):
        pass


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)
    
    def to(self, device):
        return super().to(device)


class TestDPFedAvgIntegration(unittest.TestCase):
    """Integration tests for DP-FedAvg implementations."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock configuration with simplified DP parameters
        self.config = OmegaConf.create({
            'common': {
                'seed': 42,
                'use_cuda': False,
                'batch_size': 16,
                'local_epoch': 1,
                'reset_optimizer_on_global_epoch': True,
                'buffers': 'global',
                'client_side_evaluation': False,
                'test': {
                    'client': {
                        'finetune_epoch': 0,
                        'train': False,
                        'val': False,
                        'test': False
                    }
                }
            },
            'optimizer': {
                'name': 'sgd',
                'lr': 0.01,
                'momentum': 0,
                'weight_decay': 0
            },
            'dp_fedavg_local': {
                'clip_norm': 1.0,
                'sigma': 0.1
            },
            'dp_fedavg_central': {
                'sigma': 0.1
            }
        })
        
        # Create mock data indices
        self.data_indices = [
            {'train': list(range(50)), 'val': [], 'test': []},
            {'train': list(range(50, 100)), 'val': [], 'test': []}
        ]
        
        # Create mock dataset
        self.dataset = MockDataset()
        
        # Create mock model
        self.model = MockModel()
    
    def test_local_dp_client_initialization(self):
        """Test Local DP client initialization."""
        client = DPFedAvgLocalClient(
            model=self.model,
            optimizer_cls=torch.optim.SGD,
            lr_scheduler_cls=None,
            args=self.config,
            dataset=self.dataset,
            data_indices=self.data_indices,
            device=torch.device('cpu'),
            return_diff=False
        )
        
        # Check simplified DP parameters are set correctly
        self.assertEqual(client.clip_norm, 1.0)
        self.assertEqual(client.sigma, 0.1)
    
    def test_local_dp_client_training(self):
        """Test Local DP client training process."""
        client = DPFedAvgLocalClient(
            model=self.model,
            optimizer_cls=torch.optim.SGD,
            lr_scheduler_cls=None,
            args=self.config,
            dataset=self.dataset,
            data_indices=self.data_indices,
            device=torch.device('cpu'),
            return_diff=False
        )
        
        # Test that training runs without errors
        try:
            client.fit()
            success = True
        except Exception as e:
            success = False
            print(f"Training failed with error: {e}")
        
        self.assertTrue(success, "Local DP client training should complete without errors")
    
    def test_local_dp_client_package(self):
        """Test Local DP client package method."""
        client = DPFedAvgLocalClient(
            model=self.model,
            optimizer_cls=torch.optim.SGD,
            lr_scheduler_cls=None,
            args=self.config,
            dataset=self.dataset,
            data_indices=self.data_indices,
            device=torch.device('cpu'),
            return_diff=False
        )
        
        package = client.package()
        
        # Check that DP parameters are included in package
        self.assertIn('dp_parameters', package)
        self.assertIn('clip_norm', package['dp_parameters'])
        self.assertIn('sigma', package['dp_parameters'])
        self.assertEqual(package['dp_parameters']['clip_norm'], 1.0)
        self.assertEqual(package['dp_parameters']['sigma'], 0.1)
    
    def test_central_dp_client_initialization(self):
        """Test Central DP client initialization."""
        client = DPFedAvgCentralClient(
            model=self.model,
            optimizer_cls=torch.optim.SGD,
            lr_scheduler_cls=None,
            args=self.config,
            dataset=self.dataset,
            data_indices=self.data_indices,
            device=torch.device('cpu'),
            return_diff=False
        )
        
        # Check DP parameter is set correctly
        self.assertEqual(client.sigma, 0.1)
    
    def test_central_dp_client_package(self):
        """Test Central DP client package method."""
        client = DPFedAvgCentralClient(
            model=self.model,
            optimizer_cls=torch.optim.SGD,
            lr_scheduler_cls=None,
            args=self.config,
            dataset=self.dataset,
            data_indices=self.data_indices,
            device=torch.device('cpu'),
            return_diff=False
        )
        
        package = client.package()
        
        # Check that DP config is included in package
        self.assertIn('dp_config', package)
        self.assertIn('sigma', package['dp_config'])
        self.assertEqual(package['dp_config']['sigma'], 0.1)
    
    def test_local_dp_server_hyperparams(self):
        """Test Local DP server hyperparameters."""
        hyperparams = DPFedAvgLocalServer.get_hyperparams([])
        
        # Check that simplified hyperparameters exist
        self.assertTrue(hasattr(hyperparams, 'clip_norm'))
        self.assertTrue(hasattr(hyperparams, 'sigma'))
        self.assertEqual(hyperparams.clip_norm, 1.0)
        self.assertEqual(hyperparams.sigma, 0.1)
    
    def test_central_dp_server_hyperparams(self):
        """Test Central DP server hyperparameters."""
        hyperparams = DPFedAvgCentralServer.get_hyperparams([])
        
        # Check that simplified hyperparameter exists
        self.assertTrue(hasattr(hyperparams, 'sigma'))
        self.assertEqual(hyperparams.sigma, 0.1)


class TestDPConfigurationLoading(unittest.TestCase):
    """Test DP configuration loading and validation."""
    
    def test_config_file_exists(self):
        """Test that the DP configuration file exists."""
        config_path = Path('/home/local/ASURITE/kfang11/FL-bench/config/dp_fedavg.yaml')
        self.assertTrue(config_path.exists())
    
    def test_config_file_structure(self):
        """Test configuration file structure."""
        config_path = Path('/home/local/ASURITE/kfang11/FL-bench/config/dp_fedavg.yaml')
        
        if config_path.exists():
            config = OmegaConf.load(config_path)
            
            # Check required sections exist
            self.assertIn('dp_fedavg_local', config)
            self.assertIn('dp_fedavg_central', config)
            
            # Check Local DP parameters
            local_dp = config.dp_fedavg_local
            self.assertIn('clip_norm', local_dp)
            self.assertIn('sigma', local_dp)
            
            # Check Central DP parameters
            central_dp = config.dp_fedavg_central
            self.assertIn('sigma', central_dp)


if __name__ == '__main__':
    unittest.main()