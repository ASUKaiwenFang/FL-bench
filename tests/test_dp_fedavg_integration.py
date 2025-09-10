"""
Integration tests for DP-FedAvg algorithms.
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Mock the FL-bench environment
import sys
sys.path.append('/home/local/ASURITE/kfang11/FL-bench')

from src.client.dp_fedavg_local import DPFedAvgLocalClient
from src.server.dp_fedavg_local import DPFedAvgLocalServer
from src.client.dp_fedavg_central import DPFedAvgCentralClient
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
        # Create mock configuration
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
                'epsilon': 8.0,
                'delta': 1e-5,
                'clip_norm': 1.0,
                'noise_multiplier': 1.1,
                'lot_size': 16,
                'accounting_mode': 'rdp',
                'adaptive_clipping': False,
                'clip_percentile': 50
            },
            'dp_fedavg_central': {
                'epsilon': 8.0,
                'delta': 1e-5,
                'server_noise_multiplier': 0.8,
                'noise_type': 'gaussian',
                'sensitivity': 1.0,
                'clipping_mode': 'automatic',
                'aggregation_weights': 'uniform'
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
        
        # Check DP parameters are set correctly
        self.assertEqual(client.epsilon, 8.0)
        self.assertEqual(client.delta, 1e-5)
        self.assertEqual(client.clip_norm, 1.0)
        self.assertEqual(client.noise_multiplier, 1.1)
        
        # Check privacy engine is initialized
        self.assertIsNotNone(client.privacy_engine)
    
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
        
        # Set client ID and load data
        client.client_id = 0
        client.load_data_indices()
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in client.model.named_parameters()}
        
        # Perform training
        client.fit()
        
        # Check that parameters changed
        for name, param in client.model.named_parameters():
            self.assertFalse(torch.allclose(initial_params[name], param))
        
        # Check privacy consumption
        self.assertGreater(client.privacy_consumed['epsilon'], 0)
        self.assertGreater(client.privacy_consumed['delta'], 0)
    
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
        
        client.client_id = 0
        client.load_data_indices()
        
        # Get package
        package = client.package()
        
        # Check DP-specific information is included
        self.assertIn('privacy_consumed', package)
        self.assertIn('training_steps', package)
        self.assertIn('privacy_budget_exhausted', package)
        self.assertIn('dp_parameters', package)
        
        # Check privacy parameters
        dp_params = package['dp_parameters']
        self.assertEqual(dp_params['epsilon'], 8.0)
        self.assertEqual(dp_params['noise_multiplier'], 1.1)
    
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
        
        # Check DP parameters are set correctly
        self.assertEqual(client.epsilon, 8.0)
        self.assertEqual(client.delta, 1e-5)
        self.assertEqual(client.server_noise_multiplier, 0.8)
        self.assertEqual(client.noise_type, 'gaussian')
    
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
        
        client.client_id = 0
        client.load_data_indices()
        
        # Get package
        package = client.package()
        
        # Check DP-specific tracking information is included
        self.assertIn('dp_tracking', package)
        self.assertIn('dp_config', package)
        
        # Check tracking information
        dp_tracking = package['dp_tracking']
        self.assertIn('training_rounds', dp_tracking)
        self.assertIn('samples_processed', dp_tracking)
        self.assertIn('client_contribution', dp_tracking)
        
        # Check DP configuration
        dp_config = package['dp_config']
        self.assertEqual(dp_config['epsilon'], 8.0)
        self.assertEqual(dp_config['noise_type'], 'gaussian')
    
    def test_local_dp_server_hyperparams(self):
        """Test Local DP server hyperparameters."""
        hyperparams = DPFedAvgLocalServer.get_hyperparams()
        
        # Check default values
        self.assertEqual(hyperparams.epsilon, 8.0)
        self.assertEqual(hyperparams.delta, 1e-5)
        self.assertEqual(hyperparams.clip_norm, 1.0)
        self.assertEqual(hyperparams.noise_multiplier, 1.1)
        self.assertEqual(hyperparams.lot_size, 64)
        self.assertEqual(hyperparams.accounting_mode, 'rdp')
    
    def test_central_dp_server_hyperparams(self):
        """Test Central DP server hyperparameters."""
        hyperparams = DPFedAvgCentralServer.get_hyperparams()
        
        # Check default values
        self.assertEqual(hyperparams.epsilon, 8.0)
        self.assertEqual(hyperparams.delta, 1e-5)
        self.assertEqual(hyperparams.server_noise_multiplier, 0.8)
        self.assertEqual(hyperparams.noise_type, 'gaussian')
        self.assertEqual(hyperparams.sensitivity, 1.0)


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
            self.assertIn('epsilon', local_dp)
            self.assertIn('delta', local_dp)
            self.assertIn('clip_norm', local_dp)
            self.assertIn('noise_multiplier', local_dp)
            
            # Check Central DP parameters
            central_dp = config.dp_fedavg_central
            self.assertIn('epsilon', central_dp)
            self.assertIn('delta', central_dp)
            self.assertIn('server_noise_multiplier', central_dp)
            self.assertIn('noise_type', central_dp)


if __name__ == '__main__':
    unittest.main()
