"""
Unit tests for differential privacy mechanisms.
"""

import math
import unittest
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.utils.dp_mechanisms import (
    PrivacyEngine,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
    compute_noise_multiplier,
    privacy_accountant
)


class TestDPMechanisms(unittest.TestCase):
    """Test cases for differential privacy mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_add_gaussian_noise(self):
        """Test Gaussian noise addition."""
        # Create test tensor
        tensor = torch.ones(10, 5)
        noise_multiplier = 1.0
        sensitivity = 1.0
        
        # Add noise
        noisy_tensor = add_gaussian_noise(tensor, noise_multiplier, sensitivity)
        
        # Check shape is preserved
        self.assertEqual(tensor.shape, noisy_tensor.shape)
        
        # Check that noise was actually added (tensors should be different)
        self.assertFalse(torch.allclose(tensor, noisy_tensor))
        
        # Check noise properties (statistical test)
        noise = noisy_tensor - tensor
        noise_mean = torch.mean(noise).item()
        noise_std = torch.std(noise).item()
        
        # Mean should be close to 0
        self.assertAlmostEqual(noise_mean, 0.0, delta=0.2)
        
        # Standard deviation should be close to noise_multiplier * sensitivity
        expected_std = noise_multiplier * sensitivity
        self.assertAlmostEqual(noise_std, expected_std, delta=0.2)
    
    def test_add_laplace_noise(self):
        """Test Laplace noise addition."""
        # Create test tensor
        tensor = torch.zeros(100, 10)
        epsilon = 1.0
        sensitivity = 1.0
        
        # Add noise
        noisy_tensor = add_laplace_noise(tensor, epsilon, sensitivity)
        
        # Check shape is preserved
        self.assertEqual(tensor.shape, noisy_tensor.shape)
        
        # Check that noise was actually added
        self.assertFalse(torch.allclose(tensor, noisy_tensor))
        
        # Check noise properties
        noise = noisy_tensor - tensor
        noise_mean = torch.mean(noise).item()
        
        # Mean should be close to 0
        self.assertAlmostEqual(noise_mean, 0.0, delta=0.2)
    
    def test_clip_gradients_single_tensor(self):
        """Test gradient clipping with single tensor."""
        # Create tensor with known gradient norm
        tensor = torch.ones(3, 4, requires_grad=True)
        
        # Create artificial gradients
        tensor.grad = torch.ones_like(tensor) * 2.0  # Gradient norm = sqrt(48) ≈ 6.93
        
        clip_norm = 1.0
        original_norm = clip_gradients([tensor], clip_norm)
        
        # Check that original norm was calculated correctly
        expected_original_norm = math.sqrt(48)
        self.assertAlmostEqual(original_norm, expected_original_norm, places=2)
        
        # Check that gradients were clipped
        clipped_norm = torch.norm(tensor.grad).item()
        self.assertAlmostEqual(clipped_norm, clip_norm, places=5)
    
    def test_clip_gradients_model_parameters(self):
        """Test gradient clipping with model parameters."""
        # Create simple model
        model = nn.Linear(10, 5)
        
        # Create artificial gradients
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.randn_like(param) * 10.0  # Large gradients
        
        clip_norm = 2.0
        original_norm = clip_gradients(model.parameters(), clip_norm)
        
        # Check that gradients were clipped
        total_norm = 0.0
        for param in model.parameters():
            total_norm += param.grad.data.norm() ** 2
        total_norm = total_norm ** 0.5
        
        self.assertAlmostEqual(total_norm, clip_norm, places=5)
        self.assertGreater(original_norm, clip_norm)  # Original norm should be larger
    
    def test_compute_noise_multiplier(self):
        """Test noise multiplier computation."""
        epsilon = 8.0
        delta = 1e-5
        lot_size = 64
        steps = 100
        
        noise_multiplier = compute_noise_multiplier(epsilon, delta, lot_size, steps)
        
        # Check that noise multiplier is positive
        self.assertGreater(noise_multiplier, 0)
        
        # Check that noise multiplier is reasonable (not too large or too small)
        self.assertGreater(noise_multiplier, 0.1)
        self.assertLess(noise_multiplier, 10.0)
    
    def test_privacy_accountant(self):
        """Test privacy accounting function."""
        noise_multiplier = 1.1
        lot_size = 64
        steps = 50
        dataset_size = 10000
        
        epsilon, delta = privacy_accountant(noise_multiplier, lot_size, steps, dataset_size)
        
        # Check that privacy parameters are positive
        self.assertGreater(epsilon, 0)
        self.assertGreater(delta, 0)
        
        # Check that privacy parameters are reasonable
        self.assertLess(epsilon, 100)  # Should not be extremely large
        self.assertLess(delta, 1.0)    # Delta should be < 1
    
    def test_privacy_accountant_edge_cases(self):
        """Test privacy accounting edge cases."""
        # Zero noise multiplier should give infinite epsilon
        epsilon, delta = privacy_accountant(0.0, 64, 50, 10000)
        self.assertEqual(epsilon, float('inf'))
        self.assertEqual(delta, 1.0)
        
        # Zero steps should give zero privacy cost
        epsilon, delta = privacy_accountant(1.0, 64, 0, 10000)
        self.assertEqual(epsilon, float('inf'))
        self.assertEqual(delta, 1.0)


class TestPrivacyEngine(unittest.TestCase):
    """Test cases for PrivacyEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.epsilon = 8.0
        self.delta = 1e-5
        self.lot_size = 64
        self.engine = PrivacyEngine(self.epsilon, self.delta, self.lot_size)
    
    def test_initialization(self):
        """Test PrivacyEngine initialization."""
        self.assertEqual(self.engine.target_epsilon, self.epsilon)
        self.assertEqual(self.engine.target_delta, self.delta)
        self.assertEqual(self.engine.lot_size, self.lot_size)
        self.assertEqual(self.engine.steps, 0)
        self.assertEqual(self.engine.consumed_epsilon, 0.0)
        self.assertEqual(self.engine.consumed_delta, 0.0)
    
    def test_step_updates(self):
        """Test that step() updates privacy consumption."""
        noise_multiplier = 1.1
        dataset_size = 10000
        
        # Take one step
        self.engine.step(noise_multiplier, dataset_size)
        
        # Check that steps were incremented
        self.assertEqual(self.engine.steps, 1)
        
        # Check that privacy was consumed
        self.assertGreater(self.engine.consumed_epsilon, 0)
        self.assertGreater(self.engine.consumed_delta, 0)
    
    def test_get_privacy_spent(self):
        """Test privacy spent reporting."""
        noise_multiplier = 1.1
        dataset_size = 10000
        
        # Initially no privacy spent
        eps, delta = self.engine.get_privacy_spent()
        self.assertEqual(eps, 0.0)
        self.assertEqual(delta, 0.0)
        
        # After step, privacy should be consumed
        self.engine.step(noise_multiplier, dataset_size)
        eps, delta = self.engine.get_privacy_spent()
        self.assertGreater(eps, 0)
        self.assertGreater(delta, 0)
    
    def test_get_remaining_budget(self):
        """Test remaining privacy budget calculation."""
        noise_multiplier = 1.1
        dataset_size = 10000
        
        # Initially full budget remaining
        remaining_eps, remaining_delta = self.engine.get_remaining_budget()
        self.assertEqual(remaining_eps, self.epsilon)
        self.assertEqual(remaining_delta, self.delta)
        
        # After step, budget should decrease
        self.engine.step(noise_multiplier, dataset_size)
        remaining_eps, remaining_delta = self.engine.get_remaining_budget()
        self.assertLess(remaining_eps, self.epsilon)
        self.assertLess(remaining_delta, self.delta)
    
    def test_budget_exhaustion(self):
        """Test budget exhaustion detection."""
        # Initially budget not exhausted
        self.assertFalse(self.engine.is_budget_exhausted())
        
        # Manually set consumed values to exceed budget
        self.engine.consumed_epsilon = self.epsilon + 1.0
        self.engine.consumed_delta = self.delta + 1e-6
        
        # Now budget should be exhausted
        self.assertTrue(self.engine.is_budget_exhausted())


if __name__ == '__main__':
    unittest.main()
