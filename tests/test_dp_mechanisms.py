"""
Unit tests for differential privacy mechanisms.
"""

import math
import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.dp_mechanisms import (
    add_gaussian_noise,
    clip_gradients
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
        sigma = 1.0
        
        # Add noise
        noisy_tensor = add_gaussian_noise(tensor, sigma)
        
        # Check shape is preserved
        self.assertEqual(tensor.shape, noisy_tensor.shape)
        
        # Check that noise was actually added (tensors should be different)
        self.assertFalse(torch.allclose(tensor, noisy_tensor))
        
        # Check noise properties (statistical test)
        noise = noisy_tensor - tensor
        noise_mean = torch.mean(noise).item()
        noise_std = torch.std(noise).item()
        
        # For large tensors, the mean should be close to 0
        self.assertLess(abs(noise_mean), 0.2, "Gaussian noise mean should be close to 0")
        
        # Standard deviation should be close to sigma
        expected_std = sigma
        self.assertAlmostEqual(noise_std, expected_std, delta=0.2)
    
    def test_clip_gradients_single_tensor(self):
        """Test gradient clipping with single tensor."""
        # Create tensor with known gradient norm
        tensor = torch.ones(3, 4, requires_grad=True)
        
        # Set gradient to have norm > clip_norm
        tensor.grad = torch.ones(3, 4) * 2  # Each element is 2, so norm = sqrt(12*4) = sqrt(48) ≈ 6.93
        clip_norm = 1.0
        
        # Get original norm for comparison
        original_norm = torch.norm(tensor.grad).item()
        
        # Clip gradients
        returned_norm = clip_gradients(tensor, clip_norm)
        
        # Check that returned norm matches original norm
        self.assertAlmostEqual(returned_norm, original_norm, places=5)
        
        # Check that gradient norm is now clipped
        clipped_norm = torch.norm(tensor.grad).item()
        self.assertAlmostEqual(clipped_norm, clip_norm, places=5)
        self.assertGreater(original_norm, clip_norm)  # Original norm should be larger
    
    def test_clip_gradients_model_parameters(self):
        """Test gradient clipping with model parameters."""
        # Create simple model
        model = nn.Linear(5, 3)
        
        # Set gradients manually
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.ones_like(param) * 3  # Large gradient
        
        clip_norm = 2.0
        
        # Clip gradients
        original_norm = clip_gradients(model.parameters(), clip_norm)
        
        # Check that total norm is now clipped
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        self.assertAlmostEqual(total_norm, clip_norm, places=5)
        self.assertGreater(original_norm, clip_norm)  # Original norm should be larger
    
    def test_add_gaussian_noise_device(self):
        """Test that noise is generated on correct device."""
        tensor = torch.ones(2, 2)
        sigma = 0.5
        
        # Test with default device (should match tensor device)
        noisy_tensor = add_gaussian_noise(tensor, sigma)
        self.assertEqual(noisy_tensor.device, tensor.device)
        
        # Test with explicit device
        noisy_tensor = add_gaussian_noise(tensor, sigma, device=torch.device("cpu"))
        self.assertEqual(noisy_tensor.device, torch.device("cpu"))
    
    def test_add_gaussian_noise_zero_sigma(self):
        """Test Gaussian noise with zero sigma."""
        tensor = torch.ones(3, 3)
        sigma = 0.0
        
        noisy_tensor = add_gaussian_noise(tensor, sigma)
        
        # With zero sigma, tensor should remain unchanged
        self.assertTrue(torch.allclose(tensor, noisy_tensor))
    
    def test_clip_gradients_no_clipping_needed(self):
        """Test gradient clipping when no clipping is needed."""
        # Create tensor with small gradient
        tensor = torch.ones(2, 2, requires_grad=True)
        tensor.grad = torch.ones(2, 2) * 0.1  # Small gradient
        
        clip_norm = 2.0  # Large clip norm
        original_grad = tensor.grad.clone()
        
        # Clip gradients
        returned_norm = clip_gradients(tensor, clip_norm)
        
        # Gradient should remain unchanged
        self.assertTrue(torch.allclose(tensor.grad, original_grad))
        
        # Returned norm should be the original norm
        expected_norm = torch.norm(original_grad).item()
        self.assertAlmostEqual(returned_norm, expected_norm, places=5)


if __name__ == '__main__':
    unittest.main()