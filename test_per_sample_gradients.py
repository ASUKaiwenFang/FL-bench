#!/usr/bin/env python3
"""
Per-Sample Gradient Computation Validation Tests
===============================================

This test suite validates the correctness of the torch.func.grad-based
per-sample gradient computation implementation against manual calculations
and theoretical expectations.

Author: FL-bench DP Methods Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import tracemalloc
from typing import Dict, Tuple, List, Any
import sys
from pathlib import Path

# Add FL-bench root to path
FLBENCH_ROOT = Path(__file__).parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())

from src.client.dp_fedavg_local import _compute_per_sample_grads, _compute_per_sample_norms

# Try to import opacus for comparison tests
try:
    from opacus import GradSampleModule
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("⚠️  Opacus not available - skipping opacus comparison tests")


class TestResults:
    """Container for test results and statistics."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.timing_results = {}
        self.memory_results = {}

    def add_pass(self, test_name: str):
        """Record a passed test."""
        self.passed += 1
        print(f"✓ {test_name} PASSED")

    def add_fail(self, test_name: str, error_msg: str):
        """Record a failed test."""
        self.failed += 1
        self.errors.append(f"{test_name}: {error_msg}")
        print(f"✗ {test_name} FAILED: {error_msg}")

    def add_timing(self, test_name: str, duration: float):
        """Record timing information."""
        self.timing_results[test_name] = duration
        print(f"  ⏱️  {test_name}: {duration:.4f}s")

    def add_memory(self, test_name: str, peak_memory: float):
        """Record memory usage information."""
        self.memory_results[test_name] = peak_memory
        print(f"  🧠 {test_name}: {peak_memory:.2f}MB")

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"TEST SUMMARY: {self.passed}/{total} tests passed")
        print(f"{'='*50}")

        if self.errors:
            print("FAILURES:")
            for error in self.errors:
                print(f"  - {error}")

        if self.timing_results:
            print("\nTIMING RESULTS:")
            for test_name, duration in self.timing_results.items():
                print(f"  {test_name}: {duration:.4f}s")

        if self.memory_results:
            print("\nMEMORY USAGE:")
            for test_name, memory in self.memory_results.items():
                print(f"  {test_name}: {memory:.2f}MB")


def create_simple_model(input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1) -> nn.Module:
    """
    Create a simple linear model for testing.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension

    Returns:
        Simple neural network model
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )


def create_test_data(batch_size: int = 4, input_dim: int = 2, num_classes: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate deterministic test data.

    Args:
        batch_size: Number of samples in the batch
        input_dim: Input feature dimension
        num_classes: Number of output classes (1 for regression)

    Returns:
        Tuple of (inputs, targets)
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    inputs = torch.randn(batch_size, input_dim)

    if num_classes == 1:
        # Regression targets
        targets = torch.randn(batch_size, 1)
    else:
        # Classification targets
        targets = torch.randint(0, num_classes, (batch_size,))

    return inputs, targets


def manual_compute_single_sample_grad(
    model: nn.Module,
    input_sample: torch.Tensor,
    target_sample: torch.Tensor,
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Manually compute gradient for a single sample using standard PyTorch backward.

    Args:
        model: Neural network model
        input_sample: Single input sample [1, input_dim]
        target_sample: Single target sample [1] or [1, output_dim]
        criterion: Loss function

    Returns:
        Dictionary mapping parameter names to gradients
    """
    model.zero_grad()

    # Forward pass for single sample
    prediction = model(input_sample)
    loss = criterion(prediction, target_sample)

    # Backward pass
    loss.backward()

    # Collect gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone()
        else:
            gradients[name] = torch.zeros_like(param)

    return gradients


def compute_opacus_per_sample_grads(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute per-sample gradients using Opacus for comparison.

    Args:
        model: Neural network model
        inputs: Input batch tensor
        targets: Target batch tensor
        criterion: Loss function

    Returns:
        Dictionary mapping parameter names to per-sample gradients
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus not available for comparison")

    # Create a copy of the model to avoid modifying the original
    import copy
    model_copy = copy.deepcopy(model)

    # Fix model compatibility with Opacus
    if not ModuleValidator.is_valid(model_copy):
        model_copy = ModuleValidator.fix(model_copy)

    # Wrap with GradSampleModule
    grad_sample_model = GradSampleModule(model_copy)

    # Forward and backward pass
    grad_sample_model.zero_grad()
    predictions = grad_sample_model(inputs)
    loss = criterion(predictions, targets)
    loss.backward()

    # Extract per-sample gradients
    per_sample_grads = {}
    for name, param in grad_sample_model.named_parameters():
        if hasattr(param, 'grad_sample') and param.grad_sample is not None:
            # Remove '_module.' prefix from parameter names
            clean_name = name.replace("_module.", "") if name.startswith("_module.") else name
            per_sample_grads[clean_name] = param.grad_sample.clone()

    return per_sample_grads


def test_opacus_comparison(results: TestResults):
    """
    Test our implementation against Opacus per-sample gradient computation.

    This test compares the output of our torch.func.grad implementation
    with the established Opacus GradSampleModule implementation.
    """
    test_name = "opacus_comparison"

    if not OPACUS_AVAILABLE:
        results.add_fail(test_name, "Opacus not available - install opacus to run this test")
        return

    try:
        # Use a simple model for comparison
        model = create_simple_model(input_dim=3, hidden_dim=4, output_dim=2)
        inputs, targets = create_test_data(batch_size=5, input_dim=3, num_classes=2)
        criterion = nn.CrossEntropyLoss()

        # Compute gradients using our implementation
        our_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Compute gradients using Opacus
        opacus_grads = compute_opacus_per_sample_grads(model, inputs, targets, criterion)

        # Compare parameter names
        our_param_names = set(our_grads.keys())
        opacus_param_names = set(opacus_grads.keys())

        if our_param_names != opacus_param_names:
            missing_in_ours = opacus_param_names - our_param_names
            missing_in_opacus = our_param_names - opacus_param_names
            error_msg = f"Parameter name mismatch. Missing in ours: {missing_in_ours}, Missing in opacus: {missing_in_opacus}"
            results.add_fail(test_name, error_msg)
            return

        # Compare gradient values
        max_diff = 0.0
        total_params = 0
        differences = {}

        for param_name in our_param_names:
            our_grad = our_grads[param_name]
            opacus_grad = opacus_grads[param_name]

            # Check shapes match
            if our_grad.shape != opacus_grad.shape:
                results.add_fail(test_name, f"Shape mismatch for {param_name}: ours={our_grad.shape}, opacus={opacus_grad.shape}")
                return

            # Compute element-wise differences
            diff = torch.abs(our_grad - opacus_grad)
            param_max_diff = torch.max(diff).item()
            param_mean_diff = torch.mean(diff).item()

            differences[param_name] = {
                'max_diff': param_max_diff,
                'mean_diff': param_mean_diff,
                'relative_max': param_max_diff / (torch.max(torch.abs(opacus_grad)).item() + 1e-10)
            }

            max_diff = max(max_diff, param_max_diff)
            total_params += our_grad.numel()

        # Check if differences are within tolerance
        tolerance = 1e-5
        if max_diff < tolerance:
            results.add_pass(test_name)
            print(f"    📊 Max difference across all parameters: {max_diff:.2e}")
            print(f"    📊 Total parameters compared: {total_params}")
        else:
            # Provide detailed difference information
            worst_param = max(differences.keys(), key=lambda k: differences[k]['max_diff'])
            error_msg = f"Max difference: {max_diff:.2e} > tolerance: {tolerance:.2e}. Worst param: {worst_param}"
            results.add_fail(test_name, error_msg)

            # Print detailed differences for debugging
            print(f"    🔍 Detailed differences:")
            for param_name, diff_info in differences.items():
                print(f"      {param_name}: max={diff_info['max_diff']:.2e}, mean={diff_info['mean_diff']:.2e}, rel_max={diff_info['relative_max']:.2e}")

    except Exception as e:
        results.add_fail(test_name, f"Exception during comparison: {str(e)}")


def test_opacus_different_models(results: TestResults):
    """
    Test Opacus comparison across different model architectures.

    Ensures our implementation matches Opacus across various model types.
    """
    test_name = "opacus_different_models"

    if not OPACUS_AVAILABLE:
        results.add_fail(test_name, "Opacus not available - install opacus to run this test")
        return

    try:
        # Test different model architectures
        models_and_data = [
            # Simple linear model
            (create_simple_model(input_dim=4, hidden_dim=4, output_dim=2),
             torch.randn(3, 4), torch.randint(0, 2, (3,))),

            # Multi-layer model
            (create_simple_model(input_dim=5, hidden_dim=8, output_dim=3),
             torch.randn(4, 5), torch.randint(0, 3, (4,))),
        ]

        criterion = nn.CrossEntropyLoss()
        overall_max_diff = 0.0

        for i, (model, inputs, targets) in enumerate(models_and_data):
            # Compute gradients with both methods
            our_grads = _compute_per_sample_grads(model, inputs, targets, criterion)
            opacus_grads = compute_opacus_per_sample_grads(model, inputs, targets, criterion)

            # Find maximum difference for this model
            model_max_diff = 0.0
            for param_name in our_grads.keys():
                if param_name in opacus_grads:
                    diff = torch.max(torch.abs(our_grads[param_name] - opacus_grads[param_name])).item()
                    model_max_diff = max(model_max_diff, diff)

            overall_max_diff = max(overall_max_diff, model_max_diff)
            print(f"    📊 Model {i+1} max difference: {model_max_diff:.2e}")

        tolerance = 1e-5
        if overall_max_diff < tolerance:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, f"Max difference across models: {overall_max_diff:.2e} > tolerance: {tolerance:.2e}")

    except Exception as e:
        results.add_fail(test_name, f"Exception during multi-model comparison: {str(e)}")


def test_gradient_aggregation_consistency(results: TestResults):
    """
    Test that per-sample gradients average to batch gradient.

    This is the fundamental correctness test: the mean of per-sample gradients
    should equal the gradient computed on the entire batch.
    """
    test_name = "gradient_aggregation_consistency"

    try:
        model = create_simple_model(input_dim=3, hidden_dim=5, output_dim=1)
        inputs, targets = create_test_data(batch_size=8, input_dim=3, num_classes=1)
        criterion = nn.MSELoss()

        # Compute batch gradient
        model.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()

        batch_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                batch_grads[name] = param.grad.clone()

        # Compute per-sample gradients and average them
        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)
        averaged_grads = {}
        for name, grad_tensor in per_sample_grads.items():
            averaged_grads[name] = grad_tensor.mean(dim=0)

        # Compare gradients
        max_diff = 0.0
        for name in batch_grads:
            diff = torch.max(torch.abs(batch_grads[name] - averaged_grads[name])).item()
            max_diff = max(max_diff, diff)

        # Check if difference is within tolerance
        tolerance = 1e-5
        if max_diff < tolerance:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, f"Max difference: {max_diff:.2e} > tolerance: {tolerance:.2e}")

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_individual_sample_correctness(results: TestResults):
    """
    Test correctness of individual per-sample gradients.

    Compares each per-sample gradient against manually computed single-sample gradient.
    """
    test_name = "individual_sample_correctness"

    try:
        model = create_simple_model(input_dim=2, hidden_dim=3, output_dim=1)
        inputs, targets = create_test_data(batch_size=3, input_dim=2, num_classes=1)
        criterion = nn.MSELoss()

        # Compute per-sample gradients using our function
        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Manually compute each sample's gradient
        max_diff = 0.0
        for i in range(inputs.size(0)):
            input_sample = inputs[i:i+1]
            target_sample = targets[i:i+1]

            manual_grad = manual_compute_single_sample_grad(model, input_sample, target_sample, criterion)

            # Compare with per-sample gradient
            for name in manual_grad:
                our_grad = per_sample_grads[name][i]
                manual_grad_tensor = manual_grad[name].squeeze(0) if manual_grad[name].dim() > our_grad.dim() else manual_grad[name]

                diff = torch.max(torch.abs(our_grad - manual_grad_tensor)).item()
                max_diff = max(max_diff, diff)

        tolerance = 1e-5
        if max_diff < tolerance:
            results.add_pass(test_name)
        else:
            results.add_fail(test_name, f"Max difference: {max_diff:.2e} > tolerance: {tolerance:.2e}")

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_numerical_stability(results: TestResults):
    """
    Test numerical stability and edge cases.

    Tests various edge cases that could cause numerical instability.
    """
    test_name = "numerical_stability"

    try:
        # Test 1: Single sample batch
        model = create_simple_model(input_dim=2, hidden_dim=3, output_dim=1)
        inputs = torch.randn(1, 2)
        targets = torch.randn(1, 1)
        criterion = nn.MSELoss()

        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Should have gradients for all parameters
        param_count = sum(1 for _ in model.named_parameters())
        if len(per_sample_grads) != param_count:
            results.add_fail(test_name, f"Expected {param_count} gradients, got {len(per_sample_grads)}")
            return

        # Test 2: Zero gradients (constant targets with linear model)
        simple_model = nn.Linear(2, 1)
        zero_inputs = torch.zeros(4, 2)
        constant_targets = torch.ones(4, 1)

        # This should produce small gradients
        zero_grads = _compute_per_sample_grads(simple_model, zero_inputs, constant_targets, criterion)

        # Test 3: Large values
        large_inputs = torch.randn(3, 2) * 100
        large_targets = torch.randn(3, 1) * 100

        large_grads = _compute_per_sample_grads(model, large_inputs, large_targets, criterion)

        # Check for NaN or infinite values
        has_nan = any(torch.isnan(grad).any() for grad in large_grads.values())
        has_inf = any(torch.isinf(grad).any() for grad in large_grads.values())

        if has_nan or has_inf:
            results.add_fail(test_name, "Found NaN or infinite gradients with large inputs")
        else:
            results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_different_model_architectures(results: TestResults):
    """
    Test compatibility with different model architectures.

    Ensures the per-sample gradient computation works with various model types.
    """
    test_name = "different_model_architectures"

    try:
        inputs, targets = create_test_data(batch_size=4, input_dim=8, num_classes=3)
        criterion = nn.CrossEntropyLoss()

        # Test different architectures
        architectures = [
            # Simple linear
            nn.Linear(8, 3),

            # Multi-layer perceptron
            nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            ),

            # With batch normalization (in eval mode)
            nn.Sequential(
                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )
        ]

        for i, model in enumerate(architectures):
            model.eval()  # Important for BatchNorm

            try:
                per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

                # Basic sanity checks
                if len(per_sample_grads) == 0:
                    results.add_fail(test_name, f"Architecture {i}: No gradients computed")
                    return

                # Check gradient shapes
                for name, grad in per_sample_grads.items():
                    if grad.size(0) != inputs.size(0):
                        results.add_fail(test_name, f"Architecture {i}: Wrong batch dimension for {name}")
                        return

            except Exception as e:
                results.add_fail(test_name, f"Architecture {i} failed: {str(e)}")
                return

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_different_loss_functions(results: TestResults):
    """
    Test compatibility with different loss functions.
    """
    test_name = "different_loss_functions"

    try:
        model = create_simple_model(input_dim=4, hidden_dim=8, output_dim=3)
        inputs = torch.randn(5, 4)

        # Test different loss functions
        loss_configs = [
            (nn.CrossEntropyLoss(), torch.randint(0, 3, (5,))),
            (nn.MSELoss(), torch.randn(5, 3)),
            (nn.L1Loss(), torch.randn(5, 3)),
        ]

        for criterion, targets in loss_configs:
            try:
                per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

                if len(per_sample_grads) == 0:
                    results.add_fail(test_name, f"No gradients for {type(criterion).__name__}")
                    return

            except Exception as e:
                results.add_fail(test_name, f"{type(criterion).__name__} failed: {str(e)}")
                return

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_memory_usage(results: TestResults):
    """
    Test memory usage characteristics.
    """
    test_name = "memory_usage"

    try:
        model = create_simple_model(input_dim=10, hidden_dim=20, output_dim=5)
        inputs, targets = create_test_data(batch_size=32, input_dim=10, num_classes=5)
        criterion = nn.CrossEntropyLoss()

        # Start memory tracing
        tracemalloc.start()

        # Compute per-sample gradients
        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        results.add_memory(test_name, peak_mb)

        # Basic success if no exceptions
        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_computation_time(results: TestResults):
    """
    Test computation time characteristics.
    """
    test_name = "computation_time"

    try:
        model = create_simple_model(input_dim=20, hidden_dim=50, output_dim=10)
        inputs, targets = create_test_data(batch_size=64, input_dim=20, num_classes=10)
        criterion = nn.CrossEntropyLoss()

        # Warm up
        _ = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Time the computation
        start_time = time.time()
        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)
        end_time = time.time()

        duration = end_time - start_time
        results.add_timing(test_name, duration)

        # Basic success if computation completed
        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def test_integration_with_dp_pipeline(results: TestResults):
    """
    Test integration with differential privacy pipeline.
    """
    test_name = "integration_with_dp_pipeline"

    try:
        model = create_simple_model(input_dim=5, hidden_dim=10, output_dim=2)
        inputs, targets = create_test_data(batch_size=8, input_dim=5, num_classes=2)
        criterion = nn.CrossEntropyLoss()

        # Compute per-sample gradients
        per_sample_grads = _compute_per_sample_grads(model, inputs, targets, criterion)

        # Compute per-sample norms
        per_sample_norms = _compute_per_sample_norms(per_sample_grads)

        # Basic checks
        if per_sample_norms.size(0) != inputs.size(0):
            results.add_fail(test_name, f"Wrong norm batch size: {per_sample_norms.size(0)} vs {inputs.size(0)}")
            return

        if torch.any(per_sample_norms < 0):
            results.add_fail(test_name, "Negative gradient norms detected")
            return

        # Test gradient clipping simulation
        clip_norm = 1.0
        per_sample_clip_factor = (clip_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

        # Apply clipping to first parameter as test
        first_param_name = list(per_sample_grads.keys())[0]
        first_param_grad = per_sample_grads[first_param_name]
        clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, first_param_grad)

        if torch.isnan(clipped_grad).any() or torch.isinf(clipped_grad).any():
            results.add_fail(test_name, "NaN or Inf in clipped gradients")
            return

        results.add_pass(test_name)

    except Exception as e:
        results.add_fail(test_name, f"Exception: {str(e)}")


def run_all_tests():
    """
    Run the complete test suite.
    """
    print("Per-Sample Gradient Computation Validation Tests")
    print("=" * 50)

    results = TestResults()

    # Run all tests
    test_gradient_aggregation_consistency(results)
    test_individual_sample_correctness(results)
    test_numerical_stability(results)
    test_different_model_architectures(results)
    test_different_loss_functions(results)
    test_memory_usage(results)
    test_computation_time(results)
    test_integration_with_dp_pipeline(results)

    # Opacus comparison tests (if available)
    test_opacus_comparison(results)
    test_opacus_different_models(results)

    # Print summary
    results.summary()

    return results.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)