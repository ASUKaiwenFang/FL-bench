"""
James-Stein Estimator (JSE) utilities for differential privacy in federated learning.

This module provides enhanced JSE processing capabilities specifically designed for
FL-bench's differential privacy implementations. It offers numerical stability,
performance optimization, and seamless integration with Opacus-based DP mechanisms.
"""

import torch
from typing import Dict, List


class JSEProcessor:
    """James-Stein Estimator processor with enhanced numerical stability.

    This class provides static methods for applying James-Stein shrinkage to tensors,
    specifically optimized for differential privacy applications in federated learning.
    All methods are designed to work seamlessly with Opacus GradSampleModule and
    FL-bench's DP-FedAvg implementations.
    """

    # Numerical stability constants
    DEFAULT_EPS = 1e-12
    DEFAULT_MAX_SHRINKAGE = 0.99

    @staticmethod
    def _validate_noise_variance(noise_variance: float) -> None:
        """Validate noise variance parameter."""
        if noise_variance < 0:
            raise ValueError(f"noise_variance must be non-negative, got {noise_variance}")

    @staticmethod
    def _validate_k_factor(k_factor: int) -> None:
        """Validate k_factor parameter."""
        if k_factor < 0:
            raise ValueError(f"k_factor must be >= 0, got {k_factor}")

    @staticmethod
    def _check_tensor_applicability(tensor: torch.Tensor) -> bool:
        """Check if JSE is applicable to the tensor."""
        d = tensor.numel()
        return d > 2

    @staticmethod
    def _is_numerically_stable(tensor_norm_sq: torch.Tensor) -> bool:
        """Check if tensor norm is numerically stable."""
        return tensor_norm_sq >= JSEProcessor.DEFAULT_EPS

    @staticmethod
    def apply_jse_shrinkage(
        tensor: torch.Tensor,
        noise_variance: float,
        k_factor: int = 1
    ) -> torch.Tensor:
        """
        Apply James-Stein shrinkage to a tensor with enhanced numerical stability.

        This method implements the James-Stein estimator shrinkage formula:
        shrinkage_factor = max(0, 1 - (d-2) * k_factor * σ² / ||tensor||²)
        where d is the effective dimension of the tensor.

        Args:
            tensor: Input tensor to apply shrinkage to
            noise_variance: Noise variance σ² from DP mechanism
            k_factor: Accumulation factor for multi-step scenarios (default: 1)

        Returns:
            Tensor after applying James-Stein shrinkage
        """
        # Input validation
        JSEProcessor._validate_noise_variance(noise_variance)
        JSEProcessor._validate_k_factor(k_factor)

        # Handle empty tensors
        if tensor.numel() == 0:
            return tensor

        # Check if JSE is applicable
        if not JSEProcessor._check_tensor_applicability(tensor):
            return tensor

        # Compute tensor norm squared using stable method
        tensor_norm_sq = torch.sum(tensor ** 2)

        # Numerical stability check
        if not JSEProcessor._is_numerically_stable(tensor_norm_sq):
            return tensor

        # Calculate shrinkage factor with numerical stability
        d = tensor.numel()
        shrinkage_numerator = (d - 2) * k_factor * noise_variance
        shrinkage_factor = shrinkage_numerator / tensor_norm_sq

        # Apply bounds to shrinkage factor
        shrinkage_factor = torch.clamp(shrinkage_factor, 0.0, JSEProcessor.DEFAULT_MAX_SHRINKAGE)

        # Apply shrinkage: result = (1 - shrinkage_factor) * tensor
        return (1.0 - shrinkage_factor) * tensor

    @staticmethod
    def apply_layerwise_jse_to_gradients(
        model_parameters: List[torch.nn.Parameter],
        noise_variance: float
    ) -> None:
        """
        Apply layerwise JSE to model parameter gradients in-place.

        This method directly modifies the .grad attribute of model parameters,
        applying James-Stein shrinkage to each gradient tensor independently.
        Designed for use in gradient-level JSE variants (e.g., step_noise_step_jse).

        Args:
            model_parameters: List of model parameters with gradients
            noise_variance: Noise variance σ² from DP mechanism
        """
        JSEProcessor._validate_noise_variance(noise_variance)

        for param in model_parameters:
            if param.grad is not None:
                # Apply JSE shrinkage to gradient in-place
                param.grad.data = JSEProcessor.apply_jse_shrinkage(
                    param.grad.data, noise_variance
                )


    @staticmethod
    def apply_global_jse_to_gradients(
        model_parameters: List[torch.nn.Parameter],
        noise_variance: float
    ) -> None:
        """
        Apply global JSE to model parameter gradients in-place.

        This method computes global statistics across all gradients, calculates a unified
        shrinkage factor, and applies it directly to each gradient tensor. This implementation
        avoids intermediate data structures for optimal performance.

        Mathematical formula: shrinkage_factor = (d-2) * noise_variance / ||gradients||²
        Result: grad := (1 - shrinkage_factor) * grad for each gradient

        Args:
            model_parameters: List of model parameters with gradients
            noise_variance: Noise variance σ² from DP mechanism
        """
        # Input validation
        JSEProcessor._validate_noise_variance(noise_variance)

        # Step 1: Compute global statistics directly without intermediate structures
        total_norm_sq = torch.tensor(0.0)
        total_elements = 0
        grad_params = []

        for param in model_parameters:
            if param.grad is not None and param.grad.numel() > 0:
                total_norm_sq += torch.sum(param.grad.data ** 2)
                total_elements += param.grad.numel()
                grad_params.append(param)

        # Handle edge cases
        if total_elements == 0:
            return
        if total_elements <= 2:
            # JSE is not applicable for d <= 2, return without modification
            return

        # Numerical stability check
        if not JSEProcessor._is_numerically_stable(total_norm_sq):
            return

        # Step 2: Calculate global shrinkage factor with numerical stability
        shrinkage_numerator = (total_elements - 2) * noise_variance
        shrinkage_factor = shrinkage_numerator / total_norm_sq

        # # Apply bounds to shrinkage factor
        # shrinkage_factor = torch.clamp(shrinkage_factor, 0.0, JSEProcessor.DEFAULT_MAX_SHRINKAGE)

        # Step 3: Apply shrinkage to each gradient directly in-place
        shrinkage_multiplier = 1.0 - shrinkage_factor

        for param in grad_params:
            param.grad.data *= shrinkage_multiplier

    @staticmethod
    def apply_layerwise_jse_to_parameter_diff(
        param_diff_dict: Dict[str, torch.Tensor],
        noise_variance: float,
        k_factor: int = 1
    ) -> None:
        """
        Apply layerwise JSE to parameter differences in-place.

        This method applies James-Stein shrinkage to each parameter difference tensor
        independently, computing JSE shrinkage factors based on each layer's own
        dimensions and norm. The input dictionary is modified in-place.
        Typically used for most JSE variants where per-layer shrinkage is desired.

        Args:
            param_diff_dict: Dictionary of parameter difference tensors (modified in-place)
            noise_variance: Noise variance σ² from DP mechanism
            k_factor: Accumulation factor for multi-step scenarios (default: 1)
        """
        for key, param_tensor in param_diff_dict.items():
            param_diff_dict[key] = JSEProcessor.apply_jse_shrinkage(
                param_tensor, noise_variance, k_factor
            )

    @staticmethod
    def apply_global_jse_to_parameter_diff(
        param_diff_dict: Dict[str, torch.Tensor],
        noise_variance: float,
        k_factor: int = 1
    ) -> None:
        """
        Apply global JSE to parameter differences in-place.

        This method computes global statistics across all parameter difference tensors
        without concatenation, calculates a unified shrinkage factor, and applies it
        directly to each parameter tensor. The input dictionary is modified in-place.
        This optimized implementation avoids intermediate tensor allocations while
        maintaining mathematical equivalence. Typically used for last_noise_server_jse
        variant where unified shrinkage across all parameters is desired.

        Args:
            param_diff_dict: Dictionary of parameter difference tensors (modified in-place)
            noise_variance: Noise variance σ² from DP mechanism
            k_factor: Accumulation factor for multi-step scenarios (default: 1)
        """
        # Input validation
        JSEProcessor._validate_noise_variance(noise_variance)
        JSEProcessor._validate_k_factor(k_factor)

        # Step 1: Compute global statistics without concatenation
        total_norm_sq = torch.tensor(0.0)
        total_elements = 0

        for tensor in param_diff_dict.values():
            if tensor.numel() == 0:
                continue
            total_norm_sq += torch.sum(tensor ** 2)
            total_elements += tensor.numel()

        # Handle edge cases
        if total_elements == 0:
            return
        if total_elements <= 2:
            # JSE is not applicable for d <= 2, return without modification
            return

        # Numerical stability check
        if not JSEProcessor._is_numerically_stable(total_norm_sq):
            return

        # Step 2: Calculate global shrinkage factor with numerical stability
        shrinkage_numerator = (total_elements - 2) * k_factor * noise_variance
        shrinkage_factor = shrinkage_numerator / total_norm_sq

        # # Apply bounds to shrinkage factor
        # shrinkage_factor = torch.clamp(shrinkage_factor, 0.0, JSEProcessor.DEFAULT_MAX_SHRINKAGE)

        # Step 3: Apply shrinkage to each parameter individually in-place
        shrinkage_multiplier = 1.0 - shrinkage_factor

        for key, tensor in param_diff_dict.items():
            if tensor.numel() > 0:
                # Apply shrinkage: result = (1 - shrinkage_factor) * tensor
                param_diff_dict[key] = tensor * shrinkage_multiplier



