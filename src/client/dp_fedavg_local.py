from typing import Any, Iterator
from copy import deepcopy
from enum import Enum
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from src.client.fedavg import FedAvgClient
from src.utils.dp_mechanisms import compute_per_sample_grads, compute_per_sample_norms


class AlgorithmVariant(Enum):
    """Algorithm variants for DP-FedAvg Local implementation."""
    LAST_NOISE = 1
    STEP_NOISE = 2



class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with Per-Sample Gradients.

    This client implements local differential privacy using torch.func.grad_and_value and vmap
    for efficient per-sample gradient computation. This approach provides
    excellent performance and compatibility with PyTorch 2.0+ compilation features.
    Each sample's gradients are computed and clipped independently,
    then averaged and noised before parameter updates.

    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step (current implementation)
    - last_noise: Add noise to parameter differences after training completion
    """

    # Configuration constants
    numerical_epsilon = 1e-6

    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader = None

        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        self.sigma_dp = None
        self.model_params_diff = None

        # Validate DP configuration
        self._validate_dp_config()

        # Support string or numeric configuration with enum
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = getattr(AlgorithmVariant, variant_config.upper())
        else:
            # Legacy numeric support
            self.algorithm_variant = AlgorithmVariant(variant_config)

        # Pre-compile gradient computation for PyTorch 2.0+
        self._compiled_grad_fn = None
        self._setup_compiled_gradient_function()

        # Cache for model parameters and buffers to avoid repeated extraction
        self._cached_params = None
        self._cached_buffers = None
        self._params_cache_valid = False


        # Chunking configuration for large models
        self._max_params_per_chunk = getattr(self.args.dp_fedavg_local, 'max_params_per_chunk', 10_000_000)
        self._enable_chunking = getattr(self.args.dp_fedavg_local, 'enable_chunking', True)

    def _validate_dp_config(self):
        """Validate differential privacy configuration parameters.

        Raises:
            ValueError: If any DP parameter is invalid
        """
        if self.clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {self.clip_norm}")

        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

        # Check for reasonable bounds
        if self.clip_norm > 100:
            import warnings
            warnings.warn(f"clip_norm={self.clip_norm} is unusually large")

        if self.sigma > 10:
            import warnings
            warnings.warn(f"sigma={self.sigma} is unusually large and may hurt utility")

    def _setup_compiled_gradient_function(self):
        """Setup pre-compiled gradient function for better performance.

        This method sets up a compiled version of the per-sample gradient computation
        for PyTorch 2.0+ when available and beneficial.
        """
        try:
            # Check PyTorch version
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version >= (2, 0) and hasattr(torch, 'compile'):

                def compiled_grad_fn(model, inputs, targets, criterion, cached_params=None, cached_buffers=None):
                    """Compiled version of per-sample gradient computation."""
                    return compute_per_sample_grads(model, inputs, targets, criterion, cached_params, cached_buffers)

                # Use torch.compile for optimization
                self._compiled_grad_fn = torch.compile(compiled_grad_fn, mode='reduce-overhead')
            else:
                self._compiled_grad_fn = compute_per_sample_grads

        except Exception as e:
            # Fall back to uncompiled version on any compilation error
            import warnings
            warnings.warn(f"Failed to compile gradient function, using fallback: {e}")
            self._compiled_grad_fn = compute_per_sample_grads

    def _get_performance_metrics(self):
        """Get performance metrics for monitoring.

        Returns:
            dict: Performance metrics including memory usage and timing info
        """
        metrics = {
            'clip_norm': self.clip_norm,
            'sigma': self.sigma,
            'sigma_dp': self.sigma_dp,
            'algorithm_variant': self.algorithm_variant.name,
            'compiled_gradients': self._compiled_grad_fn != compute_per_sample_grads
        }

        # Add memory info if available
        if torch.cuda.is_available():
            metrics.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated(self.device),
                'gpu_memory_cached': torch.cuda.memory_reserved(self.device)
            })

        return metrics

    def _get_cached_model_params(self):
        """Get cached model parameters and buffers to avoid repeated extraction.

        Returns:
            tuple: (params_dict, buffers_dict)
        """
        if not self._params_cache_valid or self._cached_params is None:
            self._cached_params = {name: param for name, param in self.model.named_parameters()}
            self._cached_buffers = {name: buffer for name, buffer in self.model.named_buffers()}
            self._params_cache_valid = True

        return self._cached_params, self._cached_buffers

    def _invalidate_params_cache(self):
        """Invalidate the parameters cache when model structure changes."""
        self._params_cache_valid = False


    def set_parameters(self, package: dict[str, Any]):
        self.iter_trainloader = iter(self.trainloader)

        # Reset DP processed difference for new training round
        self.model_params_diff = None

        # Invalidate parameter cache since model parameters are being updated
        self._invalidate_params_cache()

        super().set_parameters(package)

    
    def fit(self):
        """Train the model with local differential privacy using per-sample gradients.

        Supports two algorithm variants:
        - step_noise: Add noise to gradients at each training step
        - last_noise: Add noise to parameter differences after training completion
        """
        if self.algorithm_variant == AlgorithmVariant.LAST_NOISE:
            self._last_noise_training()
        elif self.algorithm_variant == AlgorithmVariant.STEP_NOISE:
            self._step_noise_training()
        else:
            raise ValueError(f"Unknown algorithm variant: {self.algorithm_variant}")
    
    def _step_noise_training(self):
        """Gradient-level noise addition using per-sample gradients.

        This method implements the per-sample DP-SGD algorithm using torch.func:
        1. Compute per-sample gradients using torch.func.grad + vmap
        2. Clip each sample's gradients independently
        3. Average the clipped gradients and add noise
        4. Apply noisy gradients to parameters
        """
        self.model.train()
        self.dataset.train()

        for _ in range(self.local_epoch):
            x, y = self.get_data_batch()

            self.optimizer.zero_grad()
            self._clip_and_add_noise(x, y)
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._step_noise_post_processing()
    
    def _last_noise_training(self):
        """Parameter-level noise addition.

        Train without noise, then add noise to parameter differences.
        Uses noise standard deviation: σ_DP = C * K * η_l * σ_g / b
        """
        self.model.train()
        self.dataset.train()


        # Standard training without noise
        for _ in range(self.local_epoch):
            self.optimizer.zero_grad()
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self._last_noise_post_processing()

    def _batch_process_gradients(self, per_sample_grads, per_sample_clip_factor, sigma_dp, device):
        """Advanced vectorized gradient processing with memory optimization.

        This implementation uses tensor concatenation for true batch processing,
        reducing memory fragmentation and improving GPU utilization.

        Args:
            per_sample_grads: Dictionary of per-sample gradients
            per_sample_clip_factor: Clipping factors for each sample
            sigma_dp: Noise standard deviation
            device: Device for computation

        Returns:
            dict: Processed gradients ready for parameter assignment
        """
        if not per_sample_grads:
            return {}

        # Try advanced batch processing for better performance
        try:
            return self._advanced_batch_process(per_sample_grads, per_sample_clip_factor, sigma_dp, device)
        except (RuntimeError, torch.OutOfMemoryError) as e:
            # Fallback to per-parameter processing if memory issues
            import warnings
            warnings.warn(f"Falling back to per-parameter processing due to: {e}")
            return self._fallback_batch_process(per_sample_grads, per_sample_clip_factor, sigma_dp, device)

    def _advanced_batch_process(self, per_sample_grads, per_sample_clip_factor, sigma_dp, device):
        """Advanced batch processing by concatenating all gradients with chunking support."""
        # Flatten all gradients for true batch processing
        flattened_grads = []
        param_shapes = []
        param_names = []
        total_params = 0

        for param_name, per_sample_grad in per_sample_grads.items():
            batch_size = per_sample_grad.size(0)
            param_shape = per_sample_grad.shape[1:]  # Shape without batch dimension

            # Flatten the parameter gradients: [batch_size, param_dims...] -> [batch_size, -1]
            flattened_grad = per_sample_grad.reshape(batch_size, -1)
            flattened_grads.append(flattened_grad)
            param_shapes.append(param_shape)
            param_names.append(param_name)
            total_params += flattened_grad.size(1)

        # Check if chunking is needed
        if self._enable_chunking and total_params > self._max_params_per_chunk:
            return self._chunked_batch_process(flattened_grads, param_shapes, param_names,
                                             per_sample_clip_factor, sigma_dp, device)

        # Standard full batch processing
        # Concatenate all gradients: [batch_size, total_params]
        all_grads = torch.cat(flattened_grads, dim=1)

        # Apply clipping to the entire batch at once
        clip_factors_expanded = per_sample_clip_factor.unsqueeze(1)  # [batch_size, 1]
        clipped_all_grads = all_grads * clip_factors_expanded

        # Add Gaussian noise directly for reproducibility
        noise = torch.randn_like(clipped_all_grads, device=device) * sigma_dp
        noisy_all_grads = clipped_all_grads + noise

        # Split back to individual parameters
        processed_grads = {}
        start_idx = 0

        for i, (param_name, param_shape) in enumerate(zip(param_names, param_shapes)):
            param_size = torch.prod(torch.tensor(param_shape)).item()
            end_idx = start_idx + param_size

            # Extract this parameter's gradients and reshape back
            param_grads = noisy_all_grads[:, start_idx:end_idx]
            param_grads = param_grads.reshape(per_sample_clip_factor.size(0), *param_shape)
            processed_grads[param_name] = param_grads

            start_idx = end_idx

        return processed_grads

    def _chunked_batch_process(self, flattened_grads, param_shapes, param_names,
                              per_sample_clip_factor, sigma_dp, device):
        """Process gradients in chunks to handle very large models."""
        processed_grads = {}
        current_chunk = []
        current_shapes = []
        current_names = []
        current_chunk_size = 0

        def process_current_chunk():
            if not current_chunk:
                return

            # Concatenate current chunk
            chunk_grads = torch.cat(current_chunk, dim=1)

            # Apply clipping and noise
            clip_factors_expanded = per_sample_clip_factor.unsqueeze(1)
            clipped_chunk = chunk_grads * clip_factors_expanded

            noise = torch.randn_like(clipped_chunk, device=device) * sigma_dp

            noisy_chunk = clipped_chunk + noise

            # Split back to parameters
            start_idx = 0
            for param_name, param_shape in zip(current_names, current_shapes):
                param_size = torch.prod(torch.tensor(param_shape)).item()
                end_idx = start_idx + param_size

                param_grads = noisy_chunk[:, start_idx:end_idx]
                param_grads = param_grads.reshape(per_sample_clip_factor.size(0), *param_shape)
                processed_grads[param_name] = param_grads

                start_idx = end_idx

            # Clear current chunk
            current_chunk.clear()
            current_shapes.clear()
            current_names.clear()

        # Process gradients in chunks
        for i, (grad, shape, name) in enumerate(zip(flattened_grads, param_shapes, param_names)):
            param_size = grad.size(1)

            # Check if adding this parameter would exceed chunk size
            if current_chunk_size + param_size > self._max_params_per_chunk and current_chunk:
                process_current_chunk()
                current_chunk_size = 0

            current_chunk.append(grad)
            current_shapes.append(shape)
            current_names.append(name)
            current_chunk_size += param_size

        # Process remaining chunk
        process_current_chunk()

        return processed_grads

    def _fallback_batch_process(self, per_sample_grads, per_sample_clip_factor, sigma_dp, device):
        """Fallback per-parameter processing for memory-constrained scenarios."""
        processed_grads = {}

        for param_name, per_sample_grad in per_sample_grads.items():
            # Vectorized clipping using optimized tensor multiplication
            clip_shape = [per_sample_clip_factor.size(0)] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)

            # Add Gaussian noise directly for reproducibility
            noise = torch.randn_like(clipped_grad, device=device) * sigma_dp
            noisy_grad = clipped_grad + noise

            processed_grads[param_name] = noisy_grad

        return processed_grads

    def _vectorized_clip_gradients(self, per_sample_grads, per_sample_norms, clip_norm):
        """Vectorized gradient clipping implementation.

        Args:
            per_sample_grads: Dictionary of per-sample gradients
            per_sample_norms: L2 norms of per-sample gradients
            clip_norm: Clipping threshold

        Returns:
            tuple: (clipped_grads_dict, clip_factors)
        """
        # Compute clipping factors in batch
        per_sample_clip_factor = (clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0)

        clipped_grads = {}
        for param_name, per_sample_grad in per_sample_grads.items():
            # Apply clipping vectorially
            clip_shape = [per_sample_clip_factor.size(0)] + [1] * (per_sample_grad.ndim - 1)
            clipped_grad = per_sample_grad * per_sample_clip_factor.view(clip_shape)
            clipped_grads[param_name] = clipped_grad

        return clipped_grads, per_sample_clip_factor

    @torch.no_grad()
    def _step_noise_post_processing(self):
        """Post-processing for step_noise variant: calculate DP-processed parameter differences."""
        self.model_params_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                self.model_params_diff[name] = param_diff.clone().cpu()

    @torch.no_grad()
    def _last_noise_post_processing(self):
        """Post-processing for last_noise variant: integrated parameter difference calculation and noise addition."""
        # Use configured batch size for noise calculation
        batch_size = self.args.common.batch_size

        # σ_DP = C * K * η_l * σ_g / b
        sigma_dp = self.clip_norm * self.local_epoch * self.args.optimizer.lr * self.sigma / batch_size
        self.sigma_dp = sigma_dp

        # Calculate noisy parameter differences and store them
        self.model_params_diff = {}

        for name, param in self.model.named_parameters():
            if name in self.regular_model_params:
                param_diff = param.data - self.regular_model_params[name].to(param.device)
                # Generate Gaussian noise (inlined for efficiency)
                noise = torch.randn_like(param_diff, device=param.device) * sigma_dp
                noisy_diff = param_diff + noise
                self.model_params_diff[name] = noisy_diff.clone().cpu()

    
                
    def _clip_and_add_noise(self, inputs, targets):
        """Clip per-sample gradients and add noise with optimized vectorized processing.

        This method implements the core DP-SGD algorithm with several performance optimizations:
        - Uses compiled gradient functions when available (PyTorch 2.0+)
        - Vectorized gradient clipping and noise addition
        - Dynamic device detection for consistency
        - Batch processing for improved efficiency

        Args:
            inputs: Input batch tensor [batch_size, ...]
            targets: Target batch tensor [batch_size, ...]
        """
        # Dynamic device detection to ensure data consistency
        device = inputs.device if hasattr(inputs, 'device') else self.device
        if device != self.device:
            import warnings
            warnings.warn(f"Input device {device} differs from model device {self.device}")

        # Get cached parameters to avoid repeated extraction
        cached_params, cached_buffers = self._get_cached_model_params()

        # Compute per-sample gradients and losses using compiled function if available
        if self._compiled_grad_fn is not None:
            per_sample_grads, per_sample_losses = self._compiled_grad_fn(
                self.model, inputs, targets, self.criterion, cached_params, cached_buffers
            )
        else:
            per_sample_grads, per_sample_losses = compute_per_sample_grads(
                self.model, inputs, targets, self.criterion, cached_params, cached_buffers
            )

        # Compute per-sample gradient norms
        per_sample_norms = compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Calculate DP noise standard deviation: σ_DP = C * σ_g / b
        batch_size = per_sample_norms.size(0)
        sigma_dp = self.clip_norm * self.sigma / batch_size
        self.sigma_dp = sigma_dp

        # Use vectorized gradient processing for improved performance
        processed_grads = self._batch_process_gradients(
            per_sample_grads,
            (self.clip_norm / (per_sample_norms + self.numerical_epsilon)).clamp(max=1.0),
            sigma_dp,
            device
        )

        # Assign processed gradients to model parameters (average over batch)
        for param_name, param in self.model.named_parameters():
            if param_name in processed_grads:
                param.grad = processed_grads[param_name].mean(dim=0)
    
    
    def package(self):
        """Package client data including DP parameters.

        Optimized implementation that avoids redundant calculations
        based on the algorithm variant.
        """

        # Optimized package components with conditional copying
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            model_params_diff=self.model_params_diff,
            sigma_dp=self.sigma_dp
        )

        # Only copy personal parameters if they exist
        if self.personal_params_name:
            model_params = self.model.state_dict()
            client_package['personal_model_params'] = {
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            }
        else:
            client_package['personal_model_params'] = {}

        # Conditional optimizer state copying based on reset setting
        if not self.args.common.reset_optimizer_on_global_epoch:
            client_package['optimizer_state'] = deepcopy(self.optimizer.state_dict())
        else:
            client_package['optimizer_state'] = {}

        # Conditional scheduler state copying
        if self.lr_scheduler is not None and not self.args.common.reset_optimizer_on_global_epoch:
            client_package['lr_scheduler_state'] = deepcopy(self.lr_scheduler.state_dict())
        else:
            client_package['lr_scheduler_state'] = {}

        return client_package
    
    
    def get_data_batch(self):
        """Get a batch of data with improved validation and error handling."""
        max_retry_attempts = 3
        retry_count = 0

        while retry_count < max_retry_attempts:
            try:
                x, y = next(self.iter_trainloader)

                # Enhanced batch size validation
                if len(x) <= 1:
                    import warnings
                    warnings.warn(f"Batch size {len(x)} is too small for DP, retrying...")
                    if retry_count < max_retry_attempts - 1:
                        x, y = next(self.iter_trainloader)
                        retry_count += 1
                        continue
                    else:
                        raise ValueError("Unable to get batch with size > 1 after retries")

                # Validate data consistency
                if len(x) != len(y):
                    raise ValueError(f"Input-target size mismatch: {len(x)} vs {len(y)}")

                return x.to(self.device), y.to(self.device)

            except StopIteration:
                self.iter_trainloader = iter(self.trainloader)
                retry_count += 1
                if retry_count >= max_retry_attempts:
                    raise RuntimeError("Failed to get data batch after maximum retries")
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retry_attempts:
                    raise RuntimeError(f"Failed to get data batch: {e}")
                import warnings
                warnings.warn(f"Error getting batch, retrying: {e}")

        raise RuntimeError("Unexpected error in get_data_batch")

