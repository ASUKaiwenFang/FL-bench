from typing import Any, Iterator
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from src.client.fedavg import FedAvgClient


def _compute_per_sample_grads(model, inputs, targets, criterion):
    """
    Compute per-sample gradients using torch.func.grad and vmap.

    Args:
        model: The neural network model
        inputs: Input batch tensor [batch_size, ...]
        targets: Target batch tensor [batch_size, ...]
        criterion: Loss function

    Returns:
        dict: {param_name: per_sample_grad_tensor} where per_sample_grad_tensor
              has shape [batch_size, *param_shape]
    """
    def compute_loss_for_sample(params, buffers, sample_input, sample_target):
        predictions = torch.func.functional_call(model, (params, buffers), sample_input.unsqueeze(0))
        return criterion(predictions, sample_target.unsqueeze(0))

    params = {name: param for name, param in model.named_parameters()}
    buffers = {name: buffer for name, buffer in model.named_buffers()}

    # Use vmap to vectorize over the batch dimension
    per_sample_grads = torch.vmap(
        torch.func.grad(compute_loss_for_sample, argnums=0),
        in_dims=(None, None, 0, 0)
    )(params, buffers, inputs, targets)

    return per_sample_grads


def _generate_gaussian_noise(tensor, std, device):
    """
    Generate Gaussian noise with specified standard deviation.

    Args:
        tensor: Reference tensor for shape and dtype
        std: Standard deviation of the noise
        device: Device to generate noise on

    Returns:
        torch.Tensor: Noise tensor with same shape as input tensor
    """
    return torch.randn_like(tensor, device=device) * std


def _compute_per_sample_norms(per_sample_grads):
    """
    Compute L2 norms of per-sample gradients.

    Args:
        per_sample_grads: dict of {param_name: per_sample_grad_tensor}

    Returns:
        torch.Tensor: Per-sample gradient norms [batch_size]
    """
    batch_size = None
    per_param_norms = []

    for param_name, grad_tensor in per_sample_grads.items():
        if batch_size is None:
            batch_size = grad_tensor.shape[0]

        # Reshape to [batch_size, -1] and compute norm for each sample
        param_norms = grad_tensor.reshape(batch_size, -1).norm(2, dim=-1)
        per_param_norms.append(param_norms)

    if not per_param_norms:
        return torch.zeros(0)

    # Stack parameter norms and compute total norm per sample
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
    return per_sample_norms


class DPFedAvgLocalClient(FedAvgClient):
    """Local Differential Privacy FedAvg Client with Per-Sample Gradients.

    This client implements local differential privacy using torch.func.grad and vmap
    for efficient per-sample gradient computation. This approach provides
    excellent performance and compatibility with PyTorch 2.0+ compilation features.
    Each sample's gradients are computed and clipped independently,
    then averaged and noised before parameter updates.

    Supports two algorithm variants:
    - step_noise: Add noise to gradients at each training step (current implementation)
    - last_noise: Add noise to parameter differences after training completion
    """
    
    # Define algorithm variant constants
    ALGORITHM_VARIANTS = {
        'last_noise': 1,
        'step_noise': 2
    }
    
    def __init__(self, **commons):
        super().__init__(**commons)
        self.iter_trainloader = None

        # Initialize DP parameters
        self.clip_norm = self.args.dp_fedavg_local.clip_norm
        self.sigma = self.args.dp_fedavg_local.sigma
        self.sigma_dp = None
        self.model_params_diff = None

        # Support string or numeric configuration
        variant_config = getattr(self.args.dp_fedavg_local, 'algorithm_variant', 'step_noise')
        if isinstance(variant_config, str):
            self.algorithm_variant = self.ALGORITHM_VARIANTS[variant_config]
        else:
            self.algorithm_variant = variant_config
        
    
    def set_parameters(self, package: dict[str, Any]):
        self.iter_trainloader = iter(self.trainloader)

        # Reset DP processed difference for new training round
        self.model_params_diff = None

        super().set_parameters(package)

    
    def fit(self):
        """Train the model with local differential privacy using per-sample gradients.

        Supports two algorithm variants:
        - step_noise: Add noise to gradients at each training step
        - last_noise: Add noise to parameter differences after training completion
        """
        if self.algorithm_variant == 1:  # last_noise
            self._last_noise_training()
        elif self.algorithm_variant == 2:  # step_noise
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
                noise = _generate_gaussian_noise(
                    param_diff, sigma_dp, param.device
                )
                noisy_diff = param_diff + noise
                self.model_params_diff[name] = noisy_diff.clone().cpu()

    
                
    def _clip_and_add_noise(self, inputs, targets):
        """
        Clip per-sample gradients and add noise.

        Args:
            inputs: Input batch tensor
            targets: Target batch tensor
        """
        # Compute per-sample gradients
        per_sample_grads = _compute_per_sample_grads(
            self.model, inputs, targets, self.criterion
        )

        # Compute per-sample gradient norms
        per_sample_norms = _compute_per_sample_norms(per_sample_grads)

        if len(per_sample_norms) == 0:
            return

        # Compute clipping factors
        per_sample_clip_factor = (
            self.clip_norm / (per_sample_norms + 1e-6)
        ).clamp(max=1.0)

        # Calculate DP noise standard deviation: σ_DP = C * σ_g / b
        batch_size = per_sample_norms.size(0)
        sigma_dp = self.clip_norm * self.sigma / batch_size
        self.sigma_dp = sigma_dp

        # Apply clipping and noise to each parameter
        for param_name, param in self.model.named_parameters():
            if param_name in per_sample_grads:
                per_sample_grad = per_sample_grads[param_name]

                # Clip per-sample gradients
                clipped_grad = torch.einsum("i,i...", per_sample_clip_factor, per_sample_grad)

                # Add Gaussian noise
                noisy_grad = clipped_grad + _generate_gaussian_noise(
                    clipped_grad, sigma_dp, self.device
                )

                # Set the parameter's gradient
                param.grad = noisy_grad
    
    
    def package(self):
        """Package client data including DP parameters.

        Optimized implementation that avoids redundant calculations
        based on the algorithm variant.
        """

        # Common package components
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            model_params_diff=self.model_params_diff,
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {} if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
            sigma_dp=self.sigma_dp
        )

        return client_package
    
    
    def get_data_batch(self):
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)

