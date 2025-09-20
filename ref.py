from typing import Any
import torch
import logging
from opacus import GradSampleModule
from opacus.optimizers.optimizer import _generate_noise
from opacus.validators import ModuleValidator
from collections import OrderedDict
from src.client.scaffold import SCAFFOLDClient


class DPScaffoldWrongClient(SCAFFOLDClient):
    """DP-SCAFFOLD Wrong Client - Incorrect Use of GradSampleModule.

    This client demonstrates INCORRECT usage of GradSampleModule for educational/comparison purposes.
    It wraps the model with GradSampleModule but ignores the per-sample gradients it computes,
    instead using the same global gradient clipping and noise addition as dp_scaffold_new.py.

    This approach is WRONG because:
    1. It incurs the computational overhead of per-sample gradient computation without benefits
    2. It does not provide proper differential privacy guarantees
    3. The global clipping and noise addition approach is insufficient for DP

    This implementation is intended for comparison studies to demonstrate why proper
    per-sample gradient processing is necessary for differential privacy.
    """

    def __init__(self, **commons):
        super().__init__(**commons)

        # Initialize DP parameters from config
        self.clip_norm = self.args.dp_scaffold_wrong.clip_norm
        self.sigma = self.args.dp_scaffold_wrong.sigma

        # Initialize random generator (will be seeded when client_id is set)
        self.noise_generator = torch.Generator()

    def _should_log_stats(self):
        """Check if detailed statistics logging is enabled"""
        return getattr(self.args.dp_scaffold_wrong, 'enable_detailed_stats', False)

    def _print_tensor_stats(self, tensor_dict_or_list, prefix, client_id=None):
        """打印张量统计信息的通用函数"""
        if not self._should_log_stats():
            return

        if client_id is not None:
            header = f"[CLIENT] [ID:{client_id}] {prefix}"
        else:
            header = f"[CLIENT] {prefix}"

        if isinstance(tensor_dict_or_list, dict):
            for name, tensor in tensor_dict_or_list.items():
                if isinstance(tensor, torch.Tensor):
                    stats = {
                        'shape': tuple(tensor.shape),
                        'mean': tensor.float().mean().item(),
                        'std': tensor.float().std().item(),
                        'norm': tensor.float().norm().item()
                    }
                    logging.info(f"{header} {name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.6f}")
        elif isinstance(tensor_dict_or_list, list):
            for i, tensor in enumerate(tensor_dict_or_list):
                if isinstance(tensor, torch.Tensor):
                    stats = {
                        'shape': tuple(tensor.shape),
                        'mean': tensor.float().mean().item(),
                        'std': tensor.float().std().item(),
                        'norm': tensor.float().norm().item()
                    }
                    logging.info(f"{header} idx_{i}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.6f}")

    def _print_model_params_stats(self, model_params, prefix, client_id=None):
        """打印模型参数统计信息"""
        self._print_tensor_stats(model_params, f"{prefix} MODEL_PARAMS", client_id)

    def _print_control_variates_stats(self, c_global, c_local, prefix, client_id=None):
        """打印控制变量统计信息"""
        self._print_tensor_stats(c_global, f"{prefix} C_GLOBAL", client_id)
        self._print_tensor_stats(c_local, f"{prefix} C_LOCAL", client_id)

    def set_parameters(self, package: dict[str, Any]):
        # Wrap model with GradSampleModule BEFORE setting parameters (but will misuse it)
        if not isinstance(self.model, GradSampleModule):
            # Fix model compatibility with Opacus before wrapping
            if not ModuleValidator.is_valid(self.model):
                self.model = ModuleValidator.fix(self.model)

            original_device = self.model.device
            self.model = GradSampleModule(self.model)
            # Preserve device attribute for compatibility
            self.model.device = original_device

            # Update parameter names after wrapping for package compatibility
            self.regular_params_name = list(key for key, _ in self.model.named_parameters())

        # Create a modified package with updated parameter names for GradSampleModule
        modified_package = package.copy()
        if isinstance(self.model, GradSampleModule):
            # Add _module. prefix to parameter names to match GradSampleModule structure
            modified_regular_model_params = {}
            for orig_name, param in package["regular_model_params"].items():
                new_name = f"_module.{orig_name}"
                modified_regular_model_params[new_name] = param
            modified_package["regular_model_params"] = modified_regular_model_params

        super().set_parameters(modified_package)

        # Seed the noise generator when client_id is available
        if hasattr(self, 'noise_generator') and self.client_id is not None:
            self.noise_generator.manual_seed(self.args.common.seed + self.client_id)

        # Print received parameters statistics
        self._print_model_params_stats(package["regular_model_params"], "RECEIVED", self.client_id)
        self._print_control_variates_stats(package["c_global"], package["c_local"], "RECEIVED", self.client_id)

    def _get_clean_param_name(self, name: str) -> str:
        """Remove _module. prefix from parameter names for compatibility."""
        return name.replace("_module.", "") if name.startswith("_module.") else name

    def _clip_gradients(self):
        """Clip gradients using global L2 norm clipping.

        WRONG: This ignores the per-sample gradients computed by GradSampleModule
        and uses global clipping instead, which doesn't provide proper DP guarantees.
        """
        # Calculate total gradient norm
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # Apply global clipping
        clip_coef = self.clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

    def _add_noise_to_gradients(self):
        """Add Gaussian noise to gradients after clipping.

        WRONG: This adds the same noise to all parameters globally,
        rather than using the proper per-sample noise scaling.
        """
        # Calculate noise standard deviation
        sigma_noise = self.clip_norm * self.sigma / self.args.common.batch_size

        for param in self.model.parameters():
            if param.grad is not None:
                noise = _generate_noise(
                    std=sigma_noise,
                    reference=param.grad,
                    generator=self.noise_generator,
                    secure_mode=False
                )
                param.grad.data.add_(noise)

    def fit(self):
        """Training with DP processing followed by SCAFFOLD control variates.

        WRONG: This wastes the computational resources of GradSampleModule
        by not using the per-sample gradients it computes.
        """
        self.model.train()
        self.dataset.train()

        # Print initial control variates at start of fit
        if self._should_log_stats():
            logging.info(f"[CLIENT] [ID:{self.client_id}] FIT_START C_GLOBAL stats:")
        self._print_tensor_stats(self.c_global, "FIT_START C_GLOBAL", self.client_id)
        if self._should_log_stats():
            logging.info(f"[CLIENT] [ID:{self.client_id}] FIT_START C_LOCAL stats:")
        self._print_tensor_stats(self.c_local, "FIT_START C_LOCAL", self.client_id)

        # Print dataloader info at start
        if self._should_log_stats():
            logging.info(f"[CLIENT] [ID:{self.client_id}] FIT_START DATALOADER info:")
            logging.info(f"[CLIENT] [ID:{self.client_id}] trainloader_length={len(self.trainloader)}")
            logging.info(f"[CLIENT] [ID:{self.client_id}] dataset_size={len(self.dataset)}")
            logging.info(f"[CLIENT] [ID:{self.client_id}] batch_size={self.args.common.batch_size}")

        for epoch in range(self.args.common.local_epoch):
            if self._should_log_stats():
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} starting")

            # Print model parameters at start of each epoch
            model_params_dict = {name: param.data for name, param in self.model.named_parameters()}
            self._print_model_params_stats(model_params_dict, f"LOCAL_EPOCH_{epoch}_START", self.client_id)
            self.optimizer.zero_grad()

            # Clear any existing grad_sample (though we won't use them)
            for param in self.model.parameters():
                if hasattr(param, 'grad_sample'):
                    param.grad_sample = None
                    
            x, y = self.get_data_batch()

            # Print batch information for debugging
            if self._should_log_stats():
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} BATCH_INFO:")
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} batch_size={x.shape[0]}, x_mean={x.mean().item():.6f}, x_std={x.std().item():.6f}")
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} y_labels={y.cpu().numpy().tolist()}")
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} x_hash={hash(x.cpu().detach().numpy().tobytes())}")

            logits = self.model(x)
            loss = self.criterion(logits, y)
            if self._should_log_stats():
                logging.info(f"[CLIENT] [ID:{self.client_id}] LOCAL_EPOCH_{epoch} loss={loss.item():.6f}")
            # self.optimizer.zero_grad()

            # # Clear any existing grad_sample (though we won't use them)
            # for param in self.model.parameters():
            #     if hasattr(param, 'grad_sample'):
            #         param.grad_sample = None

            loss.backward()

            # Print gradients before clipping
            grad_stats = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats[f"{name}_grad"] = param.grad.data
            self._print_tensor_stats(grad_stats, f"LOCAL_EPOCH_{epoch}_GRAD_BEFORE_CLIP", self.client_id)

            # Step 1: Apply global gradient clipping (WRONG approach)
            # self._clip_gradients()

            # Print gradients after clipping
            grad_stats_clipped = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats_clipped[f"{name}_grad_clipped"] = param.grad.data
            self._print_tensor_stats(grad_stats_clipped, f"LOCAL_EPOCH_{epoch}_GRAD_AFTER_CLIP", self.client_id)

            # Step 2: Add DP noise (WRONG approach)
            # self._add_noise_to_gradients()

            # Print gradients after noise
            grad_stats_noisy = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats_noisy[f"{name}_grad_noisy"] = param.grad.data
            self._print_tensor_stats(grad_stats_noisy, f"LOCAL_EPOCH_{epoch}_GRAD_AFTER_NOISE", self.client_id)

            # Step 3: Apply SCAFFOLD control variate correction after DP processing
            for param, c, c_i in zip(
                self.model.parameters(), self.c_global, self.c_local
            ):
                if param.requires_grad:
                    param.grad.data += (c - c_i).to(self.device)

            # Print gradients after SCAFFOLD correction
            grad_stats_scaffold = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_stats_scaffold[f"{name}_grad_scaffold"] = param.grad.data
            self._print_tensor_stats(grad_stats_scaffold, f"LOCAL_EPOCH_{epoch}_GRAD_AFTER_SCAFFOLD", self.client_id)

            self.optimizer.step()

            # Print model parameters after optimizer step
            model_params_dict_after = {name: param.data for name, param in self.model.named_parameters()}
            self._print_model_params_stats(model_params_dict_after, f"LOCAL_EPOCH_{epoch}_END", self.client_id)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train(self, server_package: dict[str, Any]):
        """Override parent train method to handle GradSampleModule parameter naming."""
        self.set_parameters(server_package)
        self.train_with_eval()

        with torch.no_grad():
            self.y_delta = []
            c_plus = []
            self.c_delta = []

            model_params = self.model.state_dict()
            for key in server_package["regular_model_params"].keys():
                # Handle parameter name mismatch due to GradSampleModule
                model_key = key
                if isinstance(self.model, GradSampleModule):
                    # Try with _module. prefix first
                    if f"_module.{key}" in model_params:
                        model_key = f"_module.{key}"
                    elif key in model_params:
                        model_key = key
                    else:
                        continue

                x, y_i = server_package["regular_model_params"][key], model_params[model_key]
                self.y_delta.append(y_i.cpu() - x)

            coef = 1 / (self.local_epoch * self.args.optimizer.lr)
            for c, c_i, y_del in zip(self.c_global, self.c_local, self.y_delta):
                c_plus.append(c_i - c - coef * y_del)

            for c_p, c_l in zip(c_plus, self.c_local):
                self.c_delta.append(c_p - c_l)

            self.c_local = c_plus

        # Print training results statistics
        self._print_tensor_stats(self.y_delta, "TRAIN_RESULT Y_DELTA", self.client_id)
        self._print_tensor_stats(self.c_delta, "TRAIN_RESULT C_DELTA", self.client_id)
        self._print_control_variates_stats(self.c_global, self.c_local, "TRAIN_RESULT", self.client_id)

        return self.package()