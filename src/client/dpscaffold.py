"""
DP-SCAFFOLD Client implementation.
Combines SCAFFOLD's control variate mechanism with differential privacy.
"""
import logging
from copy import deepcopy
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from src.client.dpfedavg import DPFedAvgClient


class DPSCAFFOLDClient(DPFedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # SCAFFOLD control variates
        self.iter_trainloader: Iterator[DataLoader]
        self.c_local: list[torch.Tensor]
        self.c_global: list[torch.Tensor]
        self.y_delta: list[torch.Tensor]
        self.c_delta: list[torch.Tensor]

    def set_parameters(self, package: dict[str, Any]):
        """Set parameters including control variates."""
        # Setup DP training first (from DPFedAvgClient)
        self.dp_config = package["dp_config"]
        self._setup_dp_training()

        # Set basic parameters (copied from FedAvgClient.set_parameters but adapted for DP)
        from collections import OrderedDict
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        # Set deterministic random seed for parallel mode
        if self.args.mode == "parallel":
            from src.utils.functional import fix_random_seed
            current_epoch = package["current_epoch"]
            client_seed = self.args.common.seed + self.client_id + current_epoch * 10000
            fix_random_seed(client_seed, use_cuda=self.device.type == "cuda")

        # Reset optimizer if needed
        if (
            package["optimizer_state"]
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        # Load model parameters - use _module for DP wrapped model
        self.model._module.load_state_dict(package["regular_model_params"], strict=False)
        self.model._module.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model._module.load_state_dict(self.init_buffers, strict=False)

        # Save regular_model_params for diff calculation (use _module state_dict)
        if self.return_diff:
            model_params = self.model._module.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

        # Set SCAFFOLD control variates
        self.c_global = package["c_global"]
        self.c_local = package["c_local"]
        self.iter_trainloader = iter(self.trainloader)

    def fit(self):
        """DP training with SCAFFOLD control variate correction."""
        self.model.train()
        self.dataset.train()

        steps_before = self.privacy_stats["steps_taken"]

        for epoch in range(self.local_epoch):
            # Sample single batch using Poisson sampling (from DPFedAvgClient)
            x, y = self._sample_single_batch_with_poisson()
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            logit = self.model(x)
            loss = self.criterion(logit, y)

            # Backward pass with DP
            self.optimizer.zero_grad()
            loss.backward()

            # Add SCAFFOLD control variate correction to gradients
            # This happens AFTER Opacus computes per-sample gradients
            # but BEFORE clipping and noise addition (which happens in optimizer.step())
            for param, c, c_i in zip(
                self.model.parameters(), self.c_global, self.c_local
            ):
                if param.requires_grad:
                    param.grad.data += (c - c_i).to(self.device)

            # DP optimizer step (handles clipping and noise addition)
            self.optimizer.step()

            self.privacy_stats["steps_taken"] += 1

            # Update learning rate if scheduler is available
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Update privacy statistics
        self._update_privacy_stats()

        steps_taken = self.privacy_stats["steps_taken"] - steps_before
        logging.debug(f"Client {self.client_id}: Completed {steps_taken} DP-SCAFFOLD training steps")

    def train(self, server_package: dict[str, Any]):
        """Train and update control variates."""
        self.set_parameters(server_package)
        self.train_with_eval()

        # Update control variates after training
        with torch.no_grad():
            self.y_delta = []
            c_plus = []
            self.c_delta = []

            # Get trained model parameters from DP wrapped model
            model_params = self.model._module.state_dict()

            # Calculate model parameter delta using self.regular_model_params (saved before training)
            # and current model params (after training)
            for key in self.regular_params_name:
                x = self.regular_model_params[key]  # Before training (from server)
                y_i = model_params[key]  # After training
                self.y_delta.append(y_i.cpu() - x)

            # Calculate new local control variate
            coef = 1 / (self.local_epoch * self.args.optimizer.lr)
            for c, c_i, y_del in zip(self.c_global, self.c_local, self.y_delta):
                c_plus.append(c_i - c - coef * y_del)

            # Calculate control variate delta
            for c_p, c_l in zip(c_plus, self.c_local):
                self.c_delta.append(c_p - c_l)

            # Update local control variate
            self.c_local = c_plus

        return self.package()

    def package(self):
        """Package client data including control variate deltas."""
        client_package = super().package()

        # Add SCAFFOLD-specific data
        client_package["c_delta"] = self.c_delta
        client_package["y_delta"] = self.y_delta

        return client_package

    def get_data_batch(self):
        """Get next batch from iterator (SCAFFOLD-style batch iteration)."""
        try:
            x, y = next(self.iter_trainloader)
            if len(x) <= 1:
                x, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            x, y = next(self.iter_trainloader)
        return x.to(self.device), y.to(self.device)
