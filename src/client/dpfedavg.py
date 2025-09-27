"""
Simplified DP-FedAvg client that only handles DP training.
All parameter mapping and model compatibility is handled by the server.
"""
import logging
from copy import deepcopy
from typing import Any

import torch

from src.client.fedavg import FedAvgClient

from opacus import PrivacyEngine
from opacus.data_loader import UniformWithReplacementSampler
from src.utils.dp_manager import DPManager


class DPFedAvgClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # DP state
        self.privacy_engine = None
        self.dp_config = None

        # Privacy tracking - use standardized structure
        self.privacy_stats = DPManager.create_privacy_stats_dict()

    def set_parameters(self, package: dict[str, Any]):
        """Set parameters and setup DP training."""
        self.dp_config = package["dp_config"]
        self._setup_dp_training()

        # Call parent set_parameters
        super().set_parameters(package)

    def _setup_dp_training(self):
        """Setup DP training with provided configuration."""
        # Skip if already setup
        if self.privacy_engine is not None:
            return

        # Use noise multiplier calculated by server
        noise_multiplier = self.dp_config["noise_multiplier"]

        # Initialize Privacy Engine with correct accountant
        accountant = self.dp_config.get("privacy_accountant", "rdp")
        self.privacy_engine = PrivacyEngine(accountant=accountant)

        # Make model, optimizer, and data loader private
        self.model, self.optimizer, self.trainloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.dp_config["max_grad_norm"],
        )

        self.privacy_stats["noise_multiplier"] = noise_multiplier

        # Update parameter names to match DP model (use underlying module to avoid _module prefix)
        underlying_model = self.model._module
        self.regular_params_name = list(key for key, _ in underlying_model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name = [name for name, _ in underlying_model.named_buffers()]

        logging.info(f"DP training setup complete for client {self.client_id}. "
                    f"ε={self.dp_config['epsilon']}, δ={self.dp_config['delta']}, "
                    f"noise_multiplier={noise_multiplier:.3f}, accountant={accountant}")

    def _sample_single_batch_with_poisson(self):
        """Generate a single batch using Poisson sampling"""
        sample_rate = self.dp_config["sample_rate"]

        sampler = UniformWithReplacementSampler(
            num_samples=len(self.trainset),
            sample_rate=sample_rate,
            steps=1
        )

        from torch.utils.data import DataLoader
        temp_loader = DataLoader(self.trainset, batch_sampler=sampler)
        batch = next(iter(temp_loader))
        return batch

    def fit(self):
        """DP training implementation with single batch sampling per epoch."""
        self.model.train()
        self.dataset.train()

        steps_before = self.privacy_stats["steps_taken"]

        for epoch in range(self.local_epoch):
            # Sample single batch using Poisson sampling
            x, y = self._sample_single_batch_with_poisson()

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            logit = self.model(x)
            loss = self.criterion(logit, y)

            # Backward pass with DP
            # The DP optimizer automatically handles:
            # 1. Per-sample gradient computation
            # 2. Gradient clipping
            # 3. Noise addition
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.privacy_stats["steps_taken"] += 1

            # Update learning rate if scheduler is available
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Update privacy statistics
        self._update_privacy_stats()

        steps_taken = self.privacy_stats["steps_taken"] - steps_before
        logging.debug(f"Client {self.client_id}: Completed {steps_taken} DP training steps")

    def _update_privacy_stats(self):
        """Update privacy budget consumption statistics."""
        accountant = self.privacy_engine.accountant
        if len(accountant.history) > 0:
            epsilon_spent = accountant.get_epsilon(delta=self.dp_config["delta"])
            if not (epsilon_spent != epsilon_spent):  # Check for NaN
                self.privacy_stats["epsilon_spent"] = epsilon_spent

    def package(self):
        """Package client data including privacy statistics."""
        # Get parameters from the underlying DP wrapped model
        model_params = self.model._module.state_dict()

        # Create the package manually to handle DP model parameters
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
                if key in model_params
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
                if key in model_params
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )

        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")

        # Add privacy statistics
        client_package["privacy_stats"] = deepcopy(self.privacy_stats)

        # Log privacy consumption
        eps_spent = self.privacy_stats["epsilon_spent"]
        if eps_spent > 0:
            privacy_remaining = max(0, self.dp_config["epsilon"] - eps_spent)
            logging.info(f"Client {self.client_id}: Privacy budget consumed: "
                       f"{eps_spent:.4f}/{self.dp_config['epsilon']:.4f} (ε), "
                       f"remaining: {privacy_remaining:.4f}")

        return client_package

    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Any]:
        """Evaluate model, handling DP-wrapped models."""
        target_model = self.model if model is None else model

        # Use underlying model from DP wrapper for evaluation
        unwrapped_model = target_model._module
        return super().evaluate(unwrapped_model)