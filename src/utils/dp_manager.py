"""
Differential Privacy Manager for FL-bench integration with Opacus.
Handles all DP-related operations in a centralized manner.
"""
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier


class DPManager:
    """
    Centralized manager for differential privacy operations.
    Handles model conversion, parameter mapping, and privacy tracking.
    """

    def __init__(self, args):
        self.args = args
        self.dp_config = args.dpfedavg

        # DP parameters
        self.epsilon = self.dp_config.epsilon
        self.delta = self.dp_config.delta
        self.max_grad_norm = self.dp_config.max_grad_norm
        self.noise_multiplier = getattr(self.dp_config, 'noise_multiplier', None)
        self.sample_rate = getattr(self.dp_config, 'sample_rate', -1.0)

        # Privacy tracking
        self.privacy_stats = {
            "epsilon_spent": 0.0,
            "delta_spent": 0.0,
            "noise_multiplier": 0.0,
            "steps_taken": 0
        }

        logging.info(f"DPManager initialized with ε={self.epsilon}, δ={self.delta}")

    @staticmethod
    def create_privacy_stats_dict() -> Dict[str, Any]:
        """Create standardized privacy statistics dictionary structure."""
        return {
            "epsilon_spent": 0.0,
            "delta_spent": 0.0,
            "noise_multiplier": 0.0,
            "steps_taken": 0
        }

    def prepare_model_for_dp(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare and convert model for DP training.
        Returns DP-compatible model.
        """
        # Check and fix model compatibility
        validator = ModuleValidator()
        if not validator.is_valid(model):
            try:
                fixed_model = validator.fix(model)
                logging.info("Model automatically fixed for DP compatibility")
                return fixed_model
            except Exception as e:
                errors = validator.validate(model, strict=False)
                raise ValueError(f"Model incompatible with Opacus: {errors}")
        else:
            # Model is already compatible
            return model

    def get_dp_config_for_client(self, batch_size: int, dataset_size: int, local_epoch: int, trainloader_length: int) -> Dict[str, Any]:
        """Get DP configuration to send to clients with calculated sample_rate and noise_multiplier."""
        # Calculate effective sample_rate
        if self.sample_rate > 0:
            effective_sample_rate = min(0.99, self.sample_rate)
            logging.debug(f"Using configured sample_rate={effective_sample_rate:.3f}")
        else:
            effective_sample_rate = min(0.99, batch_size / dataset_size)
            logging.debug(f"Calculated sample_rate={effective_sample_rate:.3f} from batch_size={batch_size}, dataset_size={dataset_size}")

        # Calculate noise_multiplier if not provided
        if self.noise_multiplier is None:
            # For Poisson sampling implementation: each epoch = 1 step (1 sampled batch)
            steps = local_epoch
            accountant = getattr(self.dp_config, 'privacy_accountant', 'rdp')
            effective_noise_multiplier = self.calculate_noise_multiplier(
                sample_rate=effective_sample_rate,
                steps=steps,
                accountant=accountant
            )
            logging.debug(f"Calculated noise_multiplier={effective_noise_multiplier:.3f} for sample_rate={effective_sample_rate:.3f}, steps={steps}, accountant={accountant}")
        else:
            effective_noise_multiplier = self.noise_multiplier
            logging.debug(f"Using configured noise_multiplier={effective_noise_multiplier:.3f}")

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": effective_noise_multiplier,
            "sample_rate": effective_sample_rate,
            "privacy_accountant": getattr(self.dp_config, 'privacy_accountant', 'rdp')
        }

    def update_privacy_stats(self, client_privacy_stats: Dict[str, Any]):
        """Update global privacy statistics from client reports."""
        if client_privacy_stats:
            # Take the maximum privacy consumption across clients
            self.privacy_stats["epsilon_spent"] = max(
                self.privacy_stats["epsilon_spent"],
                client_privacy_stats.get("epsilon_spent", 0.0)
            )
            self.privacy_stats["steps_taken"] += client_privacy_stats.get("steps_taken", 0)

    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        epsilon_remaining = max(0, self.epsilon - self.privacy_stats["epsilon_spent"])

        return {
            "target_epsilon": self.epsilon,
            "target_delta": self.delta,
            "epsilon_spent": self.privacy_stats["epsilon_spent"],
            "epsilon_remaining": epsilon_remaining,
            "privacy_exhausted": epsilon_remaining <= 0.01,  # Nearly exhausted
            "total_steps": self.privacy_stats["steps_taken"],
            "noise_multiplier": self.privacy_stats["noise_multiplier"]
        }

    def validate_privacy_budget(self) -> Tuple[bool, str]:
        """Check if privacy budget is still available."""
        report = self.get_privacy_report()

        if report["privacy_exhausted"]:
            return False, f"Privacy budget exhausted. Used: {report['epsilon_spent']:.4f}/{self.epsilon}"

        return True, f"Privacy budget OK. Remaining: {report['epsilon_remaining']:.4f}"

    def calculate_noise_multiplier(
        self,
        sample_rate: float,
        steps: int,
        accountant: str = "rdp"
    ) -> float:
        """
        Calculate noise multiplier for given privacy parameters.

        Args:
            sample_rate: Sampling rate (batch_size / dataset_size)
            steps: Number of training steps
            accountant: Privacy accounting method ("rdp", "gdp", "prv")

        Returns:
            float: Calculated noise multiplier
        """
        if accountant == "rdp":
            return get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
                steps=steps
            )
        else:
            # For other accountants, use RDP as fallback
            logging.warning(f"Accountant {accountant} not fully implemented, using RDP")
            return get_noise_multiplier(
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                sample_rate=sample_rate,
                steps=steps
            )