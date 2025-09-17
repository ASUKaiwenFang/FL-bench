from typing import Dict, List, Any
from collections import OrderedDict
import torch
from src.server.scaffold import SCAFFOLDServer
from src.utils.jse_utils import JSEProcessor


class DPScaffSteinServer(SCAFFOLDServer):
    """DP-ScaffStein Server combining Differential Privacy, SCAFFOLD control variates, and JSE.

    This server implements DP-ScaffStein algorithms with three variants:
    1. last_noise_server_jse: DP noise at last step, JSE at server
    2. step_noise_step_jse: DP noise and JSE at each step (JSE handled at client)
    3. step_noise_final_jse: DP noise at each step, JSE at final step (JSE handled at client)

    Only variant 1 requires server-side JSE processing.
    """

    def __init__(self, **commons):
        super().__init__(**commons)

        # Get algorithm variant from config
        variant_config = getattr(self.args.dp_scaffstein, 'algorithm_variant', 'step_noise_final_jse')
        if isinstance(variant_config, str):
            variant_map = {
                'last_noise_server_jse': 1,
                'step_noise_step_jse': 2,
                'step_noise_final_jse': 3
            }
            self.algorithm_variant = variant_map[variant_config]
        else:
            self.algorithm_variant = variant_config

    def aggregate_client_updates(self, client_packages: List[Dict[str, Any]]) -> None:
        """Aggregate client updates with DP-ScaffStein processing.

        For variant 1 (last_noise_server_jse): Apply server-side JSE to aggregated parameter differences
        For variants 2 and 3: Use standard SCAFFOLD aggregation (JSE handled at client)
        """
        # First, perform standard SCAFFOLD aggregation
        super().aggregate_client_updates(client_packages)

        # Apply server-side JSE for variant 1 only
        if self.algorithm_variant == 1:  # last_noise_server_jse
            self._apply_server_jse(client_packages)

    def _apply_server_jse(self, client_packages: List[Dict[str, Any]]) -> None:
        """Apply server-side JSE to aggregated parameter differences.

        This method is only used for algorithm variant 1 (last_noise_server_jse).
        It applies global JSE shrinkage to the aggregated parameter differences.

        Args:
            client_packages: List of client data packages
        """
        if not client_packages:
            return

        # Extract sigma_dp from first client package (should be consistent across clients)
        sigma_dp = client_packages[0].get('sigma_dp', 0.0)
        if sigma_dp <= 0:
            return  # No JSE processing if no DP noise

        # Apply global JSE to the aggregated parameter differences
        # self.avg_params contains the aggregated parameter differences
        JSEProcessor.apply_global_jse_to_parameter_diff(
            self.avg_params, sigma_dp ** 2, k_factor=1  # k_factor=1 for server-side JSE
        )

    def get_hyperparams(self) -> Dict[str, Any]:
        """Get hyperparameters for DP-ScaffStein method."""
        local_hyperparams = {
            "clip_norm": 1.0,
            "sigma": 1.0,
            "algorithm_variant": "step_noise_final_jse"
        }
        return local_hyperparams