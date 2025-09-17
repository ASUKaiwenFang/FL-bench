#!/usr/bin/env python3
"""
DP-ScaffStein Last Noise Server JSE Variant Demo
================================================

This script demonstrates the last_noise_server_jse variant of DP-ScaffStein, which combines
SCAFFOLD control variates with differential privacy noise added to parameter differences
at the final training step and James-Stein Estimator applied at the server side.

Algorithm: Parameter-level noise addition with server-side JSE
- Noise position: Added to parameter differences after training completion
- Noise standard deviation: σ_DP = C × K × η_l × σ_g / b
- Control variates: SCAFFOLD mechanism to reduce client drift
- JSE processing: Applied at server to aggregated parameter differences
- Training: Clean training with final noise addition and server-side shrinkage

Usage:
    python demo/main_dp_scaffstein_last_noise_server_jse.py

Author: FL-bench DP-ScaffStein Implementation
"""

import importlib
import inspect
import sys
from pathlib import Path

FLBENCH_ROOT = Path(__file__).parent.parent.absolute()
if FLBENCH_ROOT not in sys.path:
    sys.path.append(FLBENCH_ROOT.as_posix())

import hydra
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer
from src.utils.functional import parse_args


@hydra.main(config_path="config", config_name="dp_scaffstein_last_noise_server_jse", version_base=None)
def main(config: DictConfig):
    method_name = config.method.lower()

    try:
        fl_method_server_module = importlib.import_module(f"src.server.{method_name}")
    except:
        raise ImportError(f"Can't import `src.server.{method_name}`.")

    module_attributes = inspect.getmembers(fl_method_server_module)
    server_class = [
        attribute
        for attribute in module_attributes
        if attribute[0].lower() == method_name.replace('_', '') + "server"
    ][0][1]

    get_method_hyperparams_func = getattr(server_class, f"get_hyperparams", None)

    config = parse_args(config, method_name, get_method_hyperparams_func)

    useful_config_groups = [
        "dataset",
        "model",
        "method",
        "common",
        "mode",
        "parallel",
        "optimizer",
        "lr_scheduler",
        config.method,
    ]

    # target method is not inherited from FedAvgServer
    if server_class.__bases__[0] != FedAvgServer and server_class != FedAvgServer:
        parent_server_class = server_class.__bases__[0]
        if hasattr(parent_server_class, "get_hyperparams"):
            get_parent_method_hyperparams_func = getattr(
                parent_server_class, f"get_hyperparams", None
            )
            # class name: <METHOD_NAME>Server, only want <METHOD_NAME>
            parent_method_name = parent_server_class.__name__.lower()[:-6]
            # extract the hyperparameters of the parent method
            parent_config = parse_args(
                config, parent_method_name, get_parent_method_hyperparams_func
            )
            setattr(
                config, parent_method_name, getattr(parent_config, parent_method_name)
            )

        useful_config_groups.append(parent_method_name)

    # remove all unused config groups
    for config_group in list(config.keys()):
        if config_group not in useful_config_groups:
            delattr(config, config_group)

    server = server_class(args=config)
    server.run_experiment()


if __name__ == "__main__":
    # For gather the Fl-bench logs and hydra logs
    # Otherwise the hydra logs are stored in ./outputs/...
    sys.argv.append(
        "hydra.run.dir=./out/${method}/${dataset.name}/${now:%Y-%m-%d-%H-%M-%S}"
    )
    main()