from argparse import ArgumentParser, Namespace
from copy import deepcopy
from typing import Any

import torch
from omegaconf import DictConfig

from src.client.dp_scaffold_new import DPScaffoldNewClient
from src.server.dp_fedavg_local import DPFedAvgLocalServer


class DPScaffoldNewServer(DPFedAvgLocalServer):
    algorithm_name: str = "DP-SCAFFOLD-NEW"
    all_model_params_personalized = False
    return_diff = True
    client_cls = DPScaffoldNewClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--global_lr", type=float, default=1.0)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.c_global = [
            torch.zeros_like(param) for param in self.public_model_params.values()
        ]
        self.c_local = [deepcopy(self.c_global) for _ in self.train_clients]

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["c_global"] = self.c_global
        server_package["c_local"] = self.c_local[client_id]
        return server_package

    @torch.no_grad()
    def aggregate_client_updates(self, client_packages: dict[int, dict[str, Any]]):
        c_delta_list = [package["c_delta"] for package in client_packages.values()]
        y_delta_list = [package["y_delta"] for package in client_packages.values()]
        weights = torch.ones(len(y_delta_list)) / len(y_delta_list)
        for param, y_delta in zip(
            self.public_model_params.values(), zip(*y_delta_list)
        ):
            param.data += self.args.dp_scaffold_new.global_lr * torch.sum(
                torch.stack(y_delta, dim=-1) * weights, dim=-1
            )

        # update global control
        for c_global, c_delta in zip(self.c_global, zip(*c_delta_list)):
            c_global.data += torch.stack(c_delta, dim=-1).sum(dim=-1) / self.client_num


# Create an alias for main.py's naming convention compatibility
# main.py expects class name to match "method_name + server" pattern
Dp_scaffold_newServer = DPScaffoldNewServer