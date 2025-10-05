"""
Generate federated learning data partitions with equal samples per client.

This script creates data partitions where each client receives exactly the same
number of samples. The similarity parameter controls the degree of data heterogeneity
by mixing IID and non-IID samples.

Usage:
    python generate_data_equal_samples.py -d mnist -cn 100 -sim 0.5
"""

import json
import os
import pickle
import random
import hashlib
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Set, Optional

import numpy as np
import torch
import torchvision

from data.utils.datasets import DATASETS, BaseDataset
from src.utils.functional import fix_random_seed
from data.utils.process import (
    plot_distribution,
    prune_args,
    process_celeba,
    process_femnist,
)

CURRENT_DIR = Path(__file__).parent.absolute()


def load_train_only_dataset(dataset_name: str, dataset_root: Path, args):
    """Load only the training set of the dataset (not train+test).

    This matches the paper's approach of using only 60,000 training samples
    for MNIST instead of 70,000 (train+test).

    Args:
        dataset_name: Name of the dataset
        dataset_root: Root directory for the dataset
        args: Arguments object

    Returns:
        BaseDataset object containing only training data
    """
    if dataset_name == "mnist":
        train_part = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
        data = torch.Tensor(train_part.data).float().unsqueeze(1)
        targets = torch.Tensor(train_part.targets).long().squeeze()

        return BaseDataset(
            data=data,
            targets=targets,
            classes=list(range(10)),
        )

    elif dataset_name == "fmnist":
        train_part = torchvision.datasets.FashionMNIST(dataset_root, train=True, download=True)
        data = torch.Tensor(train_part.data).float().unsqueeze(1)
        targets = torch.Tensor(train_part.targets).long().squeeze()

        return BaseDataset(
            data=data,
            targets=targets,
            classes=list(range(10)),
        )

    elif dataset_name == "emnist":
        train_part = torchvision.datasets.EMNIST(
            dataset_root, split=args.emnist_split, train=True, download=True
        )
        data = torch.Tensor(train_part.data).float().unsqueeze(1)
        targets = torch.Tensor(train_part.targets).long().squeeze()

        return BaseDataset(
            data=data,
            targets=targets,
            classes=list(range(len(train_part.classes))),
        )

    elif dataset_name == "cifar10":
        train_part = torchvision.datasets.CIFAR10(dataset_root, train=True, download=True)
        data = torch.Tensor(train_part.data).float().permute(0, 3, 1, 2)
        targets = torch.Tensor(train_part.targets).long()

        return BaseDataset(
            data=data,
            targets=targets,
            classes=list(range(10)),
        )

    elif dataset_name == "cifar100":
        train_part = torchvision.datasets.CIFAR100(dataset_root, train=True, download=True)
        data = torch.Tensor(train_part.data).float().permute(0, 3, 1, 2)

        if args.super_class:
            targets = torch.Tensor(train_part.coarse_labels).long()
            classes = list(range(20))
        else:
            targets = torch.Tensor(train_part.targets).long()
            classes = list(range(100))

        return BaseDataset(
            data=data,
            targets=targets,
            classes=classes,
        )

    else:
        # For other datasets, fall back to FL-bench's default loading
        # (which may load train+test)
        return DATASETS[dataset_name](dataset_root, args)


def equal_samples_partition(
    targets: np.ndarray,
    target_indices: np.ndarray,
    label_set: Set[int],
    client_num: int,
    similarity: float,
    partition: Dict[str, Any],
    stats: Dict[int, Dict[str, Any]],
):
    """Partition the dataset with equal samples per client.

    Each client receives exactly the same number of samples. The similarity parameter
    controls the mix of IID and non-IID data:
    - IID portion: similarity * samples_per_client (randomly sampled from all classes)
    - non-IID portion: (1-similarity) * samples_per_client (from one specific class)

    Args:
        targets (np.ndarray): Array of data labels.
        target_indices (np.ndarray): Indices of targets.
        label_set (Set[int]): Set of unique labels.
        client_num (int): Number of clients.
        similarity (float): Similarity parameter (0.0-1.0); higher values mean more IID data.
        partition (Dict[str, Any]): Dictionary to hold output data indices for each client.
        stats (Dict[int, Dict[str, Any]]): Dictionary to record clients' data distribution.
    """
    # Calculate samples per client
    samples_per_client = len(targets) // client_num
    iid_samples = int(similarity * samples_per_client)
    non_iid_samples = samples_per_client - iid_samples

    print(f"Total samples: {len(targets)}")
    print(f"Samples per client: {samples_per_client}")
    print(f"IID samples per client: {iid_samples}")
    print(f"Non-IID samples per client: {non_iid_samples}")

    # Organize indices by label
    indices_per_label = {label: np.where(targets == label)[0] for label in label_set}

    # Shuffle indices for each label
    for label in label_set:
        np.random.shuffle(indices_per_label[label])

    # Initialize partition
    partition["data_indices"] = [[] for _ in range(client_num)]

    # Track the starting index for each label (for non-IID allocation)
    label_idx_tracker = {label: 0 for label in label_set}

    # Allocate IID portion
    if iid_samples > 0:
        # Calculate the proportion of each label in the dataset
        label_proportions = {label: len(indices_per_label[label]) / len(targets)
                            for label in label_set}

        # Track IID allocation index for each label
        iid_idx_tracker = {label: 0 for label in label_set}

        for client_id in range(client_num):
            client_iid_samples = []
            remaining_iid = iid_samples

            # Allocate samples from each label proportionally
            for i, label in enumerate(sorted(label_set)):
                if i == len(label_set) - 1:
                    # Last label gets all remaining samples
                    samples_from_label = remaining_iid
                else:
                    samples_from_label = int(iid_samples * label_proportions[label])
                    remaining_iid -= samples_from_label

                if samples_from_label > 0:
                    available = indices_per_label[label]
                    start_idx = iid_idx_tracker[label]

                    # Handle wrap-around if we run out of samples
                    if start_idx + samples_from_label <= len(available):
                        selected = available[start_idx:start_idx + samples_from_label]
                    else:
                        # Wrap around and reuse samples
                        first_part = available[start_idx:]
                        wrap_needed = samples_from_label - len(first_part)
                        second_part = available[:wrap_needed]
                        selected = np.concatenate([first_part, second_part])
                        iid_idx_tracker[label] = wrap_needed

                    iid_idx_tracker[label] = (iid_idx_tracker[label] + samples_from_label) % len(available)
                    client_iid_samples.extend(selected.tolist())

            partition["data_indices"][client_id].extend(client_iid_samples)

    # Allocate non-IID portion
    if non_iid_samples > 0:
        for client_id in range(client_num):
            # Assign one label per client (cycling through labels)
            assigned_label = client_id % len(label_set)
            available = indices_per_label[assigned_label]
            start_idx = label_idx_tracker[assigned_label]

            # Handle case where label doesn't have enough samples
            if start_idx + non_iid_samples <= len(available):
                selected = available[start_idx:start_idx + non_iid_samples]
                label_idx_tracker[assigned_label] += non_iid_samples
            else:
                # Need to wrap around
                first_part = available[start_idx:]
                wrap_needed = non_iid_samples - len(first_part)

                # Reuse samples from the beginning
                if wrap_needed <= len(available):
                    second_part = available[:wrap_needed]
                    selected = np.concatenate([first_part, second_part])
                    label_idx_tracker[assigned_label] = wrap_needed
                else:
                    # If still not enough, use random sampling with replacement
                    selected = np.random.choice(available, non_iid_samples, replace=True)
                    label_idx_tracker[assigned_label] = 0

            partition["data_indices"][client_id].extend(selected.tolist())

    # Update statistics
    for client_id in range(client_num):
        client_indices = partition["data_indices"][client_id]
        stats[client_id]["x"] = len(client_indices)
        stats[client_id]["y"] = dict(
            Counter(targets[client_indices].tolist())
        )

        # Convert to original target indices
        partition["data_indices"][client_id] = target_indices[client_indices]

    # Calculate global statistics
    sample_counts = np.array([stat["x"] for stat in stats.values()])
    stats["samples_per_client"] = {
        "mean": sample_counts.mean().item(),
        "stddev": sample_counts.std().item(),
    }

    print(f"Samples per client - Mean: {stats['samples_per_client']['mean']:.2f}, "
          f"Stddev: {stats['samples_per_client']['stddev']:.2f}")


def main(args):
    """Main function to generate equal-sample partitions."""
    dataset_root = CURRENT_DIR / "data" / args.dataset

    fix_random_seed(args.seed, args.use_cuda)

    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    # Check if partition already exists
    if os.path.isfile(dataset_root / "partition_md5.txt"):
        with open(dataset_root / "partition_md5.txt", "r") as f:
            md5 = f.read()
            if md5 == hashlib.md5(json.dumps(args.__dict__).encode()).hexdigest():
                print("Partition file already exists. Skip partitioning.")
                return

    # Validate parameters
    if not 0.0 <= args.similarity <= 1.0:
        raise ValueError(f"similarity must be between 0.0 and 1.0, got {args.similarity}")

    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0, got {args.val_ratio + args.test_ratio}")

    client_num = args.client_num
    partition = {"separation": None, "data_indices": [[] for _ in range(client_num)]}
    stats = {i: {"x": 0, "y": {}} for i in range(args.client_num)}
    dataset: BaseDataset = None

    print("=" * 80)
    print(f"Generating equal-sample partition for {args.dataset}")
    print(f"Clients: {client_num}, Similarity: {args.similarity}")
    print("=" * 80)

    # Handle special datasets
    if args.dataset == "femnist":
        dataset = process_femnist(args, partition, stats)
        partition["val"] = []
        print("Warning: FEMNIST uses its own partition logic, equal samples not guaranteed.")
    elif args.dataset == "celeba":
        dataset = process_celeba(args, partition, stats)
        partition["val"] = []
        print("Warning: CelebA uses its own partition logic, equal samples not guaranteed.")
    else:
        # Load dataset (train only)
        print("Loading dataset (training set only)...")
        dataset = load_train_only_dataset(args.dataset, dataset_root, args)
        targets = np.array(dataset.targets, dtype=np.int32)
        target_indices = np.arange(len(targets), dtype=np.int32)
        valid_label_set = set(range(len(dataset.classes)))

        # Perform equal-sample partition
        print("Partitioning data...")
        equal_samples_partition(
            targets=targets,
            target_indices=target_indices,
            label_set=valid_label_set,
            client_num=client_num,
            similarity=args.similarity,
            partition=partition,
            stats=stats,
        )

    # Split into train/val/test
    if partition["separation"] is None:
        if args.split == "user":
            test_clients_num = int(args.client_num * args.test_ratio)
            val_clients_num = int(args.client_num * args.val_ratio)
            train_clients_num = args.client_num - test_clients_num - val_clients_num
            clients_4_train = list(range(train_clients_num))
            clients_4_val = list(
                range(train_clients_num, train_clients_num + val_clients_num)
            )
            clients_4_test = list(
                range(train_clients_num + val_clients_num, args.client_num)
            )
        elif args.split == "sample":
            clients_4_train = list(range(args.client_num))
            clients_4_val = clients_4_train
            clients_4_test = clients_4_train

        partition["separation"] = {
            "train": clients_4_train,
            "val": clients_4_val,
            "test": clients_4_test,
            "total": args.client_num,
        }

    # Split samples for each client
    if args.dataset not in ["femnist", "celeba"]:
        if args.split == "sample":
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                np.random.shuffle(indices)
                testset_size = int(len(indices) * args.test_ratio)
                valset_size = int(len(indices) * args.val_ratio)
                trainset, valset, testset = (
                    indices[testset_size + valset_size :],
                    indices[testset_size : testset_size + valset_size],
                    indices[:testset_size],
                )
                partition["data_indices"][client_id] = {
                    "train": trainset,
                    "val": valset,
                    "test": testset,
                }
        elif args.split == "user":
            for client_id in partition["separation"]["train"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": indices,
                    "val": np.array([], dtype=np.int64),
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["val"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": indices,
                    "test": np.array([], dtype=np.int64),
                }

            for client_id in partition["separation"]["test"]:
                indices = partition["data_indices"][client_id]
                partition["data_indices"][client_id] = {
                    "train": np.array([], dtype=np.int64),
                    "val": np.array([], dtype=np.int64),
                    "test": indices,
                }

    # Plot distribution
    if args.plot_distribution:
        print("Plotting distribution...")
        counts = np.zeros((len(dataset.classes), args.client_num), dtype=np.int64)
        client_ids = range(args.client_num)
        for i, client_id in enumerate(client_ids):
            for j, cnt in stats[client_id]["y"].items():
                counts[j][i] = cnt
        plot_distribution(
            client_num=args.client_num,
            label_counts=counts,
            save_path=f"{dataset_root}/class_distribution.png",
        )

    # Save files
    print("Saving files...")
    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(dataset_root / "args.json", "w") as f:
        json.dump(prune_args(args), f, indent=4)

    with open(dataset_root / "partition_md5.txt", "w") as f:
        f.write(hashlib.md5(json.dumps(args.__dict__).encode()).hexdigest())

    print("=" * 80)
    print("Partition generation completed!")
    print(f"Files saved to: {dataset_root}")
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Required parameters
    parser.add_argument(
        "-d", "--dataset", type=str, choices=DATASETS.keys(), required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "-cn", "--client_num", type=int, default=100,
        help="Number of clients"
    )

    # Core parameter
    parser.add_argument(
        "-sim", "--similarity", type=float, default=0.5,
        help="Similarity parameter (0.0-1.0): controls the proportion of IID data"
    )

    # General parameters
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--iid", type=float, default=0.0,
        help="Compatibility parameter (not used, set similarity instead)"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=0,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "-c", "--classes", type=int, default=0,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "-s", "--shards", type=int, default=0,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "-sm", "--semantic", type=int, default=0,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "-ms", "--min_samples_per_client", type=int, default=10,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "--pca_components", type=Optional[int], default=None,
        help="Compatibility parameter (not used)"
    )
    parser.add_argument(
        "-sp", "--split", type=str, choices=["sample", "user"], default="sample",
        help="Split mode: 'sample' splits each client's data, 'user' splits clients into groups"
    )
    parser.add_argument(
        "-vr", "--val_ratio", type=float, default=0.0,
        help="Validation set ratio"
    )
    parser.add_argument(
        "-tr", "--test_ratio", type=float, default=0.2,
        help="Test set ratio"
    )
    parser.add_argument(
        "-pd", "--plot_distribution", type=int, default=1,
        help="Whether to plot class distribution (1=yes, 0=no)"
    )
    parser.add_argument(
        "--use_cuda", type=int, default=1,
        help="Whether to use CUDA (1=yes, 0=no)"
    )

    # Dataset-specific parameters
    parser.add_argument(
        "--super_class", type=int, default=0,
        help="For CIFAR-100: use super classes"
    )
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
        help="For EMNIST: split type"
    )
    parser.add_argument(
        "--ood_domains", nargs="+", default=None,
        help="For domain datasets: out-of-distribution domains to exclude"
    )

    args = parser.parse_args()
    main(args)
