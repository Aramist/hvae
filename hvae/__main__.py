import argparse
import json
from pathlib import Path

import torch
from torch import multiprocessing as mp

from .training.trainer import train

default_config = {
    "optimization": {
        "num_weight_updates": 1_000_000,
        "initial_learning_rate": 1e-3,
        "min_learning_rate": 1e-5,
        "num_warmup_steps": 10_000,
        "num_decay_steps": 100_000,
        "momentum": 0.9,
        "num_updates_per_log": 100,
        "num_updates_per_ckpt": 50_000,
    },
    "dataloader": {
        "batch_size": 16,
        "nfft": 1024,
        "noverlap": 1024 - 256,
        "nmels": 80,
        "snippet_length": int(2**17),
    },
    "model": {
        "input_dim": 80,
        "num_layers_per_block": 2,
        "kernel_size": [7, 7, 7, 7],
        "num_channels": [64, 64, 64, 64],
        "dilation": [3, 3, 2, 2],
        "downsample_factors": [16, 16, 16, 16],
        "latent_dim": [16, 16, 16, 16],
    },
}


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON file containing configuration options",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to a directory containing the data",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Path to a directory where the output will be stored",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    defaults = default_config.copy()

    with open(config_path, "r") as f:
        config = json.load(f)
        config = update_recursively(config, defaults)
    return config


def main():
    args = parse_args()
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = default_config

    if torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs, using DDP")
        mp.spawn(
            fn=train,
            args=(world_size, config, args.data, args.save_path),
            join=True,
            nprocs=world_size,
        )
    else:

        train(0, 1, config, args.data, args.save_path)


if __name__ == "__main__":
    main()
