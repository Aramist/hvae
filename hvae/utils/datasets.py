"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class SpecDataset(IterableDataset):
    def __init__(
        self,
        data_path: Path,
        snippet_length: int = int(2**17),
        nfft: int = 256,
        noverlap: int = 0,
        nmels: int = 80,
        spec_min: float = 7,
        spec_max: float = 12,
        make_train_subset: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.snippet_length = snippet_length

        self.nfft = nfft
        self.nmels = nmels
        self.spec_min = spec_min
        self.spec_max = spec_max

        # Very important for iterable datasets that you use different seeds for each worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rand_seed = worker_info.id
        else:
            rand_seed = 42
        self.np_rng = np.random.RandomState(rand_seed)

        with h5py.File(data_path, "r") as f:
            self.sr = f.attrs["sr"]
            self.nfft = f.attrs["nfft"]
            self.hop_length = f.attrs["hop_length"]

        with h5py.File(data_path, "r") as f:
            self.dates_for_training = filter(
                lambda k: not k.endswith("events"), f.keys()
            )
            self.dates_for_training = filter(
                lambda k: f"{k}_events" in f, self.dates_for_training
            )
            self.dates_for_training = list(self.dates_for_training)

            self.all_dates = self.dates_for_training.copy()

        if make_train_subset:
            shuffling = self.np_rng.permutation(len(self.dates_for_training))
            self.dates_for_training = [
                self.dates_for_training[i]
                for i in shuffling[: int(0.9 * len(shuffling))]
            ]

        self.hdf = None

    @property
    def dt(self):
        # return the time step of the spectrogram
        return self.hop_length / self.sr

    @property
    def train_keys(self):
        return self.dates_for_training

    @property
    def val_keys(self):
        return sorted(list(set(self.all_dates) - set(self.dates_for_training)))

    def idx_for_time(self, time: float) -> int:
        return int(time / self.dt)

    def __iter__(self):
        return self

    def __next__(self):
        if self.hdf is None:
            self.hdf = h5py.File(self.data_path, "r")
        # sample a random date
        date_idx = self.np_rng.randint(0, len(self.dates_for_training))
        date = self.dates_for_training[date_idx]
        events = self.hdf[f"{date}_events"]

        # sample a random event
        event = self.np_rng.randint(0, len(events))
        event_idx = self.idx_for_time(events[event])
        start = max(0, event_idx - self.snippet_length)

        return self.get_snippet(date, start)

    def get_snippet_time(self, date: str, start_idx: int) -> float:
        time_since_start = start_idx * self.dt
        # year_month_day_hour_minute_second
        h, m, s = map(int, date.split("_")[3:6])
        seconds_since_midnight = 3600 * h + 60 * m + s
        return seconds_since_midnight + time_since_start

    def get_snippet(self, date: str, start_idx: int):
        spec = self.hdf[date][start_idx : start_idx + self.snippet_length, :].T
        spec = np.log(spec + 1e-8)
        spec = (spec - self.spec_min) / (self.spec_max - self.spec_min)
        spec = np.clip(spec, 0, 1)
        spec = torch.from_numpy(spec).float()
        return spec  # shape is (time, features)


def make_dataloader(data_path: Path, config: dict):
    dataset = SpecDataset(data_path, **config["dataloader"])
    try:
        # I think this function only exists on linux
        avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)
    except:
        avail_cpus = max(1, os.cpu_count() - 1)
    return DataLoader(
        dataset, batch_size=config["dataloader"]["batch_size"], num_workers=avail_cpus
    )


if __name__ == "__main__":
    # test the statistics of the dataset using the default config
    default_config = {
        "optimization": {
            "num_weight_updates": 100_000,
            "initial_learning_rate": 1e-3,
            "min_learning_rate": 1e-5,
            "num_warmup_steps": 10_000,
            "num_decay_steps": 90_000,
            "momentum": 0.9,
            "num_updates_per_log": 100,
            "num_updates_per_ckpt": 10_000,
        },
        "dataloader": {
            "batch_size": 16,
            "nfft": 1024,
            "noverlap": 1024 - 128,
            "nmels": 80,
            "snippet_length": int(2**17),
            "spec_min": 7,
            "spec_max": 13,
        },
        "model": {
            "input_dim": 80,
            "num_timescales": 4,
            "num_layers_per_block": 2,
            "kernel_size": [7, 7, 7, 7],
            "num_channels": [64, 64, 64, 64],
            "dilation": [3, 3, 2, 2],
            "downsample_factors": [16, 16, 16, 16],
            "latent_dim": [16, 32, 64, 128],
            "mode": "top-down",
        },
    }

    data_path = Path("/mnt/home/atanelus/ceph/hvae_dataset.hdf5")
    dataset: SpecDataset = make_dataloader(data_path, default_config).dataset

    # Sample a random snippet
    snippet = next(iter(dataset))
    print("Snippet shape:")
    print(snippet.shape)
    print("Snippet statistics:")
    print("min, max")
    print(snippet.min(), snippet.max())
    print("mean, std")
    print(snippet.mean(), snippet.std())
    print("quantiles: 5%, 50%, 90%, 95%")
    print(np.quantile(snippet.numpy(), [0.05, 0.50, 0.90, 0.95]))
    print()
