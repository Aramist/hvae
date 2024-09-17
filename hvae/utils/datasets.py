"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torch.utils.data import DataLoader, IterableDataset
from torchaudio.transforms import MelSpectrogram


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
        self.noverlap = noverlap
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
            self.dates_for_training = list(f.keys())
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

    def __iter__(self):
        return self

    def __next__(self):
        if self.hdf is None:
            self.hdf = h5py.File(self.data_path, "r")
        # sample a random date
        date_idx = self.np_rng.random_integers(0, len(self.dates_for_training) - 1)
        date = self.dates_for_training[date_idx]

        # sample a random snippet
        start = self.np_rng.random_integers(
            0, self.hdf[date].shape[0] - self.snippet_length - 1
        )

        return self.get_snippet(date, start)

    def get_snippet(self, date: str, start_idx: int):
        spec = self.hdf[date][start_idx : start_idx + self.snippet_length, :].T
        spec = (spec - self.spec_min) / (self.spec_max - self.spec_min)
        spec = np.clip(spec, 0, 1)
        spec = torch.from_numpy(spec).float()
        return spec  # shape is (time, features)

    def _compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        transform = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.nfft,
            hop_length=self.nfft - self.noverlap,
            n_mels=self.nmels,
            f_min=500,
        )
        audio = transform(audio)
        return audio


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
