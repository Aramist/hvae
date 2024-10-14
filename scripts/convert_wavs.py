import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import wavfile
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out",
        type=Path,
        help="Path to a directory where the output will be stored",
    )
    parser.add_argument(
        "--wavs",
        type=Path,
        nargs="+",
        help="Paths to the wav files to convert",
    )
    parser.add_argument(
        "--hdfs",
        type=Path,
        nargs="+",
        help="Paths to the HDF files to process",
    )

    return parser.parse_args()


def make_mel_spectrogram(
    audio: np.ndarray,
    nfft: int,
    noverlap: int,
    nmels: int,
    fmin: int = 500,
    sr: int = 125000,
) -> np.ndarray:
    t_audio = torch.from_numpy(audio)
    transform = MelSpectrogram(
        sample_rate=sr,
        n_fft=nfft,
        hop_length=nfft - noverlap,
        n_mels=nmels,
        f_min=fmin,
    )
    audio = transform(t_audio).T
    return audio.numpy()


def process_wavs(wav_files: list[Path], output_hdf: Path):
    nfft, noverlap, nmels, fmin, sr = 1024, 1024 - 128, 80, 500, 125000
    with h5py.File(output_hdf, "a") as f:
        f.attrs["nfft"] = nfft
        f.attrs["noverlap"] = noverlap
        f.attrs["nmels"] = nmels
        f.attrs["fmin"] = fmin
        f.attrs["sr"] = sr
        f.attrs["hop_length"] = nfft - noverlap

        for wav_file in tqdm(wav_files):
            if wav_file.stem in f:
                continue
            sr, audio = wavfile.read(wav_file)
            audio = audio.astype(np.float32)
            audio /= 0.005  # approximate standard deviation of the data
            mel_spec = make_mel_spectrogram(
                audio, nfft=nfft, noverlap=noverlap, nmels=nmels
            )  # shape is (time, freq)
            f.create_dataset(wav_file.stem, data=mel_spec, chunks=(1024, nmels))


def get_events(processed_hdfs: list[Path]) -> dict[str, np.ndarray]:
    """Load events from the processed HDF files.
    Events are stored as a dictionary of numpy arrays where the key is the
    datetime (filename stem) of the corresponding audio clip and the value
    is an array of event timestamps

    Args:
        processed_hdfs (list[Path]): List of paths to the processed HDF files

    Returns:
        dict[str, np.ndarray]: Dictionary of event timestamps in seconds
    """

    events = {}

    def insert(fn: str, evt: float):
        if fn in events:
            events[fn] = events[fn] + [evt]
        else:
            events[fn] = [evt]

    for hdf in tqdm(processed_hdfs):
        with h5py.File(hdf, "r") as f:
            fns = f["audio_filenames"][:]
            fns = map(lambda x: x.decode("utf-8"), fns)
            get_stem = lambda x: x.split("\\")[-1].split(".")[0]
            fns = list(map(get_stem, fns))

            onsets = f["onsets"][:]
            for fn, onset in zip(fns, onsets):
                insert(fn, onset)

    events = {k: np.array(v) for k, v in events.items()}
    return events


def add_events_to_dataset(events: dict[str, np.ndarray], dataset: Path):
    with h5py.File(dataset, "r+") as f:
        for fn, evt in events.items():
            key = f"{fn}_events"
            if key in f:
                del f[key]

            f[key] = evt


def main():
    args = parse_args()
    # process_wavs(args.wavs, args.out)
    events = get_events(args.hdfs)
    add_events_to_dataset(events, args.out)


if __name__ == "__main__":
    main()
