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
    with h5py.File(output_hdf, "a") as f:
        f.attrs["nfft"] = 1024
        f.attrs["noverlap"] = 1024 - 256
        f.attrs["nmels"] = 80
        f.attrs["fmin"] = 500
        f.attrs["sr"] = 125000

        for wav_file in tqdm(wav_files):
            if wav_file.stem in f:
                continue
            sr, audio = wavfile.read(wav_file)
            audio = audio.astype(np.float32)
            audio /= 0.005  # approximate standard deviation of the data
            mel_spec = make_mel_spectrogram(
                audio, nfft=1024, noverlap=1024 - 256, nmels=80
            )  # shape is (time, freq)
            f.create_dataset(wav_file.stem, data=mel_spec, chunks=(1024, 80))


def main():
    args = parse_args()
    process_wavs(args.wavs, args.out)


if __name__ == "__main__":
    main()
