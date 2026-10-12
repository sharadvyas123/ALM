"""Shared features matching the completed B0 notebook (natural log, no augmentation)."""
from __future__ import annotations

import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

PREPROCESSING = dict(sample_rate=16000, num_samples=160000, n_fft=1024,
                     hop_length=512, n_mels=128, f_min=0.0, f_max=8000.0,
                     log_epsilon=1e-6, pad_mode="reflect")


class LogMel(nn.Module):
    """Power mel with HTK filters, periodic Hann, centered reflection padding."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = dict(PREPROCESSING if config is None else config)
        cfg = self.config
        self.mel = MelSpectrogram(
            sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
            f_min=cfg["f_min"], f_max=cfg["f_max"], power=2.0,
            center=True, pad_mode=cfg["pad_mode"], norm=None, mel_scale="htk",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map ``(..., samples)`` to ``(..., 1, mel, time)``."""
        if waveform.shape[-1] != self.config["num_samples"] or not torch.isfinite(waveform).all():
            raise ValueError("waveform must be finite and satisfy the configured sample count")
        result = torch.log(self.mel(waveform) + self.config["log_epsilon"])
        if result.ndim == 3:  # (B, mel, time) or (1, mel, time)
            return result.unsqueeze(1)
        if result.ndim == 2:  # defensive path for an unusual torchaudio input
            return result.unsqueeze(0).unsqueeze(0)
        if result.ndim == 4 and result.shape[1] == 1:
            return result
        raise ValueError(f"Unexpected mel shape: {tuple(result.shape)}")
