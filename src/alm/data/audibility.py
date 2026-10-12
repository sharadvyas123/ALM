"""Audibility measurements and deterministic source-mask construction.

The functions in this module operate on *rendered source contributions*, not
on source metadata alone.  For a mixture ``x = sum_c s_c`` and a contribution
``s_c`` we define the source-to-rest ratio as

``10 * log10((sum(s_c**2) + eps) / (sum((x - s_c)**2) + eps))``.

This is an energy ratio for the synthetic rendering process.  It is useful for
diagnostics and controlled label policies, but it is not a psychoacoustic
audibility model.  The ideal ratio masks use the power of the gain-adjusted
source contributions and therefore retain overlapping sources rather than
forcing a single active class per time-frequency bin.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable

import numpy as np


def _validate_contributions(contributions: np.ndarray) -> np.ndarray:
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("contributions must have shape (num_sources, num_samples)")
    if not np.isfinite(values).all():
        raise ValueError("contributions contain non-finite values")
    return values


def source_to_rest_ratio_db(
    source: np.ndarray,
    mixture: np.ndarray,
    *,
    epsilon: float = 1e-12,
    active_samples: np.ndarray | None = None,
) -> float:
    """Return the source-to-rest energy ratio in dB.

    ``mixture - source`` is used for the rest term so the calculation follows
    the rendered mixture, including the same gain and common peak scale.  An
    optional boolean mask excludes padded source tails.  A nonzero source in a
    silent rest is ``+inf``; a silent source is ``-inf``.  These conventions
    avoid treating silence as perfectly audible.
    """

    source_values = np.asarray(source, dtype=np.float64).reshape(-1)
    mixture_values = np.asarray(mixture, dtype=np.float64).reshape(-1)
    if (source_values.size == 0 or not np.isfinite(source_values).all()
            or not np.isfinite(mixture_values).all()):
        raise ValueError("waveforms must be non-empty and finite")
    if source_values.shape != mixture_values.shape:
        raise ValueError("source and mixture must have the same number of samples")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number")
    if active_samples is None:
        active = np.ones(source_values.shape, dtype=bool)
    else:
        active = np.asarray(active_samples, dtype=bool).reshape(-1)
        if active.shape != source_values.shape:
            raise ValueError("active_samples must match the waveform shape")
    source_energy = float(np.sum(np.square(source_values[active]), dtype=np.float64))
    rest = mixture_values - source_values
    rest_energy = float(np.sum(np.square(rest[active]), dtype=np.float64))
    if source_energy <= epsilon:
        return float("-inf")
    if rest_energy <= epsilon:
        return float("inf")
    return float(10.0 * np.log10((source_energy + epsilon) / (rest_energy + epsilon)))


def stft_power(
    waveform: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Return centered periodic-Hann STFT power with shape ``(F, T)``.

    Reflection padding matches the existing B0 torchaudio transform. With
    160,000 samples, ``n_fft=1024`` and
    ``hop_length=512`` this produces 513 frequency bins and 313 frames.
    """

    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be non-empty and finite")
    if n_fft <= 0 or hop_length <= 0 or n_fft % 2:
        raise ValueError("n_fft must be positive and even; hop_length must be positive")
    pad = n_fft // 2 if center else 0
    if pad_mode not in {"reflect", "constant"}:
        raise ValueError("pad_mode must be reflect or constant")
    padded = np.pad(values, (pad, pad), mode=pad_mode) if center else values
    if padded.size < n_fft:
        padded = np.pad(padded, (0, n_fft - padded.size))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1]
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    return np.square(np.abs(spectrum), dtype=np.float64).T


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=32)
def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Construct an unnormalised HTK triangular mel filterbank (B0 convention).

    The filterbank is used only to put source power masks on the same
    frequency coordinate as the classifier's log-mel input.  It does not
    claim that mel power is a perceptual source separator.
    """

    if sample_rate <= 0 or n_fft <= 0 or n_mels <= 0:
        raise ValueError("sample_rate, n_fft, and n_mels must be positive")
    nyquist = sample_rate / 2.0
    f_max = nyquist if f_max is None else float(f_max)
    if not 0 <= f_min < f_max <= nyquist:
        raise ValueError("mel frequency limits must satisfy 0 <= f_min < f_max <= Nyquist")
    mel_points = np.linspace(_hz_to_mel(np.array([f_min]))[0],
                             _hz_to_mel(np.array([f_max]))[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    frequencies = np.linspace(0.0, nyquist, n_fft // 2 + 1)
    filters = np.zeros((n_mels, frequencies.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = hz_points[index:index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters


def source_time_frequency_masks(
    contributions: np.ndarray,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int | None = 128,
    f_min: float = 0.0,
    f_max: float | None = 8_000.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return ideal ratio masks with shape ``(sources, frequency, time)``.

    For source power ``P_c`` the mask is ``P_c / (sum_j P_j + epsilon)``.
    When ``n_mels`` is set, source powers are projected to mel power before
    normalisation.  The masks are finite in ``[0, 1]`` and can overlap.
    """

    values = _validate_contributions(contributions)
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number")
    powers = np.stack([
        stft_power(source, n_fft=n_fft, hop_length=hop_length)
        for source in values
    ], axis=0)
    if n_mels is not None:
        filters = mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                 f_min=f_min, f_max=f_max)
        # tensordot avoids rebuilding an einsum contraction path for every
        # mixture. The result is (sources, time, mel), then transposed.
        powers = np.tensordot(powers, filters, axes=([1], [1])).transpose(0, 2, 1)
    denominator = powers.sum(axis=0, keepdims=True)
    masks = np.divide(powers, denominator, out=np.zeros_like(powers), where=denominator > epsilon)
    return np.clip(np.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def dominant_mask_share(
    masks: np.ndarray,
    *,
    source_index: int,
    threshold: float = 0.5,
    epsilon: float = 1e-12,
) -> float:
    """Return the fraction of non-silent bins where one source dominates."""

    values = np.asarray(masks, dtype=np.float64)
    if values.ndim != 3 or not 0 <= source_index < values.shape[0]:
        raise ValueError("masks must have shape (sources, frequency, time)")
    active = values.sum(axis=0) > epsilon
    return float(np.mean((values[source_index] > threshold)[active])) if active.any() else 0.0


def resize_masks(masks: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Bilinearly align ``(sources, H, W)`` masks to a model attention grid."""

    import torch
    import torch.nn.functional as functional

    values = np.asarray(masks, dtype=np.float32)
    if values.ndim != 3 or target_height <= 0 or target_width <= 0:
        raise ValueError("masks must be (sources, H, W) and targets must be positive")
    resized = functional.interpolate(
        torch.from_numpy(values[:, None]), size=(target_height, target_width),
        mode="bilinear", align_corners=False,
    )[:, 0]
    return resized.clamp(0.0, 1.0).numpy()


def audibility_summary(records: Iterable[dict], *, threshold_db: float = -15.0) -> list[dict]:
    """Summarise audibility records by class and mixture type."""

    from collections import defaultdict

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(str(record["class_name"]), str(record["mix_type"]))].append(record)
    result = []
    for (class_name, mix_type), group in sorted(grouped.items()):
        finite = np.asarray([r["snr_db"] for r in group if math.isfinite(float(r["snr_db"]))])
        shares = np.asarray([float(r.get("mask_share", 0.0)) for r in group], dtype=float)
        below = [float(r["snr_db"]) < threshold_db for r in group if math.isfinite(float(r["snr_db"]))]
        result.append({
            "class_name": class_name,
            "mix_type": mix_type,
            "records": len(group),
            "finite_snr_records": int(finite.size),
            "infinite_snr_records": int(len(group) - finite.size),
            "snr_median_db": float(np.median(finite)) if finite.size else float("nan"),
            "snr_q25_db": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
            "snr_q75_db": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
            "mask_share_mean": float(np.nanmean(shares)) if shares.size else float("nan"),
            "below_threshold": int(sum(below)),
            "below_threshold_fraction": float(sum(below) / len(group)) if group else 0.0,
            "threshold_db": float(threshold_db),
        })
    return result
