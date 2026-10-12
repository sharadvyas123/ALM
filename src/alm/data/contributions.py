"""Read-only reconstruction adapter around the checksum-locked v1 mixer.

Kept separate from mixing.py because that file's hash is part of the existing
dataset completion certificate. No generation protocol or certificate changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .mixing import check_audio, render_recipe
from .provenance import relative_id, sha256


def reconstruct_contributions(recipe: dict, source_audio: Path, mixture_audio: Path,
                              source_by_file: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Verify PCM16 replay, then return actual mixture and pre-quantisation stems.

    Stems include full precision gains and the common peak scale. Integer
    truncation cannot be uniquely allocated to sources: the final residual is
    bounded by one PCM16 unit and is retained in x-s_c for SIR calculations.
    """
    if recipe["split"] not in {"train", "val"}:
        raise ValueError("Only train/val recipes are permitted")
    components = json.loads(recipe["components"])
    files = recipe["source_files"].split("|")
    weights = [float(v) for v in recipe["mix_weights"].split("|")]
    if len(components) != len(files) or len(files) != len(weights):
        raise ValueError("Recipe component count mismatch")
    stems = []
    for component, file, weight in zip(components, files, weights):
        source = source_by_file[relative_id(file)]
        expected = dict(file=file, source_id=source["source_id"], group_id=source["group_id"],
                        video_id=source["video_id"], weight=weight, offset_samples=0)
        if (component != expected or source["split"] != recipe["split"]
                or source["official_split"] != "training"):
            raise ValueError("Component provenance or source-pool mismatch")
        path = source_audio / relative_id(file)
        if sha256(path) != source["audio_sha256"]:
            raise ValueError(f"Source checksum mismatch: {file}")
        audio, _ = sf.read(path, dtype="int16")
        stems.append(audio.astype(np.float64) * weight / 32768.0)
    expected, scale = render_recipe(recipe, source_audio, config)
    path = mixture_audio / relative_id(recipe["file"])
    check_audio(path, config["sample_rate"], config["sample_rate"] * config["duration_seconds"])
    actual, _ = sf.read(path, dtype="int16")
    if (sha256(path) != recipe["audio_sha256"] or not np.array_equal(actual, expected)
            or scale != float(recipe["peak_normalization_scale"])):
        raise ValueError(f"Mixture reconstruction/checksum mismatch: {recipe['file']}")
    stems = np.stack(stems) * scale
    mixture = actual.astype(np.float64) / 32768.0
    tolerance = (len(stems) + 1.00001) / 32768
    if np.max(np.abs(stems.sum(axis=0) - mixture)) > tolerance:
        raise ValueError("Unexplained rendering residual")
    return mixture, stems
