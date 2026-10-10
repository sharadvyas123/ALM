"""Measure source audibility in the v1 synthetic mixtures.

For each weighted source in each generated example, this computes:

* ``snr_db = 10 log10(E_i / sum(E_other))``;
* ideal-binary-mask share: the fraction of active time-frequency bins where
  source i has strictly more than 50% of the source energy.

Only frames whose center occurs before the source's original duration are used
for that source. The zero-padded tail is excluded. This script reads the source
audio and manifests and writes analysis artifacts outside the dataset directory.

Example:

    venv/bin/python -B scripts/audibility_analysis.py \
      --output-dir research/audibility/mad_source_disjoint_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alm.data.provenance import CLASS_NAMES, read_csv, write_csv, write_json


def stft_power(audio: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Return (frequency, time) power using centered periodic-Hann frames."""
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")
    pad = n_fft // 2
    padded = np.pad(audio.astype(np.float64), (pad, pad))
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length]
    window = np.hanning(n_fft + 1)[:-1]
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    return (np.abs(spectrum) ** 2).T


def parse_weights(value: str) -> list[float]:
    weights = [float(item) for item in value.split("|")]
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError(f"Invalid weights: {value}")
    if abs(sum(weights) - 1.0) > 0.0002:
        raise ValueError(f"Weights do not sum to one: {value}")
    return weights


def analyze_row(row, source_by_file, source_audio, sample_rate, n_fft, hop_length, below_db):
    files = row["source_files"].split("|")
    weights = parse_weights(row["mix_weights"])
    if len(files) != len(weights):
        raise ValueError(f"Source/weight count mismatch in {row['file']}")

    weighted_powers = []
    durations = []
    for file, weight in zip(files, weights):
        source = source_by_file.get(file)
        if source is None:
            raise ValueError(f"No processed metadata for source {file}")
        audio, sr = sf.read(str(source_audio / file), dtype="float64")
        if sr != sample_rate or audio.ndim != 1:
            raise ValueError(f"Unexpected source audio for {file}: sr={sr}, shape={audio.shape}")
        weighted_powers.append(stft_power(audio * weight, n_fft, hop_length))
        durations.append(float(source["duration_original"]))

    shape = weighted_powers[0].shape
    if any(power.shape != shape for power in weighted_powers):
        raise ValueError(f"STFT shape mismatch in {row['file']}")
    total_power = sum(weighted_powers)
    frame_centers = np.arange(shape[1]) * hop_length
    result = []
    for index, (file, weight, power, duration) in enumerate(zip(files, weights, weighted_powers, durations)):
        active_frames = frame_centers < duration * sample_rate
        active_total = total_power[:, active_frames]
        active_source = power[:, active_frames]
        other_power = active_total - active_source
        source_energy = float(active_source.sum())
        other_energy = float(other_power.sum())
        if other_energy <= 1e-20:
            snr_db = float("inf")
        elif source_energy <= 1e-20:
            snr_db = float("-inf")
        else:
            snr_db = float(10.0 * np.log10(source_energy / other_energy))
        non_silent = active_total > 1e-20
        dominant = (active_source > 0.5 * active_total) & non_silent
        mask_share = float(dominant.sum() / non_silent.sum()) if non_silent.any() else float("nan")
        result.append({
            "generated_file": row["file"], "generated_id": row["id"], "split": row["split"],
            "mix_type": row["mix_type"], "num_sources": len(files), "source_index": index,
            "source_file": file, "class_id": int(source_by_file[file]["label"]),
            "class_name": source_by_file[file]["label_name"], "weight": weight,
            "duration_original_seconds": duration, "active_frames": int(active_frames.sum()),
            "snr_db": snr_db, "snr_is_infinite": not math.isfinite(snr_db),
            "mask_share": mask_share, "active_tf_bins": int(non_silent.sum()),
            "below_threshold": bool(math.isfinite(snr_db) and snr_db < below_db),
        })
    return result


def finite_or_empty(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize(records, below_db):
    groups = []
    keys = sorted({(r["class_name"], r["mix_type"]) for r in records},
                  key=lambda item: (CLASS_NAMES.index(item[0]), item[1]))
    for class_name, mix_type in keys:
        subset = [r for r in records if r["class_name"] == class_name and r["mix_type"] == mix_type]
        finite = finite_or_empty([r["snr_db"] for r in subset])
        groups.append({
            "class_name": class_name, "mix_type": mix_type, "records": len(subset),
            "finite_snr_records": len(finite), "infinite_snr_records": len(subset) - len(finite),
            "snr_mean_db": float(np.mean(finite)) if finite else float("nan"),
            "snr_median_db": float(np.median(finite)) if finite else float("nan"),
            "snr_min_db": float(np.min(finite)) if finite else float("nan"),
            "snr_max_db": float(np.max(finite)) if finite else float("nan"),
            "mask_share_mean": float(np.nanmean([r["mask_share"] for r in subset])),
            "below_threshold": sum(r["below_threshold"] for r in subset),
            "below_threshold_fraction": float(sum(r["below_threshold"] for r in subset) / len(subset)),
            "threshold_db": below_db,
        })
    return groups


def plot_histograms(records, output_path, below_db):
    mix_types = ("clean", "2-class", "3-class")
    figure, axes = plt.subplots(len(CLASS_NAMES), len(mix_types), figsize=(15, 22), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(CLASS_NAMES), len(mix_types))
    for row_index, class_name in enumerate(CLASS_NAMES):
        for col_index, mix_type in enumerate(mix_types):
            axis = axes[row_index, col_index]
            values = [r["snr_db"] for r in records
                      if r["class_name"] == class_name and r["mix_type"] == mix_type and math.isfinite(r["snr_db"])]
            infinite = sum(r["class_name"] == class_name and r["mix_type"] == mix_type and not math.isfinite(r["snr_db"])
                           for r in records)
            if values:
                axis.hist(values, bins=30, color="#4fc3f7", alpha=0.85)
            axis.axvline(below_db, color="#e57373", linestyle="--", linewidth=1)
            axis.set_title(f"{class_name} / {mix_type}\nfinite={len(values)}, +inf={infinite}", fontsize=8)
            axis.grid(alpha=0.25)
    figure.supxlabel("Source-to-rest SNR (dB); finite values only")
    figure.supylabel("Source components")
    figure.suptitle("MAD source audibility by class and mixture type")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/synthetic/mad_source_disjoint_v1")
    parser.add_argument("--source-metadata", type=Path, default=ROOT / "data/processed_mad/metadata.csv")
    parser.add_argument("--source-audio", type=Path, default=ROOT / "data/processed_mad/audio")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--below-db", type=float, default=-15.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse analysis output directory: {args.output_dir}")
    if args.n_fft <= 0 or args.hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")

    metadata = read_csv(args.dataset_dir / "metadata.csv")
    manifest = read_csv(args.dataset_dir.parent.parent / "manifests" / args.dataset_dir.name / "mixture_manifest.csv")
    source_metadata = read_csv(args.source_metadata)
    source_by_file = {row["file"]: row for row in source_metadata}
    recipes = {row["file"]: row for row in manifest}
    if len(metadata) != len(manifest) or set(row["file"] for row in metadata) != set(recipes):
        raise ValueError("Dataset metadata and mixture manifest do not match")

    records = []
    for index, row in enumerate(metadata, start=1):
        recipe = recipes[row["file"]]
        if any(row[key] != recipe[key] for key in ("id", "file", "mix_type", "source_files", "mix_weights", "split")):
            raise ValueError(f"Metadata/recipe mismatch for {row['file']}")
        records.extend(analyze_row(row, source_by_file, args.source_audio, 16000, args.n_fft, args.hop_length, args.below_db))
        if index % 500 == 0 or index == len(metadata):
            print(f"Analyzed {index}/{len(metadata)} mixtures", flush=True)

    args.output_dir.mkdir(parents=True)
    fields = list(records[0])
    write_csv(args.output_dir / "audibility_records.csv", records, fields)
    summary = summarize(records, args.below_db)
    write_csv(args.output_dir / "audibility_summary.csv", summary, list(summary[0]))
    plot_histograms(records, args.output_dir / "snr_histograms.png", args.below_db)
    finite = finite_or_empty([r["snr_db"] for r in records])
    write_json(args.output_dir / "analysis_config.json", {
        "dataset_dir": str(args.dataset_dir), "source_metadata": str(args.source_metadata),
        "source_audio": str(args.source_audio), "n_fft": args.n_fft, "hop_length": args.hop_length,
        "sample_rate": 16000, "below_db": args.below_db, "records": len(records),
        "infinite_snr_records": sum(not math.isfinite(r["snr_db"]) for r in records),
        "finite_snr_records": len(finite),
        "below_threshold_records": sum(r["below_threshold"] for r in records),
        "below_threshold_fraction": sum(r["below_threshold"] for r in records) / len(records),
        "padding_policy": "per-source active frames use duration_original; zero-padded tails excluded",
        "mask_policy": "strict source power > 50% of total power, non-silent active bins only",
    })
    print(json.dumps({
        "status": "PASS", "output_dir": str(args.output_dir), "source_records": len(records),
        "finite_snr_records": len(finite), "infinite_snr_records": len(records) - len(finite),
        "below_threshold_records": sum(r["below_threshold"] for r in records),
        "below_threshold_fraction": sum(r["below_threshold"] for r in records) / len(records),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
