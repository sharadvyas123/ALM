"""Run the single preflight required before B0 training.

Checks the source-disjoint v1 metadata, seven-class order, generated audio
format, provenance disjointness, real log-mel/model tensor shapes, and active
paths. It refuses the legacy ``data/multilabel_mad`` dataset.

Example:

    venv/bin/python -B scripts/preflight_v1.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

from alm.data.auditing import audit_dataset, require_disjoint
from alm.data.provenance import CLASS_NAMES, METADATA_FIELDS, load_sources, read_csv
from utils.enhanced_alm_mad_encoder import EnhancedALMEncoder


LEGACY_MARKER = "data/multilabel_mad"


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/synthetic/mad_source_disjoint_v1")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests/mad_source_disjoint_v1")
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/processed_mad/metadata.csv")
    parser.add_argument("--source-audio", type=Path, default=ROOT / "data/processed_mad/audio")
    parser.add_argument("--raw-training", type=Path, default=ROOT / "data/raw/MAD_dataset/training.csv")
    parser.add_argument("--raw-test", type=Path, default=ROOT / "data/raw/MAD_dataset/test.csv")
    args = parser.parse_args()

    for path in (args.dataset_dir, args.manifest_dir):
        if LEGACY_MARKER in path.as_posix():
            fail(f"Legacy dataset path is forbidden: {path}")
        if not path.exists():
            fail(f"Missing v1 path: {path}")

    metadata_path = args.dataset_dir / "metadata.csv"
    audio_dir = args.dataset_dir / "audio"
    source_manifest_path = args.manifest_dir / "source_splits.csv"
    mixture_manifest_path = args.manifest_dir / "mixture_manifest.csv"
    for path in (metadata_path, audio_dir, source_manifest_path, mixture_manifest_path):
        if not path.exists():
            fail(f"Missing preflight input: {path}")
        if LEGACY_MARKER in path.as_posix():
            fail(f"Legacy dataset path is forbidden: {path}")

    rows = read_csv(metadata_path, METADATA_FIELDS)
    if len(rows) != 6429:
        fail(f"Expected 6429 generated rows, found {len(rows)}")
    split_counts = Counter(row["split"] for row in rows)
    if split_counts != Counter({"train": 5464, "val": 965}):
        fail(f"Unexpected split counts: {split_counts}")
    mix_counts = Counter((row["split"], row["mix_type"]) for row in rows)
    expected_mix = {
        ("train", "clean"): 2732, ("train", "2-class"): 1912, ("train", "3-class"): 820,
        ("val", "clean"): 482, ("val", "2-class"): 338, ("val", "3-class"): 145,
    }
    if mix_counts != Counter(expected_mix):
        fail(f"Unexpected mix counts: {mix_counts}")
    for row in rows:
        if any(LEGACY_MARKER in str(value) for value in row.values()):
            fail(f"Legacy path reference in generated metadata row {row['id']}")
        vector = [int(value) for value in row["label_vector"].split(",")]
        if len(vector) != 7:
            fail(f"Label vector width is not 7 in row {row['id']}")
        labels = [int(value) for value in row["labels"].split(",")]
        if vector != [int(class_id in labels) for class_id in range(7)]:
            fail(f"Incorrect label union in row {row['id']}")

    sources, provenance = load_sources(args.source_csv, args.raw_training, args.raw_test)
    report = audit_dataset(
        metadata_path, sources, args.source_audio, source_manifest_path,
        mixture_manifest_path, audio_dir, False,
        {"sample_rate": 16000, "duration_seconds": 10},
    )
    require_disjoint(report)
    if report["split_counts"] != {"train": 5464, "val": 965}:
        fail(f"Audit split counts differ: {report['split_counts']}")

    sample = rows[0]
    info = sf.info(str(audio_dir / sample["file"]))
    if (info.samplerate, info.frames, info.channels, info.format, info.subtype) != (16000, 160000, 1, "WAV", "PCM_16"):
        fail(f"Unexpected audio format: {info}")
    audio, sample_rate = sf.read(str(audio_dir / sample["file"]), dtype="float32")
    if sample_rate != 16000 or audio.shape != (160000,):
        fail(f"Unexpected sample audio shape: {audio.shape} at {sample_rate} Hz")

    mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512,
                                     n_mels=128, f_min=20.0, f_max=8000.0)
    waveform = torch.from_numpy(audio).reshape(1, 1, -1)
    mel = mel_transform(waveform)
    model = EnhancedALMEncoder(
        embed_dim=256, num_classes=7, num_heads=8, num_layers=6,
        ff_dim=1024, dropout=0.1, use_fusion=False, use_class_queries=False,
        max_time_frames=100,
    ).eval()
    with torch.no_grad():
        features = model.backbone(mel)
        tokens = features.reshape(features.shape[0], features.shape[1] * features.shape[2], features.shape[3]).permute(0, 2, 1)
        logits = model(mel)
    if tuple(mel.shape) != (1, 1, 128, 313):
        fail(f"Hop-512 log-mel shape is not (1,1,128,313): {tuple(mel.shape)}")
    if tuple(features.shape) != (1, 256, 8, 78) or tuple(tokens.shape) != (1, 78, 2048):
        fail(f"Unexpected model token trace: features={tuple(features.shape)}, tokens={tuple(tokens.shape)}")
    if tuple(logits.shape) != (1, 7):
        fail(f"Unexpected logits shape: {tuple(logits.shape)}")

    print(json.dumps({
        "status": "PASS",
        "dataset": str(args.dataset_dir),
        "class_order": list(CLASS_NAMES),
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "mix_counts": {f"{split}/{mix}": count for (split, mix), count in sorted(mix_counts.items())},
        "audit": {"overlapping_source_count": report["overlapping_source_count"],
                  "overlapping_group_count": report["overlapping_group_count"],
                  "known_official_test_group_overlap": report["known_official_test_group_overlap"]},
        "audio": {"sample_rate": sample_rate, "frames": int(audio.shape[0]), "format": info.format, "subtype": info.subtype},
        "model": {"hop_length": 512, "mel": tuple(mel.shape), "backbone": tuple(features.shape),
                  "tokens": tuple(tokens.shape), "logits": tuple(logits.shape), "token_count": int(tokens.shape[1])},
        "provenance": provenance,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
