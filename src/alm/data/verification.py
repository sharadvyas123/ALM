"""Validate the existing certificate, then reuse the source-disjoint audit."""
from __future__ import annotations

import json
from pathlib import Path

from .auditing import audit_dataset, require_disjoint
from .generation import load_config
from .provenance import CLASS_NAMES, read_csv, sha256


def verify_dataset(root: Path, data_config: str = "configs/data/mad_source_disjoint_v1.json",
                   *, verify_waveforms: bool = True) -> dict:
    """Verify hashes, class order, counts, source pools and all generated audio.

    Official-test audio and raw test CSV are never opened. The authenticated
    source manifest carries the existing exclusion/group identities; its
    original provenance limitations continue to apply.
    """
    cfg, paths = load_config(root / data_config, root)
    manifest = paths["manifest_dir"]
    complete = json.loads((manifest / "COMPLETE.json").read_text())
    if complete["status"] != "complete" or complete["deterministic_replay"] != "passed":
        raise ValueError("Dataset is incomplete")
    for name, digest in complete["artifact_sha256"].items():
        if sha256(manifest / name) != digest:
            raise ValueError(f"Manifest checksum mismatch: {name}")
    for name, digest in complete["dataset_artifact_sha256"].items():
        if sha256(paths["output_dir"] / name) != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    saved = json.loads((manifest / "generation_config.json").read_text())
    if cfg != saved["config"]:
        raise ValueError("Generation configuration changed")
    for file, digest in saved["code_sha256"].items():
        if sha256(root / file) != digest:
            raise ValueError(f"Certified generation code changed: {file}")
    if sha256(paths["source_csv"]) != saved["input_metadata_sha256"]["source_csv"]:
        raise ValueError("Processed source metadata changed")
    labels = json.loads((manifest / "labels.json").read_text())
    if labels != {str(i): name for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Class order mismatch")
    sources = read_csv(manifest / "source_splits.csv")
    originals = [dict(s, split="eligible" if s["split"] in {"train", "val"} else s["split"])
                 for s in sources]
    report = audit_dataset(paths["output_dir"] / "metadata.csv", originals, paths["source_audio"],
                           manifest / "source_splits.csv", manifest / "mixture_manifest.csv",
                           paths["output_dir"] / "audio", verify_waveforms, cfg)
    require_disjoint(report)
    actual = {s: {k: v["total"] for k, v in counts.items()} for s, counts in report["mix_counts"].items()}
    if actual != cfg["counts_by_split"]:
        raise ValueError("Mixture counts do not match the versioned configuration")
    report["manifest_sha256"] = sha256(manifest / "mixture_manifest.csv")
    report["certificate_sha256"] = sha256(manifest / "COMPLETE.json")
    return report
