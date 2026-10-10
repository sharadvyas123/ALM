"""Versioned, exclusive-output generation and reproducibility verification."""

import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .auditing import audit_dataset, audit_rows, require_disjoint
from .mixing import check_audio, metadata_rows, plan_mixtures, render_dataset
from .provenance import (
    CLASS_NAMES, METADATA_FIELDS, load_sources, object_hash, read_csv, sha256,
    write_csv, write_json,
)
from .splitting import split_sources


def load_config(path, root):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config["eligibility_policy"] != "verified_provenance_only"
            or config["grouping"] != "connected_full_folder_or_valid_youtube_id"):
        raise ValueError("Unsupported provenance policy")
    if config["sample_rate"] != 16000 or config["duration_seconds"] != 10:
        raise ValueError("This version preserves the 16 kHz / 10 second preprocessing contract")
    if (config["alpha_range"] != [0.3, 0.7]
            or config["dirichlet_concentration"] != [1.0, 1.0, 1.0]
            or config["three_class_weight_floor"] != 0.2):
        raise ValueError("This version preserves the legacy mixture formulas")
    ratios = [config[k] for k in ("clean_ratio", "two_class_ratio", "three_class_ratio")]
    if ratios != [0.5, 0.35, 0.15]:
        raise ValueError("Expected the approved 50/35/15 mixture proportions")
    if set(config["counts_by_split"]) != {"train", "val"}:
        raise ValueError("Exactly train and val generation targets are required")
    for counts in config["counts_by_split"].values():
        if set(counts) != {"clean", "2-class", "3-class"} or any(type(v) is not int or v <= 0 for v in counts.values()):
            raise ValueError("Invalid output counts")
        total = sum(counts.values())
        if any(abs(counts[k] - total * ratio) > 1 for k, ratio in zip(("clean", "2-class", "3-class"), ratios)):
            raise ValueError("Output counts deviate from intended proportions beyond rounding")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Expected a nonnegative integer seed")
    paths = {k: (Path(root) / config[k]).resolve() for k in (
        "source_csv", "source_audio", "raw_training_csv", "raw_test_csv", "output_dir", "manifest_dir",
    )}
    for key, parent in (("output_dir", "data/synthetic"), ("manifest_dir", "data/manifests")):
        if paths[key].parent != (Path(root) / parent).resolve() or paths[key].name != config["dataset_version"]:
            raise ValueError(f"{key} must be a versioned child of {parent}")
    return config, paths


def prepare(config, paths):
    sources, provenance = load_sources(paths["source_csv"], paths["raw_training_csv"], paths["raw_test_csv"])
    assigned, split_summary = split_sources(
        sources, config["seed"], config["val_ratio"], config["split_candidates"],
        {s: config["counts_by_split"][s]["clean"] for s in ("train", "val")},
    )
    recipes = plan_mixtures(assigned, config)
    audit = audit_rows(metadata_rows(recipes), sources, paths["source_audio"], assigned)
    require_disjoint(audit)
    summary = {
        "dataset_version": config["dataset_version"], "provenance": provenance,
        "split": split_summary,
        "output_counts": config["counts_by_split"],
        "source_assignment_sha256": object_hash(assigned),
        "mixture_recipe_sha256": object_hash(recipes),
        "rng": {"source_split": [config["seed"]],
                "train_clean": [config["seed"], 0, 1], "train_mixtures": [config["seed"], 0, 2],
                "val_clean": [config["seed"], 1, 1], "val_mixtures": [config["seed"], 1, 2],
                "bit_generator": "PCG64", "numpy_version": np.__version__},
    }
    return sources, assigned, recipes, summary


def source_audio_manifest(assigned, paths, config):
    result = []
    for source in assigned:
        row = dict(source, audio_sha256="")
        if source["split"] in {"train", "val"}:
            path = paths["source_audio"] / source["file"]
            check_audio(path, config["sample_rate"], config["sample_rate"] * config["duration_seconds"])
            row["audio_sha256"] = sha256(path)
        result.append(row)
    return result


def verify_existing(config, paths, sources, assigned, recipes, summary):
    complete_path = paths["manifest_dir"] / "COMPLETE.json"
    if not complete_path.is_file():
        raise ValueError("Missing COMPLETE.json; dataset has not completed verification")
    complete = json.loads(complete_path.read_text())
    saved_config = json.loads((paths["manifest_dir"] / "generation_config.json").read_text())
    if config != saved_config["config"]:
        raise ValueError("Recipe configuration changed since generation")
    for key in ("source_assignment_sha256", "mixture_recipe_sha256"):
        if summary[key] != saved_config["plan"][key]:
            raise ValueError(f"Deterministic replay differs: {key}")
    for key, digest in saved_config["input_metadata_sha256"].items():
        if sha256(paths[key]) != digest:
            raise ValueError(f"Input metadata changed: {key}")
    root = Path(__file__).resolve().parents[3]
    for key, digest in saved_config["code_sha256"].items():
        if sha256(root / key) != digest:
            raise ValueError(f"Generation/audit code changed: {key}")
    for key, digest in complete["artifact_sha256"].items():
        if sha256(paths["manifest_dir"] / key) != digest:
            raise ValueError(f"Manifest artifact changed: {key}")
    for key, digest in complete["dataset_artifact_sha256"].items():
        if sha256(paths["output_dir"] / key) != digest:
            raise ValueError(f"Dataset artifact changed: {key}")
    expected_sources = source_audio_manifest(assigned, paths, config)
    saved_sources = read_csv(paths["manifest_dir"] / "source_splits.csv")
    if [{k: str(v) for k, v in row.items()} for row in expected_sources] != saved_sources:
        raise ValueError("Source assignments or source audio hashes changed")
    saved_recipes = read_csv(paths["manifest_dir"] / "mixture_manifest.csv")
    if [{k: row[k] for k in recipe} for row, recipe in zip(saved_recipes, recipes)] != recipes or len(saved_recipes) != len(recipes):
        raise ValueError("Mixture manifest differs from deterministic replay")
    report = audit_dataset(
        paths["output_dir"] / "metadata.csv", sources, paths["source_audio"],
        paths["manifest_dir"] / "source_splits.csv", paths["manifest_dir"] / "mixture_manifest.csv",
        paths["output_dir"] / "audio", True, config,
    )
    require_disjoint(report)
    return {"deterministic_replay": "passed", "source_audio_hashes_verified": sum(bool(r["audio_sha256"]) for r in expected_sources),
            "audio_files_checked": report["audio_files_checked"], "waveforms_reconstructed": report["waveforms_reconstructed"],
            "overlapping_source_count": report["overlapping_source_count"], "overlapping_group_count": report["overlapping_group_count"]}


def generate(config, paths, root, config_path, command):
    # Refuse before doing any writes, including on interrupted previous runs.
    for key in ("output_dir", "manifest_dir"):
        if paths[key].exists():
            raise FileExistsError(f"Refusing to reuse existing dataset version: {paths[key]}")
    input_hashes = {key: sha256(paths[key]) for key in ("source_csv", "raw_training_csv", "raw_test_csv")}
    sources, assigned, recipes, summary = prepare(config, paths)
    source_rows = source_audio_manifest(assigned, paths, config)
    # Preflight has passed; reserve both version directories exclusively.
    for key in ("output_dir", "manifest_dir"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].mkdir(exist_ok=False)
    audio_dir = paths["output_dir"] / "audio"
    audio_dir.mkdir()
    manifest = paths["manifest_dir"]
    write_csv(manifest / "source_splits.csv", source_rows)
    write_json(manifest / "labels.json", {str(i): name for i, name in enumerate(CLASS_NAMES)})
    code_paths = sorted((Path(root) / "src/alm/data").glob("*.py")) + [
        Path(root) / "scripts/generate_mixtures.py", Path(root) / "scripts/audit_source_overlap.py",
    ]
    snapshot = {
        "config": config, "plan": summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "config_file_sha256": sha256(config_path),
        "input_metadata_sha256": input_hashes,
        "code_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in code_paths},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_tracked_status": subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=root, text=True).strip(),
        "environment": {"python": platform.python_version(), "numpy": version("numpy"),
                        "soundfile": version("soundfile"), "libsndfile": sf.__libsndfile_version__},
        "limitations": ["Unknown video IDs for 150 official-test clips prevent full parent-video independence certification.",
                        "No fingerprint-based or cross-upload audio deduplication has been performed."],
    }
    write_json(manifest / "generation_config.json", snapshot)
    rendered = render_dataset(recipes, paths["source_audio"], audio_dir, config)
    write_csv(paths["output_dir"] / "metadata.csv", metadata_rows(recipes), METADATA_FIELDS)
    write_csv(manifest / "mixture_manifest.csv", rendered)
    totals = Counter()
    for counts in config["counts_by_split"].values():
        totals.update(counts)
    totals["total"] = len(recipes)
    dataset_config = dict(config, num_classes=7, label_name_by_id={str(i): n for i, n in enumerate(CLASS_NAMES)},
                          counts=dict(totals), generation_manifest=str(manifest.relative_to(root) / "generation_config.json"))
    write_json(paths["output_dir"] / "config.json", dataset_config)
    report = audit_dataset(
        paths["output_dir"] / "metadata.csv", sources, paths["source_audio"],
        manifest / "source_splits.csv", manifest / "mixture_manifest.csv", audio_dir, True, config,
    )
    actual_counts = {s: dict(Counter(r["mix_type"] for r in recipes if r["split"] == s)) for s in ("train", "val")}
    if actual_counts != config["counts_by_split"]:
        raise ValueError("Generated output count mismatch")
    write_json(manifest / "audit.json", report)
    require_disjoint(report)
    # Independently rerun pure planning and compare both complete manifests.
    _, replay_assigned, replay_recipes, _ = prepare(config, paths)
    if assigned != replay_assigned or recipes != replay_recipes:
        raise ValueError("Non-deterministic source or mixture planning")
    if source_audio_manifest(assigned, paths, config) != source_rows:
        raise ValueError("Source audio changed during generation")
    if any(sha256(paths[key]) != digest for key, digest in input_hashes.items()):
        raise ValueError("Input metadata changed during generation")
    complete = {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_replay": "passed", "source_audio_hashes_verified": sum(bool(r["audio_sha256"]) for r in source_rows),
        "artifact_sha256": {name: sha256(manifest / name) for name in (
            "source_splits.csv", "mixture_manifest.csv", "labels.json", "generation_config.json", "audit.json",
        )},
        "dataset_artifact_sha256": {name: sha256(paths["output_dir"] / name) for name in ("metadata.csv", "config.json")},
    }
    write_json(manifest / "COMPLETE.json", complete)
    return summary, report
