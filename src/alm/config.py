"""Strict JSON experiment configuration loading and path resolution."""
from __future__ import annotations

import json
from pathlib import Path

from .data.provenance import CLASS_NAMES

EXPERIMENT_SWITCHES = (
    "use_fusion", "use_class_queries", "use_mask_supervision", "use_audibility_labels",
    "use_audibility_loss", "use_asymmetric_loss", "use_confusion_weights", "use_operational_costs",
)


def load_json(path: Path) -> dict:
    """Load an object JSON document without accepting NaN/Infinity."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def validate_experiment_config(config: dict) -> dict:
    """Validate switches, class order, and required training/model sections."""
    required = {"run_id", "dataset_dir", "seed", "model", "loss", "training", "preprocessing"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment config missing keys: {sorted(missing)}")
    if not isinstance(config["run_id"], str) or not config["run_id"]:
        raise ValueError("run_id must be a nonempty string")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not set(EXPERIMENT_SWITCHES) <= set(config["model"]):
        missing = set(EXPERIMENT_SWITCHES) - set(config["model"])
        raise ValueError(f"model switch keys missing: {sorted(missing)}")
    if config.get("class_order", list(CLASS_NAMES)) != list(CLASS_NAMES):
        raise ValueError("class order does not match the canonical seven classes")
    preprocessing = config["preprocessing"]
    expected = {"sample_rate": 16000, "duration_seconds": 10, "n_fft": 1024,
                "hop_length": 512, "n_mels": 128}
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(f"preprocessing.{key} must be {value!r}")
    training = config["training"]
    for key in ("batch_size", "epochs"):
        if type(training.get(key)) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if training.get("checkpoint_metric") not in {"mAP", "macro_f1"}:
        raise ValueError("checkpoint_metric must be mAP or macro_f1")
    loss = config["loss"]
    if loss.get("lambda_mask", 0.0) < 0:
        raise ValueError("loss.lambda_mask must be nonnegative")
    if config["model"]["use_mask_supervision"] and not config["model"]["use_class_queries"]:
        raise ValueError("mask supervision requires class queries")
    if config["model"]["use_audibility_labels"] and config.get("audibility", {}).get("policy") == "union":
        raise ValueError("audibility labels are enabled but policy is union")
    if config["model"]["use_operational_costs"] and not config.get("operational_costs", {}).get("matrix"):
        raise ValueError("operational costs require an explicit matrix")
    return config


def load_experiment(path: Path) -> dict:
    return validate_experiment_config(load_json(path))


def resolve_experiment_paths(root: Path, config: dict) -> dict[str, Path]:
    """Resolve paths while refusing a legacy dataset selection."""
    dataset = (root / config["dataset_dir"]).resolve()
    if "multilabel_mad" in dataset.as_posix() or not dataset.is_dir():
        raise ValueError(f"Forbidden or missing dataset directory: {dataset}")
    required = {"metadata": dataset / "metadata.csv", "audio": dataset / "audio",
                "manifest": root / "data/manifests" / dataset.name,
                "source_metadata": root / "data/processed_mad/metadata.csv",
                "source_audio": root / "data/processed_mad/audio"}
    if config["model"]["use_mask_supervision"] or config["model"]["use_audibility_labels"]:
        required.update({"mixture_manifest": required["manifest"] / "mixture_manifest.csv",
                         "source_manifest": required["manifest"] / "source_splits.csv"})
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing experiment path {name}: {path}")
    return required
