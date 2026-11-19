"""Run resumable, nested, source-grouped 5-fold ALM training.

Prepare certified data first with scripts/prepare_cv5.py, then run:
    python scripts/train_cv5.py --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from alm.config import load_json
from alm.data.dataset import MADMixtureDataset
from alm.data.provenance import CLASS_NAMES, METADATA_FIELDS, object_hash, read_csv, sha256
from alm.evaluation.cv_figures import (class_metric_heatmap, fold_comparison, fold_curves,
                                      outer_pr_curve, spectrogram_examples)
from alm.losses.combined import CombinedLoss
from alm.losses.cost_sensitive import estimate_training_cooccurrence
from alm.models.alm_model import ALMModel
from alm.progress import tqdm
from alm.training.engine import Trainer
from alm.training.metrics import calibrate_thresholds, multilabel_metrics
from alm.training.reproducibility import environment_record, seed_everything, worker_seed


def _build_dataset(metadata: Path, audio: Path, cache: Path, split: str, config: dict,
                   source_metadata: Path, mix_manifest: Path, source_manifest: Path,
                   source_audio: Path) -> MADMixtureDataset:
    return MADMixtureDataset(
        metadata, audio, split=split, preprocessing=config["preprocessing"],
        label_policy=config["audibility"]["policy"],
        audibility_threshold_db=config["audibility"]["threshold_db"],
        soft_audibility_center_db=config["audibility"]["center_db"],
        soft_audibility_temperature_db=config["audibility"]["temperature_db"],
        soft_audibility_floor=config["audibility"]["soft_floor"],
        return_aux_targets=True, source_metadata_path=source_metadata,
        mixture_manifest_path=mix_manifest, source_manifest_path=source_manifest,
        source_audio_dir=source_audio, cache_dir=cache,
    )


def _targets(dataset: MADMixtureDataset) -> np.ndarray:
    return np.asarray([[int(v) for v in row["label_vector"].split(",")] for row in dataset.rows], dtype=np.float32)


def _collect(model, loader, device):
    model.eval()
    targets, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Collecting predictions", unit="batch",
                          leave=False, dynamic_ncols=True, mininterval=0.5):
            output = model(batch["mel"].to(device), return_aux=False)
            targets.append(batch.get("union_target", batch["target"]).numpy())
            scores.append(torch.sigmoid(output).float().cpu().numpy())
    return np.concatenate(targets), np.concatenate(scores)


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    temp = path.with_name(path.stem + ".tmp.npz")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _event(run_root: Path, event: str, **details) -> None:
    record = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "event": event, **details}
    with (run_root / "run_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fold_fingerprint(config_path: Path, fold_manifest: Path, source_config: Path) -> str:
    inputs = [config_path, fold_manifest]
    payload = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    payload["outer_eval_complete"] = sha256(fold_manifest.parent.parent / "outer_eval/COMPLETE.json")
    payload["cv_complete"] = sha256(source_config)
    payload["fold_summary"] = sha256(fold_manifest.parent.parent.parent / "fold_summary.json")
    for relative in ("scripts/train_cv5.py", "src/alm/training/engine.py", "src/alm/data/dataset.py",
                     "src/alm/models/alm_model.py", "src/alm/training/metrics.py",
                     "src/alm/losses/combined.py", "src/alm/evaluation/cv_figures.py"):
        payload[relative] = sha256(ROOT / relative)
    return object_hash(payload)


def _verify_split_files(data_dir: Path, manifest_dir: Path, role: str) -> None:
    metadata = read_csv(data_dir / "metadata.csv", METADATA_FIELDS)
    recipes = read_csv(manifest_dir / "mixture_manifest.csv",
                       (*METADATA_FIELDS, "audio_sha256", "components", "peak_normalization_scale"))
    if len(metadata) != len(recipes):
        raise ValueError(f"{role}: mixture manifest row count differs from metadata")
    expected = set()
    for row, recipe in zip(metadata, recipes):
        if any(row[key] != recipe[key] for key in METADATA_FIELDS):
            raise ValueError(f"{role}: manifest metadata mismatch for {row['file']}")
        expected.add(row["file"])
        audio_path = data_dir / "audio" / row["file"]
        if not audio_path.is_file() or sha256(audio_path) != recipe["audio_sha256"]:
            raise ValueError(f"{role}: generated audio checksum mismatch for {row['file']}")
    actual = {path.name for path in (data_dir / "audio").glob("*.wav")}
    if actual != expected:
        raise ValueError(f"{role}: generated audio inventory differs from metadata")


def _run_fold(fold: int, *, config: dict, config_path: Path, run_root: Path,
              cv_manifest_root: Path, source_paths: dict, resume: bool, device: torch.device):
    fold_id = f"fold_{fold:02d}"
    fold_run = run_root / fold_id
    fold_dataset = ROOT / "data/synthetic" / config["dataset_version"] / fold_id
    fold_manifest = cv_manifest_root / fold_id
    for role in ("inner", "outer_eval"):
        complete = json.loads((fold_manifest / role / "COMPLETE.json").read_text())
        if complete.get("status") != "complete":
            raise ValueError(f"Uncertified dataset for {fold_id}/{role}")
        for name, digest in complete.items():
            if name.endswith("_sha256"):
                artifact = {"metadata_sha256": fold_dataset / role / "metadata.csv",
                            "source_manifest_sha256": fold_manifest / role / "source_splits.csv",
                            "mixture_manifest_sha256": fold_manifest / role / "mixture_manifest.csv",
                            "config_sha256": fold_dataset / role / "config.json",
                            "audit_sha256": fold_manifest / role / "audit.json"}.get(name)
                if artifact is not None and sha256(artifact) != digest:
                    raise ValueError(f"Dataset artifact hash mismatch: {fold_id}/{role}/{artifact.name}")
        _verify_split_files(fold_dataset / role, fold_manifest / role, role)
    fingerprint = _fold_fingerprint(config_path, fold_manifest / "inner/COMPLETE.json",
                                    cv_manifest_root / "COMPLETE.json")
    fold_run.mkdir(parents=True, exist_ok=True)
    status_path = fold_run / "fold_status.json"
    previous = {}
    if status_path.is_file():
        previous = json.loads(status_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(f"Resume refused for {fold_id}: fold fingerprint changed")
        if previous.get("status") == "complete":
            for name, digest in previous.get("artifact_sha256", {}).items():
                if sha256(fold_run / name) != digest:
                    raise ValueError(f"Completed fold artifact integrity check failed: {fold_id}/{name}")
            print(f"{fold_id}: already complete; skipping", flush=True)
            return json.loads((fold_run / "outer_metrics.json").read_text())
        if not resume:
            raise FileExistsError(f"{fold_id} is incomplete; restart with --resume")
    elif any(fold_run.iterdir()) and not resume:
        raise FileExistsError(f"Fold artifacts already exist in {fold_run}; rerun with --resume")

    inner_manifest = fold_manifest / "inner"
    inner_dir = fold_dataset / "inner"
    outer_manifest = fold_manifest / "outer_eval"
    outer_dir = fold_dataset / "outer_eval"
    train_ds = _build_dataset(inner_dir / "metadata.csv", inner_dir / "audio",
                              fold_run / "cache/train", "train", config,
                              source_paths["source_csv"], inner_manifest / "mixture_manifest.csv",
                              inner_manifest / "source_splits.csv", source_paths["source_audio"])
    val_ds = _build_dataset(inner_dir / "metadata.csv", inner_dir / "audio",
                            fold_run / "cache/inner_val", "val", config,
                            source_paths["source_csv"], inner_manifest / "mixture_manifest.csv",
                            inner_manifest / "source_splits.csv", source_paths["source_audio"])
    outer_ds = _build_dataset(outer_dir / "metadata.csv", outer_dir / "audio",
                              fold_run / "cache/outer_eval", "val", config,
                              source_paths["source_csv"], outer_manifest / "mixture_manifest.csv",
                              outer_manifest / "source_splits.csv", source_paths["source_audio"])
    train_targets = torch.from_numpy(_targets(train_ds))
    positives = train_targets.sum(0)
    pos_weight = ((len(train_targets) - positives) / positives.clamp_min(1)).clamp(
        max=float(config["loss"]["pos_weight_cap"]))
    confusion = estimate_training_cooccurrence(train_targets) if config["model"]["use_confusion_weights"] else None
    model_cfg = config["model"]
    model = ALMModel(**{key: model_cfg[key] for key in (
        "use_fusion", "use_class_queries", "use_mask_supervision", "embed_dim", "num_heads",
        "num_layers", "ff_dim", "dropout", "max_time_frames")}).to(device)
    loss_cfg = config["loss"]
    aud_cfg = config["audibility"]
    loss = CombinedLoss(
        num_classes=len(CLASS_NAMES), pos_weight=pos_weight,
        use_asymmetric_loss=model_cfg["use_asymmetric_loss"],
        asl_gamma_pos=loss_cfg["asl_gamma_pos"], asl_gamma_neg=loss_cfg["asl_gamma_neg"],
        asl_clip=loss_cfg["asl_clip"], use_audibility_loss=model_cfg["use_audibility_loss"],
        audibility_floor=loss_cfg["audibility_floor"],
        audibility_center_db=aud_cfg["center_db"], audibility_temperature_db=aud_cfg["temperature_db"],
        confusion_matrix=confusion,
        confusion_strength=loss_cfg["confusion_strength"], lambda_mask=loss_cfg["lambda_mask"],
    )
    train_cfg = config["training"]
    loader_generator = torch.Generator().manual_seed(int(config["seed"]) + fold)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=train_cfg["num_workers"], generator=loader_generator,
                              worker_init_fn=worker_seed, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"] * 2, shuffle=False,
                            num_workers=train_cfg["num_workers"], worker_init_fn=worker_seed,
                            pin_memory=device.type == "cuda")
    outer_loader = DataLoader(outer_ds, batch_size=train_cfg["batch_size"] * 2, shuffle=False,
                              num_workers=train_cfg["num_workers"], pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["eta_min"])
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    if resume and not (fold_run / "last_model.pt").exists():
        resume_existing = False
    else:
        resume_existing = resume
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = {"fold": fold, "status": "running", "fingerprint": fingerprint,
              "started_utc": previous.get("started_utc", now), "last_started_utc": now,
              "attempt": int(previous.get("attempt", 0)) + 1}
    _atomic_json(status_path, status)
    if not (fold_run / "environment.json").exists():
        with (fold_run / "environment.json").open("x", encoding="utf-8") as handle:
            json.dump(environment_record(ROOT), handle, indent=2, sort_keys=True)
            handle.write("\n")
    trainer = Trainer(
        model, loss, optimizer, device=device, run_dir=fold_run, scheduler=scheduler,
        epochs=train_cfg["epochs"], grad_clip=train_cfg["grad_clip"], amp=amp,
        checkpoint_metric=train_cfg["checkpoint_metric"],
        early_stopping_patience=train_cfg["early_stopping_patience"],
        early_stopping_min_delta=train_cfg["early_stopping_min_delta"],
        resume=resume_existing, run_fingerprint=fingerprint,
    )
    training_result = trainer.fit(train_loader, val_loader)

    best = torch.load(fold_run / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    inner_y, inner_p = _collect(model, val_loader, device)
    outer_y, outer_p = _collect(model, outer_loader, device)
    thresholds = calibrate_thresholds(inner_y, inner_p,
                                      strategy=train_cfg["threshold_strategy"])
    outer_metrics = multilabel_metrics(outer_y, outer_p, thresholds=thresholds["thresholds"])
    outer_metrics.update({
        "fold": fold, "checkpoint_epoch": int(best["epoch"]),
        "inner_best_epoch": int(training_result["best_epoch"]),
        "stopping_epoch": int(training_result["history"][-1]["epoch"]),
        "outer_examples": int(len(outer_y)),
        "threshold_source": "inner_validation_only",
        "inner_threshold_objective": thresholds["objective"],
    })
    _atomic_npz(fold_run / "inner_validation_predictions.npz", targets=inner_y, scores=inner_p)
    _atomic_npz(fold_run / "outer_predictions.npz", targets=outer_y, scores=outer_p,
                thresholds=np.asarray([thresholds["thresholds"][c] for c in CLASS_NAMES]))
    _atomic_json(fold_run / "thresholds.json", thresholds)
    _atomic_json(fold_run / "outer_metrics.json", outer_metrics)
    (fold_run / "figures").mkdir(exist_ok=True)
    fold_curves(training_result["history"], fold_run / "figures/training_curves.png", fold_id)
    spectrogram_examples(outer_ds, fold_run / "figures/outer_samples.png", fold_id)
    artifact_names = ("training.log", "history.json", "metrics.csv", "best_model.pt", "last_model.pt",
                      "inner_validation_predictions.npz", "outer_predictions.npz", "outer_metrics.json",
                      "thresholds.json", "environment.json", "figures/training_curves.png",
                      "figures/outer_samples.png")
    status.update({"status": "complete", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "best_epoch": int(training_result["best_epoch"]),
                   "stopping_epoch": int(training_result["history"][-1]["epoch"]),
                   "artifact_sha256": {name: sha256(fold_run / name) for name in artifact_names
                                       if (fold_run / name).is_file()}})
    _atomic_json(status_path, status)
    return outer_metrics


def _aggregate(run_root: Path, results: dict[str, dict]) -> dict:
    aggregate_dir = run_root / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    metric_names = ("mAP", "macro_roc_auc", "macro_precision", "macro_recall", "macro_f1",
                    "micro_precision", "micro_recall", "micro_f1", "samples_f1",
                    "hamming_loss", "label_accuracy", "mean_class_accuracy",
                    "macro_balanced_accuracy", "exact_match", "label_ranking_average_precision")
    summary = {"status": "complete", "folds": results, "metrics": {}}
    for metric in metric_names:
        values = [results[name].get(metric) for name in sorted(results)]
        valid = [float(value) for value in values if value is not None and np.isfinite(value)]
        summary["metrics"][metric] = {
            "mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else (0.0 if valid else None),
            "per_fold": values,
        }
    summary["metric_protocol"] = {
        "outer_fold": "one-time evaluation after checkpoint selection on disjoint inner validation",
        "thresholds": "selected on inner validation; applied unchanged to outer fold",
        "primary": "outer-fold mAP mean and sample standard deviation",
    }
    _atomic_json(aggregate_dir / "summary.json", summary)
    with (aggregate_dir / "metrics_by_fold.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["fold", *metric_names, "checkpoint_epoch", "stopping_epoch", "outer_examples"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in sorted(results):
            writer.writerow({"fold": name, **{k: results[name][k] for k in fields if k != "fold"}})
    preds = [np.load(run_root / f"fold_{i:02d}/outer_predictions.npz") for i in range(1, 6)]
    y = np.concatenate([item["targets"] for item in preds])
    p = np.concatenate([item["scores"] for item in preds])
    fold_thresholds = np.concatenate([np.repeat(item["thresholds"][None, :], len(item["targets"]), axis=0)
                                      for item in preds])
    binary = p >= fold_thresholds
    from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                                 label_ranking_average_precision_score,
                                 precision_recall_fscore_support, roc_auc_score)
    macro_p, macro_r, _, _ = precision_recall_fscore_support(y, binary, average="macro", zero_division=0)
    micro_p, micro_r, _, _ = precision_recall_fscore_support(y, binary, average="micro", zero_division=0)
    _, _, sample_f1, _ = precision_recall_fscore_support(y, binary, average="samples", zero_division=0)
    per_class_accuracy = np.mean(binary == y, axis=0)
    balanced = []
    auc_values = []
    for index in range(y.shape[1]):
        tp = np.sum((y[:, index] == 1) & binary[:, index])
        fn = np.sum((y[:, index] == 1) & ~binary[:, index])
        tn = np.sum((y[:, index] == 0) & ~binary[:, index])
        fp = np.sum((y[:, index] == 0) & binary[:, index])
        balanced.append(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
        if np.unique(y[:, index]).size == 2:
            auc_values.append(float(roc_auc_score(y[:, index], p[:, index])))
    pooled = {
        "examples": int(len(y)), "mAP": float(average_precision_score(y, p, average="macro")),
        "label_ranking_average_precision": float(label_ranking_average_precision_score(y, p)),
        "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        "macro_precision": float(macro_p), "macro_recall": float(macro_r),
        "macro_f1": float(f1_score(y, binary, average="macro", zero_division=0)),
        "micro_precision": float(micro_p), "micro_recall": float(micro_r),
        "micro_f1": float(f1_score(y, binary, average="micro", zero_division=0)),
        "samples_f1": float(sample_f1),
        "hamming_loss": float(hamming_loss(y, binary)),
        "label_accuracy": float(1.0 - hamming_loss(y, binary)),
        "mean_class_accuracy": float(np.mean(per_class_accuracy)),
        "macro_balanced_accuracy": float(np.mean(balanced)),
        "exact_match": float(np.all(y == binary, axis=1).mean()),
    }
    _atomic_json(aggregate_dir / "pooled_out_of_fold_metrics.json", pooled)
    _atomic_npz(aggregate_dir / "pooled_out_of_fold_predictions.npz",
                targets=y, scores=p, thresholds=fold_thresholds)
    fold_comparison(results, aggregate_dir / "fold_comparison.png")
    class_metric_heatmap(results, aggregate_dir / "class_f1_by_fold.png")
    outer_pr_curve(y, p, aggregate_dir / "pooled_precision_recall.png")
    output_names = ("summary.json", "metrics_by_fold.csv", "pooled_out_of_fold_metrics.json",
                    "pooled_out_of_fold_predictions.npz", "fold_comparison.png",
                    "class_f1_by_fold.png", "pooled_precision_recall.png")
    _atomic_json(aggregate_dir / "COMPLETE.json", {
        "status": "complete", "run_id": run_root.name,
        "artifact_sha256": {name: sha256(aggregate_dir / name) for name in output_names},
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/CV5_proposed_alm_source_grouped_v1.json")
    parser.add_argument("--resume", action="store_true", help="resume incomplete folds and skip completed folds")
    parser.add_argument("--fold", type=int, choices=range(1, 6), action="append", help="run selected fold(s)")
    parser.add_argument("--run-id", help="override the configured run ID; must be new unless --resume")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    run_id = args.run_id or config["run_id"]
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a simple directory name")
    run_root = ROOT / "runs" / run_id
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cuda" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest_root = ROOT / "data/manifests" / config["dataset_version"]
    complete_path = manifest_root / "COMPLETE.json"
    if not complete_path.is_file() or json.loads(complete_path.read_text()).get("status") != "complete":
        raise FileNotFoundError("Certified CV datasets missing; run scripts/prepare_cv5.py first")
    cv_certificate = json.loads(complete_path.read_text())
    if sha256(manifest_root / "folds.csv") != cv_certificate["folds_sha256"]:
        raise ValueError("CV fold assignment checksum mismatch")
    if sha256(manifest_root / "fold_summary.json") != cv_certificate["summary_sha256"]:
        raise ValueError("CV fold summary checksum mismatch")
    fold_summary = json.loads((manifest_root / "fold_summary.json").read_text())
    for relative, expected in fold_summary["protocol_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"CV preparation protocol code changed: {relative}")
    if config["training"]["num_workers"] != 0:
        raise ValueError("Exact epoch-boundary resume currently requires num_workers=0")
    if args.resume:
        if not run_root.is_dir():
            raise FileNotFoundError(f"Cannot resume missing run directory: {run_root}")
        run_record = json.loads((run_root / "run_manifest.json").read_text())
        if (run_record["config_sha256"] != sha256(config_path)
                or run_record["fold_manifest_sha256"] != sha256(manifest_root / "folds.csv")
                or run_record["dataset_certificate_sha256"] != sha256(complete_path)):
            raise ValueError("Resume refused: config or fold manifest has changed")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(run_root / "run_manifest.json", {
            "run_id": run_id, "config_sha256": sha256(config_path),
            "fold_manifest_sha256": sha256(manifest_root / "folds.csv"),
            "dataset_certificate_sha256": sha256(complete_path),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "protocol": "nested_grouped_5fold", "official_test_used": False,
        })
        with (run_root / "environment.json").open("x", encoding="utf-8") as handle:
            json.dump(environment_record(ROOT), handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (run_root / "resolved_config.json").open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with (run_root / "run_events.jsonl").open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    run_status_path = run_root / "run_status.json"
    prior_status = json.loads(run_status_path.read_text()) if run_status_path.is_file() else {}
    run_status = {"status": "running", "started_utc": prior_status.get("started_utc",
                          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                  "last_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "expected_folds": 5, "completed_folds": prior_status.get("completed_folds", [])}
    _atomic_json(run_status_path, run_status)
    _event(run_root, "run_started", resume=args.resume, device=args.device)
    folds = args.fold or list(range(1, 6))
    all_results = {}
    for fold in folds:
        seed_everything(int(config["seed"]) + fold, deterministic=True)
        _event(run_root, "fold_started", fold=fold)
        try:
            all_results[f"fold_{fold:02d}"] = _run_fold(
                fold, config=config, config_path=config_path, run_root=run_root,
                cv_manifest_root=manifest_root, source_paths={
                    "source_csv": ROOT / "data/processed_mad/metadata.csv",
                    "source_audio": ROOT / "data/processed_mad/audio",
                }, resume=args.resume, device=device,
            )
            print(f"fold_{fold:02d}: outer mAP={all_results[f'fold_{fold:02d}']['mAP']:.4f}", flush=True)
            _event(run_root, "fold_completed", fold=fold,
                   outer_mAP=all_results[f"fold_{fold:02d}"]["mAP"])
            run_status["completed_folds"] = sorted(set(run_status.get("completed_folds", []))
                                                     | {f"fold_{fold:02d}"})
            _atomic_json(run_status_path, run_status)
        except KeyboardInterrupt:
            _event(run_root, "fold_interrupted", fold=fold)
            run_status.update({"status": "interrupted", "interrupted_fold": fold})
            _atomic_json(run_status_path, run_status)
            raise
        except Exception as exc:
            _event(run_root, "fold_failed", fold=fold, error=f"{type(exc).__name__}: {exc}")
            run_status.update({"status": "failed", "failed_fold": fold,
                               "failure": f"{type(exc).__name__}: {exc}"})
            _atomic_json(run_status_path, run_status)
            print(f"fold_{fold:02d}: failed: {exc}", file=sys.stderr, flush=True)
            raise
    completed = {}
    for fold in range(1, 6):
        fold_status = run_root / f"fold_{fold:02d}/fold_status.json"
        if fold_status.exists() and json.loads(fold_status.read_text()).get("status") == "complete":
            completed[f"fold_{fold:02d}"] = json.loads((run_root / f"fold_{fold:02d}/outer_metrics.json").read_text())
    if len(completed) == 5:
        _aggregate(run_root, completed)
        _event(run_root, "run_completed", completed_folds=sorted(completed))
        run_status["aggregate_artifact_sha256"] = {
            name: sha256(run_root / "aggregate" / name) for name in (
                "summary.json", "metrics_by_fold.csv", "pooled_out_of_fold_metrics.json",
                "pooled_out_of_fold_predictions.npz", "fold_comparison.png",
                "class_f1_by_fold.png", "pooled_precision_recall.png", "COMPLETE.json")
        }
        print(f"5-fold run complete: {run_root / 'aggregate/summary.json'}", flush=True)
    else:
        _event(run_root, "run_in_progress", completed_folds=sorted(completed))
    run_status.update({"status": "complete" if len(completed) == 5 else "in_progress",
                       "completed_folds": sorted(completed), "expected_folds": 5,
                       "event_log_sha256": sha256(run_root / "run_events.jsonl")})
    _atomic_json(run_status_path, run_status)


if __name__ == "__main__":
    main()
