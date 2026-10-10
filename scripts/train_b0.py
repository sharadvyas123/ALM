"""Train B0 on the verified source-disjoint MAD v1 dataset.

B0 is the corrected reference model: no MTFF and no class-query head. It uses
the existing weighted BCE training protocol, hop length 512, and validation-only
threshold calibration. The official test set is not read.

Run only after the separate preflight passes and the run has been approved:

    venv/bin/python -B scripts/train_b0.py --run-id B0_mad_source_disjoint_v1_seed42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alm.data.provenance import CLASS_NAMES, sha256, write_json
from utils.enhanced_alm_mad_encoder import EnhancedALMEncoder


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MADV1Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, audio_dir: Path, train: bool, cfg: dict):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.train = train
        self.cfg = cfg
        self.mel = T.MelSpectrogram(
            sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
            f_min=cfg["f_min"], f_max=cfg["f_max"],
        )
        self.to_db = T.AmplitudeToDB()
        if train:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg["freq_mask_param"])
            self.time_mask = T.TimeMasking(time_mask_param=cfg["time_mask_param"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio, sample_rate = sf.read(str(self.audio_dir / row["file"]), dtype="float32")
        if sample_rate != self.cfg["sample_rate"] or audio.ndim != 1 or len(audio) != self.cfg["num_samples"]:
            raise ValueError(f"Unexpected audio for {row['file']}: sr={sample_rate}, shape={audio.shape}")
        waveform = torch.from_numpy(audio).reshape(1, -1)
        log_mel = self.to_db(self.mel(waveform))
        if self.train:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)
        target = torch.tensor([int(v) for v in row["label_vector"].split(",")], dtype=torch.float32)
        return {"mel": log_mel, "target": target, "index": index}


def compute_map(targets, scores):
    return float(average_precision_score(targets, scores, average="macro"))


def evaluate(model, loader, device, amp_enabled):
    model.eval()
    all_targets, all_scores, all_indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(mel)
            all_targets.append(batch["target"].numpy())
            all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
            all_indices.extend(batch["index"].tolist())
    targets = np.concatenate(all_targets)
    scores = np.concatenate(all_scores)
    return targets, scores, np.asarray(all_indices, dtype=np.int64)


def calibrate_thresholds(targets, scores, minimum=0.05, maximum=0.95, step=0.05):
    thresholds = {}
    per_class = {}
    grid = np.arange(minimum, maximum + step / 2, step)
    for class_id, class_name in enumerate(CLASS_NAMES):
        best_f1, best_threshold = -1.0, 0.5
        for threshold in grid:
            prediction = scores[:, class_id] >= threshold
            value = f1_score(targets[:, class_id], prediction, zero_division=0)
            if value > best_f1:
                best_f1, best_threshold = float(value), float(threshold)
        thresholds[class_name] = round(best_threshold, 4)
        per_class[class_name] = {"threshold": round(best_threshold, 4), "f1": round(best_f1, 6)}
    return thresholds, per_class


def metrics(targets, scores, metadata, thresholds):
    threshold_vector = np.asarray([thresholds[name] for name in CLASS_NAMES])
    predictions = (scores >= threshold_vector[None, :]).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0,
    )
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]),
               "f1": float(f1[i]), "support": int(support[i]),
               "average_precision": float(average_precision_score(targets[:, i], scores[:, i]))}
        for i, name in enumerate(CLASS_NAMES)
    }
    grouped = {}
    for mix_type in ("clean", "2-class", "3-class"):
        mask = metadata["mix_type"].to_numpy() == mix_type
        if not mask.any():
            continue
        grouped[mix_type] = {
            "examples": int(mask.sum()), "mAP": compute_map(targets[mask], scores[mask]),
            "macro_f1": float(f1_score(targets[mask], predictions[mask], average="macro", zero_division=0)),
            "micro_f1": float(f1_score(targets[mask], predictions[mask], average="micro", zero_division=0)),
        }
    return {
        "mAP": compute_map(targets, scores),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(targets, predictions)),
        "exact_match": float(np.all(targets == predictions, axis=1).mean()),
        "per_class": per_class, "by_mix_type": grouped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/synthetic/mad_source_disjoint_v1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if "multilabel_mad" in args.dataset_dir.as_posix():
        raise ValueError("B0 refuses the legacy dataset")
    dataset_dir = args.dataset_dir.resolve()
    metadata_path = dataset_dir / "metadata.csv"
    audio_dir = dataset_dir / "audio"
    run_dir = (args.runs_dir / args.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse run directory: {run_dir}")
    if not metadata_path.is_file() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Missing v1 dataset: {dataset_dir}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cuda" if args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp

    metadata = pd.read_csv(metadata_path, dtype=str)
    if list(metadata["split"].value_counts().sort_index().to_dict().items()) != [("train", 5464), ("val", 965)]:
        raise ValueError("Unexpected v1 split counts")
    train_df = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_df = metadata[metadata["split"] == "val"].reset_index(drop=True)
    cfg = {
        "sample_rate": 16000, "duration_seconds": 10, "num_samples": 160000,
        "n_mels": 128, "n_fft": 1024, "hop_length": 512, "f_min": 20.0, "f_max": 8000.0,
        "freq_mask_param": 27, "time_mask_param": 40,
    }
    train_ds = MADV1Dataset(train_df, audio_dir, True, cfg)
    val_ds = MADV1Dataset(val_df, audio_dir, False, cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    train_targets = np.stack(train_df["label_vector"].map(lambda value: [int(v) for v in value.split(",")]))
    positives = train_targets.sum(axis=0)
    negatives = len(train_targets) - positives
    pos_weights = np.minimum(negatives / np.maximum(positives, 1), 8.0).astype(np.float32)
    model = EnhancedALMEncoder(
        embed_dim=256, num_classes=7, num_heads=8, num_layers=6,
        ff_dim=1024, dropout=0.1, use_fusion=False, use_class_queries=False,
        max_time_frames=100,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_dir.mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    (run_dir / "checkpoints").mkdir()
    config = {
        "run_id": args.run_id, "dataset_dir": str(dataset_dir),
        "metadata_sha256": sha256(metadata_path), "seed": args.seed,
        "device": str(device), "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs": args.epochs, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "amp": amp_enabled, "class_order": list(CLASS_NAMES), "model": {
            "name": "EnhancedALMEncoder", "use_fusion": False, "use_class_queries": False,
            "embed_dim": 256, "num_heads": 8, "num_layers": 6, "ff_dim": 1024,
            "max_time_frames": 100,
        }, "preprocessing": cfg, "pos_weight": pos_weights.tolist(),
        "optimizer": {"name": "AdamW", "lr": 1e-4, "weight_decay": 1e-4},
        "checkpoint_rule": "highest validation mAP", "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__,
    }
    write_json(run_dir / "config.json", config)

    history = []
    best_map = -float("inf")
    best_epoch = -1
    best_predictions = None
    log_path = run_dir / "train.log"
    log_handle = log_path.open("x", encoding="utf-8")
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            train_loss = 0.0
            train_count = 0
            for batch in train_loader:
                mel = batch["mel"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(mel)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += float(loss.item()) * mel.size(0)
                train_count += mel.size(0)
            scheduler.step()
            val_targets, val_scores, val_indices = evaluate(model, val_loader, device, amp_enabled)
            val_map = compute_map(val_targets, val_scores)
            record = {"epoch": epoch, "train_loss": train_loss / train_count,
                      "val_mAP": val_map, "lr": optimizer.param_groups[0]["lr"],
                      "seconds": time.time() - started}
            history.append(record)
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)
            if val_map > best_map:
                best_map, best_epoch = val_map, epoch
                best_predictions = (val_targets.copy(), val_scores.copy(), val_indices.copy())
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                            "best_val_mAP": best_map, "config": config}, run_dir / "checkpoints/best.pt")
                np.savez_compressed(run_dir / "predictions/validation_best.npz",
                                    sample_id=val_df.iloc[val_indices]["id"].to_numpy(),
                                    file=val_df.iloc[val_indices]["file"].to_numpy(),
                                    mix_type=val_df.iloc[val_indices]["mix_type"].to_numpy(),
                                    targets=val_targets, scores=val_scores)
        torch.save({"epoch": args.epochs, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "best_epoch": best_epoch, "best_val_mAP": best_map, "config": config},
                   run_dir / "checkpoints/final.pt")
    finally:
        log_handle.close()

    if best_predictions is None:
        raise RuntimeError("No validation prediction was produced")
    best_targets, best_scores, _ = best_predictions
    thresholds, threshold_details = calibrate_thresholds(best_targets, best_scores)
    run_metrics = metrics(best_targets, best_scores, val_df, thresholds)
    run_metrics.update({"best_epoch": best_epoch, "best_val_mAP": best_map,
                        "thresholds": threshold_details})
    write_json(run_dir / "thresholds.json", {"class_order": list(CLASS_NAMES), "thresholds": thresholds,
                                               "objective": "per-class F1 on validation", "best_epoch": best_epoch})
    write_json(run_dir / "metrics.json", run_metrics)
    with (run_dir / "history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "best_epoch": best_epoch,
                      "best_val_mAP": best_map, "metrics": run_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
