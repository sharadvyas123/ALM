"""Small, explicit training engine used by the notebook and smoke tests."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..progress import tqdm

from .metrics import multilabel_metrics


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Trainer:
    """Train/validate a model with checkpoint refusal and component logging."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                 *, device: torch.device, run_dir: Path, scheduler=None, epochs: int = 1,
                 grad_clip: float | None = 1.0, amp: bool = True, checkpoint_metric: str = "mAP") -> None:
        if epochs <= 0 or grad_clip is not None and grad_clip <= 0:
            raise ValueError("invalid trainer settings")
        self.model, self.loss_fn, self.optimizer = model.to(device), loss_fn.to(device), optimizer
        self.device, self.run_dir, self.scheduler = device, Path(run_dir), scheduler
        self.epochs, self.grad_clip, self.amp = epochs, grad_clip, bool(amp and device.type == "cuda")
        self.checkpoint_metric = checkpoint_metric
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.logger = self._logger()
        self.history: list[dict] = []
        self.best_value = -float("inf")
        self.best_epoch = -1

    def _logger(self) -> logging.Logger:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"alm.run.{self.run_dir.name}")
        logger.setLevel(logging.DEBUG)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        class TqdmConsoleHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    tqdm.write(self.format(record))
                except Exception:
                    self.handleError(record)

        file_handler = logging.FileHandler(self.run_dir / "training.log", mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console = TqdmConsoleHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console)
        logger.propagate = False
        return logger

    def _forward_loss(self, batch: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        mel, targets = batch["mel"].to(self.device), batch["target"].to(self.device)
        aux_needed = getattr(self.loss_fn, "lambda_mask", 0.0) > 0
        output = self.model(mel, return_aux=aux_needed)
        if isinstance(output, dict):
            logits, attention = output["logits"], output.get("attention")
        else:
            logits, attention = output, None
        mask_targets = batch["mask_target"].to(self.device) if aux_needed and "mask_target" in batch else None
        valid_mask = batch["valid_mask"].to(self.device) if aux_needed and "valid_mask" in batch else None
        if attention is not None and mask_targets is not None and mask_targets.shape[-2:] != attention.shape[-2:]:
            mask_targets = torch.nn.functional.interpolate(
                mask_targets, size=attention.shape[-2:], mode="bilinear", align_corners=False,
            )
            if valid_mask is not None:
                valid_mask = torch.nn.functional.interpolate(
                    valid_mask[:, None].float(), size=attention.shape[-2:], mode="nearest",
                )[:, 0].bool()
        kwargs = {"audibility_db": batch["audibility_db"].to(self.device) if "audibility_db" in batch else None,
                  "attention_maps": attention,
                  "mask_targets": mask_targets,
                  "valid_mask": valid_mask,
                  "class_valid": batch["class_valid"].to(self.device) if "class_valid" in batch else None}
        return self.loss_fn(logits, targets, **kwargs), logits

    def _run_epoch(self, loader: DataLoader, *, train: bool, epoch: int) -> dict:
        self.model.train(train)
        totals: dict[str, float] = {}
        count = 0
        all_targets, all_scores = [], []
        phase = "train" if train else "val"
        bar = tqdm(total=len(loader), desc=f"Epoch {epoch}/{self.epochs} [{phase}]", leave=False,
                   dynamic_ncols=True, mininterval=0.5, miniters=1, unit="batch")
        learning_rate = self.optimizer.param_groups[0]["lr"]
        try:
            for batch in loader:
                size = batch["mel"].shape[0]
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                    losses, logits = self._forward_loss(batch)
                if not torch.isfinite(losses["total"]).all():
                    raise FloatingPointError(f"non-finite {phase} loss at epoch {epoch}")
                if train:
                    self.scaler.scale(losses["total"]).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * size
                # Reuse this batch's pre-update logits without a second CNN/BN
                # pass. Training metrics are online, with training dropout.
                all_targets.append(batch.get("union_target", batch["target"]).detach().cpu().numpy())
                all_scores.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
                count += size
                postfix = f"loss={totals['total'] / count:.4f}"
                if getattr(self.loss_fn, "lambda_mask", 0.0) > 0:
                    postfix += f" cls={totals['classification'] / count:.4f} mask={totals['mask'] / count:.4f}"
                if train:
                    postfix += f" lr={learning_rate:.2e}"
                bar.set_postfix_str(postfix, refresh=False)
                # Advance immediately: an expensive next DataLoader batch must
                # not leave the display claiming zero completed batches.
                bar.update(1)
        finally:
            bar.close()
        if not count:
            raise ValueError("empty data loader")
        result = {f"{name}_loss": value / count for name, value in totals.items()}
        result.update({f"{key}": value for key, value in multilabel_metrics(np.concatenate(all_targets), np.concatenate(all_scores)).items() if key in {"mAP", "macro_f1", "micro_f1", "hamming_loss"}})
        self.logger.debug("Epoch %d [%s] components/metrics: %s", epoch, phase, result)
        return result

    def _save_history(self, status: str) -> None:
        """Persist completed epochs, including when a later epoch is interrupted."""
        _write_json(self.run_dir / "history.json", {
            "run_id": self.run_dir.name, "status": status,
            "completed_epochs": len(self.history), "best_epoch": self.best_epoch,
            "best_value": self.best_value if math.isfinite(self.best_value) else None,
            "epochs": self.history,
        })

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        if any((self.run_dir / name).exists() for name in ("best_model.pt", "last_model.pt", "history.json", "metrics.csv")):
            raise FileExistsError(f"Refusing to overwrite a previous run in {self.run_dir}; choose a new run_id")
        self.logger.info("Experiment startup | run_id=%s | dir=%s", self.run_dir.name, self.run_dir)
        self.logger.info("DataLoaders | train_batches=%d val_batches=%d | amp=%s | progress=%s",
                         len(train_loader), len(val_loader), self.amp, tqdm.__name__)
        self.logger.info("Log file: %s", self.run_dir / "training.log")
        started_training = time.time()
        phase = "target preparation"
        try:
            self._save_history("preparing")
            for split, loader in (("train", train_loader), ("validation", val_loader)):
                prepare = getattr(getattr(loader, "dataset", None), "precompute_auxiliary", None)
                if prepare is not None:
                    self.logger.info("Preparing %s targets (reusing cached targets where available)", split)
                    prepare(description=f"Preparing {split} audibility/mask targets")
            phase = "training"
            self.logger.info("TRAINING START | epochs=%d | device=%s", self.epochs, self.device)
            for epoch in range(1, self.epochs + 1):
                started = time.time()
                self.logger.debug("Epoch %d/%d start", epoch, self.epochs)
                train = self._run_epoch(train_loader, train=True, epoch=epoch)
                validation = self._run_epoch(val_loader, train=False, epoch=epoch)
                learning_rate = self.optimizer.param_groups[0]["lr"]
                if self.scheduler is not None:
                    self.scheduler.step()
                record = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                          **{f"val_{k}": v for k, v in validation.items()},
                          "learning_rate": learning_rate,
                          "epoch_seconds": time.time() - started}
                metric = float(validation[self.checkpoint_metric])
                if not math.isfinite(metric):
                    raise FloatingPointError("Validation checkpoint metric is not finite")
                improved = metric > self.best_value
                if improved:
                    self.best_value, self.best_epoch = metric, epoch
                record["is_best"] = int(improved)
                self.history.append(record)
                checkpoint = {"epoch": epoch, "model_state": self.model.state_dict(),
                              "optimizer_state": self.optimizer.state_dict(),
                              "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                              "scaler_state": self.scaler.state_dict(),
                              "best_epoch": self.best_epoch, "best_value": self.best_value}
                _save_checkpoint(self.run_dir / "last_model.pt", checkpoint)
                if improved:
                    _save_checkpoint(self.run_dir / "best_model.pt", checkpoint)
                    self.logger.info("Best checkpoint updated | epoch=%d | val_%s=%.4f | %s",
                                     epoch, self.checkpoint_metric, metric, self.run_dir / "best_model.pt")
                self._save_history("training")
                with (self.run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    if len(self.history) == 1:
                        writer.writeheader()
                    writer.writerow(record)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_%s=%.4f | macro_F1=%.4f micro_F1=%.4f | lr=%.3e | time=%.1fs | best_updated=%s",
                                 epoch, self.epochs, train["total_loss"], validation["total_loss"],
                                 self.checkpoint_metric, metric, validation["macro_f1"], validation["micro_f1"],
                                 learning_rate, record["epoch_seconds"], improved)
            self._save_history("complete")
        except KeyboardInterrupt:
            self._save_history("interrupted")
            self.logger.warning("INTERRUPTED during %s | completed_epochs=%d; artifacts preserved; use a new run_id",
                                phase, len(self.history))
            raise
        except Exception:
            self._save_history("failed")
            self.logger.exception("TRAINING FAILED")
            raise
        finally:
            for handler in self.logger.handlers:
                handler.flush()
        self.logger.info("TRAINING COMPLETE | epochs=%d | best_epoch=%d | best_val_%s=%.4f | total_time=%.1fs",
                         self.epochs, self.best_epoch, self.checkpoint_metric, self.best_value,
                         time.time() - started_training)
        return {"best_epoch": self.best_epoch, "best_value": self.best_value, "history": self.history}
