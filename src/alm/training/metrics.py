"""Multilabel metrics and validation-only threshold calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, hamming_loss,
                             label_ranking_average_precision_score,
                             multilabel_confusion_matrix, precision_recall_fscore_support)

from ..data.provenance import CLASS_NAMES


def _validate(targets: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(targets), np.asarray(scores)
    if y.ndim != 2 or p.shape != y.shape or y.shape[1] != len(CLASS_NAMES):
        raise ValueError("targets and scores must both have shape (N,7)")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or np.any((y < 0) | (y > 1)) or np.any((p < 0) | (p > 1)):
        raise ValueError("targets and scores must be finite in [0,1]")
    return y.astype(np.int32), p.astype(np.float64)


def calibrate_thresholds(targets: np.ndarray, scores: np.ndarray, *, strategy: str = "per_class_f1",
                         minimum: float = 0.05, maximum: float = 0.95, step: float = 0.05) -> dict:
    """Calibrate thresholds on validation predictions only."""
    y, p = _validate(targets, scores)
    if strategy not in {"global_f1", "per_class_f1"} or not 0 < minimum <= maximum <= 1 or step <= 0:
        raise ValueError("invalid threshold calibration settings")
    grid = np.arange(minimum, maximum + step / 2, step)
    if strategy == "global_f1":
        best = max(((float(f1_score(y, p >= value, average="macro", zero_division=0)), float(value)) for value in grid), key=lambda x: (x[0], -x[1]))
        thresholds = np.repeat(best[1], len(CLASS_NAMES))
        objective = best[0]
        details = {name: {"threshold": float(best[1]), "f1": float(f1_score(y[:, i], p[:, i] >= best[1], zero_division=0))} for i, name in enumerate(CLASS_NAMES)}
    else:
        thresholds, details = [], {}
        for index, name in enumerate(CLASS_NAMES):
            score, threshold = max(((float(f1_score(y[:, index], p[:, index] >= value, zero_division=0)), float(value)) for value in grid), key=lambda x: (x[0], -x[1]))
            thresholds.append(threshold)
            details[name] = {"threshold": threshold, "f1": score}
        objective = float(np.mean([v["f1"] for v in details.values()]))
        thresholds = np.asarray(thresholds)
    return {"strategy": strategy, "thresholds": {name: float(thresholds[i]) for i, name in enumerate(CLASS_NAMES)},
            "objective": float(objective), "objective_name": "validation_macro_f1", "per_class": details}


def multilabel_metrics(targets: np.ndarray, scores: np.ndarray, *, thresholds: dict | np.ndarray | None = None) -> dict:
    """Compute ranking and threshold-dependent metrics with class details."""
    y, p = _validate(targets, scores)
    per_ap = []
    for i in range(y.shape[1]):
        per_ap.append(float(average_precision_score(y[:, i], p[:, i])) if y[:, i].any() else float("nan"))
    map_value = float(np.nanmean(per_ap))
    result = {"mAP": map_value, "macro_average_precision": map_value,
              "label_ranking_average_precision": float(label_ranking_average_precision_score(y, p)),
              "per_class": {name: {"average_precision": per_ap[i], "support": int(y[:, i].sum())} for i, name in enumerate(CLASS_NAMES)}}
    if thresholds is None:
        threshold_vector = np.full(len(CLASS_NAMES), 0.5)
        result["thresholds"] = {name: 0.5 for name in CLASS_NAMES}
    elif isinstance(thresholds, dict):
        threshold_vector = np.asarray([thresholds[name] for name in CLASS_NAMES], dtype=float)
        result["thresholds"] = {name: float(thresholds[name]) for name in CLASS_NAMES}
    else:
        threshold_vector = np.asarray(thresholds, dtype=float)
        if threshold_vector.shape != (len(CLASS_NAMES),):
            raise ValueError("threshold vector must have shape (7,)")
        result["thresholds"] = {name: float(threshold_vector[i]) for i, name in enumerate(CLASS_NAMES)}
    predictions = (p >= threshold_vector[None, :]).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(y, predictions, average=None, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        result["per_class"][name].update({"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])})
    result.update({"macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
                   "micro_f1": float(f1_score(y, predictions, average="micro", zero_division=0)),
                   "hamming_loss": float(hamming_loss(y, predictions)),
                   "exact_match": float(np.all(y == predictions, axis=1).mean()),
                   "confusion_matrix": multilabel_confusion_matrix(y, predictions).tolist()})
    return result


def mask_metrics(attention: np.ndarray, target: np.ndarray, *, class_valid: np.ndarray | None = None) -> dict:
    """Report finite distribution metrics without claiming perceptual localisation."""
    a, t = np.asarray(attention, dtype=float), np.asarray(target, dtype=float)
    if a.shape != t.shape or a.ndim != 4:
        raise ValueError("attention and target must be (B,C,F,T)")
    target_mass = t.reshape(*t.shape[:2], -1).sum(-1)
    valid = target_mass > 1e-8
    if class_valid is not None:
        valid &= np.asarray(class_valid, dtype=bool)
    target_dist = t / np.maximum(target_mass[..., None, None], 1e-8)
    attention_dist = a / np.maximum(a.reshape(*a.shape[:2], -1).sum(-1)[..., None, None], 1e-8)
    kl = -(target_dist * np.log(np.maximum(attention_dist, 1e-8))).reshape(*t.shape[:2], -1).sum(-1)
    values = kl[valid]
    return {"valid_class_maps": int(valid.sum()), "mean_target_to_attention_kl": float(values.mean()) if values.size else 0.0}
