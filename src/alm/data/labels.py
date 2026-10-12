"""Validated multi-label policies for synthetic mixtures."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .provenance import CLASS_NAMES


def validate_labels(labels: np.ndarray, *, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Validate a finite ``(N, num_classes)`` hard or soft label matrix."""

    values = np.asarray(labels, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != num_classes:
        raise ValueError(f"labels must have shape (N, {num_classes})")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("labels must be finite and lie in [0, 1]")
    return values


def union_labels(class_ids: Iterable[int], *, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Return the ordinary logical union of source classes."""

    result = np.zeros(num_classes, dtype=np.float32)
    for class_id in class_ids:
        if isinstance(class_id, bool) or int(class_id) != class_id or not 0 <= int(class_id) < num_classes:
            raise ValueError(f"class id outside [0, {num_classes}): {class_id!r}")
        result[int(class_id)] = 1.0
    return result


def audibility_score(
    snr_db: float,
    *,
    center_db: float = -15.0,
    temperature_db: float = 6.0,
    floor: float = 0.1,
) -> float:
    """Map source-to-rest dB to a bounded continuous audibility target.

    The mapping is ``floor + (1-floor)*sigmoid((snr-center)/temperature)``.
    ``floor`` preserves a learning signal for difficult but learnable sources;
    clean sources with ``+inf`` SNR map to one and silent sources to zero.
    """

    if (not all(math.isfinite(x) for x in (floor, center_db, temperature_db))
            or not 0.0 <= floor < 1.0 or temperature_db <= 0):
        raise ValueError("floor must be in [0, 1) and temperature_db must be positive")
    if math.isinf(snr_db):
        return 1.0 if snr_db > 0 else 0.0
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite or +/-inf")
    z = (snr_db - center_db) / temperature_db
    sigmoid = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return float(floor + (1.0 - floor) * sigmoid)


def _class_audibility(
    class_ids: Sequence[int], audibility_db: Sequence[float], num_classes: int,
) -> np.ndarray:
    if len(class_ids) != len(audibility_db):
        raise ValueError("class_ids and audibility_db must have equal length")
    result = np.full(num_classes, -math.inf, dtype=np.float64)
    for class_id, score in zip(class_ids, audibility_db):
        if not 0 <= int(class_id) < num_classes:
            raise ValueError(f"class id outside [0, {num_classes}): {class_id!r}")
        if not math.isfinite(float(score)) and score not in (float("inf"), float("-inf")):
            raise ValueError("audibility scores must be finite or +/-inf")
        result[int(class_id)] = max(result[int(class_id)], float(score))
    return result


def construct_labels(
    class_ids: Sequence[int],
    *,
    policy: str = "union",
    audibility_db: Sequence[float] | None = None,
    threshold_db: float = -15.0,
    center_db: float = -15.0,
    temperature_db: float = 6.0,
    soft_floor: float = 0.1,
    num_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    """Construct union, audibility-threshold, or soft-audibility labels."""

    if policy not in {"union", "audibility_threshold", "soft_audibility"}:
        raise ValueError(f"unknown label policy: {policy!r}")
    union = union_labels(class_ids, num_classes=num_classes)
    if policy == "union":
        return union
    if audibility_db is None:
        raise ValueError(f"audibility_db is required for policy={policy!r}")
    class_scores = _class_audibility(class_ids, audibility_db, num_classes)
    if policy == "audibility_threshold":
        return ((union > 0) & (class_scores >= threshold_db)).astype(np.float32)
    result = np.zeros(num_classes, dtype=np.float32)
    for class_id in set(int(c) for c in class_ids):
        result[class_id] = audibility_score(
            class_scores[class_id], center_db=center_db,
            temperature_db=temperature_db, floor=soft_floor,
        )
    return result


def batch_union_labels(class_id_rows: Iterable[Iterable[int]], *, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Vectorised convenience wrapper for metadata/data-loader code."""

    return validate_labels(np.stack([
        union_labels(row, num_classes=num_classes) for row in class_id_rows
    ]), num_classes=num_classes)
