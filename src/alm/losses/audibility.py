"""Audibility-conditioned BCE terms with an explicit nonzero positive floor."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def audibility_positive_weight(audibility_db: torch.Tensor, *, center_db: float = -15.0,
                              temperature_db: float = 6.0, floor: float = 0.25,
                              ceiling: float = 1.0) -> torch.Tensor:
    """Map SIR/SNR scores to ``[floor, ceiling]`` positive-term weights.

    The weight is monotone in measured audibility: clearly audible sources
    retain full positive weight while hard sources retain at least ``floor``.
    ``-inf`` (silent source) maps to the floor and ``+inf`` to the ceiling.
    """
    if temperature_db <= 0 or not 0 <= floor <= ceiling or ceiling <= 0:
        raise ValueError("invalid audibility weighting bounds")
    if not torch.is_floating_point(audibility_db) or not torch.isfinite(
            torch.where(torch.isinf(audibility_db), torch.zeros_like(audibility_db), audibility_db)).all():
        raise ValueError("audibility_db must be floating and finite except for +/-inf")
    z = (audibility_db - center_db) / temperature_db
    score = torch.sigmoid(z)
    score = torch.where(torch.isposinf(audibility_db), torch.ones_like(score), score)
    score = torch.where(torch.isneginf(audibility_db), torch.zeros_like(score), score)
    return floor + (ceiling - floor) * score


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
    positive_weight: torch.Tensor | None = None,
    negative_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute stable BCE with independently weighted positive/negative terms."""
    if logits.shape != targets.shape or not torch.is_floating_point(targets):
        raise ValueError("logits and floating targets must have identical shapes")
    if torch.any((targets < 0) | (targets > 1)) or not torch.isfinite(targets).all():
        raise ValueError("targets must be finite and lie in [0,1]")
    positive = targets * F.softplus(-logits)
    negative = (1.0 - targets) * F.softplus(logits)
    if pos_weight is not None:
        positive = positive * pos_weight
    if positive_weight is not None:
        positive = positive * positive_weight
    if negative_weight is not None:
        negative = negative * negative_weight
    result = positive + negative
    if class_weight is not None:
        result = result * class_weight
    if reduction == "none":
        return result
    if reduction == "mean":
        return result.mean()
    if reduction == "sum":
        return result.sum()
    raise ValueError("reduction must be none, mean, or sum")
