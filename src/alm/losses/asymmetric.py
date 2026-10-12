"""Asymmetric Loss for multi-label classification (Ridnik et al.)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricLoss(nn.Module):
    """ASL with explicit positive/negative terms and stable logits math.

    ``gamma_neg`` focuses on hard negatives; ``clip`` shifts negative
    probabilities toward one before focusing, suppressing easy-negative
    dominance. Fractional targets are accepted as a well-defined interpolation
    of the positive and negative terms, although the usual ASL setting uses
    binary targets.
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, reduction: str = "mean") -> None:
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0 or not 0 <= clip < 1:
            raise ValueError("ASL gammas must be nonnegative and clip must be in [0,1)")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        self.gamma_pos, self.gamma_neg, self.clip, self.reduction = gamma_pos, gamma_neg, clip, reduction

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.shape != targets.shape or not torch.is_floating_point(targets):
            raise ValueError("logits and floating targets must have identical shapes")
        if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
            raise ValueError("ASL inputs must be finite")
        if torch.any((targets < 0) | (targets > 1)):
            raise ValueError("ASL targets must lie in [0,1]")
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        if self.clip > 0:
            negative_probability = (negative_probability + self.clip).clamp(max=1.0)
        positive_log = F.logsigmoid(logits)
        negative_log = torch.log(negative_probability.clamp_min(torch.finfo(logits.dtype).eps))
        pt = targets * positive_probability + (1.0 - targets) * negative_probability
        gamma = targets * self.gamma_pos + (1.0 - targets) * self.gamma_neg
        focal = (1.0 - pt).clamp_min(0.0).pow(gamma)
        return -targets * positive_log * focal, -(1.0 - targets) * negative_log * focal

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weight: torch.Tensor | None = None) -> torch.Tensor:
        positive, negative = self.components(logits, targets)
        result = positive + negative
        if weight is not None:
            result = result * weight
        if self.reduction == "none":
            return result
        return result.mean() if self.reduction == "mean" else result.sum()
