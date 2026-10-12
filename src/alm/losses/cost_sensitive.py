"""Validated confusion and operational cost weighting.

The confusion matrix convention is ``cost[negative_class, present_class]``:
when class ``present_class`` is active and ``negative_class`` is absent, its
negative BCE term is multiplied by ``1 + strength * cost``. A matrix estimated
from training-label co-occurrence is explicitly different from an operational
cost matrix and is recorded as such in experiment configuration.
"""
from __future__ import annotations

import torch


def validate_cost_matrix(matrix: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    values = torch.as_tensor(matrix, dtype=torch.float32)
    if values.shape != (num_classes, num_classes) or not torch.isfinite(values).all():
        raise ValueError(f"cost matrix must be finite with shape ({num_classes},{num_classes})")
    diagonal_zero = torch.zeros(num_classes, device=values.device, dtype=values.dtype)
    if torch.any(values < 0) or not torch.allclose(torch.diag(values), diagonal_zero):
        raise ValueError("cost matrix must be nonnegative with a zero diagonal")
    return values


def validate_cost_vector(vector: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    values = torch.as_tensor(vector, dtype=torch.float32)
    if values.shape != (num_classes,) or not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError(f"cost vector must be finite, nonnegative, and shape ({num_classes},)")
    return values


def negative_weights_from_matrix(targets: torch.Tensor, matrix: torch.Tensor,
                                 *, strength: float = 1.0) -> torch.Tensor:
    """Return per-example negative weights from active co-occurring classes."""
    if targets.ndim != 2 or not torch.is_floating_point(targets):
        raise ValueError("targets must be a floating matrix")
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    values = validate_cost_matrix(matrix, targets.shape[1]).to(targets.device, targets.dtype)
    # (B, present_class) @ (present_class, negative_class)
    return 1.0 + strength * targets.clamp(0, 1) @ values.T


def estimate_training_cooccurrence(targets: torch.Tensor) -> torch.Tensor:
    """Estimate a row-normalised co-occurrence matrix from training labels only."""
    if targets.ndim != 2 or not torch.is_floating_point(targets):
        raise ValueError("targets must be a floating matrix")
    binary = (targets > 0.5).to(torch.float32)
    counts = binary.T @ binary
    diagonal = torch.diag(counts).clamp_min(1.0)
    result = counts / diagonal[:, None]
    result.fill_diagonal_(0.0)
    return validate_cost_matrix(result, targets.shape[1])


def apply_operational_costs(loss: torch.Tensor, cost_vector: torch.Tensor) -> torch.Tensor:
    """Apply a validated class vector to an unreduced ``(B,C)`` loss."""
    if loss.ndim != 2:
        raise ValueError("loss must be unreduced with shape (B,C)")
    vector = validate_cost_vector(cost_vector, loss.shape[1]).to(loss.device, loss.dtype)
    return loss * (1.0 + vector)
