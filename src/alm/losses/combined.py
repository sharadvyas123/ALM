"""Composition of classification and spatial mask objectives."""
from __future__ import annotations

import torch
from torch import nn

from .asymmetric import AsymmetricLoss
from .audibility import audibility_positive_weight, weighted_bce_with_logits
from .cost_sensitive import apply_operational_costs, negative_weights_from_matrix, validate_cost_matrix, validate_cost_vector


def mask_cross_entropy(attention: torch.Tensor, target: torch.Tensor,
                       *, valid_mask: torch.Tensor | None = None,
                       class_valid: torch.Tensor | None = None,
                       epsilon: float = 1e-8) -> torch.Tensor:
    """Cross entropy between per-class spatial distributions.

    Attention is already a probability distribution over ``F*T``. Each target
    ratio mask is normalised over valid cells, and absent/silent classes are
    excluded through ``class_valid``. This explicitly supports multiple active
    classes in one cell.
    """
    if attention.ndim != 4 or target.shape != attention.shape:
        raise ValueError("attention and target must both have shape (B,C,F,T)")
    if not torch.isfinite(attention).all() or not torch.isfinite(target).all():
        raise ValueError("mask tensors must be finite")
    if torch.any(target < 0):
        raise ValueError("mask targets must be nonnegative")
    b, c, f, t = attention.shape
    support = torch.ones((b, 1, f, t), dtype=torch.bool, device=attention.device)
    if valid_mask is not None:
        support = valid_mask.to(torch.bool)
        if support.shape == (b, f, t):
            support = support[:, None]
        if support.shape != (b, 1, f, t):
            raise ValueError("valid_mask must be (B,F,T) or (B,1,F,T)")
    masked_target = target * support
    target_mass = masked_target.flatten(2).sum(-1)
    normalised_target = masked_target / target_mass.clamp_min(epsilon)[:, :, None, None]
    predicted = attention.clamp_min(epsilon)
    loss = -(normalised_target * predicted.log()).flatten(2).sum(-1)
    valid = target_mass > epsilon
    if class_valid is not None:
        if class_valid.shape != (b, c):
            raise ValueError("class_valid must have shape (B,C)")
        valid = valid & class_valid.to(torch.bool)
    if not valid.any():
        return attention.sum() * 0.0
    return loss.masked_select(valid).mean()


class CombinedLoss(nn.Module):
    """Switchable objective ``classification_loss + lambda_mask * mask_loss``."""

    def __init__(self, *, num_classes: int = 7, pos_weight: torch.Tensor | None = None,
                 use_asymmetric_loss: bool = False, asl_gamma_pos: float = 0.0,
                 asl_gamma_neg: float = 4.0, asl_clip: float = 0.05,
                 use_audibility_loss: bool = False, audibility_floor: float = 0.25,
                 audibility_center_db: float = -15.0, audibility_temperature_db: float = 6.0,
                 confusion_matrix: torch.Tensor | None = None, confusion_strength: float = 1.0,
                 operational_costs: torch.Tensor | None = None, lambda_mask: float = 0.0) -> None:
        super().__init__()
        if num_classes <= 0 or lambda_mask < 0:
            raise ValueError("invalid loss dimensions or mask coefficient")
        self.num_classes = num_classes
        self.use_asymmetric_loss = use_asymmetric_loss
        self.use_audibility_loss = use_audibility_loss
        self.audibility_floor = audibility_floor
        self.audibility_center_db = audibility_center_db
        self.audibility_temperature_db = audibility_temperature_db
        self.confusion_strength = confusion_strength
        self.lambda_mask = lambda_mask
        self.asl = AsymmetricLoss(asl_gamma_pos, asl_gamma_neg, asl_clip, reduction="none")
        self.register_buffer("pos_weight", torch.ones(num_classes) if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32))
        if self.pos_weight.shape != (num_classes,) or not torch.isfinite(self.pos_weight).all() or torch.any(self.pos_weight <= 0):
            raise ValueError("pos_weight must be finite and positive per class")
        self.register_buffer("confusion_matrix", None if confusion_matrix is None else validate_cost_matrix(confusion_matrix, num_classes))
        self.register_buffer("operational_costs", None if operational_costs is None else validate_cost_vector(operational_costs, num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, *,
                audibility_db: torch.Tensor | None = None,
                attention_maps: torch.Tensor | None = None,
                mask_targets: torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None,
                class_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("logits must have shape (B,num_classes)")
        if targets.shape != logits.shape:
            raise ValueError("targets must align with logits")
        positive_weight = None
        if self.use_audibility_loss:
            if audibility_db is None or audibility_db.shape != targets.shape:
                raise ValueError("audibility_db aligned to targets is required")
            positive_weight = audibility_positive_weight(
                audibility_db, center_db=self.audibility_center_db,
                temperature_db=self.audibility_temperature_db, floor=self.audibility_floor,
            )
        negative_weight = None
        if self.confusion_matrix is not None:
            negative_weight = negative_weights_from_matrix(targets, self.confusion_matrix,
                                                            strength=self.confusion_strength)
        if self.use_asymmetric_loss:
            elementwise = self.asl(logits, targets)
            if positive_weight is not None:
                elementwise = elementwise * (1.0 + (positive_weight - 1.0) * targets)
            if negative_weight is not None:
                elementwise = elementwise * (1.0 + (negative_weight - 1.0) * (1.0 - targets))
            if self.operational_costs is not None:
                elementwise = apply_operational_costs(elementwise, self.operational_costs)
            classification = elementwise.mean()
        else:
            elementwise = weighted_bce_with_logits(
                logits, targets, pos_weight=self.pos_weight.to(logits),
                positive_weight=positive_weight, negative_weight=negative_weight,
                class_weight=None, reduction="none",
            )
            if self.operational_costs is not None:
                elementwise = apply_operational_costs(elementwise, self.operational_costs)
            classification = elementwise.mean()
        if self.lambda_mask > 0:
            if attention_maps is None or mask_targets is None:
                raise ValueError("attention_maps and mask_targets are required for mask supervision")
            auxiliary = mask_cross_entropy(attention_maps, mask_targets,
                                           valid_mask=valid_mask, class_valid=class_valid)
        else:
            auxiliary = logits.sum() * 0.0
        return {"total": classification + self.lambda_mask * auxiliary,
                "classification": classification, "mask": auxiliary,
                "positive_mean": (positive_weight.mean() if positive_weight is not None else logits.sum() * 0.0),
                "negative_mean": (negative_weight.mean() if negative_weight is not None else logits.sum() * 0.0)}
