"""Spatial queries using the existing query-head parameters and public MHA API."""
from __future__ import annotations

import torch
from utils.enhanced_alm_mad_encoder import ClassQueryHead


class SpatialClassQueryHead(ClassQueryHead):
    """Return per-class probability maps over a time-frequency token grid.

    Attention dropout is zero, so the returned head-averaged weights are
    normalised distributions even during training. Backbone/Transformer
    dropout remains configured normally. Classes normalise independently.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__(embed_dim=embed_dim, num_classes=7, num_heads=num_heads, dropout=0.0)
        torch.nn.init.trunc_normal_(self.class_queries, std=0.02)

    def forward(self, grid: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
        """Accept ``(B,F,T,D)`` and optional boolean ``(B,F,T)`` valid cells."""
        if grid.ndim != 4:
            raise ValueError("grid must have shape (B,F,T,D)")
        b, f, t, d = grid.shape
        if valid is not None:
            if valid.shape != (b, f, t) or valid.dtype != torch.bool or not valid.flatten(1).any(1).all():
                raise ValueError("valid grid must be boolean, aligned and nonempty per example")
        tokens = grid.reshape(b, f * t, d)
        queries = self.class_queries.unsqueeze(0).expand(b, -1, -1)
        attended, attention = self.attn(queries, tokens, tokens,
                                        key_padding_mask=None if valid is None else ~valid.flatten(1),
                                        need_weights=True, average_attn_weights=True)
        return {"logits": self.classifier(self.norm(attended)).squeeze(-1),
                "attention": attention.reshape(b, 7, f, t)}
