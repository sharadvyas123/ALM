"""Full-framework model with a checkpoint-compatible B0 path."""
from __future__ import annotations

import torch
from torch import nn
from utils.enhanced_alm_mad_encoder import EnhancedALMEncoder

from .class_queries import SpatialClassQueryHead


class ALMModel(EnhancedALMEncoder):
    """Reuse the CNN, fusion, patch projection, and temporal Transformer.

    With queries disabled this is exactly the existing mean-pooling model,
    including state-dict keys. With queries enabled, projected local CNN
    cells are added to the temporal Transformer context at each time step.
    Queries attend to those ``8 x 78`` cells. This retains the original
    temporal encoder instead of replacing it with a much larger 2-D encoder.
    """

    def __init__(self, *, use_fusion: bool = False, use_class_queries: bool = False,
                 use_mask_supervision: bool = False,
                 embed_dim: int = 256, num_heads: int = 8, num_layers: int = 6,
                 ff_dim: int = 1024, dropout: float = 0.1, max_time_frames: int = 100) -> None:
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads or num_layers < 1:
            raise ValueError("Invalid Transformer dimensions")
        if use_mask_supervision and not use_class_queries:
            raise ValueError("mask supervision requires class queries")
        super().__init__(use_fusion=use_fusion, use_class_queries=False, embed_dim=embed_dim,
                         num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim,
                         dropout=dropout, max_time_frames=max_time_frames)
        self.use_class_queries = use_class_queries
        self.use_mask_supervision = use_mask_supervision
        if use_class_queries:
            self.pool_norm = self.pool_classifier = None
            self.spatial_projection = nn.Linear(self.backbone_channels, embed_dim)
            self.frequency_embedding = nn.Parameter(torch.empty(1, 8, 1, embed_dim))
            nn.init.trunc_normal_(self.frequency_embedding, std=0.02)
            self.class_query_head = SpatialClassQueryHead(embed_dim, num_heads)

    def forward(self, mel: torch.Tensor, return_aux: bool = False) -> torch.Tensor | dict:
        """Return raw ``(B,7)`` logits; auxiliary mode adds ``attention (B,7,8,T)``.

        Input is fixed-duration audio; source masks are never inference inputs.
        Source silence/padding is excluded from mask-loss support separately.
        """
        if mel.ndim != 4 or mel.shape[1:3] != (1, 128) or mel.shape[-1] < 4:
            raise ValueError("mel must have shape (B,1,128,T>=4)")
        if mel.shape[-1] // 4 > self.pos_embed.shape[1] or not torch.isfinite(mel).all():
            raise ValueError("Nonfinite input or temporal positional capacity exceeded")
        features = self.backbone(mel)
        if self.fusion_block is not None:
            features = self.fusion_block(features)
        b, c, f, t = features.shape
        tokens = self.patch_proj(features.reshape(b, c * f, t).permute(0, 2, 1))
        sequence = self.transformer(tokens + self.pos_embed[:, :t])
        if self.class_query_head is None:
            result = {"logits": self.pool_classifier(self.pool_norm(sequence.mean(1)))}
        else:
            local = self.spatial_projection(features.permute(0, 2, 3, 1))
            result = self.class_query_head(local + sequence[:, None] + self.frequency_embedding)
        return result if return_aux else result["logits"]
