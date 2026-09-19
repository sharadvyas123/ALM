
"""
Enhanced ALM MAD Encoder
========================
CNN backbone → Multi-Scale Time-Frequency Fusion Block → Transformer Encoder → 7 class-query cross-attention heads → 7 logits.

Extends the Audio Spectrogram Transformer (AST) design for multi-label military audio classification on the Military Audio Dataset (MAD), with
7 classes: Communication, Gunshot, Footsteps, Shelling, Vehicle, Helicopter, Fighter.

Two research contributions are implemented here as togglable modules so that all 4 ablation configurations are reachable from this single class via constructor flags:

    1. Multi-Scale Time-Frequency Fusion Block (`use_fusion`)
       — parallel frequency / time / joint convolutions that enhance the
         CNN backbone's feature map before it is tokenized.
    2. Seven learnable class-query cross-attention pooling heads
       (`use_class_queries`)
       — each class owns a query vector that cross-attends over the
         Transformer output to build a class-specific representation,
         instead of a single shared mean-pooled vector.

Audio preprocessing assumptions (see project docs):
    Sample rate:  16 kHz
    Duration:     10 seconds
    Samples:      160,000
    FFT:          1024
    Hop length:   512
    Mel bins:     128
    Time frames:  ~313  (160000 / 512 + 1)
    Input tensor: (B, 1, 128, 313)

Task: multi-label classification (7 classes) — BCEWithLogitsLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ── ConvBlock ────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout2d.

    Standard double-convolution downsampling block used by the CNN
    backbone. `pool_size` controls how much of the frequency / time axes
    are collapsed at this stage, e.g. (2, 2) or (2, 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int | tuple[int, int] = (2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ── TimeFreqFusionBlock ─────────────────────────────────────────────────────
class TimeFreqFusionBlock(nn.Module):
    """Multi-Scale Time-Frequency Fusion Block (contribution #1).

    Three parallel conv branches read the backbone feature map
    (B, C, F, T): a frequency branch (1, 3) capturing local vertical
    patterns, a time branch (3, 1) capturing local horizontal patterns,
    and a joint branch (3, 3) capturing local time-frequency patterns.
    The branches are concatenated, fused back to `channels` with a 1x1
    convolution, and added residually to the input, so the block is a
    drop-in identity-shaped enhancement of the input feature map.
    """

    def __init__(self, channels: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.freq_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.time_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.joint_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.freq_branch(x)
        time = self.time_branch(x)
        joint = self.joint_branch(x)
        fused = self.fuse(torch.cat([freq, time, joint], dim=1))
        fused = self.dropout(fused)
        return x + fused


# ── ClassQueryHead ───────────────────────────────────────────────────────────
class ClassQueryHead(nn.Module):
    """Class-query cross-attention pooling head (contribution #2).

    Holds one learnable query vector per class. Each query cross-attends
    over the full Transformer output sequence to build a class-specific
    representation, which is then normalized and projected to a single
    logit per class. This replaces a single shared mean-pooled vector
    with 7 independently-attended, class-specific summaries.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_queries = nn.Parameter(torch.zeros(num_classes, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size = sequence.shape[0]
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attn(query=queries, key=sequence, value=sequence)
        attended = self.norm(attended)
        logits = self.classifier(attended).squeeze(-1)
        return logits


# ── EnhancedALMEncoder ───────────────────────────────────────────────────────
class EnhancedALMEncoder(nn.Module):
    """Enhanced ALM MAD multi-label audio encoder.

    CNN backbone → optional Time-Freq Fusion Block → patch projection +
    positional encoding → Transformer encoder → class-query cross-attention
    head (or a fallback mean-pool head). All 4 ablation configurations
    (baseline AST, AST + Fusion, AST + Class Queries, full proposed model)
    are reachable from this single class via `use_fusion` and
    `use_class_queries`.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pooling: str = "attention",
        use_fusion: bool = True,
        use_class_queries: bool = True,
        max_time_frames: int =128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_fusion = use_fusion
        self.use_class_queries = use_class_queries
        self.pooling = pooling

        # Stage 1: CNN backbone. Blocks 3-4 use pool=(2, 1) to preserve
        # temporal resolution for the longer 10s clips.
        self.backbone = nn.Sequential(
            ConvBlock(1, 32, pool_size=(2, 2), dropout=dropout),
            ConvBlock(32, 64, pool_size=(2, 2), dropout=dropout),
            ConvBlock(64, 128, pool_size=(2, 1), dropout=dropout),
            ConvBlock(128, 256, pool_size=(2, 1), dropout=dropout),
        )
        self.backbone_channels = 256
        self.backbone_freq_bins = 8  # 128 / 2 / 2 / 2 / 2

        # Stage 2: Multi-Scale Time-Frequency Fusion Block (contribution #1).
        # Only instantiated when enabled — keeps param count honest per ablation.
        self.fusion_block = (
            TimeFreqFusionBlock(self.backbone_channels, dropout=dropout)
            if use_fusion else None
        )

        # Stage 3: Patch projection + positional encoding.
        patch_dim = self.backbone_channels * self.backbone_freq_bins
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_time_frames, embed_dim))

        # Stage 4: Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Stage 5: Classification head — only one is instantiated.
        if use_class_queries:
            # 5a: Class-query cross-attention head (contribution #2).
            self.class_query_head = ClassQueryHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.pool_norm = None
            self.pool_classifier = None
        else:
            # 5b: Fallback mean-pool head (standard AST-style).
            self.class_query_head = None
            self.pool_norm = nn.LayerNorm(embed_dim)
            self.pool_classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.class_query_head is not None:
            nn.init.trunc_normal_(self.class_query_head.class_queries, std=0.02)

    def _encode_sequence(self, mel: torch.Tensor) -> torch.Tensor:
        """Shared backbone → fusion → tokenize → Transformer pipeline.

        Args:
            mel: (B, 1, 128, 313) log-mel spectrogram batch.

        Returns:
            (B, T, embed_dim) Transformer-encoded token sequence.
        """
        features = self.backbone(mel)  # (B, 256, F, T)

        if self.fusion_block is not None:
            features = self.fusion_block(features)  # (B, 256, F, T)

        batch_size, channels, freq_bins, time_frames = features.shape
        tokens = features.reshape(batch_size, channels * freq_bins, time_frames)
        tokens = tokens.permute(0, 2, 1)  # (B, T, channels * F)
        tokens = self.patch_proj(tokens)  # (B, T, embed_dim)

        pos = self.pos_embed[:, :time_frames, :]
        tokens = tokens + pos

        return self.transformer(tokens)  # (B, T, embed_dim)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Return an L2-normalized (B, embed_dim) embedding for `mel`."""
        sequence = self._encode_sequence(mel)
        pooled = sequence.mean(dim=1)  # (B, embed_dim)
        return torch.nn.functional.normalize(pooled, p=2, dim=-1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Return (B, num_classes) raw logits (NO sigmoid applied)."""
        sequence = self._encode_sequence(mel)

        if self.class_query_head is not None:
            return self.class_query_head(sequence)

        pooled = sequence.mean(dim=1)  # (B, embed_dim)
        pooled = self.pool_norm(pooled)
        return self.pool_classifier(pooled)

    def count_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ─────────────────────────────────────────────────────────────
def _sanity_check() -> None:
    torch.manual_seed(0)
    batch_size = 2
    mel = torch.randn(batch_size, 1, 128, 313)

    ablation_configs = [
        {"use_fusion": False, "use_class_queries": False},
        {"use_fusion": True, "use_class_queries": False},
        {"use_fusion": False, "use_class_queries": True},
        {"use_fusion": True, "use_class_queries": True},
    ]

    for config in ablation_configs:
        model = EnhancedALMEncoder(**config)
        model.eval()

        with torch.no_grad():
            logits = model(mel)
            embedding = model.encode(mel)

        assert logits.shape == (batch_size, model.num_classes), f"bad logits shape: {logits.shape}"
        assert embedding.shape == (batch_size, model.embed_dim), f"bad embedding shape: {embedding.shape}"

        norms = embedding.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), f"embedding not L2-normalized: {norms}"

        print(f"[fusion={config['use_fusion']}, queries={config['use_class_queries']}] sanity check passed.")
        print(f"  Logits shape    : {tuple(logits.shape)}")
        print(f"  Embedding shape : {tuple(embedding.shape)}")
        print(f"  Trainable params: {model.count_params():,}")


if __name__ == "__main__":
    _sanity_check()