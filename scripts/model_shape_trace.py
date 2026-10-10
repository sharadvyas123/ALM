"""Trace the active EnhancedALMEncoder tensor shapes for candidate hop lengths.

Run from the repository root with the project's CUDA-enabled interpreter:

    venv/bin/python -B scripts/model_shape_trace.py --compare 320 512

The script deliberately reports a failing candidate instead of silently changing
the positional-embedding capacity.  The approved baseline uses hop length 512.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio.transforms as T

from utils.enhanced_alm_mad_encoder import EnhancedALMEncoder, TimeFreqFusionBlock


def trace(hop_length: int, use_fusion: bool, use_class_queries: bool) -> dict:
    batch_size = 2
    waveform_samples = 160_000
    mel_transform = T.MelSpectrogram(
        sample_rate=16_000,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=128,
        f_min=20.0,
        f_max=8_000.0,
    )
    waveform = torch.zeros(batch_size, 1, waveform_samples)
    mel = mel_transform(waveform)
    model = EnhancedALMEncoder(
        embed_dim=256,
        num_classes=7,
        num_heads=8,
        num_layers=6,
        ff_dim=1024,
        dropout=0.1,
        use_fusion=use_fusion,
        use_class_queries=use_class_queries,
        max_time_frames=100,
    ).eval()

    with torch.no_grad():
        features = model.backbone(mel)
        fused_features = model.fusion_block(features) if model.fusion_block is not None else features
        b, channels, freq_bins, time_frames = fused_features.shape
        flat = fused_features.reshape(b, channels * freq_bins, time_frames)
        tokens = flat.permute(0, 2, 1)
        projected = model.patch_proj(tokens)
        result = {
            "hop_length": hop_length,
            "fusion": use_fusion,
            "class_queries": use_class_queries,
            "mel": tuple(mel.shape),
            "backbone": tuple(features.shape),
            "fusion_output": tuple(fused_features.shape),
            "tokens_before_projection": tuple(tokens.shape),
            "tokens": tuple(projected.shape),
            "positional_capacity": int(model.pos_embed.shape[1]),
            "token_count_fits_positional_embedding": time_frames <= model.pos_embed.shape[1],
        }
        try:
            logits = model(mel)
            embedding = model.encode(mel)
            result["logits"] = tuple(logits.shape)
            result["embedding"] = tuple(embedding.shape)
            result["forward"] = "passed"
        except RuntimeError as exc:
            result["forward"] = "failed"
            result["forward_error"] = str(exc).splitlines()[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs="+", type=int, default=[512])
    args = parser.parse_args()

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print("fusion_branch_axis_check: input=(B,C,F,T), (1,3)=time, (3,1)=frequency")
    block = TimeFreqFusionBlock(channels=4).eval()
    print(f"time_branch_kernel={tuple(block.time_branch[0].kernel_size)}")
    print(f"freq_branch_kernel={tuple(block.freq_branch[0].kernel_size)}")

    for hop_length in args.compare:
        for use_fusion, use_class_queries in ((False, False), (True, True)):
            result = trace(hop_length, use_fusion, use_class_queries)
            print("\nTRACE")
            for key, value in result.items():
                print(f"{key}={value}")


if __name__ == "__main__":
    main()
