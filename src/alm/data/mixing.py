"""Plan mixtures within preassigned pools, then render the saved recipes."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .provenance import CLASS_NAMES, METADATA_FIELDS, relative_id, sha256


def plan_mixtures(sources, config):
    """Pure deterministic planning; independent RNG streams per split and purpose."""
    rows = []
    for split_index, split in enumerate(("train", "val")):
        pool = sorted((r for r in sources if r["split"] == split), key=lambda r: r["file"])
        targets = config["counts_by_split"][split]
        class_pools = defaultdict(list)
        for record in pool:
            class_pools[int(record["label"])].append(record)
        if set(class_pools) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"Missing class in {split} pool")
        if len(pool) < targets["clean"]:
            raise ValueError(f"Not enough unique {split} sources for clean outputs")
        # Split RNG remains default_rng(seed), matching the feasibility experiment.
        # Generation streams are derived independently, so draw counts never alter pools.
        clean_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], split_index, 1]))
        mix_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], split_index, 2]))
        clean_indices = clean_rng.permutation(len(pool))[:targets["clean"]]
        for mix_type, n_components in (("clean", 1), ("2-class", 2), ("3-class", 3)):
            for i in range(targets[mix_type]):
                if n_components == 1:
                    components = [pool[clean_indices[i]]]
                    weights = [1.0]
                else:
                    classes = mix_rng.choice(len(CLASS_NAMES), n_components, replace=False)
                    components = [class_pools[c][mix_rng.integers(len(class_pools[c]))] for c in classes]
                    if n_components == 2:
                        alpha = float(mix_rng.uniform(*config["alpha_range"]))
                        weights = [alpha, 1.0 - alpha]
                    else:
                        raw = mix_rng.dirichlet(config["dirichlet_concentration"])
                        floor = config["three_class_weight_floor"]
                        weights = raw * (1 - 3 * floor) + floor
                        weights = (weights / weights.sum()).tolist()
                if any(r["split"] != split for r in components):
                    raise ValueError("Mixture component outside assigned source pool")
                labels = [int(r["label"]) for r in components]
                row_id = len(rows) + 1
                row = {
                    "id": str(row_id), "file": f"mad_ml_{row_id:06d}.wav",
                    "labels": ",".join(map(str, labels)),
                    "label_names": ",".join(CLASS_NAMES[c] for c in labels),
                    "label_vector": ",".join(str(int(c in labels)) for c in range(len(CLASS_NAMES))),
                    "mix_type": mix_type,
                    "source_files": "|".join(r["file"] for r in components),
                    # Full-precision values used in rendering, unlike legacy 4-digit gains.
                    "mix_weights": "|".join(repr(float(w)) for w in weights), "split": split,
                    "components": json.dumps([
                        {"file": r["file"], "source_id": r["source_id"],
                         "group_id": r["group_id"], "video_id": r["video_id"],
                         "weight": float(w), "offset_samples": 0}
                        for r, w in zip(components, weights)
                    ], sort_keys=True, separators=(",", ":")),
                }
                rows.append(row)
    return rows


def metadata_rows(recipes):
    return [{key: row[key] for key in METADATA_FIELDS} for row in recipes]


def check_audio(path, sample_rate, frames):
    info = sf.info(str(path))
    if (info.samplerate != sample_rate or info.frames != frames or info.channels != 1
            or info.subtype != "PCM_16" or info.format != "WAV"):
        raise ValueError(f"Expected mono PCM16 WAV, {sample_rate} Hz, {frames} frames: {path}")


def render_recipe(recipe, source_audio, config):
    frames = config["sample_rate"] * config["duration_seconds"]
    files = recipe["source_files"].split("|")
    weights = [float(w) for w in recipe["mix_weights"].split("|")]
    if len(files) != len(weights):
        raise ValueError("Source/weight count mismatch")
    wavs = []
    for file in files:
        path = Path(source_audio) / relative_id(file)
        check_audio(path, config["sample_rate"], frames)
        audio, _ = sf.read(str(path), dtype="int16")
        wavs.append(audio)
    if recipe["mix_type"] == "clean":
        return wavs[0], 1.0
    mixed = sum(w * wav.astype(np.float64) for w, wav in zip(weights, wavs))
    peak = float(np.max(np.abs(mixed)))
    scale = 32767.0 / peak if peak > 32767 else 1.0
    # Preserve the legacy peak normalization, clipping, and integer truncation.
    return np.clip(mixed * scale, -32768, 32767).astype(np.int16), scale


def render_dataset(recipes, source_audio, output_audio, config):
    rendered = []
    for index, recipe in enumerate(recipes, start=1):
        audio, scale = render_recipe(recipe, source_audio, config)
        path = Path(output_audio) / recipe["file"]
        # Exclusive creation: a partial/old generation cannot be overwritten.
        with path.open("xb") as handle:
            sf.write(handle, audio, config["sample_rate"], format="WAV", subtype="PCM_16")
        row = dict(recipe, peak_normalization_scale=repr(scale), audio_sha256=sha256(path))
        rendered.append(row)
        if index % 1000 == 0 or index == len(recipes):
            print(f"Rendered {index}/{len(recipes)} examples", flush=True)
    return rendered
