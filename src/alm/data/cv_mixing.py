"""Fold-specific deterministic mixture planning isolated from the certified v1 generator."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .mixing import metadata_rows, render_dataset
from .provenance import CLASS_NAMES


def plan_cv_mixtures(sources: list[dict], config: dict, *, splits: tuple[str, ...]) -> list[dict]:
    """Plan v1-compatible mixtures for selected split pools using independent RNG streams."""
    rows = []
    split_indices = {"train": 0, "val": 1}
    if not splits or any(split not in split_indices for split in splits):
        raise ValueError("CV generation requires one or more train/val split names")
    for split in splits:
        split_index = split_indices[split]
        pool = sorted((r for r in sources if r["split"] == split), key=lambda r: r["file"])
        targets = config["counts_by_split"][split]
        class_pools = defaultdict(list)
        for record in pool:
            class_pools[int(record["label"])].append(record)
        if set(class_pools) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"Missing class in {split} pool")
        if len(pool) < targets["clean"]:
            raise ValueError(f"Not enough unique {split} sources for clean outputs")
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
                if any(record["split"] != split for record in components):
                    raise ValueError("Mixture component outside its assigned source pool")
                labels = [int(record["label"]) for record in components]
                row_id = len(rows) + 1
                rows.append({
                    "id": str(row_id), "file": f"mad_cv_{row_id:06d}.wav",
                    "labels": ",".join(map(str, labels)),
                    "label_names": ",".join(CLASS_NAMES[c] for c in labels),
                    "label_vector": ",".join(str(int(c in labels)) for c in range(len(CLASS_NAMES))),
                    "mix_type": mix_type,
                    "source_files": "|".join(record["file"] for record in components),
                    "mix_weights": "|".join(repr(float(weight)) for weight in weights),
                    "split": split,
                    "components": json.dumps([
                        {"file": record["file"], "source_id": record["source_id"],
                         "group_id": record["group_id"], "video_id": record["video_id"],
                         "weight": float(weight), "offset_samples": 0}
                        for record, weight in zip(components, weights)
                    ], sort_keys=True, separators=(",", ":")),
                })
    return rows
