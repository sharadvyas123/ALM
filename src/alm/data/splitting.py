"""Deterministic class-aware allocation of intact source groups."""

from collections import Counter, defaultdict

import numpy as np

from .provenance import CLASS_NAMES


def split_sources(sources, seed, val_ratio, candidates, clean_targets):
    if not 0 < val_ratio < 1 or candidates < 1:
        raise ValueError("Invalid source split settings")
    groups = defaultdict(list)
    for row in sources:
        if row["split"] == "eligible":
            groups[row["group_id"]].append(row)
    keys = sorted(groups)
    n_val = round(len(keys) * val_ratio)
    if not 0 < n_val < len(keys):
        raise ValueError("Too few eligible groups for two source pools")
    counts = np.array([
        [sum(r["label"] == c for r in groups[g]) for c in range(len(CLASS_NAMES))]
        for g in keys
    ], dtype=np.int64)
    totals = counts.sum(axis=0)
    if np.any(totals == 0):
        raise ValueError("Eligible sources do not cover all seven classes")
    rng = np.random.default_rng(seed)
    best = None
    for candidate in range(candidates):
        indices = rng.permutation(len(keys))[:n_val]
        val = counts[indices].sum(axis=0)
        train = totals - val
        if (np.any(val == 0) or np.any(train == 0)
                or val.sum() < clean_targets["val"]
                or train.sum() < clean_targets["train"]):
            continue
        # Select using source class counts only, never audio features or model metrics.
        score = float(np.mean((val / totals - val_ratio) ** 2)
                      + (val.sum() / totals.sum() - val_ratio) ** 2)
        if best is None or score < best[0]:
            best = (score, candidate, indices)
    if best is None:
        raise ValueError("No feasible group split with class coverage and clean capacity")
    score, candidate, indices = best
    val_groups = {keys[i] for i in indices}
    result = []
    for row in sources:
        row = dict(row)
        if row["split"] == "eligible":
            row["split"] = "val" if row["group_id"] in val_groups else "train"
        result.append(row)
    summary = {
        "algorithm": "seeded_group_candidates_v1", "seed": seed,
        "candidate_count": candidates, "selected_candidate_zero_based": candidate,
        "objective": "mean((val_class/total_class-ratio)^2)+(val_size/total_size-ratio)^2",
        "objective_value": score, "source_val_ratio_target": val_ratio,
        "pools": {},
    }
    for split in ("train", "val"):
        rows = [r for r in result if r["split"] == split]
        summary["pools"][split] = {
            "sources": len(rows), "groups": len({r["group_id"] for r in rows}),
            "class_counts": dict(Counter(r["label_name"] for r in rows)),
        }
    return result, summary
