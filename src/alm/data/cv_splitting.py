"""Deterministic class-aware assignment of intact provenance groups to CV folds."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .provenance import CLASS_NAMES, object_hash


def assign_group_folds(sources: list[dict], *, n_splits: int = 5, seed: int = 42,
                       candidates: int = 4096) -> tuple[dict[str, int], dict]:
    """Assign every eligible connected group exactly once, balancing class sources.

    Candidate allocations are scored only from source labels and group sizes.
    No audio features or model predictions participate in fold construction.
    """
    if n_splits < 2 or candidates < 1:
        raise ValueError("n_splits must be >=2 and candidates must be positive")
    grouped = defaultdict(list)
    for row in sources:
        if row.get("split") in {"train", "val", "eligible"}:
            grouped[row["group_id"]].append(row)
    keys = sorted(grouped)
    if len(keys) < n_splits:
        raise ValueError("Fewer eligible groups than requested folds")
    vectors = np.asarray([
        [sum(int(r["label"]) == c for r in grouped[group]) for c in range(len(CLASS_NAMES))]
        for group in keys
    ], dtype=np.int64)
    totals = vectors.sum(axis=0)
    if np.any(totals < n_splits):
        absent = [CLASS_NAMES[i] for i in np.flatnonzero(totals < n_splits)]
        raise ValueError(f"Not enough sources to put every class in every fold: {absent}")
    sizes = np.asarray([len(grouped[g]) for g in keys], dtype=np.int64)
    rng = np.random.default_rng(seed)
    best = None
    for candidate in range(candidates):
        order = rng.permutation(len(keys))
        # Place rare-class and larger groups first; random order resolves ties.
        rarity = np.sum(vectors[order] / np.maximum(totals, 1), axis=1)
        order = order[np.argsort(-(rarity + sizes[order] / max(sizes.sum(), 1) * 1e-3), kind="stable")]
        fold_counts = np.zeros((n_splits, len(CLASS_NAMES)), dtype=np.int64)
        fold_sizes = np.zeros(n_splits, dtype=np.int64)
        assignment = np.full(len(keys), -1, dtype=np.int16)
        for index in order:
            scores = []
            fold_priority = rng.permutation(n_splits)
            for fold in fold_priority:
                proposed = fold_counts.copy()
                proposed[fold] += vectors[index]
                proposed_sizes = fold_sizes.copy()
                proposed_sizes[fold] += sizes[index]
                class_balance = np.mean(((proposed / totals) - (1.0 / n_splits)) ** 2)
                size_balance = np.mean(((proposed_sizes / sizes.sum()) - (1.0 / n_splits)) ** 2)
                scores.append((class_balance + size_balance, int(fold)))
            minimum = min(score for score, _ in scores)
            fold = next(fold for score, fold in scores if score <= minimum + 1e-15)
            assignment[index] = fold
            fold_counts[fold] += vectors[index]
            fold_sizes[fold] += sizes[index]
        if np.any(fold_counts == 0):
            continue
        score = float(np.mean(((fold_counts / totals) - (1.0 / n_splits)) ** 2)
                      + np.mean(((fold_sizes / sizes.sum()) - (1.0 / n_splits)) ** 2))
        if best is None or score < best[0]:
            best = (score, candidate, assignment.copy(), fold_counts.copy(), fold_sizes.copy())
    if best is None:
        raise ValueError("No class-complete grouped fold allocation found")
    score, selected, assignment, fold_counts, fold_sizes = best
    result = {group: int(assignment[i]) + 1 for i, group in enumerate(keys)}
    summary = {
        "algorithm": "seeded_group_class_balance_candidates_v1",
        "n_splits": n_splits, "seed": seed, "candidate_count": candidates,
        "selected_candidate_zero_based": selected, "objective_value": score,
        "objective": "mean class source share deviation squared + fold source share deviation squared",
        "assignment_sha256": object_hash(result), "folds": {},
    }
    for fold in range(n_splits):
        summary["folds"][str(fold + 1)] = {
            "groups": sum(1 for value in result.values() if value == fold + 1),
            "sources": int(fold_sizes[fold]),
            "class_source_counts": {CLASS_NAMES[c]: int(fold_counts[fold, c]) for c in range(len(CLASS_NAMES))},
        }
    return result, summary
