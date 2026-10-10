"""Read-only audits of exact-file and connected-parent provenance."""

import json
import math
from collections import Counter
from pathlib import Path

from .provenance import (
    CLASS_NAMES, METADATA_FIELDS, read_csv, reference_id, relative_id, sha256,
)


def audit_rows(rows, sources, source_audio, source_manifest=None):
    errors = []
    by_file = {}
    for source in sources:
        key = reference_id(source["file"], source_audio)
        if key in by_file:
            raise ValueError(f"Ambiguous processed source file: {key}")
        by_file[key] = source
    assignments = None
    pool_sets = {}
    if source_manifest is not None:
        assignments = {}
        for source in source_manifest:
            key = reference_id(source["file"], source_audio)
            if key in assignments:
                errors.append(f"Duplicate source manifest file: {key}")
            assignments[key] = source
            original = by_file.get(key)
            if original is None:
                errors.append(f"Unknown source manifest file: {key}")
                continue
            for field in ("source_id", "source_file", "group_id", "video_id", "label",
                          "label_name", "official_split", "provenance_status", "exclusion_reason"):
                if str(source.get(field)) != str(original[field]):
                    errors.append(f"Source manifest mismatch: {key}, {field}")
            expected_split = original["split"]
            allowed = {"train", "val"} if expected_split == "eligible" else {expected_split}
            if source.get("split") not in allowed:
                errors.append(f"Ineligible or incorrectly assigned source: {key}")
        if set(assignments) != set(by_file):
            errors.append("Source manifest does not cover processed metadata exactly")
        for split in ("train", "val", "official_test"):
            pool_sets[split] = {r["group_id"] for r in assignments.values() if r["split"] == split}
        if pool_sets["train"] & pool_sets["val"]:
            errors.append("Source pools share connected groups")
        if (pool_sets["train"] | pool_sets["val"]) & pool_sets["official_test"]:
            errors.append("Source pools share official-test groups")

    refs_by_split = {s: set() for s in ("train", "val")}
    groups_by_split = {s: set() for s in ("train", "val")}
    valid_rows = []
    seen_ids, seen_files, clean_sources = set(), set(), set()
    for line, row in enumerate(rows, start=2):
        try:
            if any(not row.get(k, "").strip() for k in METADATA_FIELDS):
                raise ValueError("missing required value")
            output_id = relative_id(row["file"])
            if row["id"] in seen_ids or output_id in seen_files:
                raise ValueError("duplicate generated ID or file")
            seen_ids.add(row["id"])
            seen_files.add(output_id)
            split = row["split"]
            if split not in refs_by_split:
                raise ValueError(f"unknown split {split!r}")
            expected = {"clean": 1, "2-class": 2, "3-class": 3}.get(row["mix_type"])
            refs = [reference_id(ref, source_audio) for ref in row["source_files"].split("|")]
            if expected is None or len(refs) != expected or len(set(refs)) != len(refs):
                raise ValueError("invalid mix type, source count, or repeated component")
            if assignments is not None and expected == 1:
                if refs[0] in clean_sources:
                    raise ValueError("clean source selected more than once")
                clean_sources.add(refs[0])
            if any(ref not in by_file for ref in refs):
                raise ValueError("unresolved source reference")
            components = [by_file[ref] for ref in refs]
            if any(r["official_split"] != "training" for r in components):
                raise ValueError("component comes from original official test")
            labels = [int(c) for c in row["labels"].split(",")]
            actual_labels = [int(r["label"]) for r in components]
            if labels != actual_labels or len(set(labels)) != expected:
                raise ValueError("component classes do not match mixture labels")
            if row["label_names"].split(",") != [CLASS_NAMES[c] for c in labels]:
                raise ValueError("class-name mismatch")
            if [int(v) for v in row["label_vector"].split(",")] != [int(c in labels) for c in range(7)]:
                raise ValueError("incorrect multi-hot label union")
            weights = [float(w) for w in row["mix_weights"].split("|")]
            if (len(weights) != expected or any(not math.isfinite(w) or w <= 0 for w in weights)
                    or abs(sum(weights) - 1) > 0.0002):
                raise ValueError("invalid weights")
            if expected == 2 and any(not 0.3 <= w <= 0.7 for w in weights):
                raise ValueError("two-class gain outside configured range")
            if expected == 3 and any(w < 0.2 - 1e-12 for w in weights):
                raise ValueError("three-class gain below floor")
            if assignments is not None and any(assignments.get(ref, {}).get("split") != split for ref in refs):
                raise ValueError("component not in the correct source pool")
            groups = {r["group_id"] for r in components}
            refs_by_split[split].update(refs)
            groups_by_split[split].update(groups)
            valid_rows.append((row, set(refs), groups))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"CSV row {line}: {exc}")

    shared = refs_by_split["train"] & refs_by_split["val"]
    shared_groups = groups_by_split["train"] & groups_by_split["val"]
    split_counts = Counter(r.get("split", "") for r in rows)
    if not all(split_counts[s] for s in ("train", "val")):
        errors.append("Both train and val splits must be nonempty")
    exact_affected = {s: [r["file"] for r, refs, _ in valid_rows if r["split"] == s and refs & shared]
                      for s in ("train", "val")}
    group_affected = {s: [r["file"] for r, _, groups in valid_rows if r["split"] == s and groups & shared_groups]
                      for s in ("train", "val")}
    breakdown = {}
    for split in ("train", "val"):
        breakdown[split] = {}
        for kind in ("clean", "2-class", "3-class"):
            subset = [(r, refs, gs) for r, refs, gs in valid_rows if r["split"] == split and r["mix_type"] == kind]
            breakdown[split][kind] = {"total": len(subset), "exact_source_affected": sum(bool(refs & shared) for _, refs, _ in subset)}
    test_groups = {r["group_id"] for r in sources if r["split"] == "official_test"
                   or r["exclusion_reason"] == "official_test_linked_group"}
    group_test_overlap = (groups_by_split["train"] | groups_by_split["val"]) & test_groups
    class_counts = {s: dict(Counter(CLASS_NAMES[int(c)] for r, _, _ in valid_rows if r["split"] == s
                                   for c in r["labels"].split(","))) for s in ("train", "val")}
    if any(set(counts) != set(CLASS_NAMES) for counts in class_counts.values()):
        errors.append("A generated split is missing class coverage")
    samples = []
    for ref in sorted(shared)[:5]:
        samples.append({"file": ref, "source_file": by_file[ref]["source_file"],
                        "examples": {s: [r["file"] for r, refs, _ in valid_rows if r["split"] == s and ref in refs][:3]
                                     for s in ("train", "val")}})
    verdict = ("Inconclusive — insufficient or unreliable source metadata" if errors else
               "Source-level leakage detected" if shared or shared_groups or group_test_overlap else
               "No source-level overlap detected in the audited metadata")
    return {
        "verdict": verdict, "complete": not errors, "error_count": len(errors), "errors": errors,
        "total_examples": len(rows), "split_counts": dict(split_counts),
        "unique_sources": {s: len(v) for s, v in refs_by_split.items()},
        "unique_groups": {s: len(v) for s, v in groups_by_split.items()},
        "overlapping_source_count": len(shared), "overlapping_group_count": len(shared_groups),
        "affected_examples": {s: len(v) for s, v in exact_affected.items()},
        "affected_example_files": exact_affected,
        "affected_percentage": {s: 100 * len(exact_affected[s]) / split_counts[s] if split_counts[s] else None
                                for s in ("train", "val")},
        "group_affected_examples": {s: len(v) for s, v in group_affected.items()},
        "overlapping_sources": sorted(shared), "overlapping_groups": sorted(shared_groups),
        "known_official_test_group_overlap": len(group_test_overlap),
        "mix_counts": breakdown, "class_occurrences": class_counts, "overlap_samples": samples,
        "source_pools_checked": assignments is not None,
        "limitations": [
            "Counts are based on metadata identity, not audio fingerprint deduplication.",
            "Unverified official-test video identities prevent complete parent-video independence certification.",
            "No inference about memorization or metric inflation is made.",
        ] + (["Overlap counts cover validated rows only; see errors."] if errors else []),
    }


def audit_dataset(metadata_path, sources, source_audio, source_manifest_path=None,
                  mixture_manifest_path=None, audio_dir=None, verify_waveforms=False,
                  config=None):
    rows = read_csv(metadata_path, METADATA_FIELDS)
    source_manifest = read_csv(source_manifest_path) if source_manifest_path else None
    report = audit_rows(rows, sources, source_audio, source_manifest)
    report["metadata_sha256"] = sha256(metadata_path)
    report["audio_files_checked"] = 0
    report["waveforms_reconstructed"] = 0
    report["source_audio_hashes_checked"] = 0
    if report["errors"]:
        return report
    recipes = None
    if mixture_manifest_path:
        recipes = read_csv(mixture_manifest_path, (*METADATA_FIELDS, "components", "audio_sha256", "peak_normalization_scale"))
        if len(recipes) != len(rows):
            report["errors"].append("Mixture manifest row count mismatch")
        for row, recipe in zip(rows, recipes):
            if any(row[k] != recipe[k] for k in METADATA_FIELDS):
                report["errors"].append(f"Mixture recipe/metadata mismatch: {row['file']}")
    if verify_waveforms and (recipes is None or audio_dir is None or config is None):
        raise ValueError("Waveform verification requires audio, configuration, and mixture manifest")
    if audio_dir and not report["errors"]:
        import numpy as np
        import soundfile as sf
        from .mixing import check_audio, render_recipe

        cfg = config or {"sample_rate": 16000, "duration_seconds": 10}
        if verify_waveforms and source_manifest is not None:
            for source in source_manifest:
                if source["split"] in {"train", "val"} and "audio_sha256" in source:
                    if sha256(Path(source_audio) / source["file"]) != source["audio_sha256"]:
                        report["errors"].append(f"Source audio checksum mismatch: {source['file']}")
                    report["source_audio_hashes_checked"] += 1
        source_map = {r["file"]: r for r in sources}
        expected_files = {row["file"] for row in rows}
        actual_files = {p.relative_to(audio_dir).as_posix() for p in Path(audio_dir).rglob("*.wav")}
        if expected_files != actual_files:
            report["errors"].append("Generated WAV inventory does not match metadata")
        for index, row in enumerate(rows):
            try:
                path = Path(audio_dir) / row["file"]
                check_audio(path, cfg["sample_rate"], cfg["sample_rate"] * cfg["duration_seconds"])
                report["audio_files_checked"] += 1
                if recipes is not None:
                    recipe = recipes[index]
                    if sha256(path) != recipe["audio_sha256"]:
                        raise ValueError("Generated WAV checksum mismatch")
                    components = json.loads(recipe["components"])
                    refs = row["source_files"].split("|")
                    weights = [float(v) for v in row["mix_weights"].split("|")]
                    if len(components) != len(refs):
                        raise ValueError("Component provenance count mismatch")
                    for component, ref, weight in zip(components, refs, weights):
                        source = source_map[ref]
                        expected = {"file": ref, "source_id": source["source_id"],
                                    "group_id": source["group_id"], "video_id": source["video_id"],
                                    "weight": weight, "offset_samples": 0}
                        if component != expected:
                            raise ValueError("Component provenance mismatch")
                    if verify_waveforms:
                        expected, scale = render_recipe(recipe, source_audio, cfg)
                        actual, _ = sf.read(str(path), dtype="int16")
                        if scale != float(recipe["peak_normalization_scale"]) or not np.array_equal(expected, actual):
                            raise ValueError("Saved waveform differs from its source recipe")
                        report["waveforms_reconstructed"] += 1
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                report["errors"].append(f"{row['file']}: {exc}")
    report["error_count"] = len(report["errors"])
    report["complete"] = not report["errors"]
    if not report["complete"]:
        report["verdict"] = "Inconclusive — insufficient or unreliable source metadata"
    return report


def require_disjoint(report):
    if (not report["complete"] or report["overlapping_source_count"]
            or report["overlapping_group_count"] or report["known_official_test_group_overlap"]):
        raise ValueError("Dataset failed source-disjoint audit; inspect the audit report")
