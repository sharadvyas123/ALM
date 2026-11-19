"""Audits for nested CV datasets; leaves the hash-certified v1 audit untouched."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .mixing import check_audio
from .provenance import CLASS_NAMES, METADATA_FIELDS, read_csv, reference_id, relative_id, sha256


def audit_cv_dataset(metadata_path: Path, sources: list[dict], source_audio: Path,
                     source_manifest_path: Path, mixture_manifest_path: Path,
                     audio_dir: Path, *, required_splits: tuple[str, ...]) -> dict:
    rows = read_csv(metadata_path, METADATA_FIELDS)
    source_manifest = read_csv(source_manifest_path)
    recipes = read_csv(mixture_manifest_path,
                       (*METADATA_FIELDS, "components", "audio_sha256", "peak_normalization_scale"))
    errors = []
    by_file = {reference_id(source["file"], source_audio): source for source in sources}
    assignments = {reference_id(row["file"], source_audio): row for row in source_manifest}
    if len(by_file) != len(sources) or len(assignments) != len(source_manifest):
        errors.append("Duplicate processed source file or source-manifest file")
    if set(assignments) != set(by_file):
        errors.append("Source manifest does not cover processed source metadata exactly")
    fields = ("source_id", "source_file", "group_id", "video_id", "label", "label_name",
              "official_split", "provenance_status", "exclusion_reason")
    for file, assigned in assignments.items():
        original = by_file.get(file)
        if original is None:
            continue
        if any(str(assigned.get(key)) != str(original.get(key)) for key in fields):
            errors.append(f"Source-manifest identity mismatch: {file}")
        expected = original["split"]
        allowed = {"train", "val"} if expected == "eligible" else {expected}
        if assigned.get("split") not in allowed:
            errors.append(f"Invalid source assignment for {file}: {assigned.get('split')}")
    pool_groups = {split: {row["group_id"] for row in source_manifest if row["split"] == split}
                   for split in ("train", "val")}
    if pool_groups["train"] & pool_groups["val"]:
        errors.append("Assigned train and validation source pools share groups")

    refs_by_split = {"train": set(), "val": set()}
    groups_by_split = {"train": set(), "val": set()}
    class_counts = {split: Counter() for split in ("train", "val")}
    seen_ids, seen_files, clean_refs = set(), set(), set()
    if len(rows) != len(recipes):
        errors.append("Mixture manifest row count differs from metadata")
    for index, row in enumerate(rows):
        if index >= len(recipes):
            break
        recipe = recipes[index]
        if any(row[key] != recipe[key] for key in METADATA_FIELDS):
            errors.append(f"Recipe/metadata mismatch for {row.get('file')}")
        try:
            file = relative_id(row["file"])
            if row["id"] in seen_ids or file in seen_files:
                raise ValueError("duplicate generated ID or path")
            seen_ids.add(row["id"])
            seen_files.add(file)
            split = row["split"]
            if split not in refs_by_split:
                raise ValueError(f"unexpected generated split {split}")
            n_components = {"clean": 1, "2-class": 2, "3-class": 3}.get(row["mix_type"])
            refs = [reference_id(value, source_audio) for value in row["source_files"].split("|")]
            if n_components is None or len(refs) != n_components or len(set(refs)) != n_components:
                raise ValueError("invalid component count")
            if n_components == 1:
                if refs[0] in clean_refs:
                    raise ValueError("clean source was selected more than once")
                clean_refs.add(refs[0])
            components = [by_file[ref] for ref in refs]
            if any(component["official_split"] != "training" for component in components):
                raise ValueError("official-test source used in generated data")
            if any(assignments.get(ref, {}).get("split") != split for ref in refs):
                raise ValueError("mixture component does not match its source pool")
            labels = [int(value) for value in row["labels"].split(",")]
            if labels != [int(component["label"]) for component in components] or len(set(labels)) != n_components:
                raise ValueError("mixture labels disagree with source labels")
            if row["label_names"].split(",") != [CLASS_NAMES[label] for label in labels]:
                raise ValueError("mixture label names disagree with canonical order")
            vector = [int(value) for value in row["label_vector"].split(",")]
            if vector != [int(label in labels) for label in range(len(CLASS_NAMES))]:
                raise ValueError("invalid multi-hot label vector")
            weights = [float(value) for value in row["mix_weights"].split("|")]
            if (len(weights) != n_components or any(not math.isfinite(value) or value <= 0 for value in weights)
                    or abs(sum(weights) - 1.0) > 0.0002):
                raise ValueError("invalid mixture weights")
            if n_components == 2 and any(not 0.3 <= value <= 0.7 for value in weights):
                raise ValueError("two-source weight is outside configured range")
            if n_components == 3 and any(value < 0.2 - 1e-12 for value in weights):
                raise ValueError("three-source weight is below configured floor")
            recipe_components = json.loads(recipe["components"])
            if len(recipe_components) != len(refs):
                raise ValueError("recipe component details do not align")
            for detail, ref, weight in zip(recipe_components, refs, weights):
                source = by_file[ref]
                expected_detail = {"file": ref, "source_id": source["source_id"],
                                   "group_id": source["group_id"], "video_id": source["video_id"],
                                   "weight": weight, "offset_samples": 0}
                if detail != expected_detail:
                    raise ValueError("recipe source details mismatch")
            refs_by_split[split].update(refs)
            groups = {component["group_id"] for component in components}
            groups_by_split[split].update(groups)
            for label in labels:
                class_counts[split][CLASS_NAMES[label]] += 1
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"Generated row {index + 2}: {error}")

    if not all(Counter(row["split"] for row in rows)[split] for split in required_splits):
        errors.append(f"Required split is empty: {required_splits}")
    if any(set(class_counts[split]) != set(CLASS_NAMES) for split in required_splits):
        errors.append("A required split does not contain all seven classes")
    overlap_sources = refs_by_split["train"] & refs_by_split["val"]
    overlap_groups = groups_by_split["train"] & groups_by_split["val"]
    test_groups = {row["group_id"] for row in sources
                   if row["split"] == "official_test" or row.get("exclusion_reason") == "official_test_linked_group"}
    test_overlap = (groups_by_split["train"] | groups_by_split["val"]) & test_groups
    if overlap_sources or overlap_groups or test_overlap:
        errors.append("Generated mixture sources overlap across pools or a known official-test group")

    expected_files = {row["file"] for row in rows}
    actual_files = {path.name for path in Path(audio_dir).glob("*.wav")}
    if actual_files != expected_files:
        errors.append("Generated audio inventory differs from metadata")
    audio_checked = 0
    for recipe in recipes:
        audio_path = Path(audio_dir) / recipe["file"]
        try:
            check_audio(audio_path, 16000, 160000)
            if sha256(audio_path) != recipe["audio_sha256"]:
                raise ValueError("generated audio checksum mismatch")
            audio_checked += 1
        except (ValueError, OSError, RuntimeError) as error:
            errors.append(f"{recipe['file']}: {error}")
    return {
        "complete": not errors, "error_count": len(errors), "errors": errors,
        "verdict": "No source-level overlap detected" if not errors else "CV dataset audit failed",
        "total_examples": len(rows), "split_counts": dict(Counter(row["split"] for row in rows)),
        "class_occurrences": {split: dict(class_counts[split]) for split in ("train", "val")},
        "unique_sources": {split: len(refs_by_split[split]) for split in ("train", "val")},
        "unique_groups": {split: len(groups_by_split[split]) for split in ("train", "val")},
        "overlapping_source_count": len(overlap_sources), "overlapping_group_count": len(overlap_groups),
        "known_official_test_group_overlap": len(test_overlap),
        "metadata_sha256": sha256(metadata_path), "audio_files_checked": audio_checked,
        "source_pools_checked": True,
        "limitations": ["No audio fingerprint or cross-upload deduplication is performed."],
    }


def require_cv_disjoint(report: dict) -> None:
    if (not report["complete"] or report["overlapping_source_count"]
            or report["overlapping_group_count"] or report["known_official_test_group_overlap"]):
        raise ValueError("Nested CV dataset failed source-disjoint audit; inspect audit.json")
