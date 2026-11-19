"""Build nested, source-grouped five-fold datasets without reading test audio."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alm.data.cv_auditing import audit_cv_dataset, require_cv_disjoint
from alm.data.cv_splitting import assign_group_folds
from alm.data.generation import load_config
from alm.data.cv_mixing import plan_cv_mixtures
from alm.data.mixing import check_audio, metadata_rows, render_dataset
from alm.data.provenance import (CLASS_NAMES, METADATA_FIELDS, object_hash, read_csv,
                                sha256, write_csv, write_json)
from alm.data.verification import verify_dataset


def _mix_counts(total: int, ratios: tuple[float, float, float]) -> dict:
    clean = round(total * ratios[0])
    two = round(total * ratios[1])
    return {"clean": clean, "2-class": two, "3-class": total - clean - two}


def _build_dataset(*, destination: Path, manifest_dir: Path,
                   source_assignments: list[dict], original_sources: list[dict],
                   source_audio: Path, generation_cfg: dict, splits: tuple[str, ...],
                   mixture_sources: list[dict] | None = None) -> dict:
    destination.mkdir(parents=True)
    audio_dir = destination / "audio"
    audio_dir.mkdir()
    manifest_dir.mkdir(parents=True)
    recipes = plan_cv_mixtures(mixture_sources or source_assignments, generation_cfg, splits=splits)
    requested = sum(sum(generation_cfg["counts_by_split"][split].values()) for split in splits)
    if len(recipes) != requested:
        raise ValueError(f"Mixture planner produced {len(recipes)} examples; expected {requested}")
    rows = metadata_rows(recipes)
    fold_manifest = []
    for source in source_assignments:
        row = dict(source)
        if row["split"] in {"train", "val"} and not row.get("audio_sha256"):
            raise ValueError(f"Missing certified source audio checksum: {row['file']}")
        if row["split"] not in {"train", "val"}:
            row["audio_sha256"] = ""
        fold_manifest.append(row)
    write_csv(manifest_dir / "source_splits.csv", fold_manifest)
    write_json(manifest_dir / "labels.json", {str(i): n for i, n in enumerate(CLASS_NAMES)})
    rendered = render_dataset(recipes, source_audio, audio_dir, generation_cfg)
    write_csv(destination / "metadata.csv", rows, METADATA_FIELDS)
    write_csv(manifest_dir / "mixture_manifest.csv", rendered)
    write_json(destination / "config.json", {
        "dataset_version": generation_cfg["dataset_version"],
        "split_role": destination.name,
        "sample_rate": generation_cfg["sample_rate"],
        "duration_seconds": generation_cfg["duration_seconds"],
        "class_order": list(CLASS_NAMES),
        "counts_by_split": generation_cfg["counts_by_split"],
        "seed": generation_cfg["seed"],
    })
    audit = audit_cv_dataset(destination / "metadata.csv", original_sources, source_audio,
                             manifest_dir / "source_splits.csv", manifest_dir / "mixture_manifest.csv",
                             audio_dir, required_splits=splits)
    require_cv_disjoint(audit)
    write_json(manifest_dir / "audit.json", audit)
    complete = {
        "status": "complete", "metadata_sha256": sha256(destination / "metadata.csv"),
        "source_manifest_sha256": sha256(manifest_dir / "source_splits.csv"),
        "mixture_manifest_sha256": sha256(manifest_dir / "mixture_manifest.csv"),
        "config_sha256": sha256(destination / "config.json"),
        "audit_sha256": sha256(manifest_dir / "audit.json"),
        "recipe_sha256": object_hash(recipes), "audio_files": len(rendered),
    }
    write_json(manifest_dir / "COMPLETE.json", complete)
    return complete


def prepare(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("folds") != 5:
        raise ValueError("This protocol requires exactly five outer folds")
    source_cfg, source_paths = load_config(ROOT / config["source_config"], ROOT)
    verify_dataset(ROOT, config["source_config"], verify_waveforms=False)
    base_manifest = source_paths["manifest_dir"]
    source_complete = json.loads((base_manifest / "COMPLETE.json").read_text())
    if source_complete.get("status") != "complete":
        raise ValueError("Source-disjoint v1 dataset certificate is incomplete")
    for name, digest in source_complete["artifact_sha256"].items():
        if sha256(base_manifest / name) != digest:
            raise ValueError(f"Source manifest checksum mismatch: {name}")
    snapshot = json.loads((base_manifest / "generation_config.json").read_text())
    if sha256(source_paths["source_csv"]) != snapshot["input_metadata_sha256"]["source_csv"]:
        raise ValueError("Processed source metadata changed since certification")

    base_sources = read_csv(base_manifest / "source_splits.csv")
    # Candidate pools include all eligible official-training groups, including
    # those in the former 15% validation pool. Exclusions/test stay untouched.
    eligible_rows = [r for r in base_sources if r["split"] in {"train", "val"}]
    for source in eligible_rows:
        audio_path = source_paths["source_audio"] / source["file"]
        check_audio(audio_path, source_cfg["sample_rate"],
                    source_cfg["sample_rate"] * source_cfg["duration_seconds"])
        if sha256(audio_path) != source["audio_sha256"]:
            raise ValueError(f"Certified source audio checksum mismatch: {source['file']}")
    outer_fold_by_group, outer_summary = assign_group_folds(
        eligible_rows, n_splits=5, seed=int(config["seed"]),
    )
    processed_sources = read_csv(source_paths["source_csv"])
    base_by_file = {r["file"]: r for r in base_sources}
    original_sources = []
    for row in processed_sources:
        value = dict(row)
        # Keep source-manifest fields required by audit_rows.
        manifest_row = base_by_file[row["file"]]
        value["split"] = "eligible" if manifest_row["split"] in {"train", "val"} else manifest_row["split"]
        for key in ("source_id", "group_id", "video_id", "official_split", "provenance_status", "exclusion_reason"):
            value[key] = manifest_row[key]
        original_sources.append(value)

    data_root = ROOT / "data/synthetic" / config["dataset_version"]
    manifest_root = ROOT / "data/manifests" / config["dataset_version"]
    if data_root.exists() or manifest_root.exists():
        raise FileExistsError(f"Refusing to overwrite CV dataset version: {data_root} or {manifest_root}")
    data_root.mkdir(parents=True)
    manifest_root.mkdir(parents=True)
    generation = dict(source_cfg)
    generation.update(config["generation"])
    generation.update({"dataset_version": config["dataset_version"], "seed": int(config["seed"])})
    ratios = tuple(float(generation[key]) for key in ("clean_ratio", "two_class_ratio", "three_class_ratio"))
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Mixture proportions must sum to one")
    total_examples = int(config["examples_per_fold"])
    outer_eval_n = round(total_examples * 0.2)
    inner_n = total_examples - outer_eval_n
    inner_train_n = round(inner_n * 0.8)
    inner_val_n = inner_n - inner_train_n
    generation["counts_by_split"] = {
        "train": _mix_counts(inner_train_n, ratios),
        "val": _mix_counts(inner_val_n, ratios),
    }
    summary = {"status": "preparing", "config_sha256": sha256(config_path),
               "source_certificate_sha256": sha256(base_manifest / "COMPLETE.json"),
               "source_metadata_sha256": sha256(source_paths["source_csv"]),
               "protocol_code_sha256": {
                   relative: sha256(ROOT / relative) for relative in (
                       "scripts/prepare_cv5.py", "src/alm/data/cv_splitting.py",
                       "src/alm/data/cv_mixing.py", "src/alm/data/cv_auditing.py",
                       "src/alm/data/mixing.py", "src/alm/data/auditing.py",
                       "src/alm/data/provenance.py", "src/alm/data/generation.py",
                   )
               },
               "outer_assignment": outer_summary, "folds": {}}

    for outer in range(1, 6):
        fold_name = f"fold_{outer:02d}"
        fold_data_root, fold_manifest_root = data_root / fold_name, manifest_root / fold_name
        fold_data_root.mkdir()
        fold_manifest_root.mkdir()
        outer_group_ids = {g for g, f in outer_fold_by_group.items() if f == outer}
        development_rows = [r for r in eligible_rows if r["group_id"] not in outer_group_ids]
        inner_assignment, inner_summary = assign_group_folds(
            development_rows, n_splits=4, seed=int(config["seed"]) + outer,
        )
        inner_holdout = (outer - 1) % 4 + 1
        inner_val_groups = {g for g, f in inner_assignment.items() if f == inner_holdout}

        inner_generation = dict(generation, seed=int(config["seed"]) + outer)
        inner_sources = []
        outer_sources = []
        for source in base_sources:
            row = dict(source)
            if source["group_id"] in outer_group_ids:
                row["fold"] = str(outer)
                row["split"] = "train"
            elif source["group_id"] in inner_val_groups:
                row["fold"] = str(inner_assignment[source["group_id"]])
                row["split"] = "val"
            elif source["group_id"] in inner_assignment:
                row["fold"] = str(inner_assignment[source["group_id"]])
                row["split"] = "train"
            else:
                row["fold"] = "excluded"
            inner_sources.append(row)
            out = dict(source)
            if source["group_id"] in outer_group_ids:
                out["fold"] = str(outer)
                out["split"] = "val"
            elif source["group_id"] in outer_fold_by_group:
                out["fold"] = str(outer_fold_by_group[source["group_id"]])
                out["split"] = "train"
            outer_sources.append(out)

        # Inner training/early-stop source groups partition the 80% development
        # pool; outer groups remain exclusive to final fold scoring.
        inner_dir, inner_manifest_dir = fold_data_root / "inner", fold_manifest_root / "inner"
        inner_mixture_sources = [row for row in inner_sources
                                 if row["group_id"] not in outer_group_ids
                                 and row["split"] in {"train", "val"}]
        inner_complete = _build_dataset(
            destination=inner_dir, manifest_dir=inner_manifest_dir,
            source_assignments=inner_sources, original_sources=original_sources,
            source_audio=source_paths["source_audio"], generation_cfg=inner_generation,
            splits=("train", "val"), mixture_sources=inner_mixture_sources,
        )

        outer_dir, outer_manifest_dir = fold_data_root / "outer_eval", fold_manifest_root / "outer_eval"
        outer_total_cfg = dict(generation)
        outer_total_cfg["counts_by_split"] = {"train": {"clean": 1, "2-class": 1, "3-class": 1},
                                               "val": _mix_counts(outer_eval_n, ratios)}
        outer_cfg = dict(outer_total_cfg, seed=int(config["seed"]) + 1000 + outer)
        outer_complete = _build_dataset(
            destination=outer_dir, manifest_dir=outer_manifest_dir,
            source_assignments=outer_sources, original_sources=original_sources,
            source_audio=source_paths["source_audio"], generation_cfg=outer_cfg,
            splits=("val",),
        )
        summary["folds"][fold_name] = {
            "inner_group_assignment": inner_summary,
            "inner_validation_fold": inner_holdout,
            "inner_dataset": inner_complete,
            "outer_eval_dataset": outer_complete,
        }
        print(f"{fold_name}: inner and outer datasets prepared and audited", flush=True)

    summary["status"] = "complete"
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary["python"] = platform.python_version()
    write_csv(manifest_root / "folds.csv", [
        {"group_id": group, "outer_fold": fold} for group, fold in sorted(outer_fold_by_group.items())
    ])
    write_json(manifest_root / "fold_summary.json", summary)
    write_json(manifest_root / "COMPLETE.json", {
        "status": "complete", "folds_sha256": sha256(manifest_root / "folds.csv"),
        "summary_sha256": sha256(manifest_root / "fold_summary.json"),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/CV5_proposed_alm_source_grouped_v1.json")
    args = parser.parse_args()
    result = prepare(args.config.resolve())
    print(json.dumps({"status": result["status"], "folds": len(result["folds"])}, indent=2))


if __name__ == "__main__":
    main()
