"""MAD v1 dataset with optional deterministic audibility supervision targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from ..progress import tqdm

from .audibility import source_time_frequency_masks, source_to_rest_ratio_db
from .contributions import reconstruct_contributions
from .labels import construct_labels
from .preprocessing import PREPROCESSING, LogMel
from .provenance import CLASS_NAMES, METADATA_FIELDS, read_csv


class MADMixtureDataset(Dataset):
    """Load only one approved split of generated mixtures.

    ``return_aux_targets`` reconstructs source contributions from the signed
    recipe and verifies both source and mixture checksums. It never reads the
    official-test split. A cache, when requested, stores analysis arrays only
    and never alters the versioned dataset.
    """

    def __init__(self, metadata_path: Path, audio_dir: Path, *, split: str,
                 preprocessing: dict | None = None, label_policy: str = "union",
                 audibility_threshold_db: float = -15.0,
                 soft_audibility_center_db: float = -15.0,
                 soft_audibility_temperature_db: float = 6.0,
                 soft_audibility_floor: float = 0.1,
                 return_aux_targets: bool = False, source_metadata_path: Path | None = None,
                 mixture_manifest_path: Path | None = None, source_manifest_path: Path | None = None,
                 source_audio_dir: Path | None = None, cache_dir: Path | None = None) -> None:
        if split not in {"train", "val"}:
            raise ValueError("MADMixtureDataset permits only train or val")
        if label_policy not in {"union", "audibility_threshold", "soft_audibility"}:
            raise ValueError("unknown label policy")
        self.metadata_path, self.audio_dir, self.split = Path(metadata_path), Path(audio_dir), split
        self.config = dict(PREPROCESSING if preprocessing is None else preprocessing)
        self.config.setdefault("num_samples", int(self.config["sample_rate"] * self.config["duration_seconds"]))
        self.label_policy = label_policy
        self.threshold_db = audibility_threshold_db
        self.soft_center_db = soft_audibility_center_db
        self.soft_temperature_db = soft_audibility_temperature_db
        self.soft_floor = soft_audibility_floor
        self.return_aux_targets = return_aux_targets
        self.transform = LogMel(self.config)
        rows = read_csv(self.metadata_path, METADATA_FIELDS)
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split}")
        if any(r["split"] not in {"train", "val"} for r in rows):
            raise ValueError("Metadata contains a forbidden split")
        self._source_by_file: dict[str, dict] = {}
        self._recipes: dict[str, dict] = {}
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        if return_aux_targets:
            required = (source_metadata_path, mixture_manifest_path, source_manifest_path, source_audio_dir)
            if any(path is None for path in required):
                raise ValueError("source and recipe paths are required for auxiliary targets")
            source_metadata = read_csv(source_metadata_path)
            source_manifest = read_csv(source_manifest_path)
            by_file = {r["file"]: dict(r) for r in source_metadata}
            for record in source_manifest:
                if record["file"] not in by_file:
                    raise ValueError(f"Source manifest references unknown file: {record['file']}")
                merged = dict(by_file[record["file"]])
                merged.update(record)
                merged["label"] = int(merged["label"])
                self._source_by_file[record["file"]] = merged
            self._recipes = {r["file"]: r for r in read_csv(mixture_manifest_path)}
            if set(self._recipes) != {r["file"] for r in rows}:
                raise ValueError("Mixture manifest does not cover metadata")
            self.source_audio_dir = Path(source_audio_dir)
            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def precompute_auxiliary(self, *, description: str | None = None) -> None:
        """Materialise audibility/mask caches before a training DataLoader starts.

        Lazy mask reconstruction can make the first training progress bar look
        frozen because a batch performs source replay, several STFTs, and mel
        projection on the CPU. Explicit precomputation gives that work its own
        progress bar and leaves the training bars focused on model updates.
        """
        if not self.return_aux_targets or self._cache_dir is None:
            return
        with tqdm(total=len(self.rows), desc=description or f"Preparing {self.split} targets",
                  leave=False, dynamic_ncols=True, mininterval=0.5, miniters=1, unit="mixture") as bar:
            for row in self.rows:
                self._analysis(row)
                bar.update(1)

    def _analysis(self, row: dict) -> dict[str, np.ndarray]:
        cache_path = self._cache_dir / f"{row['id']}.npz" if self._cache_dir else None
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                result = {key: cached[key] for key in cached.files}
            expected = {"mask_target", "audibility_db", "class_valid", "valid_mask"}
            if (set(result) != expected or result["mask_target"].shape != (7, 128, 313)
                    or result["audibility_db"].shape != (7,)
                    or result["class_valid"].shape != (7,)
                    or result["valid_mask"].shape != (128, 313)
                    or not all(np.isfinite(result[key]).all() for key in ("mask_target", "class_valid", "valid_mask"))):
                raise ValueError(f"Invalid audibility cache: {cache_path}")
            if np.isnan(result["audibility_db"]).any():
                raise ValueError(f"Invalid audibility cache: {cache_path}")
            return result
        recipe = self._recipes[row["file"]]
        mixture, stems = reconstruct_contributions(recipe, self.source_audio_dir,
                                                   self.audio_dir, self._source_by_file,
                                                   {"sample_rate": self.config["sample_rate"],
                                                    "duration_seconds": self.config["num_samples"] // self.config["sample_rate"]})
        files = row["source_files"].split("|")
        classes = [int(self._source_by_file[file]["label"]) for file in files]
        masks = source_time_frequency_masks(
            stems, sample_rate=self.config["sample_rate"], n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"], n_mels=self.config["n_mels"],
            f_min=self.config["f_min"], f_max=self.config["f_max"],
        )
        class_mask = np.zeros((len(CLASS_NAMES), *masks.shape[1:]), dtype=np.float32)
        audibility = np.full(len(CLASS_NAMES), -np.inf, dtype=np.float32)
        class_valid = np.zeros(len(CLASS_NAMES), dtype=bool)
        for index, (file, stem, class_id) in enumerate(zip(files, stems, classes)):
            duration = float(self._source_by_file[file]["duration_original"])
            active = np.arange(stem.size) < duration * self.config["sample_rate"]
            audibility[class_id] = max(audibility[class_id], source_to_rest_ratio_db(
                stem, mixture, active_samples=active))
            class_mask[class_id] += masks[index]
            class_valid[class_id] = class_valid[class_id] or bool(np.sum(stem[active] ** 2) > 1e-12)
        result = {"mask_target": class_mask, "audibility_db": audibility,
                  "class_valid": class_valid, "valid_mask": masks.sum(0) > 1e-12}
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **result)
            temporary.replace(cache_path)
        return result

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = self.audio_dir / row["file"]
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != self.config["sample_rate"] or waveform.ndim != 1 or waveform.size != self.config["num_samples"]:
            raise ValueError(f"Invalid audio contract for {path}: sr={sample_rate}, shape={waveform.shape}")
        mel = self.transform(torch.from_numpy(waveform)).squeeze(0)
        classes = [int(value) for value in row["labels"].split(",")]
        if self.return_aux_targets:
            aux = self._analysis(row)
            # ``aux["audibility_db"]`` is indexed by all seven classes,
            # while construct_labels expects one audibility value per source
            # class listed in this particular mixture.
            source_audibility_db = [
                float(aux["audibility_db"][class_id]) for class_id in classes
            ]
            target = construct_labels(
                classes, policy=self.label_policy,
                audibility_db=source_audibility_db,
                threshold_db=self.threshold_db, center_db=self.soft_center_db,
                temperature_db=self.soft_temperature_db, soft_floor=self.soft_floor,
            )
            item = {"mel": mel, "target": torch.from_numpy(target),
                    "union_target": torch.tensor([int(c in classes) for c in range(7)], dtype=torch.float32),
                    "audibility_db": torch.from_numpy(aux["audibility_db"]),
                    "mask_target": torch.from_numpy(aux["mask_target"]),
                    "class_valid": torch.from_numpy(aux["class_valid"]),
                    "valid_mask": torch.from_numpy(aux["valid_mask"])}
        else:
            target = np.asarray([int(value) for value in row["label_vector"].split(",")], dtype=np.float32)
            item = {"mel": mel, "target": torch.from_numpy(target)}
        item.update({"id": row["id"], "file": row["file"], "mix_type": row["mix_type"]})
        return item
