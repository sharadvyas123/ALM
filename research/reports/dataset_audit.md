# ALM MAD Source-Level Leakage Audit and Rebuild

**Audit date:** 2026-10-10  
**Project:** ALM military-audio multi-label classification  
**Dataset version:** `mad_source_disjoint_v1`

## 1. Executive verdict

The legacy generated dataset `data/multilabel_mad/` had source-level leakage:

- **1,060** processed source recordings appeared in both generated train and validation.
- **752 / 965 validation examples (77.927461%)** contained at least one source also used by training.
- **1,679 / 5,464 training examples (30.728404%)** contained at least one source also used by validation.

A replacement dataset was generated using source-disjoint pools. Its independent
post-generation audit reports:

- **0** overlapping processed source recordings;
- **0** overlapping connected source groups;
- **0** validation examples affected by train-source overlap;
- **0** known official-test connected groups used by generated train/validation;
- **6,429 / 6,429** generated waveforms reconstructed exactly from their recipes.

The audit establishes provenance overlap or disjointness. It does not establish
model memorization or quantify metric inflation.

## 2. Inputs and schemas

Files used:

| File | Role |
|---|---|
| `script/generate_multilable_dataset.py` | Legacy generation and post-generation split implementation |
| `data/multilabel_mad/metadata.csv` | Legacy generated examples and `source_files` provenance |
| `data/multilabel_mad/config.json` | Legacy seed, ratios, and counts |
| `data/processed_mad/metadata.csv` | Processed-file to original `source_file` mapping |
| `data/raw/MAD_dataset/training.csv` | Original training manifest and YouTube provenance |
| `data/raw/MAD_dataset/test.csv` | Original official-test manifest and YouTube provenance |
| `script/enhanced_alm_mad_training.ipynb` | Consumer of generated train/validation rows |

Legacy generated metadata fields:

```text
id,file,labels,label_names,label_vector,mix_type,source_files,mix_weights,split
```

The `source_files` field is pipe-delimited. Paths were normalized for separator
and redundant-component differences while preserving case and full relative path
identity. Generated references were resolved through `processed_mad/metadata.csv`
to the original `source_file` values.

## 3. Legacy audit results

| Measure | Result |
|---|---:|
| Generated examples | 6,429 |
| Training examples | 5,464 |
| Validation examples | 965 |
| Unique sources in generated training | 5,053 |
| Unique sources in generated validation | 1,437 |
| Sources in both splits | **1,060** |
| Affected validation examples | **752** |
| Affected validation percentage | **77.927461%** |
| Affected training examples | **1,679** |
| Affected training percentage | **30.728404%** |
| Malformed provenance rows | 0 |
| Unresolved generated source references | 0 |

The original processed metadata contains 6,429 official training clips and 1,037
official test clips. All unique source files referenced by the legacy generated
dataset resolve to the original training split. The legacy audit still has leakage
because its generated train/validation split is not source-disjoint.

## 4. Root cause

The legacy generator:

1. loads all original rows with `split == "training"`;
2. creates all clean, two-class, and three-class outputs;
3. records component provenance in `source_files`;
4. assigns generated rows to `train` and `val` at lines 253–254 of
   `script/generate_multilable_dataset.py`.

Mixture components are sampled with replacement across outputs from per-class
pools. The clean reservation is not a train/validation source split and does not
prevent reserved sources from being used by later mixtures. Consequently, the
same source can contribute to generated rows on both sides of the later split.

`script/enhanced_alm_mad_training.ipynb` reads the generated `split` column and
constructs `train_df` and `val_df`; it does not perform a second provenance-aware
split.

## 5. Approved source policy for the replacement dataset

The rebuild uses `verified_provenance_only`:

- Exclude the **110 training clips** in connected groups linked to official-test
  video IDs.
- Exclude the **294 training clips** in folders without a usable video ID.
- Retain **6,025 eligible training clips**.
- Connect source clips transitively when they share a full source folder or a
  validated YouTube video ID.
- Assign connected groups to train or validation before mixture generation.

The raw metadata has 446 missing or malformed video-URL rows, six folders with
multiple valid video IDs, and 16 video IDs appearing in multiple folders. The
grouping implementation handles these conservatively. URL timestamps are ignored;
malformed IDs are not guessed. There are 150 official-test clips without usable
video IDs, so complete parent-video independence from those particular test clips
cannot be certified from the available metadata. Their unknown training-side
counterparts are excluded under the selected policy.

## 6. Replacement dataset results

Output locations:

```text
data/synthetic/mad_source_disjoint_v1/
    audio/                 # 6,429 generated mono PCM16 WAV files
    metadata.csv
    config.json

data/manifests/mad_source_disjoint_v1/
    labels.json
    source_splits.csv
    mixture_manifest.csv
    generation_config.json
    audit.json
    audit_external.json
    COMPLETE.json
```

### Source pools

| Pool | Connected groups | Source clips |
|---|---:|---:|
| Training | 372 | 5,127 |
| Validation | 66 | 898 |

All seven classes occur in both source pools:

| Class | Training sources | Validation sources |
|---|---:|---:|
| Communication | 601 | 112 |
| Gunshot | 984 | 172 |
| Footsteps | 641 | 104 |
| Shelling | 722 | 120 |
| Vehicle | 758 | 124 |
| Helicopter | 696 | 140 |
| Fighter | 725 | 126 |

### Generated examples

| Mix type | Training | Validation | Total |
|---|---:|---:|---:|
| Clean | 2,732 | 482 | 3,214 |
| Two-class | 1,912 | 338 | 2,250 |
| Three-class | 820 | 145 | 965 |
| **Total** | **5,464** | **965** | **6,429** |

The generated-example mix proportions are the approved 50% / 35% / 15% targets,
subject only to integer counts. Mixture components are drawn only from their
assigned source pool. Clean sources are unique within each split; mixture sources
may be reused within the same split.

## 7. Verification

The new dataset passed:

- source manifest coverage and processed-source identity checks;
- zero train/validation source-file overlap;
- zero train/validation connected-group overlap;
- zero known official-test connected-group overlap;
- seven-class label and multi-hot union checks;
- source-count and weight checks for clean, two-class, and three-class rows;
- exact output count checks;
- 6,429 generated WAV inventory checks;
- 6,025 source-audio SHA-256 checks;
- 6,429 source-recipe waveform reconstructions;
- deterministic replay of source assignments and mixture recipes;
- 15 focused unit tests covering path normalization, conservative grouping,
  split determinism, mixture provenance, audio tamper detection, and malformed
  metadata handling.

The independently written external audit result is in
`data/manifests/mad_source_disjoint_v1/audit_external.json` and has
`complete: true`, `overlapping_source_count: 0`,
`overlapping_group_count: 0`, and `waveforms_reconstructed: 6429`.

## 8. Commands executed

All commands were run from the repository root with the existing environment:

```bash
venv/bin/python -B -m unittest discover -s tests -v
venv/bin/python -B scripts/audit_source_overlap.py \
  --metadata data/multilabel_mad/metadata.csv \
  --output research/reports/dataset_audit_legacy.json
venv/bin/python -B scripts/generate_mixtures.py \
  --config configs/data/mad_source_disjoint_v1.json --dry-run
venv/bin/python -B scripts/generate_mixtures.py \
  --config configs/data/mad_source_disjoint_v1.json
venv/bin/python -B scripts/generate_mixtures.py \
  --config configs/data/mad_source_disjoint_v1.json --verify-existing
venv/bin/python -B scripts/audit_source_overlap.py \
  --metadata data/synthetic/mad_source_disjoint_v1/metadata.csv \
  --source-manifest data/manifests/mad_source_disjoint_v1/source_splits.csv \
  --mixture-manifest data/manifests/mad_source_disjoint_v1/mixture_manifest.csv \
  --audio-dir data/synthetic/mad_source_disjoint_v1/audio \
  --verify-waveforms --require-disjoint \
  --output data/manifests/mad_source_disjoint_v1/audit_external.json
```

No model training was started. The existing `data/multilabel_mad/` dataset was
not deleted, overwritten, or renamed. The legacy generator and training notebook
were not modified.

## 9. Next action

Before training, update the experiment configuration or training entry point to
select:

```text
data/synthetic/mad_source_disjoint_v1/metadata.csv
data/synthetic/mad_source_disjoint_v1/audio/
```

Then perform a separate training run with a new run ID and record the manifest
version, generation configuration, and audit path in the experiment log. Do not
compare new source-disjoint validation metrics with legacy metrics without labeling
the split protocol change.
