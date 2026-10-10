# Dataset audit artifacts

`dataset_audit.md` records the legacy leakage audit and the source-disjoint rebuild.
`dataset_audit_legacy.json` is the machine-readable legacy audit, including affected
example IDs and overlapping sources. The new dataset's full audit is stored with
its versioned manifests under `data/manifests/mad_source_disjoint_v1/audit.json`.

## Commands (from the repository root)

```bash
venv/bin/python -B -m unittest discover -s tests -v
venv/bin/python -B scripts/model_shape_trace.py --compare 320 512
venv/bin/python -B scripts/preflight_v1.py
venv/bin/python -B scripts/audibility_analysis.py --output-dir research/audibility/mad_source_disjoint_v1
venv/bin/python -B scripts/audit_source_overlap.py --metadata data/multilabel_mad/metadata.csv --output research/reports/dataset_audit_legacy.json
venv/bin/python -B scripts/generate_mixtures.py --config configs/data/mad_source_disjoint_v1.json --dry-run
venv/bin/python -B scripts/generate_mixtures.py --config configs/data/mad_source_disjoint_v1.json
venv/bin/python -B scripts/generate_mixtures.py --config configs/data/mad_source_disjoint_v1.json --verify-existing
venv/bin/python -B scripts/audit_source_overlap.py --metadata data/synthetic/mad_source_disjoint_v1/metadata.csv --source-manifest data/manifests/mad_source_disjoint_v1/source_splits.csv --mixture-manifest data/manifests/mad_source_disjoint_v1/mixture_manifest.csv --audio-dir data/synthetic/mad_source_disjoint_v1/audio --verify-waveforms --require-disjoint
```

After the shape trace and preflight pass, obtain explicit approval before starting
B0. The B0 runner uses a new run directory and does not read the official test set:

```bash
venv/bin/python -B scripts/train_b0.py \
  --run-id B0_mad_source_disjoint_v1_seed42
```

Generation creates **new version directories only**. Reusing an existing version
fails, including after an interrupted run. `COMPLETE.json` is written only after
full provenance/audio checks and deterministic plan replay succeed. To repeat an
audit of the legacy dataset, omit `--output` or supply a new report filename;
reports are exclusively created rather than overwritten.

No installation is required with the current `venv`: the data tools use the
standard library, NumPy, and SoundFile; tests use `unittest`. Entry points resolve
the project from their file location, and configuration paths are project-relative.

The training notebook still selects `data/multilabel_mad/`. Training integration
must explicitly select the new version; running the old notebook unchanged would
continue to use the legacy dataset. Training and checkpoint configuration are a
separate follow-up.
