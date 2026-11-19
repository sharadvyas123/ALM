# Nested source-grouped five-fold ALM protocol

This protocol estimates generalisation to unseen source groups. It does not read
official-test audio or labels. The five outer folds are the evaluation units;
each outer training pool is divided again into inner training and validation
groups. Checkpoint selection, early stopping, and threshold calibration use only
the inner validation subset. The outer fold is scored after the best checkpoint
has been selected.

## Why the nested split is used

If the same fold both selects the checkpoint epoch and reports its score, the
reported score can benefit from that repeated validation feedback. Here, the
outer fold is isolated from all training choices. Model settings are fixed in
the experiment config. Do not compare many configurations on these same outer
folds and present the winning configuration's score as an unbiased estimate;
that requires another outer evaluation layer or a separate final test.

## Files and outputs

- `configs/experiments/CV5_proposed_alm_source_grouped_v1.json` fixes model,
  optimizer, seed, fold count, early-stopping patience, and minimum improvement.
- `scripts/prepare_cv5.py` creates per-fold inner and outer datasets, with
  mixtures generated only from sources assigned to the corresponding pool.
- `data/manifests/mad_source_grouped_cv5_v1/` stores group assignments,
  per-split provenance manifests, checksums, and audit certificates.
- `scripts/train_cv5.py` trains folds sequentially and supports epoch-boundary
  resume with `--resume`.
- `scripts/evaluate_cv5.py` rebuilds aggregate tables and figures from completed
  fold outputs without retraining.
- `runs/<run-id>/fold_XX/` stores append-only training logs, histories,
  checkpoints, predictions, thresholds, metrics, and diagnostic figures.
- `runs/<run-id>/aggregate/` stores fold tables, pooled out-of-fold metrics,
  precision-recall curves, fold-comparison figures, and an aggregate artifact
  checksum certificate.

Each fold checkpoint records model, optimizer, scheduler, AMP scaler, early
stopping state, epoch history, and Python/NumPy/PyTorch/CUDA/DataLoader RNG
states. Resume verifies config, source assignment, fold dataset, generated-audio
checksums, and code hashes.
The runner requires `num_workers=0` for reproducible epoch-boundary resume.
Artifact SHA-256 checksums are recorded when a fold completes.

## Metric protocol

The primary score is outer-fold macro mAP, summarized as mean and sample
standard deviation over five folds. Report every fold as well as the aggregate.
Thresholds are optimized per class on inner validation only, then frozen for
outer evaluation. Each fold reports mAP, ROC-AUC, label-ranking average
precision, per-class average precision, precision, recall, F1, accuracy,
specificity, balanced accuracy and support; macro, micro, and sample-averaged
precision/recall/F1; Hamming loss; label accuracy; exact-match accuracy; and
per-class confusion matrices. Pooled out-of-fold ranking metrics use all outer
predictions; pooled threshold metrics use each example's fold-specific
inner-validation thresholds.

Figures include per-fold loss/mAP history, outer-fold log-mel and source-mask
examples, fold metric comparisons, per-class F1 heatmaps, and pooled outer-fold
precision-recall curves. Training and prediction collection display TQDM
progress, while fold logs persist to disk.

## Commands

From the repository root:

```bash
python scripts/prepare_cv5.py
python scripts/train_cv5.py --device auto
python scripts/train_cv5.py --resume --device auto
python scripts/evaluate_cv5.py --run-id CV5_proposed_alm_source_grouped_v1_seed42
```

Preparation creates approximately 32,145 ten-second mixture waveforms, plus
manifests and training caches. It is intentionally exclusive: if interrupted,
the partial version is retained for inspection, and the same version is not
silently overwritten. Move the partial version aside and choose a new dataset
version before regenerating it.

The final official test evaluation is a separate step. It should happen once,
after the model and reporting protocol are fixed, and only if official test
audio and labels are available in a verified compatible format.
