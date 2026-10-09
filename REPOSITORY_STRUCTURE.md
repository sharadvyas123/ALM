# Proposed MAD Research Repository Structure

```text
ALM/
├── README.md                           # MAD project overview, setup, and reproduction commands
├── REPOSITORY_STRUCTURE.md              # Proposed target layout
├── pyproject.toml                      # Installable alm package and tooling configuration
├── requirements.txt                    # Reproducible environment dependencies
├── .gitignore
├── .gitattributes
│
├── src/
│   └── alm/                            # Reusable research implementation
│       ├── __init__.py
│       ├── config.py                   # Configuration loading and explicit path resolution
│       ├── inference.py                # Multi-label prediction using saved thresholds
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py             # MAD PyTorch datasets and multi-hot targets
│       │   ├── preprocessing.py        # 16 kHz, 10-second mono audio and shared log-Mel transforms
│       │   ├── splitting.py            # Source-disjoint splits before mixture generation
│       │   └── mixing.py               # Synthetic mixtures and source provenance
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mad_singlelabel.py      # Earlier MAD single-label reference model
│       │   └── enhanced_mad.py         # CNN–Transformer with optional MTFF and class queries
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py              # Shared training loop and checkpoint selection
│       │   └── losses.py               # BCE, positive weighting, and selected loss experiments
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py             # mAP, macro/micro F1, per-class metrics, Hamming loss
│           └── thresholds.py          # Validation-only class-specific threshold calibration
│
├── scripts/                            # Thin command-line entry points importing alm
│   ├── prepare_data.py
│   ├── generate_mixtures.py
│   ├── train.py
│   ├── calibrate.py
│   └── evaluate.py
│
├── configs/                            # Versioned recipes used by scripts and notebooks
│   ├── data/
│   │   └── mad.yaml                    # Preprocessing, source splits, and mixture settings
│   └── experiments/
│       ├── a0_baseline.yaml            # Multi-label baseline: no MTFF or class queries
│       ├── a1_mtff.yaml                # A0 + MTFF
│       ├── a2_class_query.yaml         # A0 + class-query attention
│       ├── a3_mtff_class_query.yaml    # A0 + both components
│       └── clean_only.yaml            # Clean-only training control
│
├── notebooks/                          # Interactive research importing shared alm code
│   ├── 01_mad_exploration.ipynb
│   ├── 02_preprocessing_inspection.ipynb
│   ├── 03_singlelabel_baseline.ipynb
│   ├── 04_multilabel_training.ipynb
│   └── 05_error_analysis.ipynb
│
├── data/
│   ├── README.md                       # Dataset acquisition, schemas, and version conventions
│   ├── raw/                            # Original MAD recordings and official split metadata
│   ├── processed/                      # Standardized clean audio, organized by dataset version
│   ├── synthetic/                      # Generated mixtures, organized by generation version
│   └── manifests/
│       └── <dataset-version>/
│           ├── labels.json             # Canonical ordered seven-class vocabulary
│           ├── source_splits.csv       # Source IDs and train/validation/official-test assignment
│           ├── mixture_manifest.csv    # Source IDs, labels, gains, offsets, and output paths
│           └── generation_config.yaml # Exact configuration used to produce this dataset
│
├── runs/                               # Generated artifacts grouped by unique experiment run
│   └── <run-id>/
│       ├── config.yaml                 # Fully resolved experiment settings and random seeds
│       ├── provenance.json             # Code state, manifest versions, environment, hardware
│       ├── checkpoints/
│       │   ├── best.pth
│       │   └── last.pth
│       ├── thresholds.json             # Class order, checkpoint ID, validation calibration details
│       ├── predictions/
│       │   ├── validation.npz          # Sample IDs, targets, and class probabilities
│       │   └── final_test.npz          # Created only for final test evaluation
│       ├── metrics/
│       │   ├── validation.json
│       │   ├── validation_per_class.csv
│       │   ├── final_test.json         # Created only for final test evaluation
│       │   └── final_test_per_class.csv
│       ├── figures/                    # Run-specific curves and diagnostic plots
│       └── train.log
│
├── research/
│   ├── context/
│   │   ├── MAD_ALM_MASTER_CONTEXT.md
│   │   └── military_audio_research_plan.md
│   ├── literature/
│   │   ├── MAD_RESEARCH.md
│   │   ├── RESEARCH_PAPERS.md
│   │   ├── paper_matrix.csv
│   │   └── gap_analysis.md
│   ├── experiments/
│   │   ├── hypotheses.md               # Research questions and falsifiable hypotheses
│   │   ├── protocol.md                 # Shared split, training, calibration, evaluation policy
│   │   └── experiment_log.md           # Run IDs, observations, and research decisions
│   ├── reports/
│   │   ├── dataset_audit.md
│   │   ├── implementation_audit.md
│   │   ├── architecture.md             # Verified architecture and tensor-shape trace
│   │   └── experiment_summary.csv      # Selected cross-run results linked to run IDs
│   └── paper/
│       ├── manuscript/
│       ├── presentation/
│       └── figures/                    # Selected publication figures linked to source runs
│
└── tests/                              # Research correctness and reproducibility checks
    ├── test_splits.py                  # Source disjointness and official-test isolation
    ├── test_mixing.py                  # Label unions and source provenance
    ├── test_preprocessing.py           # Shared training/inference preprocessing contract
    ├── test_models.py                  # Tensor shapes, finite loss, and gradients
    ├── test_class_order.py             # Consistent targets, logits, and threshold ordering
    └── test_evaluation.py              # Known-answer metrics and threshold calibration checks
```
