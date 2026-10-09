# Master Context Prompt — ALM Military Audio Classification Research Project

## 1. Project Overview

I am working on a research project titled:

**“AST-Based Multi-Label Military Audio Classification: Multi-Scale Time–Frequency Fusion with Class-Query Attention for Acoustic Situational Awareness.”**

The project is associated with the Department of Computer Science and Engineering at IIIT Nagpur.

The goal is to develop a reproducible multi-label military audio classification system using the Military Audio Dataset (MAD). Unlike conventional single-label classification, the system should detect multiple military sound events occurring in the same audio clip.

The project repository is:

https://github.com/sharadvyas123/ALM

## 2. Reference Materials

Use these materials as the primary project context:

1. **Original context paper:** Describes the initial methodology, dataset, preprocessing, synthetic audio mixing, CNN–Transformer model, class imbalance handling, and threshold optimization.
2. **Project presentation:** `AST-Based-Multi-Label-Military-Audio-Classification.pdf`. Describes the research gap, objectives, dataset split strategy, proposed architecture, training methodology, reported metrics, and future work.
3. **GitHub repository:** Contains the actual implementation, model encoder, dataset-generation script, research notes, notebooks, and evaluation utilities.
4. **Updated context paper:** Incorporates the proposed research direction, implementation discrepancies, reported presentation results, and experimental recommendations.

When these sources disagree, distinguish between:
- What the paper proposes.
- What the presentation claims.
- What the repository currently implements.
- What has actually been experimentally verified.

Do not assume that a proposed feature is implemented correctly or that a reported result is reproducible without checking the relevant source code, logs, or evaluation artifacts.

## 3. Dataset and Classification Task

The project uses seven military audio classes:

| Index | Class |
|---|---|
| 0 | Communication |
| 1 | Gunshot |
| 2 | Footsteps |
| 3 | Shelling |
| 4 | Vehicle |
| 5 | Helicopter |
| 6 | Fighter |

The initial context paper reports approximately 6,429 training samples and 1,037 test samples before preprocessing. Verify the precise counts against the dataset metadata when needed.

Each model input represents a standardized 10-second mono audio clip at 16 kHz, corresponding to 160,000 waveform samples.

The task is multi-label classification. The target is a seven-dimensional multi-hot vector, and each class has an independent sigmoid probability. A class-specific threshold converts its probability into a final binary prediction.

## 4. Synthetic Multi-Label Audio Generation

The original dataset contains single-label recordings. The project creates synthetic overlapping acoustic scenes by combining recordings from different classes.

The general mixture equation is:

x_mix = Σ αᵢ xᵢ

The target is the logical union of the source labels, rather than numerical addition.

The repository's dataset-generation script currently specifies:
- 50% clean examples.
- 35% two-class mixtures.
- 15% three-class mixtures.
- Sample rate of 16 kHz.
- Mixing coefficient range of 0.3–0.7.
- Random seed of 42.
- A configured validation ratio of 15%.

These are implementation settings, not proof that the data split is leakage-free. The presentation separately describes a 70% training, 10% validation, and 20% test strategy. Reconcile the actual split policy before describing either as the definitive experimental setup.

**Critical concern:** The existing generation workflow may split generated examples after mixtures are constructed. If the same source recordings contribute to both training and validation examples, validation leakage can occur. Prefer splitting original source recordings first, generating training mixtures only from training sources, and keeping validation sources disjoint. The official test set must remain untouched until final evaluation.

## 5. Model Architecture and Research Direction

The intended model is an AST-inspired audio architecture with these components:

1. Log-Mel spectrogram representation.
2. Patch or feature embedding.
3. Convolutional feature extraction.
4. Multi-Scale Time–Frequency Fusion (MTFF).
5. Transformer encoder.
6. Class-query cross-attention.
7. Seven independent output logits.

The research direction combines:
- Synthetic multi-label training data.
- CNN–Transformer audio representation learning.
- Multi-scale time–frequency feature fusion.
- Learnable class-specific query vectors for class-specific pooling.
- Imbalance-aware loss.
- Per-class threshold optimization.

The repository's `utils/enhanced_alm_mad_encoder.py` implements a custom CNN–Transformer encoder with optional time-frequency fusion and optional class-query attention. Do not call it an unchanged implementation of the original Audio Spectrogram Transformer (AST).

The repository implementation has used six Transformer layers and eight attention heads. Some configuration defaults differ from the original context paper, so inspect the actual active configuration before documenting architecture details.

**Potential code issue:** The time-frequency fusion block appears to have branch names or comments reversed relative to the convolution kernel dimensions. Check the tensor layout and intended time/frequency axes before changing the code.

**Potential preprocessing mismatch:** The encoder and original paper describe a hop length of 512, while the training notebook has used 320. This may alter the spectrogram frame count and cause a positional-embedding shape mismatch. Confirm the active configuration, tensor dimensions, and model forward pass.

## 6. Training and Calibration

The proposed training loss is `BCEWithLogitsLoss`, with per-class positive weights calculated as:

pos_weight[c] = min(N_negative[c] / N_positive[c], 8.0)

Compute these weights from the effective training targets, including the effects of synthetic mixing, and handle classes with zero positive examples safely.

The model should output raw logits during training. Apply sigmoid during inference to obtain independent class probabilities.

Select per-class decision thresholds using validation predictions, with candidate thresholds initially ranging from 0.05 to 0.95. Define the optimization metric clearly, such as per-class F1, and save the selected thresholds in a file such as `thresholds.json`.

Never optimize thresholds on the test set.

## 7. Reported Presentation Results

The presentation reports the following metrics:

- mAP: 87.99%.
- Macro F1: 78.92%.
- Micro F1: 81.14%.
- Hamming loss: 0.0591.

Treat these as **reported presentation results, not independently verified results**. Before making scientific claims, establish:
- Which model checkpoint produced them.
- Which dataset split was evaluated.
- Whether the evaluation was multi-label and used the correct thresholds.
- Whether the metric calculations are correct.
- Whether there was any train–validation or train–test leakage.
- Whether the numbers can be reproduced from saved predictions or evaluation logs.

The presentation identifies mAP as the primary metric and also mentions per-class precision, recall, F1, macro/micro F1, Hamming loss, and exact-match ratio.

## 8. Known Repository Issues to Investigate

Prioritize the following before adding more architectural complexity:

1. Align the hop length across the training configuration, preprocessing, and model input assumptions.
2. Check the time-frequency fusion branch dimensions and names.
3. Prevent source-recording overlap between training and validation mixtures.
4. Inspect the old `script/evaluate_test_set.py`, which appears to use a legacy single-label encoder, `argmax`, and a single-label confusion matrix. Develop a proper multi-label evaluator if needed.
5. Confirm that the active training notebook actually uses the enhanced encoder and intended loss.
6. Check whether the model's Transformer receives a meaningful sequence of temporal tokens rather than a sequence of length one.
7. Ensure that the seven output classes are consistently ordered throughout data generation, training, inference, threshold storage, and evaluation.
8. Record random seeds, split definitions, preprocessing parameters, mixture ratios, and checkpoint details.

Inspect current repository files before assuming any issue is still present; implementations may have changed.

## 9. Experimental Plan

Develop a defensible experimental comparison using controlled ablations:

- Baseline model without MTFF or class-query attention.
- Baseline plus MTFF.
- Baseline plus class-query attention.
- Baseline with both components.
- Clean-only training versus synthetic-mixture training.

Evaluate each relevant configuration using the same source-disjoint data split and consistent training conditions.

Additional analyses should include:
- Per-class performance.
- Clean versus synthetic-mixture performance.
- Robustness under varying overlap and noise conditions.
- Threshold sensitivity.
- Representative false positives and false negatives.
- Ablation results showing whether each proposed component provides a measurable improvement.

Do not claim architectural novelty merely because components are combined. Position the contribution in terms of the specific research question, controlled implementation, and experimentally demonstrated benefits.

## 10. Python Environment

I am considering Python 3.12 for the project environment.

The current pinned requirements include packages such as:
- `librosa==0.11.0`
- `numpy==2.2.6`
- `numba==0.64.0`
- `llvmlite==0.46.0`
- `scipy==1.15.3`
- `pandas==2.3.3`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.8`
- Jupyter and supporting packages.

The listed requirements do not include PyTorch, which is needed for the neural-network implementation. Python 3.12 appears to be a reasonable candidate, but verify installation compatibility for the exact operating system and package pins. Install PyTorch separately using its official installation instructions, then verify imports, GPU availability if applicable, and a complete model forward pass.

## 11. How I Want You to Help

Act as a careful research and implementation assistant for this project.

When I ask you to continue working:
1. Preserve the project context above.
2. Inspect the actual repository or provided files when necessary.
3. Prefer fixing reproducibility, correctness, and evaluation before proposing extra features.
4. Explain changes clearly and connect them to the research objectives.
5. Provide practical code, commands, and file-specific recommendations when useful.
6. Distinguish confirmed facts from assumptions and proposed experiments.
7. Do not invent results, citations, implementations, or successful tests.
8. Keep the context paper, presentation, repository, and actual experiments consistent.
9. When discussing scientific performance, prioritize fair evaluation and leakage prevention.
10. If essential information is missing, ask a concise clarification rather than silently inventing it.

**Immediate priority:** Verify the current code and configuration, fix preprocessing and data-split issues, validate the model's tensor shapes, and establish a reproducible multi-label evaluation pipeline before trusting the reported results or extending the architecture.