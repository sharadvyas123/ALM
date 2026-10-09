# Research Roadmap: Novel Multi-Label Military Audio Classification

## Project context

**Working title:** *AST-Based Multi-Label Military Audio Classification:
Multi-Scale Time--Frequency Fusion with Class-Query Attention for
Acoustic Situational Awareness*

**Dataset:** Military Audio Dataset (MAD)\
**Task:** Multi-label classification of seven sound classes:
Communication, Gunshot, Footsteps, Shelling, Vehicle, Helicopter, and
Fighter.\
**Input setup:** Standardized 10-second, mono audio at 16 kHz (160,000
waveform samples), represented using log-Mel spectrograms.\
**Current direction:** Synthetic multi-label mixture generation, a
CNN--Transformer audio encoder, multi-scale time--frequency fusion,
class-query attention, imbalance-aware training, and class-specific
threshold calibration.

> This document is a research and implementation plan, not a claim that
> the proposed ideas are novel or that any expected improvement has
> already been achieved. Novelty must be established by reviewing prior
> work and validating the implementation experimentally.

------------------------------------------------------------------------

## 1. Overall objective

Develop a reproducible and scientifically defensible multi-label
military audio classification system by investigating three possible
contribution areas:

1.  **Data-generation contribution:** a more meaningful method for
    synthesizing overlapping multi-label acoustic scenes.
2.  **Architecture contribution:** a classification architecture that
    better represents overlapping events and extracts class-specific
    evidence.
3.  **Objective/loss contribution:** a training objective that addresses
    the project's actual error patterns, class imbalance, and/or
    confusing co-occurring classes.

The final paper should only claim contributions that are supported by
literature review, correct implementation, controlled experiments, and
statistically credible evaluation. It is not necessary to force a new
contribution into every stage if experiments show that a simpler method
is stronger.

## 2. Research principles and non-negotiable safeguards

-   **Research the gap before naming a method novel.** Combining known
    blocks is not automatically a novel contribution.
-   **Reproduce a reliable baseline first.** Preserve a baseline
    checkpoint, configuration, split definitions, and evaluation
    predictions.
-   **Prevent data leakage.** Split original source recordings before
    constructing synthetic mixtures. Training mixtures must use training
    sources only; validation mixtures must use disjoint validation
    sources. Keep the official test set untouched until final
    evaluation.
-   **Use consistent evaluation.** Every experiment must use the same
    split policy, preprocessing, class order, and metric implementation.
-   **Separate validation from test.** Use validation data for
    architecture selection, loss tuning, and threshold calibration. Use
    the test set only for final reporting.
-   **Record all configurations.** Save random seeds, source IDs,
    mixture parameters, model configuration, thresholds, optimizer
    settings, checkpoint identity, and software versions.
-   **Distinguish verified facts from hypotheses.** A proposed block is
    not an implemented feature, and a reported metric is not
    reproducible until its evaluation path is verified.

------------------------------------------------------------------------

# Phase 0 --- Audit and establish a trustworthy baseline

Do this before introducing new methods. Otherwise, improvements may be
caused by bug fixes or leakage rather than the proposed contribution.

## 0.1 Audit the dataset-generation pipeline

Inspect the data-generation scripts and answer:

-   Are source recordings split before mixture generation?
-   Can a source recording, or a near-duplicate segment of it, appear in
    both training and validation mixtures?
-   Are labels created as the logical union of source labels?
-   Are mixture coefficients, clipping, normalization, silence, and
    source duration handled consistently?
-   Are the clean/two-source/three-source mixture proportions actually
    implemented as configured?
-   Are class counts calculated from the final generated targets?
-   Is the random seed effective and are generated manifests saved?
-   Are the validation and official test sets truly independent?

**Deliverable:** a dataset audit note, source-disjoint split manifest,
and reproducible mixture-generation configuration.

## 0.2 Audit preprocessing and architecture

Verify:

-   Sample rate, waveform duration, channel conversion, and log-Mel
    parameters.
-   The active hop length in preprocessing and training. Resolve any
    mismatch (the project context notes that 512 and 320 have both
    appeared).
-   Input and tensor shapes at every major model stage.
-   Whether the Transformer receives a meaningful sequence of tokens
    rather than a sequence of length one.
-   Whether time and frequency axes are correctly identified in the
    time-frequency fusion branches.
-   Whether the training notebook actually instantiates the intended
    enhanced encoder and loss.
-   Whether the seven classes retain exactly the same order in targets,
    logits, metrics, and threshold files.

**Deliverable:** architecture diagram based on the actual code,
tensor-shape trace, and a verified forward/backward pass.

## 0.3 Build a correct multi-label evaluation pipeline

Use raw model logits during training and apply sigmoid for inference.
The evaluation code should support:

-   Mean average precision (mAP), as the proposed primary metric.
-   Macro F1 and micro F1.
-   Per-class precision, recall, F1, and average precision.
-   Hamming loss and exact-match/subset accuracy.
-   Per-class precision--recall curves and threshold sensitivity.
-   Results on clean clips and clips with overlapping events, when these
    subsets are defined without test leakage.

Confirm that evaluation does not use `argmax` as a substitute for
multi-label prediction. Each class is predicted independently using its
probability and a threshold.

**Deliverable:** one evaluator used by every experiment, plus saved
validation/test predictions and a machine-readable metrics file.

## 0.4 Reproduce the baseline

Train and evaluate the simplest credible version of the existing model
under the corrected data and evaluation protocol. Freeze its
configuration before experimentation.

**Baseline record should include:** data split, source counts, mixture
proportions, preprocessing, model blocks, loss, optimizer, training
duration, best-checkpoint rule, thresholds, per-class metrics, and
random seeds.

Do not use previously reported presentation metrics as a verified
baseline until the exact checkpoint, split, thresholds, and evaluator
have been confirmed.

------------------------------------------------------------------------

# Phase 1 --- Literature review and novelty mapping

The first research task is to discover what has already been done, not
to immediately invent a replacement.

## 1.1 Review three related-work groups

### A. Multi-label audio classification and sound-event detection

Search for work involving:

-   Multi-label sound event classification on general audio datasets.
-   Polyphonic/overlapping sound-event detection.
-   CNN--Transformer and AST-inspired architectures.
-   Class-wise pooling, class-query attention, and query-based
    prediction.
-   Multi-scale temporal, frequency, or time--frequency feature fusion.
-   Military audio classification, acoustic scene understanding, and
    overlapping tactical sounds.

For each paper, record the dataset, task, labels, architecture, training
objective, mixing/data augmentation method, metrics, and limitations.

### B. Synthetic mixtures and audio augmentation

Search for methods involving:

-   Linear waveform mixing and random gain mixing.
-   Signal-to-noise-ratio (SNR)-controlled mixing.
-   Loudness or energy normalization.
-   Event-aware mixing and overlap-duration control.
-   Foreground/background or source-to-scene mixing.
-   Mixup-style methods and their differences from physical waveform
    mixtures.
-   Noise/reverberation augmentation and realistic acoustic scene
    simulation.
-   Synthetic data generation for imbalanced or rare sound classes.

Do not treat all methods as equivalent: waveform superposition,
feature-space Mixup, and physically motivated acoustic simulation have
different assumptions and label semantics.

### C. Multi-label losses and class imbalance

Review:

-   Binary cross-entropy with logits and class-wise positive weighting.
-   Focal loss and asymmetric loss.
-   Class-balanced or effective-number weighting.
-   Ranking-oriented objectives for multi-label prediction.
-   Methods for label co-occurrence and hard negative examples.
-   Calibration and class-specific threshold selection.

## 1.2 Build a literature matrix

Create a spreadsheet or Markdown table with one row per paper:

  -----------------------------------------------------------------------
  Field                               What to capture
  ----------------------------------- -----------------------------------
  Citation                            Full reference, year, venue,
                                      DOI/link

  Task/dataset                        Multi-label classification,
                                      sound-event detection, domain

  Data generation                     Exact mixing/augmentation method

  Architecture                        Encoder, fusion, attention,
                                      pooling, output head

  Loss                                Formula and weighting strategy

  Evaluation                          Split policy, metrics, baselines

  Main result                         Reported result and comparison
                                      conditions

  Limitation                          Explicit limitation or carefully
                                      justified gap

  Relevance                           Directly relevant to this project,
                                      adjacent, or background

  Candidate experiment                What can be tested in our setup
  -----------------------------------------------------------------------

## 1.3 Turn the literature into testable hypotheses

For each candidate contribution, write:

1.  What existing methods do.
2.  What limitation remains for this task.
3.  Why the limitation matters for military audio mixtures.
4.  What mechanism is proposed to address it.
5.  What experiment would falsify the hypothesis.
6.  What baseline and ablation would establish whether it helps.

**Exit criterion:** do not freeze a novelty claim until the literature
matrix and baseline comparisons identify a specific, defensible gap.

------------------------------------------------------------------------

# Phase 2 --- Contribution candidate 1: Better synthetic mixture generation

## 2.1 Current method to document

The project context describes a mixture of source waveforms:

`x_mix = sum(alpha_i * x_i)`

The target is the logical union of source labels, not numerical addition
of label vectors. The current generation configuration has been
described as 50% clean examples, 35% two-class mixtures, 15% three-class
mixtures, and mixing coefficients in the range 0.3--0.7. Verify these
values in the current code before treating them as definitive.

## 2.2 Questions to investigate

-   Do uniform random mixing coefficients make one source inaudible or
    excessively dominant?
-   Does the current mixing process preserve a realistic relative
    loudness between sources?
-   Does it control overlap duration, event timing, silence, or source
    activity?
-   Are all source classes sampled fairly, especially rare classes?
-   Does the training mixture distribution resemble the intended
    evaluation conditions?
-   Does the model improve on genuinely overlapping examples, or only on
    the synthetic distribution used in training?

## 2.3 Candidate approaches to compare

These are candidates, not assumed novelties. Keep the simplest method as
the baseline.

1.  **Existing random-gain mixing:** reproduce the current approach
    exactly.
2.  **SNR-controlled mixing:** choose a target relative level for the
    added source, then compute gain using a clearly defined SNR
    convention.
3.  **Loudness/energy-aware mixing:** control source contribution using
    a consistent loudness or energy measure, while avoiding excessive
    normalization that erases meaningful level differences.
4.  **Event-aware temporal overlap:** vary onset offset and overlap
    ratio so events need not occupy the entire 10-second clip
    simultaneously.
5.  **Class-aware mixture sampling:** sample source combinations to
    improve rare-class exposure without creating unrealistic or biased
    co-occurrences.
6.  **Combined candidate:** only combine the most promising elements
    after their individual effects have been measured.

Do not implement every option at once. First identify the failure mode
and test one change at a time.

## 2.4 Required controls and measurements

Compare clean-only training, the current mixture method, and each
selected candidate under the same model and split policy.

Measure:

-   mAP and macro/micro F1.
-   Per-class average precision, recall, and F1.
-   Performance grouped by number of active labels.
-   Performance grouped by overlap duration and relative source level.
-   Robustness to controlled background noise or other defined
    perturbations.
-   Mixture validity: clipping rate, source audibility, label
    correctness, and source overlap statistics.

A method that increases overall scores but makes rare events less
detectable or produces unrealistic mixtures may not be a useful
contribution.

**Potential contribution claim, only if supported:** a task-specific,
controlled mixture-generation strategy that improves multi-label
recognition under defined overlap and source-level conditions compared
with conventional random mixing.

**Deliverable:** documented algorithm, mixture-generation configuration,
example waveforms/spectrograms, quality checks, and an ablation table.

------------------------------------------------------------------------

# Phase 3 --- Contribution candidate 2: Classification architecture

## 3.1 Establish the architectural baseline

Document the exact current model from the active code, not just the
intended diagram. The project direction includes:

-   Log-Mel spectrogram input.
-   Patch or feature embedding.
-   Convolutional feature extraction.
-   Multi-scale time--frequency fusion (MTFF).
-   Transformer encoder.
-   Class-query cross-attention.
-   Seven independent output logits.

The custom CNN--Transformer should be described accurately as
AST-inspired, not as an unchanged AST implementation.

## 3.2 Identify the architectural limitation

Use error analysis and feature-shape tracing to identify a specific
issue. Possible questions:

-   Are short transient events such as gunshots and footsteps being
    diluted by global pooling?
-   Are sustained events such as vehicles, helicopters, or fighters
    represented differently from brief events?
-   Does the fusion block preserve useful temporal and frequency
    information?
-   Does class-query attention retrieve class-specific evidence, or does
    it collapse to similar representations for all classes?
-   Are rare classes confused with acoustically similar classes?
-   Does the model process enough temporal tokens to represent event
    timing?

These are hypotheses to investigate, not assumed defects.

## 3.3 Controlled architecture ablation

Use the same data, training protocol, parameter reporting, and
evaluation for each configuration.

  -----------------------------------------------------------------------
  ID                      Model configuration     Purpose
  ----------------------- ----------------------- -----------------------
  A0                      Simple CNN/Transformer  Reference
                          baseline without        
                          proposed MTFF or        
                          class-query attention   

  A1                      A0 + MTFF               Isolate multi-scale
                                                  fusion effect

  A2                      A0 + class-query        Isolate class-specific
                          attention               pooling effect

  A3                      A0 + MTFF + class-query Test the combined
                          attention               design

  A4                      Best justified design + Test the new
                          one proposed            architectural
                          modification            hypothesis
  -----------------------------------------------------------------------

Only add A4 after the first four experiments reveal where the main
limitation lies. Keep parameter count and compute cost in the report;
where feasible, compare against a similarly sized baseline.

## 3.4 Candidate modifications

Choose based on the literature and error analysis, rather than adding
blocks indiscriminately:

-   Multi-resolution temporal or frequency receptive fields.
-   Explicit temporal event localization or event-aware pooling.
-   Class-conditioned feature aggregation.
-   A carefully designed interaction between class queries and
    multi-scale features.
-   A lightweight attention or fusion alternative if the current model
    is too expensive.

A modification should have a clear mechanism, a simple implementation
specification, and an ablation that can show whether it actually helps.

**Potential contribution claim, only if supported:** a specific
architectural mechanism that improves class-wise detection of
overlapping military audio events, with measurable gains over
component-level and compute-aware baselines.

**Deliverable:** architecture diagram, tensor-shape documentation,
parameter/FLOP or runtime comparison, ablation results, and per-class
error analysis.

------------------------------------------------------------------------

# Phase 4 --- Contribution candidate 3: Loss/objective redesign

## 4.1 Correct understanding of BCE loss

For a multi-label problem with seven classes, binary cross-entropy is
calculated independently for each class logit and target. A common form
is:

`L = mean over samples and classes of BCEWithLogits(z[b,c], y[b,c])`

The `mean` reduction averages the element-wise losses into one scalar
for optimization. **It does not remove class-specific learning
signals.** Each class logit contributes its own error and gradient. The
network can still learn different features for different classes.

Therefore, the fact that the losses are averaged is not, by itself, a
reason to replace BCE. A new loss should be motivated by measured
problems such as imbalance, excessive false positives, missed rare
events, or difficult class confusions.

## 4.2 Establish loss baselines

At minimum, compare:

1.  BCEWithLogitsLoss without class weights.
2.  BCEWithLogitsLoss with positive-class weights computed from the
    effective training targets.
3.  One literature-supported alternative chosen for the observed error
    pattern, such as focal loss or asymmetric loss.

Compute weights using the final training labels, including synthetic
mixtures. Cap or otherwise control extreme weights, and handle classes
with zero positive examples safely. Report the exact formula and values.

## 4.3 Decide whether a custom loss is justified

Use validation predictions and per-class metrics to diagnose:

-   Which classes have low recall?
-   Which classes have excessive false positives?
-   Are the errors driven by class imbalance, easy negatives dominating
    optimization, or class co-occurrence?
-   Does improving one rare class severely damage other classes?
-   Are probability calibration and threshold selection responsible for
    apparent metric changes?

Only then consider a new objective. Possible research directions
include:

-   Asymmetric treatment of positive and negative examples.
-   A class-balanced objective derived from the effective training
    distribution.
-   A carefully justified ranking or hard-example component.
-   A co-occurrence-aware term, only if the data support reliable
    modeling of label relationships and the literature reveals a real
    gap.

Do not invent a complex compound loss by adding several known terms
without a clear hypothesis. If combining terms, define every term
mathematically, state its purpose, and ablate each term independently.

## 4.4 Loss evaluation protocol

Keep the model architecture and training data fixed while comparing loss
functions. Use the same optimizer, training budget, checkpoint
selection, and threshold-selection procedure.

Report mAP, macro/micro F1, per-class precision/recall/F1, Hamming loss,
and relevant calibration or ranking metrics. Distinguish improvements
caused by the loss from improvements caused by per-class threshold
tuning.

**Potential contribution claim, only if supported:** an objective
tailored to a demonstrated failure mode of multi-label military audio
classification that improves the chosen primary metric without
unacceptable regressions in other classes.

**Deliverable:** equations, implementation tests, weight values,
training curves, per-class results, and loss ablation table.

------------------------------------------------------------------------

# Phase 5 --- Integrated experiments and ablation study

After testing each contribution in isolation, combine only the validated
components.

## 5.1 Recommended experiment order

  -----------------------------------------------------------------------
  Step                    Experiment              Question
  ----------------------- ----------------------- -----------------------
  E0                      Corrected baseline      What is the trustworthy
                                                  starting point?

  E1                      Existing mixture method Does synthetic mixing
                          vs clean-only           help at all?

  E2                      Candidate mixture       Which data-generation
                          methods                 choice helps and under
                                                  what conditions?

  E3                      Baseline vs MTFF        Does fusion contribute
                                                  independently?

  E4                      Baseline vs class-query Does class-specific
                          attention               pooling contribute
                                                  independently?

  E5                      Best architecture       Is there a useful
                          combination             interaction between
                                                  blocks?

  E6                      BCE vs weighted BCE vs  Which objective
                          selected alternative    addresses the observed
                                                  errors?

  E7                      Best validated data +   Do the contributions
                          architecture + loss     work together?

  E8                      Robustness and          Does the improvement
                          sensitivity             persist under defined
                                                  conditions?

  E9                      Final untouched test    What is the final
                          evaluation              generalization
                                                  performance?
  -----------------------------------------------------------------------

If compute is limited, run a small pilot to eliminate clearly weak
options, then repeat promising configurations with multiple seeds. Do
not choose a winner solely from one lucky run.

## 5.2 Factorial analysis

Once the candidate methods are narrowed down, use a small factorial
ablation to measure individual and interaction effects. For example,
evaluate:

-   Existing vs proposed mixing.
-   Baseline vs proposed architecture.
-   Standard vs selected loss.

This can reveal whether the proposed loss only helps with a particular
architecture or whether improved mixing makes a loss change unnecessary.
Keep training and evaluation conditions consistent.

## 5.3 Robustness experiments

Define test conditions before inspecting final test results. Depending
on the dataset, consider:

-   Number of simultaneously active classes.
-   Relative source level/SNR.
-   Overlap duration and event onset.
-   Controlled background noise.
-   Rare versus frequent classes.
-   Confusable class pairs.

Use source-disjoint validation data for iterative design. Use the
official test set only for the final, pre-specified comparisons.

## 5.4 Statistical credibility

For the strongest configurations:

-   Repeat training with multiple random seeds where feasible.
-   Report mean and standard deviation, not only the best run.
-   Include per-class results and confusion/co-occurrence analysis.
-   Use confidence intervals or an appropriate paired comparison when
    the evaluation design supports it.
-   Report model size, runtime, and computational cost alongside
    accuracy metrics.

------------------------------------------------------------------------

# Phase 6 --- Threshold calibration and metric discipline

Threshold optimization is an important evaluation component, but should
not be presented as a new architectural contribution unless there is a
genuinely new method.

1.  Train with raw logits.
2.  Apply sigmoid to obtain independent class probabilities.
3.  Choose class-specific thresholds on validation predictions only.
4.  Define the threshold objective in advance, for example per-class F1
    or a stated aggregate objective.
5.  Save thresholds with class names, checkpoint ID, and validation
    configuration.
6.  Freeze the thresholds before evaluating the official test set.
7.  Report threshold-free metrics such as mAP as well as
    threshold-dependent metrics such as F1 and Hamming loss.

Do not tune thresholds on the test set or compare methods using
inconsistent threshold procedures.

------------------------------------------------------------------------

# Phase 7 --- Reproducibility and implementation discipline

Maintain a results directory with a consistent structure, for example:

``` text
research/
├── literature/
│   ├── paper_matrix.csv
│   └── gap_analysis.md
├── configs/
│   ├── baseline.yaml
│   └── experiment_*.yaml
├── manifests/
│   ├── source_splits.csv
│   └── mixture_manifest.csv
├── predictions/
│   ├── validation/
│   └── final_test/
├── metrics/
│   ├── per_class.csv
│   └── experiment_summary.csv
├── figures/
└── notes/
    ├── implementation_audit.md
    └── experiment_log.md
```

For each run, record:

-   Git commit hash.
-   Configuration file and random seeds.
-   Dataset and source-split manifest versions.
-   Model checkpoint and checkpoint-selection rule.
-   Loss formula and weights.
-   Thresholds and their validation objective.
-   Metrics, training duration, and hardware/software environment.

Add unit tests for label-union correctness, source-split disjointness,
tensor shapes, finite loss/gradients, class ordering, and metric
calculations.

------------------------------------------------------------------------

# Phase 8 --- Writing the research paper

Write the paper around evidence, not around the number of blocks.

## Suggested structure

1.  **Abstract:** task, verified contributions, experimental setup, and
    key measured results.
2.  **Introduction:** problem significance, limitations of current
    methods, research questions, and contributions.
3.  **Related Work:** multi-label audio classification, synthetic
    mixtures, architectures, and loss functions.
4.  **Dataset and Problem Formulation:** classes, labels, splits, input
    representation, and leakage safeguards.
5.  **Proposed Method:** mixture-generation method, architecture,
    objective, and threshold calibration.
6.  **Experimental Setup:** baselines, implementation details, metrics,
    seeds, and fair-comparison protocol.
7.  **Results:** main comparison, component ablations, per-class
    analysis, and robustness.
8.  **Discussion:** why components helped or failed, limitations,
    computational cost, and validity threats.
9.  **Conclusion and Future Work.**
10. **Appendix/Supplement:** equations, hyperparameters, split protocol,
    additional class-wise results, and reproducibility details.

## Contribution statements

Write contribution claims only after the experiments are complete. A
defensible contribution statement should specify:

-   What is different from the closest prior work.
-   What technical mechanism was introduced.
-   What evaluation demonstrates its value.
-   Which conditions or limitations constrain the claim.

Avoid claiming "first," "state of the art," or "novel" unless the
literature and experimental evidence support the exact wording.

------------------------------------------------------------------------

# Phase 9 --- Decision gates

Proceed to the next phase only when the current phase has met its exit
criteria.

-   [ ] **Gate 1 --- Reliable baseline:** source-disjoint data splits,
    aligned preprocessing, verified tensor shapes, correct multi-label
    evaluator, reproducible baseline.
-   [ ] **Gate 2 --- Literature gap:** comparison matrix completed and
    candidate hypotheses linked to prior work.
-   [ ] **Gate 3 --- Mixing decision:** candidate methods compared under
    fixed conditions; retain a new method only if its benefit is
    measurable and its data generation is valid.
-   [ ] **Gate 4 --- Architecture decision:** MTFF and class-query
    attention evaluated independently; retain modifications supported by
    ablation.
-   [ ] **Gate 5 --- Loss decision:** baseline losses compared; custom
    loss pursued only if a clear error pattern and literature gap
    justify it.
-   [ ] **Gate 6 --- Integration:** strongest validated components
    combined and checked for interactions.
-   [ ] **Gate 7 --- Final evaluation:** thresholds frozen, final test
    run performed once the experimental protocol is set, results
    independently checked.
-   [ ] **Gate 8 --- Paper claims:** every contribution and reported
    metric traceable to code, configuration, predictions, and documented
    evaluation.

------------------------------------------------------------------------

# Immediate next actions: first two weeks

## Week 1 --- Trustworthiness and literature

1.  Inspect the current repository and identify the active generation
    script, training notebook/script, encoder, loss, and evaluator.
2.  Fix or document split leakage, preprocessing mismatches,
    time/frequency-axis handling, and token-shape issues.
3.  Produce a clean baseline and save validation predictions.
4.  Begin the literature matrix, prioritizing multi-label audio
    classification and synthetic mixture generation.

## Week 2 --- Narrow the hypotheses

1.  Complete an initial focused review of mixing strategies, class-query
    attention, multi-scale fusion, and multi-label losses.
2.  Identify the most important baseline error patterns by class and
    overlap condition.
3.  Select no more than a few high-value candidate experiments.
4.  Write an experiment specification for each: hypothesis, method,
    controls, metrics, compute estimate, and falsification criterion.
5.  Run a small pilot for the highest-priority mixing experiment before
    changing the architecture and loss simultaneously.

## Recommended priority

**Correctness and leakage prevention → reproducible baseline →
literature review → mixing ablation → architecture ablation → loss
ablation → integrated study → final test → paper writing.**

This sequence reduces wasted GPU time and makes it possible to attribute
improvements to the right change.

## Final reminder

The objective is not to maximize the number of modifications. It is to
identify a specific scientific gap, design a method that addresses it,
and demonstrate---through fair comparisons and ablations---that the
method provides a real benefit for multi-label military audio
classification.
