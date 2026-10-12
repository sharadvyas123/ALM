# Proposed-framework experiments

All configurations use the authenticated `mad_source_disjoint_v1` train/validation
dataset, hop 512, `log(mel + 1e-6)` preprocessing, seed 42, AdamW, and validation
mAP checkpoint selection. The ordinary metadata union is retained unless a
configuration explicitly enables an audibility policy.

`B7_confusion_weights_source_disjoint_v1.json` estimates a nonnegative
off-diagonal matrix from **training-label co-occurrence only**. It is not an
operational cost matrix. `B7_operational_costs_template.json` intentionally
remains disabled until a domain expert supplies and documents a 7x7 matrix.

`FULL_proposed_alm_source_disjoint_v1.json` enables all implemented and
justified components: fusion, spatial class queries, mask supervision,
soft-audibility targets, audibility-conditioned ASL, and training-derived
confusion weighting. It does not invent operational costs.
