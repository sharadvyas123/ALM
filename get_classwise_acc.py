
# ============================================================
# ALM: POST-TRAINING CLASS-WISE EVALUATION
# Loads saved checkpoint. DOES NOT TRAIN OR MODIFY THE MODEL.
# ============================================================

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# ------------------------------------------------------------
# 1. PROJECT ROOT AND IMPORTS
# ------------------------------------------------------------

ROOT = Path.cwd().resolve()

while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

if not (ROOT / "src").is_dir():
    raise RuntimeError("Could not identify ALM repository root.")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alm.config import load_experiment, resolve_experiment_paths
from alm.data.provenance import CLASS_NAMES
from alm.data.dataset import MADMixtureDataset
from alm.models.alm_model import ALMModel
from alm.training.metrics import (
    calibrate_thresholds,
    multilabel_metrics
)


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

CONFIG_PATH = (
    ROOT / "configs/experiments/"
    "FULL_proposed_alm_source_disjoint_v1.json"
)

CFG = load_experiment(CONFIG_PATH)
PATHS = resolve_experiment_paths(ROOT, CFG)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RUN_DIR = ROOT / "runs" / CFG["run_id"]

CHECKPOINT_PATH = RUN_DIR / "best_model.pt"

OUTPUT_DIR = RUN_DIR / "post_training_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ALM POST-TRAINING EVALUATION")
print("=" * 65)

print("Device:", DEVICE)
print("Run:", CFG["run_id"])
print("Checkpoint:", CHECKPOINT_PATH)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {CHECKPOINT_PATH}"
    )

print("Classes:", list(CLASS_NAMES))


# ------------------------------------------------------------
# 3. RECONSTRUCT THE MODEL ARCHITECTURE
# ------------------------------------------------------------

model_cfg = CFG["model"]

model = ALMModel(
    **{
        key: model_cfg[key]
        for key in (
            "embed_dim",
            "num_heads",
            "num_layers",
            "ff_dim",
            "dropout",
            "max_time_frames"
        )
    },
    use_fusion=model_cfg["use_fusion"],
    use_class_queries=model_cfg["use_class_queries"],
    use_mask_supervision=model_cfg["use_mask_supervision"]
).to(DEVICE)

# Load previously trained weights
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state"])

model.eval()

print("\nSaved model loaded successfully!")
print("Checkpoint epoch:", checkpoint.get("epoch", "Not recorded"))


# ------------------------------------------------------------
# 4. RECONSTRUCT VALIDATION DATASET
# ------------------------------------------------------------

aux_kwargs = dict(
    metadata_path=PATHS["metadata"],
    audio_dir=PATHS["audio"],
    preprocessing=CFG["preprocessing"],
    source_metadata_path=PATHS["source_metadata"],
    mixture_manifest_path=(
        PATHS["manifest"] / "mixture_manifest.csv"
    ),
    source_manifest_path=(
        PATHS["manifest"] / "source_splits.csv"
    ),
    source_audio_dir=PATHS["source_audio"]
)

needs_aux_targets = any(
    model_cfg[key]
    for key in (
        "use_mask_supervision",
        "use_audibility_labels",
        "use_audibility_loss"
    )
)

eval_dataset = MADMixtureDataset(
    **aux_kwargs,
    split="val",
    return_aux_targets=needs_aux_targets,

    label_policy=(
        CFG["audibility"]["policy"]
        if model_cfg["use_audibility_labels"]
        else "union"
    ),

    audibility_threshold_db=CFG["audibility"]["threshold_db"],
    soft_audibility_center_db=CFG["audibility"]["center_db"],
    soft_audibility_temperature_db=CFG["audibility"]["temperature_db"],
    soft_audibility_floor=CFG["audibility"]["soft_floor"],

    # No cache generation or training required
    cache_dir=None
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=CFG["training"]["batch_size"] * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

print("\nValidation samples:", len(eval_dataset))
print("Validation batches:", len(eval_loader))


# ------------------------------------------------------------
# 5. COLLECT VALIDATION PREDICTIONS
# ------------------------------------------------------------

all_targets = []
all_scores = []

model.eval()

with torch.inference_mode():

    for batch_idx, batch in enumerate(eval_loader):

        mel = batch["mel"].to(DEVICE)

        output = model(mel, return_aux=True)

        probabilities = torch.sigmoid(
            output["logits"]
        )

        # Ground-truth labels
        if "union_target" in batch:
            targets = batch["union_target"]
        else:
            targets = batch["target"]

        all_targets.append(targets.cpu().numpy())

        all_scores.append(
            probabilities.cpu().numpy()
        )

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {batch_idx + 1}/"
                f"{len(eval_loader)} batches"
            )


y_true = np.concatenate(all_targets, axis=0)
y_score = np.concatenate(all_scores, axis=0)

print("\nInference completed!")
print("Ground truth shape:", y_true.shape)
print("Prediction shape:", y_score.shape)

assert y_true.shape == y_score.shape
assert y_true.shape[1] == len(CLASS_NAMES)


# ------------------------------------------------------------
# 6. CLASS-WISE ACCURACY AT DEFAULT THRESHOLD (0.5)
# ------------------------------------------------------------

y_pred_default = (
    y_score >= 0.5
).astype(np.int32)

correct_default = (
    y_true == y_pred_default
).sum(axis=0)

class_accuracy_default = (
    correct_default / y_true.shape[0]
)

# ------------------------------------------------------------
# 7. CALIBRATE PER-CLASS THRESHOLDS
# ------------------------------------------------------------

calibration = calibrate_thresholds(
    y_true,
    y_score,
    strategy="per_class_f1"
)

thresholds = calibration["thresholds"]

# Support both dictionary and array/list threshold formats
if isinstance(thresholds, dict):
    thresholds = np.array(
        [thresholds[name] for name in CLASS_NAMES],
        dtype=float
    )
else:
    thresholds = np.asarray(thresholds, dtype=float)

assert thresholds.shape == (len(CLASS_NAMES),)

y_pred_calibrated = (
    y_score >= thresholds.reshape(1, -1)
).astype(np.int32)

correct_calibrated = (
    y_true == y_pred_calibrated
).sum(axis=0)

class_accuracy_calibrated = (
    correct_calibrated / y_true.shape[0]
)


# ------------------------------------------------------------
# 8. CALCULATE CLASS-WISE METRICS
# ------------------------------------------------------------

tp = ((y_true == 1) & (y_pred_calibrated == 1)).sum(axis=0)

tn = ((y_true == 0) & (y_pred_calibrated == 0)).sum(axis=0)

fp = ((y_true == 0) & (y_pred_calibrated == 1)).sum(axis=0)

fn = ((y_true == 1) & (y_pred_calibrated == 0)).sum(axis=0)

precision = np.divide(
    tp,
    tp + fp,
    out=np.zeros(len(CLASS_NAMES), dtype=float),
    where=(tp + fp) != 0
)

recall = np.divide(
    tp,
    tp + fn,
    out=np.zeros(len(CLASS_NAMES), dtype=float),
    where=(tp + fn) != 0
)

f1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros(len(CLASS_NAMES), dtype=float),
    where=(precision + recall) != 0
)

# Per-class average accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

positive_support = tp + fn
negative_support = tn + fp


# ------------------------------------------------------------
# 9. CREATE FINAL CLASS-WISE RESULTS TABLE
# ------------------------------------------------------------

results = pd.DataFrame({

    "Class": CLASS_NAMES,

    "Threshold": thresholds,

    "Positive_Samples": positive_support,

    "Negative_Samples": negative_support,

    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,

    "Accuracy_0.5": class_accuracy_default,

    "Accuracy_Calibrated": class_accuracy_calibrated,

    "Precision": precision,

    "Recall": recall,

    "F1_Score": f1,

    "Accuracy_Percent": accuracy * 100,

    "Precision_Percent": precision * 100,

    "Recall_Percent": recall * 100,

    "F1_Percent": f1 * 100
})


# ------------------------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CLASS-WISE EVALUATION RESULTS")
print("=" * 65)

print(
    results.round(4)
)

print("\nMean Class Accuracy:",
      round(float(accuracy.mean() * 100), 2), "%")

print("Mean Class Precision:",
      round(float(precision.mean() * 100), 2), "%")

print("Mean Class Recall:",
      round(float(recall.mean() * 100), 2), "%")

print("Mean Class F1:",
      round(float(f1.mean() * 100), 2), "%")


# ------------------------------------------------------------
# 11. EXISTING MULTI-LABEL EVALUATION METRICS
# ------------------------------------------------------------

uncalibrated_metrics = multilabel_metrics(
    y_true,
    y_score
)

calibrated_metrics = multilabel_metrics(
    y_true,
    y_score,
    thresholds=thresholds
)

print("\nUNCALIBRATED METRICS")

for key in (
    "mAP",
    "macro_f1",
    "micro_f1",
    "hamming_loss"
):
    print(f"{key}: {uncalibrated_metrics[key]}")


print("\nCALIBRATED METRICS")

for key in (
    "mAP",
    "macro_f1",
    "micro_f1",
    "hamming_loss"
):
    print(f"{key}: {calibrated_metrics[key]}")


# ------------------------------------------------------------
# 12. SAVE ALL RESULTS
# ------------------------------------------------------------

# Main class-wise accuracy and metrics
results.to_csv(
    OUTPUT_DIR / "class_wise_accuracy.csv",
    index=False
)

# Existing detailed metrics format
pd.DataFrame.from_dict(
    calibrated_metrics["per_class"],
    orient="index"
).to_csv(
    OUTPUT_DIR / "class_metrics.csv"
)

# Save calibrated thresholds
with open(
    OUTPUT_DIR / "thresholds.json",
    "w"
) as f:
    json.dump(calibration, f, indent=4)

# Save global metrics
global_results = {
    "uncalibrated": {
        key: float(uncalibrated_metrics[key])
        for key in (
            "mAP",
            "macro_f1",
            "micro_f1",
            "hamming_loss"
        )
    },

    "calibrated": {
        key: float(calibrated_metrics[key])
        for key in (
            "mAP",
            "macro_f1",
            "micro_f1",
            "hamming_loss"
        )
    },

    "mean_class_accuracy": float(accuracy.mean()),

    "mean_class_precision": float(precision.mean()),

    "mean_class_recall": float(recall.mean()),

    "mean_class_f1": float(f1.mean())
}

with open(
    OUTPUT_DIR / "global_metrics.json",
    "w"
) as f:
    json.dump(global_results, f, indent=4)


# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSaved files:")

for file in OUTPUT_DIR.iterdir():
    print(" -", file.name)

print("\nOutput directory:", OUTPUT_DIR)

print("\nNo training was performed.")
print("Original checkpoint remains unchanged.")
