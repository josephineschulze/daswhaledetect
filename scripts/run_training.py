""" run file for stage 2 (training of random forest)

loads labeled npz window, balances classes, trains RF, saves the model + metadata, and reports / saved OOB evaluation 
metrics and diagnostic stacked-spectra plots

ignore if you want to use the pre-trained model :) 
"""


import os

import joblib
import numpy as np
import pandas as pd

from training.dataset import load_npz_dataset, balance_classes
from training.model import train_random_forest, get_oob_predictions
from training.evaluate import (
    compute_confusion_matrices,
    compute_per_class_metrics,
    plot_confusion_matrix,
    compute_oob_average_precision,
    plot_stacked_spectra,
)


# Settings

# npz directory - change to needs
npz_dirs = [
    "/Volumes/Extreme Pro/Chapter1_publication_files/reprocessed_publication/training_list/npz", 
] 

feature_key = "spectrum_after_detrended" #this is the final spectrum that RF runs of, its been FK filtered, background removed and gaussian baseline subtracted (detrended)
freq_key = "freqs"
feature_length = 132 # rn i am working with 14-80 Hz, with 0.5 hz steps

random_state = 42
max_per_class = 133

allowed_classes_raw = [ # decide which classes are even going into the training dataset
    "20-40hz", #sweeps
    "20hz", # fin whale 20hz
    "40-70hz", # sweeps
    "backbeat", #fin whale backbeat
    "background noise",
    "bwab", # blue whale a/b
    "ds", # downsweep, but actually 40-70hz sweeps
    "earthquake",
    "ship",
]

# group classes together
sweep_source_classes = {"ds", "40-70hz"}
pulse_source_classes = {"20hz"}
noise_source_classes = {"background noise", "ship", "earthquake"}

class_thresholds = {
    "sweeps": {"min": 0.9},
    "pulses": {"min": 1.0},
    "bwab": {"min": 3.8},
    "backbeat": {"min": 1.6},
    "20-40hz": {"min": 1.1},
}

VMIN = -1
VMAX = 5
CMAP = "viridis"
MAX_WINDOWS_PER_FIG = 500
FIG_HEIGHT = 6

OUTPUT_DIR = "/Volumes/Extreme Pro/Chapter1_publication_files/reprocessed_publication/training_RF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = "/Volumes/Extreme Pro/Chapter1_publication_files/rf_noisesupp_model.pkl"
metadata_path = "/Users/josephsc/Documents/GitHub/DAS_detector_5/publication_files/rf_model_metadata_noisesupp.pkl"


# load + filter + group + gate/threshold
df, skipped = load_npz_dataset(
    npz_dirs=npz_dirs,
    feature_key=feature_key,
    feature_length=feature_length,
    allowed_classes_raw=allowed_classes_raw,
    sweep_source_classes=sweep_source_classes,
    pulse_source_classes=pulse_source_classes,
    noise_source_classes=noise_source_classes,
    class_gates=class_thresholds,
)

if len(df) == 0:
    raise ValueError("No usable NPZ files remained after filtering / grouping / gating")

print(f"\nLoaded {len(df)} usable windows, skipped {len(skipped)}")
print("\nGrouped class counts after gating:")
print(df["CallType_grouped"].value_counts())
print("\nRaw class counts after gating:")
print(df["CallType_raw"].value_counts())


# Balance classes
train_df = balance_classes(df, max_per_class=max_per_class, random_state=random_state)

print(f"\nTotal annotations passing energy gates: {len(df)}")
print(f"Total annotations used after balancing: {len(train_df)}")
print("\nBalanced grouped class counts:")
print(train_df["CallType_grouped"].value_counts())


# train RF
X_train = np.vstack(train_df["spectrum"].to_numpy())
y_train = train_df["CallType_grouped"].to_numpy()
print(f"\nX_train shape: {X_train.shape}")

rf = train_random_forest(X_train, y_train, random_state=random_state)
print(f"\nOOB score: {rf.oob_score_:.4f}")

joblib.dump(rf, model_path)
print("Saved RF model to:", model_path)


# oob predictions + classification metrics
y_oob_true, y_oob_pred, oob_probs_valid, rf_classes = get_oob_predictions(rf, y_train)
print(f"Valid OOB samples: {len(y_oob_true)} / {len(y_train)}")

oob_results_df = pd.DataFrame({"y_true": y_oob_true, "y_pred": y_oob_pred})
for i, cls in enumerate(rf_classes):
    oob_results_df[f"prob_{cls}"] = oob_probs_valid[:, i]
oob_predictions_path = os.path.join(OUTPUT_DIR, "rf_window_predictions_with_truth.csv")
oob_results_df.to_csv(oob_predictions_path, index=False)
print(f"Saved OOB predictions to: {oob_predictions_path}")

labels = sorted(np.unique(y_oob_true))
cm_df, row_pct, col_pct = compute_confusion_matrices(y_oob_true, y_oob_pred, labels)

cm_df.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_raw.csv"))
row_pct.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_row_normalized.csv"))
col_pct.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_col_normalized.csv"))

metrics_df = compute_per_class_metrics(cm_df, labels)
metrics_path = os.path.join(OUTPUT_DIR, "rf_oob_class_metrics.csv")
metrics_df.to_csv(metrics_path)
print("\nOOB metrics by class:")
print(metrics_df)

plot_confusion_matrix(row_pct, "Confusion Matrix (Row Normalized, OOB)",
                       os.path.join(OUTPUT_DIR, "rf_confusion_matrix_row_normalized.png"), fmt=".1f")
plot_confusion_matrix(col_pct, "Confusion Matrix (Column Normalized, OOB)",
                       os.path.join(OUTPUT_DIR, "rf_confusion_matrix_col_normalized.png"), fmt=".1f")


# oob average precision
ap_scores, macro_ap = compute_oob_average_precision(y_oob_true, oob_probs_valid, rf_classes)
for cls, ap in ap_scores.items():
    print(f"OOB AP for class {cls}: {ap:.4f}")
print(f"\nMacro-average OOB AP: {macro_ap:.4f}")

ap_df = pd.DataFrame(ap_scores, index=["average_precision"]).T
ap_df.to_csv(os.path.join(OUTPUT_DIR, "rf_oob_average_precision.csv"))


# save model metadata
model_metadata = {
    "feature_key": feature_key,
    "freq_key": freq_key,
    "feature_length": feature_length,
    "allowed_classes_raw": allowed_classes_raw,
    "sweep_source_classes": sorted(sweep_source_classes),
    "pulse_source_classes": sorted(pulse_source_classes),
    "noise_source_classes": sorted(noise_source_classes),
    "class_gates": class_thresholds,
    "rf_classes": list(rf.classes_),
    "oob_score": float(rf.oob_score_),
    "macro_average_precision": float(macro_ap),
}
joblib.dump(model_metadata, metadata_path)
print("Saved metadata to:", metadata_path)


#diagnostic stacked spectra / stack spectrograms, one set of pages per grouped class
plot_stacked_spectra(
    train_df, feature_key=feature_key, freq_key=freq_key, feature_length=feature_length,
    group_col="CallType_grouped", outdir=os.path.join(OUTPUT_DIR, "stack_plots"),
    max_windows_per_fig=MAX_WINDOWS_PER_FIG, fig_height=FIG_HEIGHT,
    vmin=VMIN, vmax=VMAX, cmap=CMAP,
    title_prefix="Stacked detrended spectra: grouped class = ",
    filename_prefix="stack_train",
)

print("\nTRAINING COMPLETE")
print(f"Model: {model_path}")
print(f"Output directory: {OUTPUT_DIR}")