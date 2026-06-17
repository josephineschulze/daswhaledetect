"""run file for stage 3 (detector)

loads the saved random forest, predicts on a folder of new NPZ windows, aggregates predictions to the group level,
makes diagnostic stack spectrogram plots, and (where applicable) evaluates predictions against ground truth"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from detector.predict import (
    load_unknown_npz,
    predict_windows,
    aggregate_to_groups,
    attach_true_labels,
)
from training.dataset import sanitize_filename
from training.evaluate import (
    compute_confusion_matrices,
    compute_per_class_metrics,
    plot_confusion_matrix,
    plot_stacked_spectra,
)



# Paths
model_path = "/Volumes/Extreme Pro/Chapter1_publication_files/rf_noisesupp_model.pkl"
unknown_npz_dir = "/Volumes/Extreme Pro/Chapter1_publication_files/reprocessed_publication/20220822_list/npz"
OUTPUT_DIR = "/Volumes/Extreme Pro/Chapter1_publication_files/reprocessed_publication/classification_RF_20220822"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Settings
feature_key = "spectrum_after_detrended"
freq_key = "freqs"
feature_length = 132

vmin = -1
vmax = 5
cmap = "viridis"
max_windows_per_fig = 500
fig_height = 6

# maps raw CallType (normalized: lowercase, no spaces/underscores) -> RF class space
calltype_to_class = {
    "20hz": "pulses", # fin whale
    "backbeat": "backbeat", # fin whale
    "backgroundnoise": "noise",
    "20-40hz": "20-40hz", #fin/blue sweeps
    "40-70hz": "sweeps", # fin/blue sweeps
    "ds": "sweeps", # fin/blue sweeps
    "bwab": "bwab", #blue whale A/B
    "earthquake": "noise",
    "ship": "noise",
}


# load model
rf = joblib.load(model_path)
print("Loaded saved random forest from:", model_path)
print("Model classes:", rf.classes_)


#load unknown npz files (no gating)
unknown_df, skipped = load_unknown_npz(unknown_npz_dir, feature_key, feature_length)
print(f"\nUsable unknown records: {len(unknown_df)}")
print(f"Skipped unknown records: {len(skipped)}")

if len(unknown_df) == 0:
    raise ValueError("No usable unknown NPZ files found.")

# predict and aggregate
unknown_pred_df, class_names = predict_windows(rf, unknown_df)
group_pred_df = aggregate_to_groups(unknown_pred_df, class_names)

window_csv = os.path.join(OUTPUT_DIR, "rf_window_predictions.csv")
group_csv = os.path.join(OUTPUT_DIR, "rf_group_predictions.csv")
skipped_csv = os.path.join(OUTPUT_DIR, "rf_unknown_skipped.csv")

unknown_pred_df.to_csv(window_csv, index=False)
group_pred_df.to_csv(group_csv, index=False)
pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(skipped_csv, index=False)
print("\nSaved:", window_csv, group_csv, skipped_csv, sep="\n")


# stack plots by predictd class (whole dataset)
plot_stacked_spectra(
    unknown_pred_df, feature_key=feature_key, freq_key=freq_key, feature_length=feature_length,
    group_col="predicted_class", outdir=os.path.join(OUTPUT_DIR, "stack_plots_by_predicted_class"),
    max_windows_per_fig=max_windows_per_fig, fig_height=fig_height,
    vmin=vmin, vmax=vmax, cmap=cmap,
    title_prefix="Stacked detrended spectra: predicted class = ",
    filename_prefix="stack_predicted",
)


# stack plots by segment block, then by predicted class within each segment
unknown_pred_df["segment_label"] = unknown_pred_df["segment_label"].fillna("").astype(str).str.strip()
unknown_pred_df["segment_label_clean"] = unknown_pred_df["segment_label"].replace("", "unknown_segment")

stack_segment_dir = os.path.join(OUTPUT_DIR, "stack_plots_by_segment")
os.makedirs(stack_segment_dir, exist_ok=True)

for seg_label, seg_df in unknown_pred_df.groupby("segment_label_clean"):
    seg_dir = os.path.join(stack_segment_dir, sanitize_filename(seg_label))
    plot_stacked_spectra(
        seg_df, feature_key=feature_key, freq_key=freq_key, feature_length=feature_length,
        group_col="predicted_class", outdir=seg_dir,
        max_windows_per_fig=max_windows_per_fig, fig_height=fig_height,
        vmin=vmin, vmax=vmax, cmap=cmap,
        title_prefix=f"Segment = {seg_label} | Predicted class = ",
        filename_prefix=f"stack_segment_{sanitize_filename(seg_label)}_pred",
    )



# evaluate against ground truth (rows with a recognised annotation/ true CallType)
unknown_pred_df = attach_true_labels(unknown_pred_df, calltype_to_class)

true_label_summary_raw = (
    unknown_pred_df["true_class_raw"].value_counts(dropna=False)
    .rename_axis("true_class_raw").reset_index(name="n_windows")
)
true_label_summary_raw["fraction"] = true_label_summary_raw["n_windows"] / len(unknown_pred_df)
print("\nRaw true label summary:")
print(true_label_summary_raw)

true_label_summary_mapped = (
    unknown_pred_df["true_class_mapped"].value_counts(dropna=False)
    .rename_axis("true_class_mapped").reset_index(name="n_windows")
)
true_label_summary_mapped["fraction"] = true_label_summary_mapped["n_windows"] / len(unknown_pred_df)
print("\nMapped true label summary:")
print(true_label_summary_mapped)

unmapped_df = unknown_pred_df[
    unknown_pred_df["true_class_mapped"].astype(str).str.startswith("unmapped::", na=False)
].copy()

true_label_summary_raw.to_csv(os.path.join(OUTPUT_DIR, "rf_true_label_summary_raw.csv"), index=False)
true_label_summary_mapped.to_csv(os.path.join(OUTPUT_DIR, "rf_true_label_summary_mapped.csv"), index=False)
unmapped_df.to_csv(os.path.join(OUTPUT_DIR, "rf_unmapped_true_labels.csv"), index=False)

eval_df = unknown_pred_df[
    (unknown_pred_df["true_class_mapped"] != "Unknown")
    & (~unknown_pred_df["true_class_mapped"].astype(str).str.startswith("unmapped::", na=False))
].copy()

if len(eval_df) == 0:
    raise ValueError("No evaluable rows found after mapping true labels.")
print(f"\nEvaluable rows: {len(eval_df)} / {len(unknown_pred_df)}")

y_true = eval_df["true_class_mapped"]
y_pred = eval_df["predicted_class"]
labels_for_eval = list(rf.classes_)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels_for_eval, zero_division=0
)
pr_table = pd.DataFrame({
    "Class": labels_for_eval, "Precision": precision, "Recall": recall,
    "F1_score": f1, "Support": support,
})
print("\nPrecision / recall table:")
print(pr_table)
pr_table.to_csv(os.path.join(OUTPUT_DIR, "rf_precision_recall_table.csv"), index=False)

print("\nClassification report:")
print(classification_report(y_true, y_pred, labels=labels_for_eval, zero_division=0))

cm_raw_df, cm_row_norm_df, cm_col_norm_df = compute_confusion_matrices(y_true, y_pred, labels_for_eval)
cm_raw_df.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_raw.csv"))
cm_row_norm_df.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_row_normalized.csv"))
cm_col_norm_df.to_csv(os.path.join(OUTPUT_DIR, "rf_confusion_matrix_col_normalized.csv"))

eval_metrics_df = compute_per_class_metrics(cm_raw_df, labels_for_eval)
eval_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "rf_eval_class_metrics.csv"))

unknown_pred_df.to_csv(os.path.join(OUTPUT_DIR, "rf_window_predictions_with_truth.csv"), index=False)


# plot confusion matrices
plot_confusion_matrix(
    cm_raw_df, "Raw confusion matrix",
    os.path.join(OUTPUT_DIR, "rf_confusion_matrix_raw.png"), fmt=".0f",
)
plot_confusion_matrix(
    cm_row_norm_df, "Row-wise normalized confusion matrix (%)",
    os.path.join(OUTPUT_DIR, "rf_confusion_matrix_row_normalized.png"), fmt=".1f",
)
plot_confusion_matrix(
    cm_col_norm_df, "Column-wise normalized confusion matrix (%)",
    os.path.join(OUTPUT_DIR, "rf_confusion_matrix_col_normalized.png"), fmt=".1f",
)

print("\nDETECTION COMPLETE")
print(f"Output directory: {OUTPUT_DIR}")