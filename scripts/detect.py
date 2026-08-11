"""
Stage 3: detector.

Loads a saved random forest, predicts on a folder of new NPZ windows,
aggregates predictions to the group level, makes diagnostic stack
spectrogram plots, and (where applicable) evaluates predictions against
ground truth.

Usage:
    python predict.py --config configs/detection.yaml
"""

import os
import argparse
import logging
import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

from daswhaledetect.utils import load_config_detect
from daswhaledetect.dataload import sanitize_filename, load_unknown_npz
from daswhaledetect.evaluate import (
    predict,
    save_prediction_tables,
    plot_stacked_spectra,
    attach_true_labels,
    build_eval_dataframe,
    compute_confusion_matrices,
    compute_per_class_metrics,
)

logger = logging.getLogger("daswhale.predict")


def run_detection(cfg):
    """Runs the full stage-3 pipeline end to end, using settings from cfg."""

    output_dir = cfg["paths"]["output_dir"]

    # 1. Load model and data
    rf = joblib.load(cfg["paths"]["model_path"])
    logger.info("Loaded saved random forest from: %s", cfg["paths"]["model_path"])
    logger.info("Model classes: %s", rf.classes_)

    unknown_df, skipped = load_unknown_npz(cfg["paths"]["npz_dir"], cfg["features"]["feature_key"], cfg["features"]["feature_length"])
    unknown_pred_df, group_pred_df, class_names = predict(rf, unknown_df)

    save_prediction_tables(unknown_pred_df, group_pred_df, skipped, output_dir)

    plot_stacked_spectra(
        unknown_pred_df, 
        feature_key=cfg["features"]["feature_key"], 
        freq_key=cfg["features"]["freq_key"], 
        feature_length=cfg["features"]["feature_length"],
        group_col="predicted_class", 
        outdir=os.path.join(output_dir, "stack_plots_by_predicted_class"),
        max_windows_per_fig=cfg["plotting"]["max_windows_per_fig"], 
        fig_height=cfg["plotting"]["fig_height"],
        vmin=cfg["plotting"]["vmin"], 
        vmax=cfg["plotting"]["vmax"], 
        cmap=cfg["plotting"]["cmap"],
        title_prefix="Stacked detrended spectra: predicted class = ",
        filename_prefix="stack_predicted"
    )

    # stack plots by segment block, then by predicted class within each segment
    unknown_pred_df["segment_label"] = unknown_pred_df["segment_label"].fillna("").astype(str).str.strip()
    unknown_pred_df["segment_label_clean"] = unknown_pred_df["segment_label"].replace("", "unknown_segment")
    
    stack_segment_dir = os.path.join(output_dir, "stack_plots_by_segment")
    os.makedirs(stack_segment_dir, exist_ok=True)
    

    for seg_label, seg_df in unknown_pred_df.groupby("segment_label_clean"):
        seg_dir = os.path.join(stack_segment_dir, sanitize_filename(seg_label))
        plot_stacked_spectra(
            seg_df, 
            feature_key=cfg["features"]["feature_key"], 
            freq_key=cfg["features"]["freq_key"], 
            feature_length=cfg["features"]["feature_length"],
            group_col="predicted_class", 
            outdir=seg_dir,
            max_windows_per_fig=cfg["plotting"]["max_windows_per_fig"], 
            fig_height=cfg["plotting"]["fig_height"],
            vmin=cfg["plotting"]["vmin"], 
            vmax=cfg["plotting"]["vmax"], 
            cmap=cfg["plotting"]["cmap"],
            title_prefix=f"Segment = {seg_label} | Predicted class = ",
            filename_prefix=f"stack_segment_{sanitize_filename(seg_label)}_pred"
        )

    # 4. evaluate against ground truth, if any labels are available
    unknown_pred_df = attach_true_labels(unknown_pred_df, cfg["label_mapping"])
    eval_df = build_eval_dataframe(unknown_pred_df, output_dir)
    if eval_df is None:
        return
    y_true = eval_df["true_class_mapped"]
    y_pred = eval_df["predicted_class"]
    labels_for_eval = list(rf.classes_)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_for_eval, zero_division=0)
    pr_table = pd.DataFrame({
        "class": labels_for_eval,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support
    })
    logger.info("Precision/Recall/F1 table:")
    for line in pr_table.to_string(index=False).splitlines():
        logger.info("%s", line)
    pr_table.to_csv(os.path.join(output_dir, "rf_eval_pr_table.csv"), index=False)

    logger.info("=" * 80)
    logger.info("Classification report")
    logger.info("=" * 80)
    report = classification_report(y_true, y_pred, labels=labels_for_eval, zero_division=0)
    for line in report.strip().splitlines():
        logger.info("%s", line)

    cm_raw_df, cm_row_norm_df, cm_col_norm_df = compute_confusion_matrices(y_true, y_pred, labels_for_eval, output_dir, save_to_csv=True)
    eval_metrics_df = compute_per_class_metrics(cm_raw_df, labels_for_eval, output_dir, save_to_csv=True)
    eval_metrics_df.to_csv(os.path.join(output_dir, "rf_eval_class_metrics.csv"))

    unknown_pred_df.to_csv(os.path.join(output_dir, "rf_window_predictions_with_truth.csv"), index=False)

    # plot_confusion_matrix(
    #     cm_raw_df, "Raw confusion matrix",
    #     os.path.join(output_dir, "rf_confusion_matrix_raw.png"), fmt=".0f",
    # )
    # plot_confusion_matrix(
    #     cm_row_norm_df, "Row-wise normalized confusion matrix (%)",
    #     os.path.join(output_dir, "rf_confusion_matrix_row_normalized.png"), fmt=".1f",
    # )
    # plot_confusion_matrix(
    #     cm_col_norm_df, "Column-wise normalized confusion matrix (%)",
    #     os.path.join(output_dir, "rf_confusion_matrix_col_normalized.png"), fmt=".1f",
    # )

    logger.info("DETECTION COMPLETE")
    logger.info("Output directory: %s", output_dir)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run stage 3 (detector) of the DASWhaleDetect pipeline.")
    parser.add_argument("--config", default="configs/detect.yaml", help="Path to a detection YAML config file.")
    args = parser.parse_args()
    cfg = load_config_detect(args.config)

    log_path = os.path.join(cfg["paths"]["output_dir"], "detection.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]
    )

    run_detection(cfg)


if __name__ == "__main__":
    main()