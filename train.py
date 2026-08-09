"""
Stage 2: Train the Random Forest classifier.

This script:

1. Loads labeled preprocessed NPZ windows.
2. Filters, groups, and optionally gates raw CallType labels.
3. Balances classes.
4. Trains a Random Forest classifier.
5. Saves the model and metadata.
6. Computes out-of-bag evaluation metrics.
7. Saves tables, confusion matrices, and stacked-spectrum plots.

Run:

    python train.py --config configs/train.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from daswhaledetect.utils import load_config_train
from daswhaledetect.dataload import load_npz_dataset, balance_classes
from daswhaledetect.evaluate import (
    oob_predictions, 
    compute_confusion_matrices, 
    compute_per_class_metrics, 
    compute_oob_average_precision,
    plot_stacked_spectra,
    plot_confusion_matrix,
    )

logger = logging.getLogger("train")

def run_training(cfg):
    """Run the full Random Forest training pipeline."""
    paths = cfg["paths"]
    npz_dirs = [p for p in Path(paths["npz_dir"]).rglob("*npz") if p.is_dir()]
    output_dir = Path(paths["output_dir"])
    model_path = os.path.join(output_dir, f"{paths['model_name']}.pkl")
    metadata_path = os.path.join(output_dir, f"{paths['model_name']}_metadata.pkl")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING RANDOM FOREST TRAINING")
    logger.info("Input NPZ directories:")
    for npz_dir in npz_dirs:
        logger.info("  %s", npz_dir)

    logger.info("Output directory: %s", cfg["paths"]["output_dir"])
    logger.info("Feature key: %s", cfg["features"]["feature_key"])
    logger.info("Expected feature length: %d", cfg["features"]["feature_length"])
    logger.info("Random seed: %d", cfg["sampling"]["random_state"])
    logger.info("Maximum samples per class: %d", cfg["sampling"]["max_per_class"])
    logger.info("=" * 80)


    dataset_df, skipped = load_npz_dataset(
        npz_dirs=npz_dirs,
        feature_key=cfg["features"]["feature_key"],
        feature_length=cfg["features"]["feature_length"],
        allowed_classes_raw=cfg["classes"]["allowed_classes_raw"],
        sweep_source_classes=cfg["classes"]["sweep_source_classes"],
        pulse_source_classes=cfg["classes"]["pulse_source_classes"],
        noise_source_classes=cfg["classes"]["noise_source_classes"],
        class_gates=cfg["classes"]["class_gates"],
    )
    # -------------------------------------------------------------------------
    # Balance classes
    # -------------------------------------------------------------------------

    train_df = balance_classes(
        dataset_df,
        max_per_class=cfg["sampling"]["max_per_class"],
        random_state=cfg["sampling"]["random_state"],
    )

    X_train = np.vstack(train_df["spectrum"].to_numpy())
    y_train = train_df["CallType_grouped"].to_numpy()

    logger.info("Training matrix shape: %s", X_train.shape)

    rf_model = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        min_samples_split=cfg["model"]["min_samples_split"],
        min_samples_leaf=cfg["model"]["min_samples_leaf"],
        max_features=cfg["model"]["max_features"],
        class_weight=cfg["model"]["class_weight"],
        bootstrap=cfg["model"]["bootstrap"],
        oob_score=cfg["model"]["oob_score"],
        n_jobs = cfg["n_workers"],
        random_state=cfg["sampling"]["random_state"],
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, model_path)

    logger.info("Saved Random Forest model: %s", model_path)
    logger.info("OOB score: %.4f", rf_model.oob_score_)

    y_oob_true, y_oob_pred, oob_probs_valid, rf_classes = oob_predictions(rf_model, y_train, cfg["paths"]["output_dir"])

    cf_dir = os.path.join(cfg["paths"]["output_dir"], "confusion_matrices")
    metrics_dir = os.path.join(cfg["paths"]["output_dir"], "metrics")
    os.makedirs(cf_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    labels = sorted(np.unique(y_oob_true))
    cm_df, row_pct, col_pct = compute_confusion_matrices(y_oob_true, y_oob_pred, labels, cf_dir, save_to_csv=True)
    metrics_df = compute_per_class_metrics(cm_df, labels, metrics_dir, save_to_csv=True)
    ap_scores, macro_ap = compute_oob_average_precision(y_oob_true, oob_probs_valid, rf_classes, metrics_dir, save_to_csv=True)

    plot_confusion_matrix(row_pct, "Confusion Matrix (Row Normalized, OOB)", 
                          os.path.join(cf_dir, "rf_confusion_matrix_row_normalized.png"))
    plot_confusion_matrix(col_pct, "Confusion Matrix (Column Normalized, OOB)", 
                          os.path.join(cf_dir, "rf_confusion_matrix_col_normalized.png"))
    plot_confusion_matrix(cm_df, "Confusion Matrix (Raw Counts, OOB)", 
                          os.path.join(cf_dir, "rf_confusion_matrix_raw.png"))


    model_metadata = {
        "feature_key": cfg["features"]["feature_key"],
        "freq_key": cfg["features"]["freq_key"],
        "feature_length": int(cfg["features"]["feature_length"]),
        "allowed_classes_raw": cfg["classes"]["allowed_classes_raw"],
        "sweep_source_classes": sorted(cfg["classes"]["sweep_source_classes"]),
        "pulse_source_classes": sorted(cfg["classes"]["pulse_source_classes"]),
        "noise_source_classes": sorted(cfg["classes"]["noise_source_classes"]),
        "class_gates": cfg["classes"]["class_gates"],
        "rf_classes": list(rf_model.classes_),
        "oob_score": float(rf_model.oob_score_),
        "macro_average_precision": float(macro_ap)
    }
    joblib.dump(model_metadata, metadata_path)
    logger.info("Saved model metadata: %s", metadata_path)
    
    plot_stacked_spectra(
        train_df,
        feature_key=cfg["features"]["feature_key"],
        freq_key=cfg["features"]["freq_key"],
        feature_length=cfg["features"]["feature_length"],
        group_col="CallType_grouped",
        outdir=os.path.join(cfg["paths"]["output_dir"], "stack_plots"),
        max_windows_per_fig=cfg["plotting"]["max_windows_per_fig"],
        fig_height=cfg["plotting"]["fig_height"],
        vmin=cfg["plotting"]["vmin"],
        vmax=cfg["plotting"]["vmax"],
        cmap=cfg["plotting"]["cmap"],
        title_prefix="Stacked detrended spectra: grouped class = ",
        filename_prefix="stack_train",
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("Model: %s", model_path)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run stage 2 (training) of the DASWhaleDetect pipeline.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to a training YAML config file.")
    args = parser.parse_args()
    cfg = load_config_train(args.config)

    log_path = os.path.join(cfg["paths"]["output_dir"], "training.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )


    run_training(cfg)


if __name__ == "__main__":
    main()