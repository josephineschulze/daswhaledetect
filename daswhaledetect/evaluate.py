import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from daswhaledetect.utils import majority_vote, safe_scalar_str, normalize_label

logger = logging.getLogger("daswhaledetect.evaluate")

def save_prediction_tables(unknown_pred_df, group_pred_df, skipped, output_dir):
    window_csv = os.path.join(output_dir, "rf_window_predictions.csv")
    group_csv = os.path.join(output_dir, "rf_group_predictions.csv")
    skipped_csv = os.path.join(output_dir, "rf_unknown_skipped.csv")


    unknown_pred_df.drop(columns=["spectrum"], errors="ignore").to_csv(window_csv, index=False)
    group_pred_df.drop(columns=["spectrum"], errors="ignore").to_csv(group_csv, index=False)
    pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(skipped_csv, index=False)
    logger.info("Saved prediction tables to: %s", output_dir)

def oob_predictions(rf_model, y_train, output_dir):
    oob_probs = rf_model.oob_decision_function_
    rf_classes = rf_model.classes_

    valid_oob_mask = np.isfinite(oob_probs).all(axis=1) & (oob_probs.sum(axis=1) > 0)
    y_oob_true = np.asarray(y_train)[valid_oob_mask]
    oob_probs_valid = oob_probs[valid_oob_mask]
    y_oob_pred = rf_classes[np.argmax(oob_probs_valid, axis=1)]

    logger.info("Valid OOB samples: %d / %d", len(y_oob_true),len(y_train),)

    oob_results_df = pd.DataFrame({"y_true": y_oob_true, "y_pred": y_oob_pred})
    for i, cls in enumerate(rf_classes):
        oob_results_df[f"prob_{cls}"] = oob_probs_valid[:, i]
    oob_predictions_path = os.path.join(output_dir, "rf_window_predictions_with_truth.csv")
    oob_results_df.to_csv(oob_predictions_path, index=False)  
    logger.info("Saved OOB predictions: %s", oob_predictions_path)

    return y_oob_true, y_oob_pred, oob_probs_valid, rf_classes


def compute_confusion_matrices(y_true, y_pred, labels, output_dir, save_to_csv=True):
    """
    raw counts
    row-normalized (recall: % of each true class's
    windows)
    column-normalized (precision: % of each predicted class's windows) versions, all as labeled dataframes.

    returns (cm_df, row_pct, col_pct)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).rename_axis("true_class", axis="index")
    row_pct = (cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0) * 100).rename_axis("true_class", axis="index") #recall confusion matrix
    col_pct = (cm_df.div(cm_df.sum(axis=0).replace(0, np.nan), axis=1) * 100).rename_axis("true_class", axis="index") # precision confusion matrix

    if save_to_csv:
        cm_df.to_csv(os.path.join(output_dir, "rf_confusion_matrix_raw.csv"))
        row_pct.to_csv(os.path.join(output_dir, "rf_confusion_matrix_row_normalized.csv"))
        col_pct.to_csv(os.path.join(output_dir, "rf_confusion_matrix_col_normalized.csv"))

    return cm_df, row_pct, col_pct

def compute_per_class_metrics(cm_df, labels, output_dir, save_to_csv=True):
    """precision/recall/f1/support per class, computed directly from a raw confusion-matrix dataframe"""
    cm = cm_df.values
    class_metrics = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[label] = {
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "support": int(cm[i, :].sum()),
        }

    if save_to_csv:
        logger.info("OOB metrics by class:\n%s", pd.DataFrame(class_metrics).T.to_string())
        metrics_path = os.path.join(output_dir, "rf_oob_class_metrics.csv")
        pd.DataFrame(class_metrics).T.rename_axis("class", axis="index").to_csv(metrics_path)
        logger.info("Saved OOB metrics by class: %s", metrics_path)

    
    return pd.DataFrame(class_metrics).T

def compute_oob_average_precision(y_oob_true, oob_probs_valid, rf_classes, output_dir, save_to_csv=True):
    """
    per-class average precision from OOB probabilities (one-vs-rest), plus
    the macro average across classes. returns (ap_scores_dict, macro_ap)
    """
    y_true_bin = label_binarize(y_oob_true, classes=rf_classes)
    ap_scores = {}
    for i, cls in enumerate(rf_classes):
        ap_scores[cls] = float(average_precision_score(y_true_bin[:, i], oob_probs_valid[:, i]))
    macro_ap = float(np.mean(list(ap_scores.values())))

    for cls, ap in ap_scores.items():
        logger.info("OOB AP for class %s: %.4f", cls, ap)
    logger.info("Macro-average OOB AP: %.4f", macro_ap)

    ap_df = pd.DataFrame(ap_scores, index=["average_precision"]).T.rename_axis("class", axis="index")
    if save_to_csv:
        ap_df.to_csv(os.path.join(output_dir, "rf_oob_average_precision.csv"))

    return ap_scores, macro_ap


def predict(rf, df):
    """
    Runs RF prediction on every window and aggregates predictions by group.

    Window-level output:
        - predicted_class
        - max_probability
        - prob_<class> columns

    Group-level output:
        - majority vote prediction
        - mean probability prediction
        - confidence statistics

    Returns:
        pred_df: dataframe with window-level predictions
        group_df: dataframe with group-level predictions
        class_names: RF class names
    """

    # ---------- window-level prediction ----------
    X = np.vstack(df["spectrum"].to_numpy())

    pred_labels = rf.predict(X)
    pred_probs = rf.predict_proba(X)
    class_names = rf.classes_

    pred_df = df.copy()
    pred_df["predicted_class"] = pred_labels
    pred_df["max_probability"] = pred_probs.max(axis=1)

    for i, cls in enumerate(class_names):
        pred_df[f"prob_{cls}"] = pred_probs[:, i]

    sort_cols = [
        c for c in ["predicted_class", "group_id", "window_index", "npz_path"]
        if c in pred_df.columns
    ]
    pred_df = pred_df.sort_values(sort_cols).reset_index(drop=True)


    # ---------- group-level aggregation ----------
    prob_cols = [f"prob_{cls}" for cls in class_names]
    group_rows = []

    for group_id, g in pred_df.groupby("group_id"):
        row = {
            "group_id": group_id,
            "file_path": g["file_path"].iloc[0],
            "segment_label": g["segment_label"].iloc[0],
            "n_windows": len(g),
            "group_pred_majority_vote": majority_vote(g["predicted_class"]),
            "group_mean_confidence": g["max_probability"].mean(),
        }

        mean_probs = g[prob_cols].mean(axis=0)
        for cls in class_names:
            row[f"mean_prob_{cls}"] = mean_probs[f"prob_{cls}"]

        row["group_pred_mean_prob"] = class_names[
            np.argmax([row[f"mean_prob_{cls}"] for cls in class_names])
        ]
        group_rows.append(row)

    group_df = (pd.DataFrame(group_rows).sort_values("group_id").reset_index(drop=True))

    
    logger.info("%d predicted classes: %s", len(class_names), list(class_names))
    logger.info("%d usable unknown windows", len(df))
    logger.info("%d window-level predictions", len(pred_df))
    logger.info("%d group-level predictions", len(group_df))

    return pred_df, group_df, class_names

def attach_true_labels(pred_df, calltype_to_class):
    """
    reads the true CallType (falling back to "calltype" or "label" if
    CallType isn't present) out of each row's npz_path, normalizes it, and
    maps it to RF class space via calltype_to_class

    rows with no usable label get true_class_mapped="Unknown"; rows whose
    normalized label isn't a key in calltype_to_class get
    "unmapped::<raw value>" so they're easy to spot and exclude from scoring.

    returns the updated dataframe (adds true_class_raw, true_class_mapped, label_key_used)
    """
    true_labels_raw = []
    true_labels_mapped = []
    label_key_used = []

    for npz_path in pred_df["npz_path"]:
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "CallType" in data:
                raw_label = safe_scalar_str(data["CallType"])
                key_used = "CallType"
            elif "calltype" in data:
                raw_label = safe_scalar_str(data["calltype"])
                key_used = "calltype"
            elif "label" in data:
                raw_label = safe_scalar_str(data["label"])
                key_used = "label"
            else:
                raw_label = "Unknown"
                key_used = "missing"
        except Exception as e:
            raw_label = "Unknown"
            key_used = f"error: {e}"

        if raw_label == "Unknown":
            mapped_label = "Unknown"
        else:
            mapped_label = calltype_to_class.get(normalize_label(raw_label), f"unmapped::{raw_label}")

        true_labels_raw.append(raw_label)
        true_labels_mapped.append(mapped_label)
        label_key_used.append(key_used)

    out_df = pred_df.copy()
    out_df["true_class_raw"] = true_labels_raw
    out_df["true_class_mapped"] = true_labels_mapped
    out_df["label_key_used"] = label_key_used
    return out_df

def build_eval_dataframe(pred_df, output_dir):
    """
    returns a dataframe with only rows that have a usable true label (not
    "Unknown" or "unmapped::<raw value>"), and only the columns needed for
    evaluation.
    """
    true_label_summary_raw = (
        pred_df["true_class_raw"].value_counts(dropna=False)
        .rename_axis("true_class_raw").reset_index(name="n_windows")
    )
    true_label_summary_raw["fraction"] = true_label_summary_raw["n_windows"] / len(pred_df)
    logger.info("==========Raw true label summary:==========")
    summary_text = true_label_summary_raw.to_string(index=False)
    for line in summary_text.splitlines():
        logger.info("%s", line)

    true_label_summary_mapped = (
        pred_df["true_class_mapped"].value_counts(dropna=False)
        .rename_axis("true_class_mapped").reset_index(name="n_windows")
    )
    true_label_summary_mapped["fraction"] = true_label_summary_mapped["n_windows"] / len(pred_df)
    logger.info("==========Mapped true label summary:==========")
    summary_text = true_label_summary_mapped.to_string(index=False)
    for line in summary_text.splitlines():
        logger.info("%s", line)

    unmapped_df = pred_df[
        pred_df["true_class_mapped"].astype(str).str.startswith("unmapped::", na=False)
    ].copy()

    true_label_summary_raw.to_csv(os.path.join(output_dir, "rf_true_label_summary_raw.csv"), index=False)
    true_label_summary_mapped.to_csv(os.path.join(output_dir, "rf_true_label_summary_mapped.csv"), index=False)
    unmapped_df.to_csv(os.path.join(output_dir, "rf_unmapped_true_labels.csv"), index=False)

    eval_df = pred_df[
        (pred_df["true_class_mapped"] != "Unknown")
        & (~pred_df["true_class_mapped"].astype(str).str.startswith("unmapped::", na=False))
    ].copy()

    if len(eval_df) == 0:
        logger.warning("No rows with usable true labels for evaluation.")
        logger.warning("DETECTION COMPLETE: no ground truth available.")
        return None
    logger.info("Evaluable rows: %d / %d", len(eval_df), len(pred_df))
    return eval_df

def plot_bad_channel_fraction(dist_ref, frac_files_persistent, threshold, group_key, diag_dir):
    """Line plot of how often each channel was flagged persistently noisy, across sampled files for better reference."""

    safe_key = group_key.replace(" ", "_")
    fig = plt.figure(figsize=(12, 4))
    plt.plot(dist_ref / 1000.0, frac_files_persistent)
    plt.axhline(threshold, linestyle="--")
    plt.xlabel("Distance (km)")
    plt.ylabel("Fraction of sampled files persistently noisy")
    plt.title(f"Persistent bad-channel score ({group_key})")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f"persistent_bad_channels_fraction_{safe_key}.png"), dpi=200)
    plt.close(fig)

def plot_bad_channel_matrix(persistent_matrix, group_key, diag_dir):
    """Image plot of which channels were flagged hot in each sampled file  one row per file."""

    safe_key = group_key.replace(" ", "_")
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(persistent_matrix, aspect="auto", origin="lower")
    plt.xlabel("Channel index")
    plt.ylabel("Sampled file #")
    plt.title(f"Per-file persistently noisy channel flags ({group_key})")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f"persistent_bad_channels_matrix_{safe_key}.png"), dpi=200)
    plt.close(fig)

def plot_stacked_spectra(
    df, feature_key, freq_key, feature_length, group_col,
    outdir=None, max_windows_per_fig=500, fig_height=6,
    vmin=-1, vmax=5, cmap="viridis",
    title_prefix="", filename_prefix="stack", show=False,
):
    """
    in text referred to as stack spectrograms 

    essentially builds a stack of images of frequence x window of the preprocessed spectra that feed into the 
    training RF, when plotted chronologically it resembles something similar to a spectrogram view (but not here
    as we are plotting all the individual annotations over many different datasets)

    df must have an "npz_path" column; sorts windows within each page by
    group_id/window_index/npz_path so repeated calls are reproducible.

    saves each page as a PNG under outdir if given, and/or shows it inline.
    """
    saved_paths = []

    for group_val in sorted(df[group_col].dropna().unique()):
        cls_df = df[df[group_col] == group_val].copy()
        sort_cols = [c for c in ["group_id", "window_index", "npz_path"] if c in cls_df.columns]
        cls_df = cls_df.sort_values(sort_cols).reset_index(drop=True)

        n = len(cls_df)
        if n == 0:
            continue
        n_pages = int(np.ceil(n / max_windows_per_fig))

        for page in range(n_pages):
            a = page * max_windows_per_fig
            b = min((page + 1) * max_windows_per_fig, n)
            chunk = cls_df.iloc[a:b]

            freqs_ref = None
            cols = []

            for _, row in chunk.iterrows():
                try:
                    data = np.load(row["npz_path"], allow_pickle=True)

                    if feature_key not in data:
                        continue

                    spec = np.asarray(data[feature_key], dtype=float).ravel()
                    if spec.size < feature_length:
                        continue
                    spec = spec[:feature_length]

                    if freq_key in data:
                        freqs = np.asarray(data[freq_key], dtype=float).ravel()[:len(spec)]
                    else:
                        freqs = np.arange(len(spec), dtype=float)

                    if len(spec) != len(freqs):
                        continue
                    if not np.isfinite(spec).any():
                        continue

                    order = np.argsort(freqs)
                    freqs = freqs[order]
                    spec = spec[order]

                    if freqs_ref is None:
                        freqs_ref = freqs

                    spec_interp = np.interp(freqs_ref, freqs, spec, left=spec[0], right=spec[-1])
                    cols.append(spec_interp)
                except Exception:
                    continue

            if freqs_ref is None or len(cols) == 0:
                continue

            M = np.column_stack(cols)
            fig_width = max(10, 0.18 * M.shape[1])

            plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(
                M, origin="lower", aspect="auto",
                extent=[0, M.shape[1], float(freqs_ref[0]), float(freqs_ref[-1])],
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            plt.colorbar(label=feature_key)
            plt.xlabel("Window (sorted index)")
            plt.ylabel("Frequency (Hz)" if not np.array_equal(freqs_ref, np.arange(len(freqs_ref))) else "Frequency bin")
            plt.title(f"{title_prefix}{group_val}\nWindows {a + 1}-{b} of {n}")
            plt.tight_layout()

            fig_path = None
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                fig_name = f"{filename_prefix}_{group_val}_page_{page + 1}.png".replace(" ", "_")
                fig_path = os.path.join(outdir, fig_name)
                plt.savefig(fig_path, dpi=200, bbox_inches="tight")
                saved_paths.append(fig_path)

            if show:
                plt.show()
            plt.close()

    return saved_paths

def plot_fk_before_after(fk_before, fk_after, freqs, ks, calltype, seg_label, widx, png_dir, tag):
    """Save the fk spectrum of one window, before and after background removal, as two PNGs."""
    fig = plt.figure(figsize=(8, 4))
    vmin, vmax = np.nanpercentile(fk_before, [30, 99])
    plt.imshow(
        fk_before, origin="lower", aspect="auto",
        extent=[freqs.min(), freqs.max(), ks.min(), ks.max()],
        cmap="viridis", vmin=vmin, vmax=vmax,
    )
    plt.colorbar(label="dB")
    plt.title(f"FK before background removal | {calltype} | {seg_label} | w={widx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Wavenumber (1/m)")
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"{tag}_FK_before.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    vmin_after = np.nanpercentile(fk_after, 5)
    vmax_after = np.nanpercentile(fk_after, 95)
    plt.imshow(
        fk_after, origin="lower", aspect="auto",
        extent=[freqs.min(), freqs.max(), ks.min(), ks.max()],
        cmap="viridis", vmin=vmin_after, vmax=vmax_after,
    )
    plt.colorbar(label="Relative dB")
    plt.title(f"FK after background removal | {calltype} | {seg_label} | w={widx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Wavenumber (1/m)")
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"{tag}_FK_after.png"), dpi=200)
    plt.close(fig)

def plot_spectrum_detrended(freqs, spec_before, baseline, spec_detrended, calltype, seg_label, widx, png_dir, tag):
    """Save one PNG showing the raw, gaussian baseline, and detrended spectrum together for better reference."""

    fig = plt.figure(figsize=(6, 4))
    plt.plot(freqs, spec_before, label="Raw")
    plt.plot(freqs, baseline, "--", label="Baseline")
    plt.plot(freqs, spec_detrended, label="Detrended")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Median FK (dB)")
    plt.title(f"Spectrum | {calltype} | {seg_label} | w={widx}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"{tag}_spectrum_detrended.png"), dpi=200)
    plt.close(fig)

def plot_confusion_matrix(cm_df, title, outpath=None, fmt=".2f", figsize=(10, 8), show=True):
    """confusion-matrix-shaped (raw counts or normalized %), optionally saved to outpath"""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_df.values, aspect="auto", origin="upper", cmap="viridis")

    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=90)
    ax.set_yticklabels(cm_df.index)

    finite_vals = cm_df.values[np.isfinite(cm_df.values)]
    half_max = (np.nanmax(finite_vals) * 0.5) if finite_vals.size else 0.0

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            val = cm_df.iloc[i, j]
            ax.text(
                j, i,
                format(val, fmt) if pd.notna(val) else "nan",
                ha="center", va="center", fontsize=8,
                color="white" if pd.notna(val) and val > half_max else "black",
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)



