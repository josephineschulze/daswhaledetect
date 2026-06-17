"""evaluation helpers - compute confusion matrices, per-class metrics, stacked spectra and evalutes
predictions against ground truth"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize

def compute_confusion_matrices(y_true, y_pred, labels):
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
    )
    row_pct = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0) * 100 #recall confusion matrix
    col_pct = cm_df.div(cm_df.sum(axis=0).replace(0, np.nan), axis=1) * 100 # precision confusion matrix
    return cm_df, row_pct, col_pct

def compute_per_class_metrics(cm_df, labels):
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
    return pd.DataFrame(class_metrics).T


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


def compute_oob_average_precision(y_oob_true, oob_probs_valid, rf_classes):
    """
    per-class average precision from OOB probabilities (one-vs-rest), plus
    the macro average across classes. returns (ap_scores_dict, macro_ap)
    """
    y_true_bin = label_binarize(y_oob_true, classes=rf_classes)
    ap_scores = {}
    for i, cls in enumerate(rf_classes):
        ap_scores[cls] = float(average_precision_score(y_true_bin[:, i], oob_probs_valid[:, i]))
    macro_ap = float(np.mean(list(ap_scores.values())))
    return ap_scores, macro_ap

def plot_stacked_spectra(
    df, feature_key, freq_key, feature_length, group_col,
    outdir=None, max_windows_per_fig=500, fig_height=6,
    vmin=-1, vmax=5, cmap="viridis",
    title_prefix="", filename_prefix="stack", show=True,
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

