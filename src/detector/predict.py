"""last stage :) 
loads random forest saved by the trainign stage, runs it on a folder of new/unlabeled npz windows,
aggregates window-level predictions up to one prediction per (file, semgent) group, and, if annotations (CallType)
are available, evaluates predictions against that ground truth

reuses the same npz helpers and confusion matrix / metric code as the trainign stage"""

import os
from glob import glob
import numpy as np
import pandas as pd
from training.dataset import safe_scalar_str, safe_scalar_int, get_group_id_from_npz, majority_vote




def load_unknown_npz(npz_dir, feature_key, feature_length):
    """

    loads every *.npz in npz_dir with no class/energy filtering (unlike
    training's load_npz_dataset)
     
    this stage is meant to run on whatever
    windows the preprocessing step produced, known class or not.

    returns (df, skipped)
    """
    records = []
    skipped = []

    npz_paths = sorted(glob(os.path.join(npz_dir, "*.npz")))

    for npz_path in npz_paths:
        try:
            data = np.load(npz_path, allow_pickle=True)

            if feature_key not in data:
                skipped.append((os.path.basename(npz_path), f"missing {feature_key}"))
                continue

            spectrum = np.asarray(data[feature_key], dtype=float).ravel()
            if spectrum.size < feature_length:
                skipped.append((os.path.basename(npz_path), f"spectrum too short: {spectrum.size}"))
                continue
            spectrum = spectrum[:feature_length]

            group_id, file_path, segment_label = get_group_id_from_npz(data, npz_path)
            window_index = safe_scalar_int(data["window_index"], default=-1) if "window_index" in data else -1

            records.append({
                "npz_path": npz_path,
                "group_id": group_id,
                "file_path": file_path,
                "segment_label": segment_label,
                "window_index": window_index,
                "spectrum": spectrum,
            })
        except Exception as e:
            skipped.append((os.path.basename(npz_path), str(e)))

    df = pd.DataFrame(records)
    return df, skipped

def predict_windows(rf, df):
    """
    runs rf.predict / rf.predict_proba on every window in df (must have a
    "spectrum" column)
    
    adds predicted_class, max_probability, and one prob_<class> column per RF class
    
    returns the updated dataframe and the array of class names
    """
    X = np.vstack(df["spectrum"].to_numpy())

    pred_labels = rf.predict(X)
    pred_probs = rf.predict_proba(X)
    class_names = rf.classes_

    out_df = df.copy()
    out_df["predicted_class"] = pred_labels
    out_df["max_probability"] = pred_probs.max(axis=1)
    for i, cls in enumerate(class_names):
        out_df[f"prob_{cls}"] = pred_probs[:, i]

    sort_cols = [c for c in ["predicted_class", "group_id", "window_index", "npz_path"] if c in out_df.columns]
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    return out_df, class_names

def aggregate_to_groups(pred_df, class_names):
    """
    one row per group id, summarises window-level predictions:

    majority vote across windows, and argmax of the mean per class probability across windows
    (usually agree but can not agree when a group has many low-confidence windows split across classes)
    one row per group_id, summarizing window-level predictions two ways:
    """
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

        row["group_pred_mean_prob"] = class_names[np.argmax([row[f"mean_prob_{cls}"] for cls in class_names])]
        group_rows.append(row)

    return pd.DataFrame(group_rows).sort_values("group_id").reset_index(drop=True)

def normalize_label(x):
    """lowercase/strip/remove spaces+underscores, so raw CallType values line up with calltype_to_class keys"""
    x = safe_scalar_str(x).strip().lower()
    x = x.replace("_", "")
    x = x.replace(" ", "")
    return x


def normalize_label(x):
    """lowercase/strip/remove spaces+underscores, so raw CallType values line up with calltype_to_class keys"""
    x = safe_scalar_str(x).strip().lower()
    x = x.replace("_", "")
    x = x.replace(" ", "")
    return x


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