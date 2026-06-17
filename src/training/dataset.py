"""loads npz feature files produced by the preprocessing pipeline into a labaled dataframe, groups raw CallType labels into
broader training classes, applies per-class energy thresholds, balances class counts before training
"""

import os
from glob import glob
from collections import Counter
import numpy as np
import pandas as pd


#low level npz field readers
#reads a value from npz field as plain python string, however it was stored (0-d array, array, scalar)
def safe_scalar_str(x):
    try:
        arr = np.array(x)
        if arr.shape == ():
            return str(arr.item()).strip()
        return str(x).strip()
    except Exception:
        return str(x).strip()


#same idea as above but for ints
def safe_scalar_int(x, default=-1):
    try:
        arr = np.array(x)
        if arr.shape == ():
            return int(arr.item())
        return int(x)
    except Exception:
        return default


def get_group_id_from_npz(data, fallback_path):

#  builds a group id (one DAS detection window groups together with the rest of its same file+segmet) from
# whichever of file_path/segment_label are present, falling ack to the npz path itself if neither is available
    file_path = ""
    segment_label = ""

    if "file_path" in data:
        file_path = safe_scalar_str(data["file_path"])

    if "segment_label" in data:
        segment_label = safe_scalar_str(data["segment_label"])

    if file_path and segment_label:
        return f"{file_path}__{segment_label}", file_path, segment_label
    elif file_path:
        return file_path, file_path, segment_label
    elif segment_label:
        return segment_label, file_path, segment_label
    else:
        return fallback_path, file_path, segment_label


def majority_vote(series): # most common value in round of predicted labels, used for group-level aggregation
    counts = Counter(series)
    return counts.most_common(1)[0][0]


def sanitize_filename(text): # take away characters that wouldnt for for use in output paths
    text = str(text)
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for ch in bad:
        text = text.replace(ch, "_")
    return text.replace(" ", "_")


# building the labeled training table
def load_npz_dataset(
    npz_dirs,
    feature_key,
    feature_length,
    allowed_classes_raw,
    sweep_source_classes,
    pulse_source_classes,
    noise_source_classes,
    class_gates,
):
    """
    scans npz_dirs for *.npz files, keeps only ones with a usable feature
    vector and a recognised CallType, groups the raw CallType into a broader
    training class (sweeps/pulses/noise/etc, via the *_source_classes sets),
    and applies per-grouped-class energy thresholds from "class_gates"

    returns (df, skipped):
      df      - one row per usable window, columns: npz_path, CallType_raw,
                CallType_grouped, energy_proxy, group_id, window_index, spectrum
      skipped - list of (filename, reason) for anything dropped along the way
    """
    records = []
    skipped = []

    all_npz_files = []
    for d in npz_dirs:
        all_npz_files.extend(glob(os.path.join(d, "*.npz")))
    all_npz_files = sorted(set(all_npz_files))

    for npz_path in all_npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)

            if feature_key not in data:
                skipped.append((os.path.basename(npz_path), f"missing {feature_key}"))
                continue

            if "CallType" not in data:
                skipped.append((os.path.basename(npz_path), "missing CallType"))
                continue

            spectrum = np.asarray(data[feature_key], dtype=float).ravel()
            if spectrum.size < feature_length:
                skipped.append((os.path.basename(npz_path), f"spectrum too short: {spectrum.size}"))
                continue
            spectrum = spectrum[:feature_length]

            raw_label = safe_scalar_str(data["CallType"]).lower()

            if raw_label in {"", "unknown", "none", "nan", "unlabeled"}:
                skipped.append((os.path.basename(npz_path), f"bad CallType: {raw_label}"))
                continue

            if raw_label not in allowed_classes_raw:
                skipped.append((os.path.basename(npz_path), f"CallType not in allowed_classes_raw: {raw_label}"))
                continue

            # group raw label into the broader training class
            if raw_label in sweep_source_classes:
                grouped_label = "sweeps"
            elif raw_label in pulse_source_classes:
                grouped_label = "pulses"
            elif raw_label in noise_source_classes:
                grouped_label = "noise"
            else:
                grouped_label = raw_label

            try:
                energy_value = float(np.nanmax(spectrum))
            except Exception:
                energy_value = np.nan

            # apply energy gate on the grouped label, if one is defined
            if grouped_label in class_gates:
                gate = class_gates[grouped_label]
                if not np.isfinite(energy_value):
                    skipped.append((os.path.basename(npz_path), "non-finite energy_proxy"))
                    continue
                if "min" in gate and energy_value < gate["min"]:
                    skipped.append((os.path.basename(npz_path), f"below gate min for {grouped_label}: {energy_value:.3f}"))
                    continue
                if "max" in gate and energy_value > gate["max"]:
                    skipped.append((os.path.basename(npz_path), f"above gate max for {grouped_label}: {energy_value:.3f}"))
                    continue

            group_id, file_path, segment_label = get_group_id_from_npz(data, npz_path)
            window_index = safe_scalar_int(data["window_index"], default=-1) if "window_index" in data else -1

            records.append({
                "npz_path": npz_path,
                "CallType_raw": raw_label,
                "CallType_grouped": grouped_label,
                "energy_proxy": energy_value,
                "group_id": group_id,
                "window_index": window_index,
                "spectrum": spectrum,
            })

        except Exception as e:
            skipped.append((os.path.basename(npz_path), str(e)))

    df = pd.DataFrame(records)
    return df, skipped


# balancing classes after gating
def balance_classes(df, max_per_class, random_state=42, verbose=True):
    """
    randomly downsamples each grouped class to at most max_per_class rows 
    
    
    "noise" is special (first balanced across its raw subclasses in order to keep the same number of samples 
    for ambient noise, ship noise, earthquakes, otherwise we'd end up with mostly background noise drowning out the other examples), 
    and then capped to fit within the overall max samples per class
    
    returns the combined, shuffled training dataframe
    """
    balanced_parts = []

    for cls, cls_df in df[df["CallType_grouped"] != "noise"].groupby("CallType_grouped"):
        n_take = min(max_per_class, len(cls_df))
        sampled_df = cls_df.sample(n=n_take, random_state=random_state)
        balanced_parts.append(sampled_df)
        if verbose:
            print(f"\n{cls}: sampled {n_take} from {len(cls_df)} total")
            print("Raw subclass counts in sample:")
            print(sampled_df["CallType_raw"].value_counts())

    noise_df = df[df["CallType_grouped"] == "noise"].copy()
    if len(noise_df) > 0:
        if verbose:
            print("\nRaw noise subclass counts before balancing:")
            print(noise_df["CallType_raw"].value_counts())

        noise_counts = noise_df["CallType_raw"].value_counts()
        n_take_per_noise_subclass = min(noise_counts.min(), max_per_class // len(noise_counts))

        noise_parts = []
        for raw_noise_cls, sub_df in noise_df.groupby("CallType_raw"):
            sampled_sub_df = sub_df.sample(n=n_take_per_noise_subclass, random_state=random_state)
            noise_parts.append(sampled_sub_df)

        balanced_noise_df = pd.concat(noise_parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
        balanced_parts.append(balanced_noise_df)

        if verbose:
            print(f"\nSelected {n_take_per_noise_subclass} samples from each noise subtype")
            print("Balanced noise subclass counts used:")
            print(balanced_noise_df["CallType_raw"].value_counts())

    train_df = pd.concat(balanced_parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df