import logging

import pandas as pd
import os
import csv
from collections import defaultdict
from daswhaledetect.utils import (
    normalize_seg_text,
    parse_dt, 
    safe_scalar_str, 
    safe_scalar_int, 
    sanitize_filename, 
    clean_str, 
    parse_int_safe, 
    parse_float_safe
)
import os
from glob import glob
from collections import Counter
import numpy as np
import re
from datetime import datetime

logger = logging.getLogger("daswhaledetect.dataload")

def parse_datetime_from_path(file_path, fixed_date_if_time_only=None):
    """
    pulls datetime out of a raw file's path/name: eg:
    .../20251012/dphi/200008.hdf5 -> 2025-10-12 20:00:08
    """
    p = str(file_path)
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]

    date_dt = None

    seg_ymd = list(re.finditer(r"(?:^|[\\/])(?P<ymd>20\d{6})(?:[\\/]|$)", p))
    if seg_ymd:
        ymd = seg_ymd[-1].group("ymd")
        Y, mo, da = int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])
        try:
            date_dt = datetime(Y, mo, da, 0, 0, 0)
        except Exception:
            date_dt = None

    if date_dt is None:
        seg_ymd2 = list(re.finditer(r"(?:^|[\\/])(?P<Y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})(?:[\\/]|$)", p))
        if seg_ymd2:
            m = seg_ymd2[-1]
            Y, mo, da = int(m.group("Y")), int(m.group("m")), int(m.group("d"))
            try:
                date_dt = datetime(Y, mo, da, 0, 0, 0)
            except Exception:
                date_dt = None

    if date_dt is None:
        any_ymd = list(re.finditer(r"(20\d{2})([01]\d)([0-3]\d)", p))
        if any_ymd:
            m = any_ymd[-1]
            Y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                date_dt = datetime(Y, mo, da, 0, 0, 0)
            except Exception:
                date_dt = None

    if date_dt is None and fixed_date_if_time_only:
        date_dt = parse_dt(fixed_date_if_time_only + " 00:00:00")

    time_hms = None

    if stem.isdigit() and len(stem) == 6:
        H, M, S = int(stem[0:2]), int(stem[2:4]), int(stem[4:6])
        if 0 <= H <= 23 and 0 <= M <= 59 and 0 <= S <= 59:
            time_hms = (H, M, S)

    if time_hms is None:
        m = re.search(r"[_\-]([0-2]\d)([0-5]\d)([0-5]\d)$", stem)
        if m:
            time_hms = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if time_hms is None:
        m = re.search(r"(?<!\d)([0-2]\d)[-_]([0-5]\d)[-_]([0-5]\d)(?!\d)", stem)
        if m:
            time_hms = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if date_dt is not None and time_hms is not None:
        H, M, S = time_hms
        dt_ = datetime(date_dt.year, date_dt.month, date_dt.day, H, M, S)
        return dt_, dt_.strftime("%Y-%m-%d_%H-%M-%S")

    if date_dt is not None and time_hms is None:
        return date_dt, date_dt.strftime("%Y-%m-%d_%H-%M-%S")

    return None, sanitize_filename(stem)

def parse_segment_label_or_range(seg_value):
    """

    give a fiber-segment annotation value into a normalized {start_m, end_m}
    range  accepts "050-069km_step10", "50-70km", "50000-70000", etc.

    all annotations were a bit different so this seemed like an easy(ish) fix
    """
    s = clean_str(seg_value)
    if not s:
        return {"raw": "", "norm": "", "start_m": None, "end_m": None}

    norm = normalize_seg_text(s)

    m = re.search(r"(\d{2,3})-(\d{2,3})km", norm)
    if m:
        a, b = int(m.group(1)) * 1000, int(m.group(2)) * 1000
        return {"raw": s, "norm": norm, "start_m": min(a, b), "end_m": max(a, b)}

    m = re.search(r"(\d{4,6})[-_](\d{4,6})", norm)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {"raw": s, "norm": norm, "start_m": min(a, b), "end_m": max(a, b)}

    return {"raw": s, "norm": norm, "start_m": None, "end_m": None}

def discover_files_by_folder_and_time(folder_dir, file_type="*.hdf5", start_dt=None, end_dt=None):
    """
    finds raw DAS file directly in a folder, optionally restricts to a time range

    start_dt, end_dt : str or datetime, optional
    either a real datetime, or a string like "2025-10-12 14:00:00".
    -> leave both as None to get every matching file in the folder.
    """
    if isinstance(start_dt, str):
        start_dt = parse_dt(start_dt)
    if isinstance(end_dt, str):
        end_dt = parse_dt(end_dt)

    pattern = os.path.join(folder_dir, "**", file_type)
    candidates = sorted(glob(pattern, recursive=True))

    if start_dt is None and end_dt is None:
        return candidates

    selected = []
    for fp in candidates:
        dt_file, _ = parse_datetime_from_path(fp)
        if dt_file is None:
            continue  # can't tell when this file is from - skip it if a range was requested
        if start_dt is not None and dt_file < start_dt:
            continue
        if end_dt is not None and dt_file > end_dt:
            continue
        selected.append(fp)

    return selected

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
                skipped.append((os.path.basename(npz_path), f"missing key: {feature_key}"))
                continue

            spectrum = np.asarray(data[feature_key], dtype=float).ravel()
            if spectrum.size < feature_length:
                skipped.append((os.path.basename(npz_path), f"spectrum length: {spectrum.size} < {feature_length}"))
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

    if len(records) == 0:
        raise ValueError(f"No usable unknown NPZ files found in {npz_dir}.")

    logger.info("Usable unknown records: %d", len(df))
    logger.info("Skipped unknown records: %d", len(skipped))

    return df, skipped

def discover_list_files(list_dir, list_glob="*.csv"):
    """finds annotation-list csvs in a directory"""
    hits = sorted(glob(os.path.join(list_dir, list_glob)))
    return [p for p in hits if not os.path.basename(p).lower().startswith(".")]

def load_annotations_from_list_csv(csv_path, allow_npz_path_as_fallback_input=False):
    """
    loads one annotation list csv, grouping rows by raw file each one refers to

    expected columns:
    npz_path, original_file_path, timestamp,
    window_index, segment_start_seconds, segment_start_realtime,
    fiber_segment, CallType
    """
    annotations_by_file = defaultdict(list)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]

        if not any(c in cols for c in ("original_file_path", "npz_path")):
            raise ValueError(
                f"{csv_path} does not contain original_file_path or npz_path.\nFound columns: {cols}"
            )

        for row in reader:
            original_file_path = clean_str(row.get("original_file_path", ""))
            npz_path = clean_str(row.get("npz_path", ""))

            input_file_path = original_file_path
            if not input_file_path and allow_npz_path_as_fallback_input:
                input_file_path = npz_path
            if not input_file_path:
                continue

            ann = {
                "source_list_file": csv_path,
                "npz_path": npz_path,
                "original_file_path": original_file_path,
                "timestamp": clean_str(row.get("timestamp", "")),
                "window_index": parse_int_safe(row.get("window_index", None), default=None),
                "segment_start_seconds": parse_float_safe(row.get("segment_start_seconds", None), default=None),
                "segment_start_realtime": clean_str(row.get("segment_start_realtime", "")),
                "fiber_segment": clean_str(row.get("fiber_segment", "")),
                "fiber_segment_spec": parse_segment_label_or_range(row.get("fiber_segment", "")),
                "CallType": clean_str(row.get("CallType", "")) or "unknown",
            }
            annotations_by_file[input_file_path].append(ann)

    return annotations_by_file

def segment_matches_requested(seg_label, seg_tuple, requested_specs):
    """check built segment against requested fiber-segment specs, by text match or range overlap"""
    if not requested_specs:
        return False

    seg_start_m, seg_end_m = int(seg_tuple[0]), int(seg_tuple[1])
    seg_label_norm = normalize_seg_text(seg_label)

    for spec in requested_specs:
        if spec["norm"] and seg_label_norm == spec["norm"]:
            return True
        if spec["start_m"] is not None and spec["end_m"] is not None:
            overlap_start = max(seg_start_m, spec["start_m"])
            overlap_end = min(seg_end_m, spec["end_m"])
            overlap = max(0, overlap_end - overlap_start)
            seg_len = max(1, seg_end_m - seg_start_m)
            req_len = max(1, spec["end_m"] - spec["start_m"])
            if overlap / seg_len > 0.5 or overlap / req_len > 0.5:
                return True

    return False

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
                input(f"{feature_key} not found in npz, press enter to continue")
                skipped.append((os.path.basename(npz_path), f"missing {feature_key}"))
                continue

            if "CallType" not in data:
                input("CallType not found in npz, press enter to continue")
                skipped.append((os.path.basename(npz_path), "missing CallType"))
                continue

            spectrum = np.asarray(data[feature_key], dtype=float).ravel()
            if spectrum.size < feature_length:
                input(f"spectrum too short: {spectrum.size}, press enter to continue")
                skipped.append((os.path.basename(npz_path), f"spectrum too short: {spectrum.size}"))
                continue
            spectrum = spectrum[:feature_length]

            raw_label = safe_scalar_str(data["CallType"]).lower()

            if raw_label in {"", "unknown", "none", "nan", "unlabeled"}:
                input(f"bad CallType: {raw_label}, press enter to continue")
                skipped.append((os.path.basename(npz_path), f"bad CallType: {raw_label}"))
                continue

            if raw_label not in allowed_classes_raw:
                input(f"CallType not in allowed_classes_raw: {raw_label}, press enter to continue")
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

    
    if df.empty:
        raise ValueError(
            "No usable NPZ files remained after filtering, grouping, and class gating."
        )


    logger.info("Loaded %d usable windows.", len(df))
    logger.info("Skipped %d NPZ files.", len(skipped))

    logger.info("")
    logger.info("Grouped class counts after gating:")
    logger.info("%-25s %10s", "class", "n_windows")

    for class_name, count in df["CallType_grouped"].value_counts().items():
        logger.info("%-25s %10d", class_name, count)

    logger.info("")
    logger.info("Raw class counts after gating:")
    logger.info("%-25s %10s", "class", "n_windows")
    for class_name, count in df["CallType_raw"].value_counts().items():
        logger.info("%-25s %10d", class_name, count)


    return df, skipped

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
            logger.info("")
            logger.info(f"Sampled {n_take} from {len(cls_df)} total for class '{cls}'")
            logger.info("Raw subclass counts in sample:")
            logger.info("%-25s %10s", "class", "n_windows")
            for class_name, count in sampled_df["CallType_raw"].value_counts().items():
                logger.info("%-25s %10d", class_name, count)
            

    noise_df = df[df["CallType_grouped"] == "noise"].copy()
    if len(noise_df) > 0:
        if verbose:
            logger.info("")
            logger.info("Raw noise subclass counts before balancing:")
            logger.info("%-25s %10s", "class", "n_windows")
            for class_name, count in noise_df["CallType_raw"].value_counts().items():
                logger.info("%-25s %10d", class_name, count)

        noise_counts = noise_df["CallType_raw"].value_counts()
        n_take_per_noise_subclass = min(noise_counts.min(), max_per_class // len(noise_counts))

        noise_parts = []
        for raw_noise_cls, sub_df in noise_df.groupby("CallType_raw"):
            sampled_sub_df = sub_df.sample(n=n_take_per_noise_subclass, random_state=random_state)
            noise_parts.append(sampled_sub_df)

        balanced_noise_df = pd.concat(noise_parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
        balanced_parts.append(balanced_noise_df)

        if verbose:
            logger.info("")
            logger.info("Selected %d samples from each noise subtype", n_take_per_noise_subclass)
            logger.info("Balanced noise subclass counts used:")
            logger.info("%-25s %10s", "class", "n_windows")
            for class_name, count in balanced_noise_df["CallType_raw"].value_counts().items():
                logger.info("%-25s %10d", class_name, count)

    train_df = pd.concat(balanced_parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.info(
        "Total windows passing gates: %d",
        len(df),
    )

    logger.info(
        "Total windows after balancing: %d",
        len(train_df),
    )

    logger.info("")
    logger.info("Balanced grouped class counts:")
    logger.info("%-25s %10s", "class", "n_windows")
    for class_name, count in train_df["CallType_grouped"].value_counts().items():
        logger.info("%-25s %10d", class_name, count)


    return train_df
