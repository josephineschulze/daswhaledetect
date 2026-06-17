"""Figure out which raw files to process (based on my needs before): 

- from annotation-list csv (discover_list_files + load_annotaitions_from_list_csv)
- directly from a folder, optionally narrowing it to a date/time range (discover_files_by_folder_and_time) for processing a chunk
of a day instead of all of it at once"""

import os
import re
import csv
import glob
from collections import defaultdict
from datetime import datetime



def sanitize_filename(s, max_len=220):
# turns arbitrary string into smoehting usable as a filename
    s = str(s)
    s = s.replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    return s[:max_len]

def clean_str(x):
    """Coerce anything, including None, into a stripped string."""
    if x is None:
        return ""
    return str(x).strip()


def parse_int_safe(x, default=None):
    """Parse something into an int, returning `default` instead of raising on failure."""
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x).strip()))
    except Exception:
        return default



def parse_float_safe(x, default=None):
    """Parse something into a float, returning 'default' instead of running into issues"""
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default


def _parse_dt(s):
    """Try a list of common datetime string formats until one matches"""
    if s is None:
        return None
    s = str(s).strip().replace("T", " ")
    fmts = [
        "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M",
        "%Y%m%d %H%M%S", "%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S",
        "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H-%M-%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
    


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
        date_dt = _parse_dt(fixed_date_if_time_only + " 00:00:00")

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


def normalize_seg_text(s):
    """Strip spaces/underscores and lowercase for easier comparison"""
    if s is None:
        return ""
    return str(s).strip().replace(" ", "").replace("_", "").lower()


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


def discover_list_files(list_dir, list_glob="*.csv"):
    """finds annotation-list csvs in a directory"""
    hits = sorted(glob.glob(os.path.join(list_dir, list_glob)))
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


def discover_files_by_folder_and_time(folder, file_glob="*.hdf5", start_dt=None, end_dt=None):
    """
    finds raw DAS file directly in a folder, optionally restricts to a time range

    start_dt, end_dt : str or datetime, optional
    either a real datetime, or a string like "2025-10-12 14:00:00".
    -> leave both as None to get every matching file in the folder.
    """
    if isinstance(start_dt, str):
        start_dt = _parse_dt(start_dt)
    if isinstance(end_dt, str):
        end_dt = _parse_dt(end_dt)

    pattern = os.path.join(folder, "**", file_glob)
    candidates = sorted(glob.glob(pattern, recursive=True))

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