""" stage 1 - preprocessing
reads raw DAS hdf5 file, matches windows against annotation list (in list mode, for training data) or 
processes every window directly without requiring annotations (folder mode, for new/unlabeled data headed to the detector stage 3)

extracts the fk based detrended spectrum feature or ach window, saved png diagnostics + npz outputs

multiprocessing pool parallelises across files
"""

import os
import csv
import random
import traceback
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict

import numpy as np

from preprocessing import data_handle, dsp, plot, data_load

# RUN MODE
RUN_MODE = "folder"  # "list" or "folder"

# CONFIG
CHANNEL_RANGE = [30000, 90000, 2]   # (start_m, end_m, step_channels) #sampling every 16ish m
SEGMENT_LENGTH_M = 20_000
MIN_CHANNELS_PER_SEGMENT = 8

WINDOW_LENGTH_S = 2.0
WINDOW_OVERLAP = 0.5

FREQ_MIN = 14
FREQ_MAX = 80

VEL_MIN = 1400
VEL_MAX = 6000

SPECTRAL_BASELINE_HZ = 10.0
N_WORKERS = 3

SUPPRESS_BAD_CHANNELS = True
SUPPRESS_MODE = "interp"   # "zero" or "interp" - interp replaces with neighbouring channel values
APPLY_BANDPASS = True
BP_ORDER = 4
REMOVE_COMMON_MODE = True
REMOVE_CHANNEL_MEAN = True

TARGET_FS = 256.0

# bad-channel estimation settings
N_SAMPLE_FILES_PER_DAY = 50
RANDOM_SEED = 42
BLOCK_LENGTH_S = 1
PER_BLOCK_Z_THRESH = 5.0
PER_FILE_BLOCK_FRACTION_THRESH = 0.6
ACROSS_FILE_FRACTION_THRESH = 0.25

# Save a PNG (FK before/after + detrended spectrum) per saved window, in
# addition to the NPZ. Fine for list mode (only annotated windows get
# saved). In folder mode EVERY window gets saved, so a full day can produce
# tens of thousands of PNGs -- set this False for folder-mode runs unless
# you actually want to eyeball every window.
SAVE_PNGS = True

# LIST MODE INPUTS
LIST_DIR = "/Volumes/Extreme Pro/Chapter1_publication_files/annotations"
LIST_GLOB = "*.csv"
ALLOW_NPZ_PATH_AS_FALLBACK_INPUT = False

# FOLDER MODE INPUTS -- fill these in if/when you use RUN_MODE = "folder"
FOLDER_PATH = "/Volumes/Extreme Pro/Chapter1_publication_files/testdata/data/20260320/dphi"
FOLDER_GLOB = "*.hdf5"
FOLDER_START_DT = None   # e.g. "2025-10-12 14:00:00", or None for no lower bound
FOLDER_END_DT = None     # e.g. "2025-10-12 18:00:00", or None for no upper bound
FOLDER_RUN_LABEL = "folder_run"   # name used for this run's output subfolder

# OUTPUT
# Each list file (list mode) or each folder run (folder mode) gets its own
# subfolder inside here, containing png/, npz/, bad_channels/, diagnostics/.
OUTPUT_ROOT = "/Users/josephsc/Desktop/test_package"


# BAD-CHANNEL ESTIMATION (per day/group)

def estimate_bad_channels_for_group(group_key, files_for_group, bad_dir, diag_dir):
    """
    samples up to N_SAMPLE_FILES_PER_DAY files from this group (day), scores
    every channel in 1s blocks, and flags channels that are persistently
    noisy across most of the sampled files. saves a txt list + npz summary +
    two diagnostic pngs.
    """
    rng = random.Random(RANDOM_SEED)
    sample_files = rng.sample(files_for_group, min(N_SAMPLE_FILES_PER_DAY, len(files_for_group)))

    print(f"[bad-ch] {group_key}: sampling {len(sample_files)} files")

    per_file_persistent = []
    dist_ref = None

    for f in sample_files:
        md = data_handle.get_acquisition_parameters(str(f))
        selected = data_handle.select_channels(md["dx"], CHANNEL_RANGE)
        trace, tx, dist = data_handle.load_das_data(str(f), selected, md)

        orig_fs = float(md["fs"])
        trace, tx, fs = data_handle.resample_trace_to_target_fs(trace, tx, orig_fs, TARGET_FS)

        zmat = data_handle.channel_block_scores(trace, fs, BLOCK_LENGTH_S)
        hot_blocks = zmat > PER_BLOCK_Z_THRESH
        frac_hot_blocks = hot_blocks.mean(axis=1)
        persistent_this_file = frac_hot_blocks >= PER_FILE_BLOCK_FRACTION_THRESH
        per_file_persistent.append(persistent_this_file.astype(np.uint8))

        if dist_ref is None:
            dist_ref = dist.copy()

    persistent_matrix = np.vstack(per_file_persistent)
    frac_files_persistent = persistent_matrix.mean(axis=0)
    bad_idx = np.where(frac_files_persistent >= ACROSS_FILE_FRACTION_THRESH)[0]

    safe_group_key = data_load.sanitize_filename(group_key)
    txt_path = os.path.join(bad_dir, f"bad_channels_{safe_group_key}.txt")
    with open(txt_path, "w") as f:
        for idx in bad_idx:
            f.write(f"{idx}\t{dist_ref[idx]:.3f}\n")

    np.savez_compressed(
        os.path.join(bad_dir, f"bad_channels_{safe_group_key}_summary.npz"),
        bad_channel_indices=bad_idx.astype(np.int32),
        bad_channel_dist_m=dist_ref[bad_idx].astype(np.float32),
        frac_files_persistent=frac_files_persistent.astype(np.float32),
        dist_m=dist_ref.astype(np.float32),
        n_sample_files=len(sample_files),
        block_length_s=float(BLOCK_LENGTH_S),
        per_block_z_thresh=float(PER_BLOCK_Z_THRESH),
        per_file_block_fraction_thresh=float(PER_FILE_BLOCK_FRACTION_THRESH),
        across_file_fraction_thresh=float(ACROSS_FILE_FRACTION_THRESH),
        target_fs=float(TARGET_FS),
    )

    plot.plot_bad_channel_fraction(dist_ref, frac_files_persistent, ACROSS_FILE_FRACTION_THRESH, group_key, diag_dir)
    plot.plot_bad_channel_matrix(persistent_matrix, group_key, diag_dir)

    return bad_idx.astype(np.int32), txt_path


# PER-WINDOW FEATURE EXTRACTION (shared by list mode and folder mode)

def compute_window_features(trace_window, fs, ks):
    """
    fft2 -> fk in dB -> trim to [FREQ_MIN, FREQ_MAX] -> remove the
    frequency-dependent background trend (estimated from outside the
    velocity band) -> median spectrum inside the velocity band -> detrend.

    identical for list mode and folder mode, so an NPZ produced either way
    carries exactly the same feature definition.
    """
    nt_win = trace_window.shape[1]

    fk = np.fft.fftshift(np.fft.fft2(trace_window), axes=(0, 1))
    fk_db = 10 * np.log10(np.abs(fk) + 1e-12)

    freqs_full = np.fft.fftshift(np.fft.fftfreq(nt_win, d=1 / fs))
    freq_mask = (freqs_full >= FREQ_MIN) & (freqs_full <= FREQ_MAX)
    freqs = freqs_full[freq_mask]
    fk_trimmed = fk_db[:, freq_mask]

    fk_clean, vel_mask, bg_spec = dsp.background_removal_fk(fk_trimmed, ks, freqs, VEL_MIN, VEL_MAX)

    fk_before = np.where(vel_mask, fk_trimmed, np.nan)
    fk_after = np.where(vel_mask, fk_clean, np.nan)

    spec_before = np.nanmedian(fk_before, axis=0)
    spec_after = np.nanmedian(fk_after, axis=0)
    spec_detrended, baseline = dsp.detrend_spectrum(spec_after, freqs, SPECTRAL_BASELINE_HZ)

    return {
        "freqs": freqs,
        "freq_mask": freq_mask,
        "fk_before": fk_before,
        "fk_after": fk_after,
        "vel_mask": vel_mask,
        "bg_spec": bg_spec,
        "spec_before": spec_before,
        "spec_after": spec_after,
        "spec_detrended": spec_detrended,
        "baseline": baseline,
    }


def save_window_outputs(tag, feats, ks, fs, original_fs, time_axis, w_start, w_end, widx,
                         calltype, fiber_segment, seg_label, seg_tuple, dist_seg,
                         n_channels, dx, selected_channels, bad_idx, group_key,
                         file_path, original_file_path, source_npz_path, source_list_file,
                         timestamp, seg_start_seconds, seg_start_realtime,
                         png_dir, npz_dir, duration_s):
    """save the PNG diagnostics (optional) + the NPZ feature file for one window/annotation."""

    if SAVE_PNGS:
        plot.plot_fk_before_after(
            feats["fk_before"], feats["fk_after"], feats["freqs"], ks,
            calltype, seg_label, widx, png_dir, tag,
        )
        plot.plot_spectrum_detrended(
            feats["freqs"], feats["spec_before"], feats["baseline"],
            feats["spec_detrended"], calltype, seg_label, widx, png_dir, tag,
        )

    np.savez_compressed(
        os.path.join(npz_dir, f"{tag}_fk_processed.npz"),
        fk_db_before=feats["fk_before"].astype(np.float32),
        fk_db_after=feats["fk_after"].astype(np.float32),
        freqs=feats["freqs"].astype(np.float32),
        ks=ks.astype(np.float32),
        time_axis=time_axis.astype(np.float32),
        freq_mask=feats["freq_mask"],
        velocity_mask=feats["vel_mask"],
        spectrum_before=feats["spec_before"].astype(np.float32),
        spectrum_after=feats["spec_after"].astype(np.float32),
        spectrum_after_detrended=feats["spec_detrended"].astype(np.float32),
        spectrum_baseline=feats["baseline"].astype(np.float32),
        background_spectrum=feats["bg_spec"].astype(np.float32),
        fs=float(fs),
        original_fs=float(original_fs),
        target_fs=float(TARGET_FS),
        nt=time_axis.shape[0],
        duration_s=float(duration_s),
        dx=dx,
        channel_indices=selected_channels,
        n_channels=n_channels,
        bad_channel_indices=bad_idx.astype(np.int32),
        day_key=str(group_key),
        group_key=str(group_key),
        file_path=file_path,
        original_file_path=str(original_file_path),
        source_npz_path=str(source_npz_path),
        source_list_file=str(source_list_file),
        timestamp=str(timestamp),
        CallType=str(calltype),
        species=str(calltype),  # kept for compatibility
        fiber_segment=str(fiber_segment),
        requested_fiber_segment=str(fiber_segment),
        channel_range=CHANNEL_RANGE,
        freq_range=[FREQ_MIN, FREQ_MAX],
        velocity_range=[VEL_MIN, VEL_MAX],
        window_index=int(widx),
        window_start_s=float(w_start / fs),
        window_end_s=float(w_end / fs),
        window_length_s=float(WINDOW_LENGTH_S),
        window_overlap=float(WINDOW_OVERLAP),
        segment_label=str(seg_label),
        segment_m=np.array(seg_tuple, dtype=np.int64),
        segment_length_m=int(SEGMENT_LENGTH_M),
        dist_segment=dist_seg.astype(np.float32),
        annotation_segment_start_seconds=np.nan if seg_start_seconds is None else float(seg_start_seconds),
        annotation_segment_start_realtime=str(seg_start_realtime),
    )


# MULTIPROCESSING GLOBALS

G_RUN_MODE = None
G_TASKS_BY_FILE = None
G_PNG_DIR = None
G_NPZ_DIR = None
G_BAD_CHANNELS_BY_GROUP = None


def pool_init(run_mode, tasks_by_file, png_dir, npz_dir, bad_channels_by_group):
    global G_RUN_MODE, G_TASKS_BY_FILE, G_PNG_DIR, G_NPZ_DIR, G_BAD_CHANNELS_BY_GROUP
    G_RUN_MODE = run_mode
    G_TASKS_BY_FILE = tasks_by_file
    G_PNG_DIR = png_dir
    G_NPZ_DIR = npz_dir
    G_BAD_CHANNELS_BY_GROUP = bad_channels_by_group


# PER-FILE PROCESSING

def process_one_file(file_path):
    try:
        if not os.path.exists(file_path):
            return (file_path, "missing", "File does not exist")

        if G_RUN_MODE == "list":
            annotations = G_TASKS_BY_FILE.get(file_path, [])
            if not annotations:
                return (file_path, "skipped", "No annotations for this file")
        else:
            annotations = None  # folder mode: every window gets kept, no annotation lookup needed

        dt_file, dt_tag = data_load.parse_datetime_from_path(file_path)
        group_key = dt_file.strftime("%Y%m%d") if dt_file is not None else "unknown"
        timestamp_file = dt_tag if dt_tag else "unknownDate_unknownTime"

        bad_idx = G_BAD_CHANNELS_BY_GROUP.get(group_key, np.array([], dtype=np.int32))

        print(
            f"\n[PID {os.getpid()}] Processing {file_path} "
            f"| mode={G_RUN_MODE} | group={group_key} | badch={len(bad_idx)}",
            flush=True,
        )

        metadata = data_handle.get_acquisition_parameters(file_path)
        selected_channels = data_handle.select_channels(metadata["dx"], CHANNEL_RANGE)

        trace_full, tx, dist_full = data_handle.load_das_data(file_path, selected_channels, metadata)

        original_fs = float(metadata["fs"])

        if SUPPRESS_BAD_CHANNELS:
            trace_full = data_handle.suppress_bad_channels(trace_full, bad_idx, mode=SUPPRESS_MODE)

        dx = float(metadata["dx"])
        n_channels_full, nt_original = trace_full.shape
        duration_s = nt_original / original_fs

        seg_defs = data_handle.build_segments_from_dist(
            dist_m=dist_full,
            requested_range_m=CHANNEL_RANGE,
            seg_len_m=SEGMENT_LENGTH_M,
            min_channels=MIN_CHANNELS_PER_SEGMENT,
        )
        if not seg_defs:
            return (file_path, "skipped", "No valid distance segments inside requested range")

        # resample the FULL trace to TARGET_FS before any windowing/filtering
        trace_full, tx, fs = data_handle.resample_trace_to_target_fs(trace_full, tx, original_fs, TARGET_FS)

        n_channels_full, nt = trace_full.shape
        win_samples = int(WINDOW_LENGTH_S * fs)
        hop_samples = int(win_samples * (1 - WINDOW_OVERLAP))
        if win_samples <= 1 or hop_samples <= 0 or nt < win_samples:
            return (file_path, "skipped", f"Too short for windowing: nt={nt}, win_samples={win_samples}")

        if G_RUN_MODE == "list":
            ann_by_window = defaultdict(list)
            for ann in annotations:
                w = ann.get("window_index", None)
                if w is not None:
                    ann_by_window[w].append(ann)
            if not ann_by_window:
                return (file_path, "skipped", "No usable window_index values in annotations")

        saved_count = 0

        for slc, seg_tuple, seg_label in seg_defs:
            trace_seg_full = trace_full[slc, :]
            dist_seg = dist_full[slc]
            n_channels = trace_seg_full.shape[0]

            if n_channels < MIN_CHANNELS_PER_SEGMENT:
                continue

            dx_eff = selected_channels[2] * dx
            ks = np.fft.fftshift(np.fft.fftfreq(n_channels, d=dx_eff))

            # common-mode removal + channel-mean removal + bandpass, applied to the
            # FULL segment trace (all time samples) BEFORE slicing into windows --
            # gives sosfiltfilt enough context to settle cleanly, so every window
            # (including the last) sees artifact-free samples at its edges.
            if REMOVE_COMMON_MODE:
                trace_seg_full = trace_seg_full - np.median(trace_seg_full, axis=0, keepdims=True)
            if REMOVE_CHANNEL_MEAN:
                trace_seg_full = trace_seg_full - np.mean(trace_seg_full, axis=1, keepdims=True)
            if APPLY_BANDPASS:
                trace_seg_full = dsp.bandpass_time_domain(trace_seg_full, fs, FREQ_MIN, FREQ_MAX, order=BP_ORDER)

            for w_start, w_end, widx in dsp.sliding_windows(nt, win_samples, hop_samples):

                if G_RUN_MODE == "list":
                    anns_this_window = ann_by_window.get(widx, [])
                    if not anns_this_window:
                        continue

                    # keep only annotations whose requested fiber segment matches this segment
                    anns_matching_segment = [
                        ann for ann in anns_this_window
                        if data_load.segment_matches_requested(
                            seg_label=seg_label,
                            seg_tuple=seg_tuple,
                            requested_specs=[ann["fiber_segment_spec"]],
                        )
                    ]
                    if not anns_matching_segment:
                        continue
                else:
                    # folder mode: one unlabeled "pseudo-annotation" per window so the
                    # save loop below stays identical to list mode
                    anns_matching_segment = [{
                        "CallType": "unknown",
                        "fiber_segment": seg_label,
                        "segment_start_seconds": None,
                        "segment_start_realtime": "",
                        "timestamp": "",
                        "original_file_path": file_path,
                        "npz_path": "",
                        "source_list_file": "",
                    }]

                trace = trace_seg_full[:, w_start:w_end].astype(np.float32)
                time_axis = np.arange(w_start, w_end) / fs

                feats = compute_window_features(trace, fs, ks)

                # save once per annotation row so CallType/fiber segment stay attached exactly
                for ann_i, ann in enumerate(anns_matching_segment):
                    calltype = ann.get("CallType", "unknown")
                    fiber_segment = ann.get("fiber_segment", "")
                    seg_start_seconds = ann.get("segment_start_seconds", None)
                    seg_start_realtime = ann.get("segment_start_realtime", "")
                    timestamp_ann = ann.get("timestamp", "") or timestamp_file

                    tag_raw = (
                        f"{timestamp_ann}"
                        f"__CT-{calltype}"
                        f"__FS-{fiber_segment}"
                        f"__SEG-{seg_label}"
                        f"__W-{widx:04d}"
                        f"__A-{ann_i:02d}"
                    )
                    tag = data_load.sanitize_filename(tag_raw)

                    save_window_outputs(
                        tag=tag, feats=feats, ks=ks, fs=fs, original_fs=original_fs,
                        time_axis=time_axis, w_start=w_start, w_end=w_end, widx=widx,
                        calltype=calltype, fiber_segment=fiber_segment, seg_label=seg_label,
                        seg_tuple=seg_tuple, dist_seg=dist_seg, n_channels=n_channels, dx=dx,
                        selected_channels=selected_channels, bad_idx=bad_idx, group_key=group_key,
                        file_path=file_path,
                        original_file_path=ann.get("original_file_path", file_path),
                        source_npz_path=ann.get("npz_path", ""),
                        source_list_file=ann.get("source_list_file", ""),
                        timestamp=timestamp_ann,
                        seg_start_seconds=seg_start_seconds,
                        seg_start_realtime=seg_start_realtime,
                        png_dir=G_PNG_DIR, npz_dir=G_NPZ_DIR, duration_s=duration_s,
                    )

                    saved_count += 1
                    print(f"[PID {os.getpid()}] Saved -> {tag}", flush=True)

        if saved_count == 0:
            return (file_path, "skipped", "No matching segment/window combinations were found")

        return (file_path, "ok", f"Saved {saved_count} outputs")

    except Exception as e:
        tb = traceback.format_exc()
        try:
            err_name = f"ERROR_{data_load.sanitize_filename(os.path.basename(str(file_path)))}_{os.getpid()}.txt"
            with open(os.path.join(G_NPZ_DIR, err_name), "w") as f:
                f.write(f"File: {file_path}\n\n{tb}\n")
        except Exception:
            pass
        return (file_path, "error", f"{e}\n{tb}")


# SHARED PIPELINE RUNNER (bad-channel estimation + multiprocessing pool)

def run_files_through_pipeline(run_mode, valid_files, tasks_by_file, output_base, run_label):
    png_dir = os.path.join(output_base, "png")
    npz_dir = os.path.join(output_base, "npz")
    bad_dir = os.path.join(output_base, "bad_channels")
    diag_dir = os.path.join(output_base, "diagnostics")
    for d in (output_base, png_dir, npz_dir, bad_dir, diag_dir):
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"PROCESSING: {run_label}  (mode={run_mode})")
    print(f"OUTPUT BASE: {output_base}")
    print(f"Files to process: {len(valid_files)}")
    print("=" * 80)

    # group files by day for bad-channel estimation
    files_by_group = defaultdict(list)
    for fp in valid_files:
        dt_file, _ = data_load.parse_datetime_from_path(fp)
        group_key = dt_file.strftime("%Y%m%d") if dt_file is not None else "unknown"
        files_by_group[group_key].append(fp)

    bad_channels_by_group = {}
    for group_key, fps in sorted(files_by_group.items()):
        bad_idx, txt_path = estimate_bad_channels_for_group(group_key, fps, bad_dir, diag_dir)
        bad_channels_by_group[group_key] = bad_idx
        print(f"[bad-ch] {group_key}: {len(bad_idx)} channels -> {txt_path}")

    print(f"\nStarting multiprocessing with {N_WORKERS} worker processes...", flush=True)
    print(f"All traces will be resampled to {TARGET_FS:.3f} Hz BEFORE windowing", flush=True)

    with Pool(
        processes=N_WORKERS,
        initializer=pool_init,
        initargs=(run_mode, tasks_by_file, png_dir, npz_dir, bad_channels_by_group),
    ) as pool:
        results = pool.map(process_one_file, valid_files, chunksize=1)

    ok = sum(1 for _, s, _ in results if s == "ok")
    err = sum(1 for _, s, _ in results if s == "error")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    missing = sum(1 for _, s, _ in results if s == "missing")

    summary_txt = os.path.join(output_base, f"{data_load.sanitize_filename(run_label)}_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Run label: {run_label}\n")
        f.write(f"Run mode: {run_mode}\n")
        f.write(f"Output base: {output_base}\n")
        f.write(f"OK: {ok}\n")
        f.write(f"Skipped: {skipped}\n")
        f.write(f"Missing: {missing}\n")
        f.write(f"Errors: {err}\n\n")
        for fp, status, msg in results:
            f.write(f"{status}\t{fp}\t{msg}\n")

    print("\n================ SUMMARY ================")
    print(f"Run label: {run_label}")
    print(f"OK:      {ok}")
    print(f"Skipped: {skipped}")
    print(f"Missing: {missing}")
    print(f"Errors:  {err}")
    print(f"Summary written: {summary_txt}")


# LIST MODE (training-data prep, driven by annotation CSVs)

def process_one_list_file(list_csv_path):
    list_name = os.path.splitext(os.path.basename(list_csv_path))[0]
    list_tag = data_load.sanitize_filename(list_name)
    output_base = os.path.join(OUTPUT_ROOT, list_tag)
    os.makedirs(output_base, exist_ok=True)

    annotations_by_file = data_load.load_annotations_from_list_csv(
        list_csv_path, allow_npz_path_as_fallback_input=ALLOW_NPZ_PATH_AS_FALLBACK_INPUT
    )
    file_paths_list = sorted(annotations_by_file.keys())

    if not file_paths_list:
        print("No usable rows found in this list file.")
        return

    print(f"Unique raw files referenced by list: {len(file_paths_list)}")

    valid_files_to_process = []
    for fp in file_paths_list:
        if not os.path.exists(fp):
            print(f"[WARN] Missing raw file in list: {fp}")
            continue
        valid_files_to_process.append(fp)

    if not valid_files_to_process:
        print("All referenced raw files are missing. Nothing to do.")
        return

    # write manifest of what this list contained
    manifest_csv = os.path.join(output_base, f"{list_tag}_resolved_manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_list_file", "original_file_path", "n_annotations", "exists"])
        for fp in file_paths_list:
            w.writerow([list_csv_path, fp, len(annotations_by_file[fp]), int(os.path.exists(fp))])
    print(f"Manifest written: {manifest_csv}")

    run_files_through_pipeline(
        run_mode="list",
        valid_files=valid_files_to_process,
        tasks_by_file=annotations_by_file,
        output_base=output_base,
        run_label=list_tag,
    )


def run_list_mode():
    list_files = data_load.discover_list_files(LIST_DIR, LIST_GLOB)
    if not list_files:
        print(f"No list CSV files found in: {LIST_DIR}")
        return

    print(f"Found {len(list_files)} list CSV files in {LIST_DIR}")
    for i, p in enumerate(list_files, 1):
        print(f"  {i:02d}. {p}")

    for list_csv_path in list_files:
        try:
            process_one_list_file(list_csv_path)
        except Exception:
            print(f"\n[ERROR] Failed while processing list file: {list_csv_path}")
            print(traceback.format_exc())


# FOLDER MODE (unlabeled data prep, for the detector stage)

def run_folder_mode():
    print("\n" + "=" * 80)
    print(f"FOLDER MODE: {FOLDER_PATH}")
    print(f"Time range: {FOLDER_START_DT} -> {FOLDER_END_DT}")
    print("=" * 80)

    files = data_load.discover_files_by_folder_and_time(
        FOLDER_PATH, file_glob=FOLDER_GLOB, start_dt=FOLDER_START_DT, end_dt=FOLDER_END_DT
    )
    if not files:
        print("No files found for the requested folder/time range.")
        return

    print(f"Found {len(files)} raw files to process")

    output_base = os.path.join(OUTPUT_ROOT, data_load.sanitize_filename(FOLDER_RUN_LABEL))
    os.makedirs(output_base, exist_ok=True)

    run_files_through_pipeline(
        run_mode="folder",
        valid_files=files,
        tasks_by_file=None,
        output_base=output_base,
        run_label=FOLDER_RUN_LABEL,
    )


# MAIN

def main():
    mp.set_start_method("spawn", force=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if RUN_MODE == "list":
        run_list_mode()
    elif RUN_MODE == "folder":
        run_folder_mode()
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}, expected 'list' or 'folder'")


if __name__ == "__main__":
    main()