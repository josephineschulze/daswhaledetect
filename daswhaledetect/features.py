import logging
from pathlib import Path
import random
import os
import numpy as np
import das4whales as dw
from collections import defaultdict
from fractions import Fraction
from tqdm import tqdm
import scipy.signal as sp
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool


from daswhaledetect.utils import sanitize_filename
from daswhaledetect.dataload import (
    parse_datetime_from_path,
    segment_matches_requested,
)
from daswhaledetect.evaluate import (
    plot_bad_channel_fraction, 
    plot_bad_channel_matrix, 
    plot_spectrum_detrended, 
    plot_fk_before_after
)


logger = logging.getLogger("daswhaledetect.features")

def get_resample_factors(orig_fs, target_fs, max_denominator = 10000):
    """ work out whole-number up/down ratio needed for resampling from orig fs to target orig_fs
    
    e.g 500 hz -> 256 hz becomes 256/500, reduced to its simplest equivalent ratio"""

    frac= Fraction(str(target_fs / orig_fs)).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator

def resample_trace_to_target_fs(trace, tx, orig_fs, target_fs, pad_seconds=2.0):
    """resample trace from orig sample rate to shared target rate, to make files for publication all 
    more comparable
    
    pad trace with mirrored copy of itself before resampling, then trim padding back afterwards (reduce edge effects)"""

    orig_fs = float(orig_fs)
    target_fs = float(target_fs)

    if orig_fs <= 0 or target_fs <= 0:
        raise ValueError(f"sampling rates must be positive. Orig fs= {orig_fs}, target fs= {target_fs}")

    # if fs already the same, nothing to do
    if np.isclose(orig_fs, target_fs, rtol=0, atol=1e-9):
        if tx is None or len(tx) != trace.shape[1]:
            tx_new = np.arange(trace.shape[1], dtype=np.float64) / target_fs
        else:
            tx_new = np.asarray(tx, dtype=np.float64)
        return trace.astype(np.float32, copy=False), tx_new, target_fs

    # pad both edges for a softer edge
    pad_samples = int(pad_seconds * orig_fs)
    trace_padded = np.pad(trace, ((0, 0), (pad_samples, pad_samples)), mode='reflect')

    # actually resample the padded trace using the whole-number ratio above
    up, down = get_resample_factors(orig_fs, target_fs)
    trace_rs_padded = sp.resample_poly(trace_padded, up, down, axis=1).astype(np.float32)

    # Strip the padding back off in the resampled domain, since the
    # padding length in samples changes when the sample rate changes.
    pad_samples_rs = int(pad_seconds * target_fs)
    trace_rs = trace_rs_padded[:, pad_samples_rs:-pad_samples_rs]

    # New time axis at the new rate, starting wherever the original did
    t0 = 0.0
    if tx is not None and len(tx) > 0:
        t0 = float(tx[0])
    tx_rs = t0 + (np.arange(trace_rs.shape[1], dtype=np.float64) / target_fs)

    return trace_rs, tx_rs, target_fs

def channel_block_scores(trace, fs, block_length_s):
    """
    chop files into 1s blocks, score each channel in every block, to find unusually 
    high energy relative to the other channels in the same block

    flags channels that are persistently noisy, rather than those briefly pickign up
    real signals
    """
    n_channels, nt = trace.shape
    block_len = max(1, int(round(block_length_s * fs)))
    n_blocks = nt // block_len
    if n_blocks < 1:
        raise ValueError("File too short for requested block length")

    # keep whole numbers of blocks, drop leftover samples that dont fill a full block
    trace_use = trace[:, :n_blocks * block_len]
    blocks = trace_use.reshape(n_channels, n_blocks, block_len)

    energy = np.mean(blocks.astype(np.float64) ** 2, axis=2)
    loge = np.log10(energy + 1e-12)  # log scale, so very different magnitudes compare fairly

    # score each time block independently (comparing channel energy against other channels within the same block)
    zmat = np.zeros_like(loge, dtype=float)
    for b in range(n_blocks):
        zmat[:, b] = robust_zscore(loge[:, b])
    return zmat

def robust_zscore(x):
    """
    z score using median and MAD (median absolute deviation) to catch 'bad channels' - 
    the ones noisy throughout the whole file
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return np.zeros_like(x, dtype=float)
    # 0.6745 rescales MAD so it behaves like a standard deviation would
    # for normally-distributed data.
    return 0.6745 * (x - med) / mad

def suppress_bad_channels(trace, bad_idx, mode="interp"):
    """
    replaces bad channels with something less disruptive before later processing

    mode= zero: set bad channels to 0 (tried, was fine)
    mode= interp: replace each bad channel with the average of it immediate neighbours
    to avoid it being full of gaps (worked better)

    """
    if bad_idx.size == 0:
        return trace

    trace = trace.copy()  # don't modify the caller's array in place
    keep = bad_idx[(bad_idx >= 0) & (bad_idx < trace.shape[0])]
    if keep.size == 0:
        return trace

    if mode == "zero":
        trace[keep, :] = 0.0
    elif mode == "interp":
        for idx in keep:
            if idx == 0:
                trace[idx, :] = trace[idx + 1, :]
            elif idx == trace.shape[0] - 1:
                trace[idx, :] = trace[idx - 1, :]
            else:
                trace[idx, :] = 0.5 * (trace[idx - 1, :] + trace[idx + 1, :])
    else:
        raise ValueError(f"unknown suppression mode: {mode}")

    return trace

def build_segments_from_dist(dist_m, requested_range_m, seg_len_m, min_channels=8):
    """split long cable into shorter spatial segments (using 20km here), so each block can be windowed/processed
    on its own
    """
    start_req_m, end_req_m, step_ch = requested_range_m
    dist = np.asarray(dist_m).astype(float)

    #too few channels/NaN dist values - bail
    if dist.size < min_channels or not np.all(np.isfinite(dist)):
        return []

    # in case dist runs high to low instead of low to high, flip so i can assume increasing distance
    if dist[0] > dist[-1]:
        dist_for_mask = dist[::-1]
        idx_map = np.arange(dist.size)[::-1]
    else:
        dist_for_mask = dist
        idx_map = np.arange(dist.size)

    in_req = (dist_for_mask >= float(start_req_m)) & (dist_for_mask <= float(end_req_m))
    if not np.any(in_req):
        return []

    idxs = np.where(in_req)[0]
    i0_req = int(idxs[0])
    i1_req = int(idxs[-1]) + 1
    dist_req = dist_for_mask[i0_req:i1_req]

    # go along the cable in seg_len_m-sized steps
    segs = []
    edge = float(start_req_m)
    last_edge = float(end_req_m)

    while edge < last_edge:
        edge_next = min(edge + float(seg_len_m), last_edge)
        mask = (dist_req >= edge) & (dist_req < edge_next)
        if np.any(mask):
            j0 = int(np.where(mask)[0][0])
            j1 = int(np.where(mask)[0][-1]) + 1
            ii0_sorted = i0_req + j0
            ii1_sorted = i0_req + j1
            if (ii1_sorted - ii0_sorted) >= min_channels:
                ii0 = int(idx_map[ii0_sorted])
                ii1 = int(idx_map[ii1_sorted - 1])
                lo = min(ii0, ii1)
                hi = max(ii0, ii1) + 1
                if (hi - lo) >= min_channels:
                    seg_start_m = int(round(np.nanmin(dist[lo:hi])))
                    seg_end_m = int(round(np.nanmax(dist[lo:hi])))
                    label = f"{seg_start_m//1000:03d}-{seg_end_m//1000:03d}km_step{int(step_ch)}"
                    segs.append((slice(lo, hi, 1), (seg_start_m, seg_end_m, int(step_ch)), label))
        edge = edge_next

    segs.sort(key=lambda x: x[1][0])
    return segs

def sliding_windows(nt, win_samples, hop_samples):
    """
    generates (start, end, window_index) tuples chopping a trace of length
    nt into overlapping windows, win_samples long, hop_samples apart.
    """
    widx = 0
    for start in range(0, nt - win_samples + 1, hop_samples):
        yield start, start + win_samples, widx
        widx += 1

# ================================================== #
# Bad channel estimation functions
# ================================================== #
def score_channels_worker(args):
    """
    Process one file for bad-channel estimation.
    """
    file_path, cfg = args

    md = dw.data_handle.get_acquisition_parameters(str(file_path),interrogator=cfg["interrogator"])

    selected_channels = [int(m // md["dx"])for m in cfg["channel_range"]]
    selected_channels[2] = cfg["channel_range"][2]

    trace, tx, dist, _ = dw.data_handle.load_das_data(str(file_path),selected_channels,md,interrogator=cfg["interrogator"])
    trace, tx, fs = resample_trace_to_target_fs(trace,tx,md["fs"],cfg["features"]["target_fs"])

    zmat = channel_block_scores(trace,fs,cfg["bad_channel_estimation"]["block_length_s"])
    hot_blocks = (zmat >cfg["bad_channel_estimation"]["per_block_z_thresh"])
    frac_hot_blocks = hot_blocks.mean(axis=1)
    persistent_this_file = (frac_hot_blocks >=cfg["bad_channel_estimation"]["per_file_block_fraction_thresh"])

    return persistent_this_file.astype(np.uint8), dist

def estimate_bad_channels_for_group(group_key, files_for_group, bad_dir, diag_dir, cfg):
    """
    Estimate bad channels for one day/group.
    """

    rng = random.Random(cfg["bad_channel_estimation"]["random_seed"])
    sample_files = rng.sample(files_for_group,min(cfg["bad_channel_estimation"]["n_sample_files_per_day"],len(files_for_group)))

    logger.info("Estimating bad channels for %s using %d files with %d workers",group_key,len(sample_files),cfg["n_workers"])
    tasks = [(fp, cfg) for fp in sample_files]

    with Pool(cfg["n_workers"]) as pool:
        results = list(
            tqdm(pool.imap(score_channels_worker, tasks),total=len(tasks),desc=f"Selected channels ({group_key})",unit="file")
        )

    per_file_persistent = []
    dist_ref = None

    for persistent, dist in results:
        per_file_persistent.append(persistent)
        if dist_ref is None:
            dist_ref = dist

    persistent_matrix = np.vstack(per_file_persistent)
    frac_files_persistent = persistent_matrix.mean(axis=0)
    bad_idx = np.where(frac_files_persistent >=cfg["bad_channel_estimation"]["across_file_fraction_thresh"])[0]

    # ----- save txt / npz / plots -----
    safe_group_key = sanitize_filename(group_key)
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
        block_length_s=float(cfg["bad_channel_estimation"]["block_length_s"]),
        per_block_z_thresh=float(cfg["bad_channel_estimation"]["per_block_z_thresh"]),
        per_file_block_fraction_thresh=float(cfg["bad_channel_estimation"]["per_file_block_fraction_thresh"]),
        across_file_fraction_thresh=float(cfg["bad_channel_estimation"]["across_file_fraction_thresh"]),
        target_fs=float(cfg["features"]["target_fs"]),
    )
    logger.info("Saved bad channel summary for %s to %s", group_key, txt_path)

    plot_bad_channel_fraction(dist_ref, frac_files_persistent, cfg["bad_channel_estimation"]["across_file_fraction_thresh"], group_key, diag_dir)
    plot_bad_channel_matrix(persistent_matrix, group_key, diag_dir)

    return bad_idx.astype(np.int32)


def estimate_bad_channels(valid_files, bad_dir, diag_dir, cfg):
    """
    samples up to N_SAMPLE_FILES_PER_DAY files from this group (day), scores
    every channel in 1s blocks, and flags channels that are persistently
    noisy across most of the sampled files. saves a txt list + npz summary +
    two diagnostic pngs.
    """
    logger.info("=" * 80)
    logger.info("=" * 28 + " BAD CHANNEL ESTIMATION " + "=" * 28)
    logger.info("=" * 80)

    files_by_day = defaultdict(list)

    for fp in valid_files:
        dt_file, _ = parse_datetime_from_path(fp)
        day = dt_file.strftime("%Y%m%d") if dt_file else "unknown"
        files_by_day[day].append(fp)

    bad_channels_by_group = {}

    for group_key, fps in sorted(files_by_day.items()):

        bad_channels_by_group[group_key] = estimate_bad_channels_for_group(group_key,fps,bad_dir, diag_dir, cfg)

    logger.info("All bad channel estimation complete. Saved summaries to %s", bad_dir)

    return files_by_day, bad_channels_by_group


# =============================================== #
# Feature extraction functions
# =============================================== #
def background_removal_fk(fk_db, k, f, vmin, vmax):
    """
    remove frequency-dependent background trend from already-computed fk spectrum
    (estimated from everything OUTSIDE the velocity band of interest)

    everything outside the fk filter gets subtracted rfom the whole spectrum,
    hopefully correcting for freq-dependent noise 
    """

    kk, ff = np.meshgrid(k, f, indexing="ij")
    vel = np.divide(ff, kk, out=np.full_like(ff, np.nan), where=kk != 0)
    vel_abs = np.abs(vel)

    # velocity mask for the fk filter speeds (most likely using 1400-6000m/s to be very broad)
    vel_mask = (vel_abs >= vmin) & (vel_abs <= vmax)
    bg_mask = ~vel_mask

    # average background region per freq column across all wavenumbers = estimate of background nosie at each freq
    bg_spec = np.nanmean(np.where(bg_mask, fk_db, np.nan), axis=0)
    bg_trend = bg_spec - np.nanmean(bg_spec)  # centre around 0 dB
    fk_clean = fk_db - bg_trend[None, :]

    return fk_clean, vel_mask, bg_spec

def detrend_spectrum(spec_db, freq, smooth_hz):
    """
    remove a baseline from a 1D spectrum (dB), leaving behind local peaks with slighlty less 
    abrupt variation - baseline is a gaussian smoothed version of the spectrum itself, following the 
    general trend (with calls showing) but not the sharp noisy features peaking out
    """
    df_ = np.mean(np.diff(freq))
    sigma_bins = smooth_hz / df_ if df_ != 0 else 1.0  # Hz -> bins
    baseline = gaussian_filter1d(spec_db, sigma=sigma_bins)
    return spec_db - baseline, baseline

def compute_window_features(trace_window, fs, ks, cfg):
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
    freq_mask = (freqs_full >= cfg["features"]["freq_min"]) & (freqs_full <= cfg["features"]["freq_max"])
    freqs = freqs_full[freq_mask]
    fk_trimmed = fk_db[:, freq_mask]

    fk_clean, vel_mask, bg_spec = background_removal_fk(fk_trimmed, ks, freqs, cfg["features"]["vel_min"], cfg["features"]["vel_max"])

    fk_before = np.where(vel_mask, fk_trimmed, np.nan)
    fk_after = np.where(vel_mask, fk_clean, np.nan)

    spec_before = np.nanmedian(fk_before, axis=0)
    spec_after = np.nanmedian(fk_after, axis=0)
    spec_detrended, baseline = detrend_spectrum(spec_after, freqs, cfg["features"]["spectral_baseline_hz"])

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

def process_one_file(args):
    import logging
    import os

    fp, tasks_by_file, run_mode, png_dir, npz_dir, bad_channels_by_group, cfg = args
    

    if not os.path.isfile(fp):
        return (fp, "skipped", "file not found")

    if run_mode == "list":
        annotations = tasks_by_file.get(fp, [])
        if not annotations:
            return (fp, "skipped", "no annotations for this file")
    else:
        annotations = None

    df_file, dt_tag = parse_datetime_from_path(fp)
    group_key = df_file.strftime("%Y%m%d") if df_file else "unknown"
    timestamp_file = dt_tag if dt_tag else "unknownDate_unknownTime"

    bad_idx = bad_channels_by_group.get(group_key, [])

    metadata = dw.data_handle.get_acquisition_parameters(fp, interrogator=cfg["interrogator"])
    selected_channels = [int(m // metadata["dx"]) for m in cfg["channel_range"]]
    selected_channels[2] = cfg["channel_range"][2]
    tr_full, tx, dist_full, _ = dw.data_handle.load_das_data(fp, selected_channels, metadata, interrogator=cfg["interrogator"])
    
    original_fs = metadata["fs"]
    
    if cfg["features"]["suppress_bad_channels"]:
        trace_full = suppress_bad_channels(tr_full, bad_idx, mode=cfg["features"]["suppress_mode"])

    dx = float(metadata["dx"])
    duration_s = tr_full.shape[1] / original_fs

    seg_defs = build_segments_from_dist(dist_full, cfg["channel_range"], cfg["features"]["segment_length_m"], min_channels=cfg["features"]["min_channels_per_segment"])
    
    if not seg_defs:
        return (fp, "skipped", "no valid segments found")

    trace_full, tx, fs = resample_trace_to_target_fs(trace_full, tx, original_fs, cfg["features"]["target_fs"])
    
    _, nt = trace_full.shape
    win_samples = int(round(cfg["features"]["window_length_s"] * fs))
    hop_samples = int(win_samples * (1 - cfg["features"]["window_overlap"]))

    if win_samples <= 1 or hop_samples <= 0:
        return (fp, "skipped", "invalid windowing parameters")

    if run_mode == "list":
        ann_by_window = defaultdict(list)
        for ann in annotations:
            w = ann.get("window_index", None)
            if w is not None:
                ann_by_window[w].append(ann)
        if not ann_by_window:
            return (fp, "skipped", "no annotations for this file")

    saved_count = 0

    for slc, seg_tuple, seg_label in seg_defs:
        trace_seg_full = trace_full[slc, :]
        dist_seg = dist_full[slc]
        n_channels = trace_seg_full.shape[0]

        if n_channels < cfg["features"]["min_channels_per_segment"]:
            return (fp, "skipped", f"segment {seg_label} has too few channels")

        dx_eff = selected_channels[2] * dx
        ks = np.fft.fftshift(np.fft.fftfreq(n_channels, d=dx_eff))

        # common-mode removal + channel-mean removal + bandpass, applied to the
        # FULL segment trace (all time samples) BEFORE slicing into windows --
        # gives sosfiltfilt enough context to settle cleanly, so every window
        # (including the last) sees artifact-free samples at its edges.
        if cfg["features"]["remove_common_mode"]:
            trace_seg_full = trace_seg_full - np.mean(trace_seg_full, axis=0, keepdims=True)    
        if cfg["features"]["remove_channel_mean"]:
            trace_seg_full = trace_seg_full - np.mean(trace_seg_full, axis=1, keepdims=True)
        if cfg["features"]["apply_bandpass"]:
            sos_bpfilter = dw.dsp.butterworth_filter(
                [cfg["features"]["bandpass_order"], 
                    [cfg["features"]["freq_min"], cfg["features"]["freq_max"]], 'bp'], fs)
            trace_seg_full = sp.sosfiltfilt(sos_bpfilter, trace_seg_full, axis=1)

        for w_start, w_end, widx in sliding_windows(nt, win_samples, hop_samples):

            if run_mode == "list":
                ann_this_window = ann_by_window.get(widx, [])
                if not ann_this_window:
                    continue

                anns_matching_segment = [
                    ann for ann in ann_this_window 
                    if segment_matches_requested(
                        seg_label=seg_label,
                        seg_tuple=seg_tuple,
                        requested_specs=[ann["fiber_segment_spec"]],)
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
                    "original_file_path": fp,
                    "npz_path": "",
                    "source_list_file": "",
                }]
            trace_window = trace_seg_full[:, w_start:w_end].astype(np.float32)
            time_window = np.arange(w_start, w_end) / fs

            feats = compute_window_features(trace_window, fs, ks, cfg)

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
                tag = sanitize_filename(tag_raw)
                """save the PNG diagnostics (optional) + the NPZ feature file for one window/annotation."""
                

                if cfg["features"]["save_pngs"]:
                    plot_fk_before_after(
                        feats["fk_before"], feats["fk_after"], feats["freqs"], ks,
                        calltype, seg_label, widx, png_dir, tag,
                    )
                    plot_spectrum_detrended(
                        feats["freqs"], feats["spec_before"], feats["baseline"],
                        feats["spec_detrended"], calltype, seg_label, widx, png_dir, tag,
                    )

                np.savez_compressed(
                    os.path.join(npz_dir, f"{tag}_fk_processed.npz"),
                    fk_db_before=feats["fk_before"].astype(np.float32),
                    fk_db_after=feats["fk_after"].astype(np.float32),
                    freqs=feats["freqs"].astype(np.float32),
                    ks=ks.astype(np.float32),
                    time_axis=time_window.astype(np.float32),
                    freq_mask=feats["freq_mask"],
                    velocity_mask=feats["vel_mask"],
                    spectrum_before=feats["spec_before"].astype(np.float32),
                    spectrum_after=feats["spec_after"].astype(np.float32),
                    spectrum_after_detrended=feats["spec_detrended"].astype(np.float32),
                    spectrum_baseline=feats["baseline"].astype(np.float32),
                    background_spectrum=feats["bg_spec"].astype(np.float32),
                    fs=float(fs),
                    original_fs=float(original_fs),
                    target_fs=float(cfg["features"]["target_fs"]),
                    nt=time_window.shape[0],
                    duration_s=float(duration_s),
                    dx=dx,
                    channel_indices=selected_channels,
                    n_channels=n_channels,
                    bad_channel_indices=bad_idx.astype(np.int32),
                    day_key=str(group_key),
                    group_key=str(group_key),
                    file_path=fp,
                    original_file_path=str(ann.get("original_file_path", fp)),
                    source_npz_path=str(ann.get("npz_path", "")),
                    source_list_file=str(ann.get("source_list_file", "")),
                    timestamp=str(timestamp_ann),
                    CallType=str(calltype),
                    species=str(calltype),  # kept for compatibility
                    fiber_segment=str(fiber_segment),
                    requested_fiber_segment=str(fiber_segment),
                    channel_range=cfg["channel_range"],
                    freq_range=[cfg["features"]["freq_min"], cfg["features"]["freq_max"]],
                    velocity_range=[cfg["features"]["vel_min"], cfg["features"]["vel_max"]],
                    window_index=int(widx),
                    window_start_s=float(w_start / fs),
                    window_end_s=float(w_end / fs),
                    window_length_s=float(cfg["features"]["window_length_s"]),
                    window_overlap=float(cfg["features"]["window_overlap"]),
                    segment_label=str(seg_label),
                    segment_m=np.array(seg_tuple, dtype=np.int64),
                    segment_length_m=int(cfg["features"]["segment_length_m"]),
                    dist_segment=dist_seg.astype(np.float32),
                    annotation_segment_start_seconds=np.nan if seg_start_seconds is None else float(seg_start_seconds),
                    annotation_segment_start_realtime=str(seg_start_realtime),
                )


                saved_count += 1

    if saved_count == 0:
        return (fp, "skipped", "no matching segment/window")
    else:
        return (fp, "ok", f"Saved {saved_count} windows")


def feature_extract(valid_files, tasks_by_file, run_mode, png_dir, npz_dir, bad_channels_by_group, cfg):

    args = [(fp, tasks_by_file, run_mode, png_dir, npz_dir, bad_channels_by_group, cfg) for fp in valid_files]

    logger.info("=" * 80)
    logger.info("=" * 30 + " FEATURE EXTRACTION " + "=" * 30)
    logger.info("=" * 80)
    logger.info("Starting feature extraction for %d files with %d workers", len(valid_files), cfg["n_workers"])

    with Pool(processes=cfg["n_workers"]) as pool:
        results = list(
            tqdm(pool.imap(process_one_file, args, chunksize=1), total=len(args), desc="Feature extraction", unit="file")
        )
        
    logger.info("Feature extraction complete. Saved NPZ files to %s", npz_dir)


    return results