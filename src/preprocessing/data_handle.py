"""
reads DAS acquistion metadata and loads raw strain data
"""

import os
import h5py
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly

def get_acquisition_parameters(filepath):
    """reads the acquisition metadata (sample rate, channel spacing etc)"""

    def _scalar(x): #unwraps numpy array
        x = np.squeeze(x) #drop any length-1 dimensions
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return None
            return x.flat[0].item() #still array - take first value
        try:
            return x.item() # numpy scalar -> plain python scalar
        except Exception:
            return x #already plain python

    def h5_exists(fp, path: str) -> bool: #checks path before reading, avoids crashing with KeyError
        cur = fp # current position
        for part in path.split("/"):
            if part == "":
                continue
            if part not in cur:
                return False
            cur = cur[part]
        return True

    def h5_read(fp, path: str, default=None): #reads fp safely, avoids raising error if path is missing, lets the function cope with slightly different file layouts across years
        try:
            if not h5_exists(fp, path):
                return default
            obj = fp
            for part in path.split("/"):
                if part == "":
                    continue
                obj = obj[part]
            return obj[()] #[()] reads the actual stored value
        except Exception:
            return default

    # step 1: work out which file 'layout' we have (which year)
    # files come from same interrogator (OptoDas) but different deployments and locations
    # NOTE for Eric : this part is probably a bit tricky for adjusting it to the new location, likely needs some hacking
    basename = os.path.basename(filepath).lower()
    fullpath = filepath.lower()
    if "2020" in fullpath or "2020" in basename:
        dataset = "2020"
    elif (
        "2022" in fullpath or "2022" in basename
        or "2025" in fullpath or "2025" in basename
        or "2026" in fullpath or "2026" in basename
    ):
        dataset = "2022"
    else:
        raise ValueError(f"Couldn't detect dataset year from {filepath}")

    # step 2: open file and read the fields for the specific layout/year:
    with h5py.File(filepath, "r", locking=False) as fp:
        if dataset == "2020":
            #sampling rate stored as the time between samples
            #(seconds), so fs (samples/sec) is just 1/dt
            dt_s = h5_read(fp, "header/dt", default=None)
            if dt_s is None:
                raise KeyError("missing header/dt in file, can't compute fs")
            fs = 1.0 / float(_scalar(dt_s))

            #channel spacing (m) - 2020 files dont store dx directly, its a fixed constant
            # (1.021m) scaled by the on-instrument decimation factor
            roi_dec = h5_read(fp, "demodSpec/roiDec", default=1)
            dx = 1.021 * float(_scalar(roi_dec))

            #number of channels (nx) and number of time samples (ns)
            nx = h5_read(fp, "acqSpec/nChannels", default=None)
            ns = h5_read(fp, "acqSpec/nSamples", default=None)
            if nx is None or ns is None:
                raise KeyError("missing acqSpec/nChannels or acqSpec/nSamples")
            nx = int(_scalar(nx))
            ns = int(_scalar(ns))

            # gauge length and refr index sometimes present, sometimes not - fall back to sensible defaults rather than crashing when theyre missing
            gauge_length = h5_read(fp, "header/GL", default=None)
            if gauge_length is None:
                gauge_length = 2.0 * dx
            else:
                gauge_length = float(_scalar(gauge_length))

            n = h5_read(fp, "cableSpec/refIndex", default=None)
            if n is None:
                n = 1.47
            else:
                n = float(_scalar(n))

        elif dataset == "2022":
            # same idea, but field paths changed in the newer file format
            dt_s = h5_read(fp, "header/dt", default=None)
            if dt_s is None:
                raise KeyError("missing header/dt in file, can't compute fs")
            fs = 1.0 / float(_scalar(dt_s))

            # dx is stored directly here, but still needs multiplying by the
            # decimation factor to get the *effective* spacing
            dx0 = h5_read(fp, "header/dx", default=None)
            if dx0 is None:
                raise KeyError("missing header/dx in file, can't compute dx")
            dx0 = float(_scalar(dx0))
            roi_dec = h5_read(fp, "demodSpec/roiDec", default=1)
            roi_dec = float(_scalar(roi_dec))
            dx = dx0 * roi_dec

            nx = h5_read(fp, "header/dimensionRanges/dimension1/size", default=None)
            ns = h5_read(fp, "header/dimensionRanges/dimension0/size", default=None)
            if nx is None or ns is None:
                raise KeyError("missing header/dimensionRanges sizes, can't read nx/ns")
            nx = int(_scalar(nx))
            ns = int(_scalar(ns))

            gauge_length = h5_read(fp, "header/gaugeLength", default=None)
            if gauge_length is None:
                gauge_length = 2.0 * dx
            else:
                gauge_length = float(_scalar(gauge_length))

            n = h5_read(fp, "cableSpec/refractiveIndex", default=None)
            if n is None:
                n = 1.47
            else:
                n = float(_scalar(n))

        else:
            raise ValueError(f"unsupported dataset {dataset}")

        # Step 3: compute scale factor
        # converts raw unwrapped optical phase into physical strain, computed from a formula used by OptaSense interrogators (NOT directly out of the file)
        scale_factor = (
            (2 * np.pi) / (2 ** 16) * (1550.12e-9) / (0.78 * 4 * np.pi * n * gauge_length)
        )

        # step 4: package into one dictionary:
        metadata = {
            "fs": float(fs),               # sampling rate, Hz
            "dx": float(dx),               # channel spacing, m
            "nx": int(nx),                 # number of channels in the file
            "ns": int(ns),                 # number of time samples in the file
            "GL": float(gauge_length),     # gauge length, m
            "n": float(n),                 # fibre refractive index
            "scale_factor": float(scale_factor),
            "dataset": dataset,            # "2020" or "2022", which layout we used
            "basename": os.path.splitext(os.path.basename(filepath))[0],
        }

    return metadata

def select_channels(dx, channel_range):
    """convert channel range into channel indices for slicing raw data
    
    channel_range= start(m), end(m), step(channels)"""

    start_m, end_m, step_channels = channel_range

    #convert the start/end positions from metres into channel index numbers
    start_idx = int(start_m // dx)
    end_idx = int(end_m // dx)

    # step is already channel count, so just need a guard against 0/neg values
    step_idx= max(int(step_channels), 1)

    selected_channels= [start_idx, end_idx, step_idx]

    print(f"channel range(m): {channel_range}")
    print (f"channel indices : {selected_channels}")
    return selected_channels

def load_das_data(filename, selected_channels, metadata, date_str=None):
    """load actual strain data for one file, only for the requested channels"""

    # opens file and reads raw data block. ".T" transposes so rows are channels and columns are time samples (file stores it the other way around)
    with h5py.File(filename, "r", locking=False) as fp:
        raw_data = fp['data'][()].T

    #keep only channels we asked for (start:stop:step), as float64
    trace = raw_data[selected_channels[0]:selected_channels[1]:selected_channels[2], :].astype(np.float64, copy=True)

    # convert raw units into strain RATE using the scale factor computed in get_acquisition_parameters()
    trace *= metadata['scale_factor']

    #this interrogator outputs strain RATE, so integrate over time (running cumulative sum, scaled by the time step 1/fs) to recover the actual strain
    # NOTE: tried both strain and strain rate, and it didn't make a big difference for my data
    trace = np.cumsum(trace, axis=1) * (1 / metadata['fs'])

    # build matching time axis (seconds) and distance axis (metres) for the channels kept
    nnx, nns = trace.shape
    tx = np.arange(nns) / metadata['fs']
    dist = (np.arange(nnx) * selected_channels[2] + selected_channels[0]) * metadata['dx']

    return trace, tx, dist


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
    trace_rs_padded = resample_poly(trace_padded, up, down, axis=1).astype(np.float32)

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


### Bad channel estimate below (and suppression):
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