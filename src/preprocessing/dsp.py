""" sign processing 
bandpass filtering, removing FK background trend, detrending spectrum, generating slidinng time windows """

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d


def butter_bandpass_sos(fs, fmin, fmax, order=5):
    """
    butterworth bp filter (here 5th order but can be changed)
    """
    
    return butter(order, [fmin, fmax], btype="bandpass", fs=fs, output="sos")


def bandpass_time_domain(trace, fs, fmin, fmax, order=5):
    """
    apply butterworth bandpass filter to every channel of a trace

    sosfiltfilt applies the filter forwards then backwards, cancelling
    out the phase shift/time delay a normal filter would introduce  (important since i 
    care about the exact timing of a signal arriving at different channels)
    """
    sos = butter_bandpass_sos(fs, fmin, fmax, order=order)
    return sosfiltfilt(sos, trace, axis=1).astype(np.float32)


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


def sliding_windows(nt, win_samples, hop_samples):
    """
    generates (start, end, window_index) tuples chopping a trace of length
    nt into overlapping windows, win_samples long, hop_samples apart.
    """
    widx = 0
    for start in range(0, nt - win_samples + 1, hop_samples):
        yield start, start + win_samples, widx
        widx += 1