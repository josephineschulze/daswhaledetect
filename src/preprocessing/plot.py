" FK-before / FK-after/ spectrum figures, plus diagnostic plots inside bad channel estimates"


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

plt.style.use("dark_background")


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