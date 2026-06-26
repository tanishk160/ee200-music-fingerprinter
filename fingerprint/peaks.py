"""
peaks.py  –  Extract constellation of local maxima from a spectrogram.
"""
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt


# ── peak extraction ─────────────────────────────────────────────────────────
def find_peaks(S_db: np.ndarray,
               neighborhood: tuple = (20, 20),
               threshold_db: float = -60.0,
               min_amplitude_db: float = -50.0):
    """
    Find local maxima (constellation points) in a dB spectrogram.

    Parameters
    ----------
    S_db           : 2-D ndarray [freq_bins x time_frames]
    neighborhood   : (freq_radius, time_radius) of the comparison window
    threshold_db   : a peak must be at least this many dB above the local
                     background (controls density)
    min_amplitude_db : absolute floor — ignore bins below this level

    Returns
    -------
    peaks : list of (freq_bin, time_frame) tuples
    """
    # local max filter
    local_max = maximum_filter(S_db, size=neighborhood)
    # a point is a peak if it IS the local maximum AND above the floor
    is_peak = (S_db == local_max) & (S_db >= min_amplitude_db)

    freq_idx, time_idx = np.where(is_peak)
    peaks = list(zip(freq_idx.tolist(), time_idx.tolist()))
    return peaks


def plot_constellation(S_db, freqs, times, peaks,
                       title="Constellation of Peaks", ax=None):
    """Overlay the constellation on the spectrogram."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.pcolormesh(times, freqs, S_db, cmap="magma",
                  shading="auto", vmin=-80, vmax=0)

    if peaks:
        p_freq = [freqs[f] for f, _ in peaks]
        p_time = [times[t] for _, t in peaks]
        ax.scatter(p_time, p_freq, s=4, c="#00e5ff",
                   alpha=0.7, linewidths=0, label=f"peaks ({len(peaks)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim(0, 8000)
    ax.legend(loc="upper right", fontsize=8)

    if created:
        plt.tight_layout()
    return ax
