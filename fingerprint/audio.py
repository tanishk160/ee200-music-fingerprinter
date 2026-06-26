"""
audio.py  –  Load audio and compute spectrograms.
"""
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ── constants ──────────────────────────────────────────────────────────────
SR = 22050          # target sample rate (Hz)
HOP_DEFAULT = 512   # default STFT hop length (samples)


# ── loaders ────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SR, mono: bool = True):
    """Load an audio file and return (y, sr)."""
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    return y, sr_out


# ── spectrogram ────────────────────────────────────────────────────────────
def compute_spectrogram(y, sr=SR, n_fft=4096, hop_length=HOP_DEFAULT):
    """
    Compute magnitude spectrogram (in dB) via STFT.

    Returns
    -------
    S_db  : 2-D ndarray  [freq_bins x time_frames]  values in dB
    freqs : 1-D ndarray  centre frequencies (Hz) for each row
    times : 1-D ndarray  centre time (s)   for each column
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]),
                                   sr=sr, hop_length=hop_length)
    return S_db, freqs, times


def plot_dft(y, sr=SR, title="DFT Magnitude Spectrum", ax=None):
    """Plot the magnitude of the full DFT (time info lost)."""
    N = len(y)
    Y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(freqs, 20 * np.log10(Y + 1e-10), lw=0.5, color="#4fc3f7")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.set_xlim(0, sr / 2)
    ax.grid(alpha=0.3)

    if created:
        plt.tight_layout()
    return ax


def plot_spectrogram(S_db, freqs, times, title="Spectrogram",
                     ax=None, cmap="magma"):
    """Plot a pre-computed dB spectrogram."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(12, 4))

    img = ax.pcolormesh(times, freqs, S_db, cmap=cmap,
                        shading="auto", vmin=-80, vmax=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim(0, 8000)          # focus on perceptual range

    if created:
        plt.colorbar(img, ax=ax, label="dB")
        plt.tight_layout()
    return ax
