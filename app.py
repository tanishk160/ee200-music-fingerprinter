"""
app.py  –  Q3B Streamlit Music Fingerprinting App

Architecture (ephemeral-cloud-safe)
------------------------------------
• No SQLite compilation on startup — works instantly on every reboot.
• Database lives as 17 small pkl chunk files committed to GitHub.
• At query time, chunks are streamed one-by-one (~25 MB peak RAM each).
• @st.cache_resource caches peaks and hashes within a session.

Modes
-----
• Single-Clip Mode  : upload one query clip → spectrogram, constellation,
                      offset histogram, and matched song.
• Batch Mode        : upload multiple clips → download results.csv
"""
import io
import os
import csv
import time
import pickle
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint.audio   import load_audio, compute_spectrogram, SR, HOP_DEFAULT
from fingerprint.peaks   import find_peaks, plot_constellation
from fingerprint.identify import (match_query, best_match,
                                   plot_offset_histogram,
                                   build_single_peak_db, match_single_peaks)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Sonic Fingerprinter",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* cards */
    .result-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin: 16px 0;
        backdrop-filter: blur(6px);
    }

    .match-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .score-badge {
        display: inline-block;
        background: rgba(0,229,255,0.15);
        border: 1px solid #00e5ff;
        border-radius: 999px;
        padding: 4px 18px;
        font-size: 0.85rem;
        color: #00e5ff;
        margin-top: 8px;
    }

    /* metric */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    h1, h2, h3 { color: #e0e0e0 !important; }

    /* buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00e5ff22, #7c4dff22);
        border: 1px solid #00e5ff66;
        color: #00e5ff;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        border-color: #00e5ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ── constants ─────────────────────────────────────────────────────────────────
DB_CHUNKS_DIR = "fingerprint"
DB_CHUNK_PREFIX = "db_chunk_"
PEAKS_CACHE   = "fingerprint/songs_peaks.pkl"
N_FFT         = 4096
HOP           = HOP_DEFAULT
NEIGHBORHOOD  = (20, 20)
MIN_AMP_DB    = -50.0


# ── cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading peaks cache …")
def get_single_db():
    """Load the small songs_peaks.pkl (~10 MB) for single-peak baseline."""
    if os.path.exists(PEAKS_CACHE):
        with open(PEAKS_CACHE, "rb") as f:
            songs_peaks = pickle.load(f)
        return build_single_peak_db(songs_peaks)
    return None


def get_chunk_list():
    """Return sorted list of chunk file names (or empty list)."""
    try:
        return sorted(
            f for f in os.listdir(DB_CHUNKS_DIR)
            if f.startswith(DB_CHUNK_PREFIX) and f.endswith(".pkl")
        )
    except FileNotFoundError:
        return []


# ── helpers ───────────────────────────────────────────────────────────────────
def fingerprint_clip(path: str):
    """Load a clip, return (S_db, freqs, times, peaks)."""
    y, sr = load_audio(path, sr=SR)
    S_db, freqs, times = compute_spectrogram(y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    peaks = find_peaks(S_db, neighborhood=NEIGHBORHOOD, min_amplitude_db=MIN_AMP_DB)
    return S_db, freqs, times, peaks


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def dark_fig(*args, **kwargs):
    """Create a matplotlib figure with a dark background."""
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#0f0c29")
    axes_list = ax.flat if hasattr(ax, '__len__') else [ax]
    for a in axes_list:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="#a0a0b0")
        a.xaxis.label.set_color("#a0a0b0")
        a.yaxis.label.set_color("#a0a0b0")
        a.title.set_color("#e0e0e0")
        for spine in a.spines.values():
            spine.set_edgecolor("#333355")
    return fig, ax


def run_match(peaks):
    """
    Run chunk-streaming match with a Streamlit progress bar.
    Returns offsets dict.
    """
    chunks = get_chunk_list()
    n = len(chunks)
    prog = st.progress(0, text="🔍 Searching database …")

    def update(frac, label):
        prog.progress(frac, text=label)

    offsets = match_query(
        peaks,
        chunks_dir=DB_CHUNKS_DIR,
        chunk_prefix=DB_CHUNK_PREFIX,
        progress_cb=update,
    )
    prog.empty()
    return offsets


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Sonic Fingerprinter")
    st.markdown("*An EE200 Project — Music ID powered by spectral hashing*")
    st.divider()

    mode = st.radio("**Mode**", ["Single-Clip", "Batch"])
    st.divider()

    chunks = get_chunk_list()
    single_db = get_single_db()

    if not chunks:
        st.error("⚠️ No database chunks found.\nRun `build_database.py` first.")
        db_ready = False
    else:
        st.success(f"✅ Database ready\n`{len(chunks)}` chunk files on disk")
        db_ready = True

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Audio → STFT spectrogram\n"
        "2. Find local-max peaks → *constellation*\n"
        "3. Pair peaks → compact hashes\n"
        "4. Stream through DB chunks → matches\n"
        "5. Peak bin = matched song ✨"
    )


# ── main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🎵 Sonic Fingerprinter")
st.markdown(
    "Upload an audio clip and watch it get identified — "
    "just like Shazam, but built from scratch with DSP."
)

if not db_ready:
    st.warning(
        "The fingerprint database chunk files are missing. "
        "Run `python build_database.py` in your project directory first, "
        "then restart the app."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-CLIP MODE
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "Single-Clip":
    st.markdown("## Single-Clip Identification")
    uploaded = st.file_uploader(
        "Upload a query audio clip",
        type=["mp3", "wav", "flac", "ogg"],
        key="single_uploader",
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(uploaded.name)[-1], delete=False
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        t0 = time.time()

        with st.spinner("Fingerprinting clip …"):
            S_db, freqs, times, peaks = fingerprint_clip(tmp_path)

        offsets = run_match(peaks)
        winner, score, scores = best_match(offsets)
        elapsed = time.time() - t0

        # ── result card ──
        if winner:
            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.9rem;color:#a0a0b0;">Matched Song</div>
                <div class="match-title">{winner}</div>
                <span class="score-badge">🏆 confidence score: {score}</span>
                <span class="score-badge" style="margin-left:8px;">⏱ {elapsed:.2f}s</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No match found. The clip may not be in the database.")

        st.divider()

        # ── intermediate visuals ──
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📊 Spectrogram")
            fig, ax = dark_fig(figsize=(8, 3))
            ax.pcolormesh(times, freqs, S_db, cmap="magma",
                          shading="auto", vmin=-80, vmax=0)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title(f"Spectrogram — {uploaded.name}")
            ax.set_ylim(0, 8000)
            plt.tight_layout()
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)

        with col2:
            st.markdown("#### ✨ Constellation of Peaks")
            fig, ax = dark_fig(figsize=(8, 3))
            ax.pcolormesh(times, freqs, S_db, cmap="magma",
                          shading="auto", vmin=-80, vmax=0)
            if peaks:
                pf = [freqs[f] for f, _ in peaks]
                pt = [times[t] for _, t in peaks]
                ax.scatter(pt, pf, s=3, c="#00e5ff", alpha=0.7,
                           linewidths=0, label=f"{len(peaks):,} peaks")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Constellation of Peaks")
            ax.set_ylim(0, 8000)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)

        if scores:
            st.markdown("#### 📈 Offset Histogram (Top Candidates)")
            top_songs = sorted(scores, key=scores.get, reverse=True)[:5]
            fig, axes = dark_fig(len(top_songs), 1,
                                 figsize=(12, 2.5 * len(top_songs)))
            if len(top_songs) == 1:
                axes = [axes]
            for idx, song_id in enumerate(top_songs):
                offs = offsets.get(song_id, np.array([]))
                color = "#00e5ff" if song_id == winner else "#546e7a"
                if len(offs):
                    lo, hi = offs.min(), offs.max()
                    bins = max(10, int((hi - lo) + 1))
                    axes[idx].hist(offs, bins=bins, color=color, edgecolor="none")
                label = (f"★ {song_id}  (score={scores[song_id]})"
                         if song_id == winner
                         else f"{song_id}  (score={scores[song_id]})")
                axes[idx].set_title(label, fontsize=9)
                axes[idx].set_xlabel("Time offset (frames)")
                axes[idx].set_ylabel("Hash count")
                axes[idx].grid(alpha=0.3)
            plt.tight_layout()
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)

            st.divider()
            st.markdown("#### 📊 All Candidate Scores")
            top20 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
            names = [x[0] for x in top20]
            vals  = [x[1] for x in top20]
            fig, ax = dark_fig(figsize=(12, 3))
            bar_colors = ["#00e5ff" if n == winner else "#546e7a" for n in names]
            ax.barh(names[::-1], vals[::-1], color=bar_colors[::-1])
            ax.set_xlabel("Confidence Score")
            ax.set_title("Top-20 Candidate Songs")
            ax.grid(alpha=0.3, axis="x")
            plt.tight_layout()
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)

        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH MODE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## Batch Identification")
    st.markdown(
        "Upload multiple query clips. "
        "The app will identify each and produce a downloadable `results.csv`."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips (multiple OK)",
        type=["mp3", "wav", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if uploaded_files:
        results = []
        outer_prog = st.progress(0, text="Processing clips …")

        for i, uf in enumerate(uploaded_files):
            outer_prog.progress(i / len(uploaded_files),
                                text=f"Clip {i+1}/{len(uploaded_files)}: {uf.name}")

            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uf.name)[-1], delete=False
            ) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            try:
                _, _, _, peaks = fingerprint_clip(tmp_path)
                offsets = run_match(peaks)
                winner, score, _ = best_match(offsets)
                prediction = winner if winner else "unknown"
            except Exception as e:
                prediction = f"error: {e}"
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            results.append({
                "filename": os.path.splitext(uf.name)[0],
                "prediction": prediction,
            })

        outer_prog.progress(1.0, text="Done!")

        # display table
        st.markdown("### Results")
        st.dataframe(results, use_container_width=True)

        # CSV download
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(results)

        st.download_button(
            label="⬇️ Download results.csv",
            data=csv_buf.getvalue().encode(),
            file_name="results.csv",
            mime="text/csv",
        )
