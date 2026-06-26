"""
app.py  –  Q3B Streamlit Music Fingerprinting App
===================================================
Run locally:
    streamlit run app.py

Architecture
------------
Uses a compact NumPy sorted-array database (fingerprint/db_*.npy) instead of
the 317 MB pickle.  All three .npy files total ~60-80 MB and load in <5s.
Identification of a 30-second clip takes ~3-8 seconds.

Modes
-----
• Single-Clip Mode  : upload one query clip → shows spectrogram, constellation,
                      offset histogram, and matched song.
• Batch Mode        : upload multiple clips → downloads results.csv
"""
import io, os, csv, time, pickle, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint.audio    import load_audio, compute_spectrogram, SR, HOP_DEFAULT
from fingerprint.peaks    import find_peaks
from fingerprint.hashes   import load_npy_db, load_database, NpyDB
from fingerprint.identify import (match_query_npy, match_query,
                                   best_match, build_single_peak_db,
                                   match_single_peaks)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Sonic Fingerprinter",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
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
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #00e5ff22, #7c4dff22);
        border: 1px solid #00e5ff66;
        color: #00e5ff;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover { border-color: #00e5ff; color: white; }
</style>
""", unsafe_allow_html=True)


# ── constants ────────────────────────────────────────────────────────────────
NPY_DIR         = "fingerprint"
DB_PATH         = "fingerprint/database.pkl"    # legacy fallback
N_FFT           = 4096
HOP             = HOP_DEFAULT
NEIGHBORHOOD    = (20, 20)
MIN_AMP_DB      = -50.0
MAX_AUDIO_SECS  = 30   # only fingerprint first 30 s (fast + accurate enough)


# ── cached DB load ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📂 Loading fingerprint database …")
def get_db():
    """
    Load the fingerprint database.
    Prefers compact NpyDB (~60 MB, loads in <5s) over the 317 MB pickle.
    Returns (db_object, db_type, n_hashes, n_songs).
    """
    # ── NpyDB (preferred) ────────────────────────────────────────────────────
    npy_db = load_npy_db(NPY_DIR)
    if npy_db is not None:
        return npy_db, "npy", npy_db.n_hashes(), npy_db.n_songs()

    # ── Pickle fallback ───────────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        db = load_database(DB_PATH)
        return db, "pickle", len(db), None

    return None, None, 0, 0


# ── helpers ───────────────────────────────────────────────────────────────────
def fingerprint_clip(path: str):
    """Load clip (≤ MAX_AUDIO_SECS), return (S_db, freqs, times, peaks)."""
    y, sr = load_audio(path, sr=SR)
    max_samples = int(MAX_AUDIO_SECS * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    S_db, freqs, times = compute_spectrogram(y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    peaks = find_peaks(S_db, neighborhood=NEIGHBORHOOD, min_amplitude_db=MIN_AMP_DB)
    return S_db, freqs, times, peaks


def run_match(peaks, db_obj, db_type):
    if db_type == "npy":
        return match_query_npy(peaks, db_obj)
    else:
        return match_query(peaks, db_obj)


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def dark_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#0f0c29")
    axes_list = ax.flat if hasattr(ax, "flat") else [ax]
    for a in axes_list:
        a.set_facecolor("#1a1a2e")
        a.tick_params(colors="#a0a0b0")
        a.xaxis.label.set_color("#a0a0b0")
        a.yaxis.label.set_color("#a0a0b0")
        a.title.set_color("#e0e0e0")
        for spine in a.spines.values():
            spine.set_edgecolor("#333355")
    return fig, ax


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Sonic Fingerprinter")
    st.markdown("*An EE200 Project — Music ID powered by spectral hashing*")
    st.divider()

    mode = st.radio("**Mode**", ["Single-Clip", "Batch"])
    st.divider()

    db_obj, db_type, n_hashes, n_songs = get_db()

    if db_obj is None:
        st.error("⚠️ No database found.\nRun `python build_npz_db.py` first.")
    else:
        backend = {"npy": "NumPy ⚡ (fast)", "pickle": "Pickle (legacy)"}[db_type]
        st.success(
            f"✅ Database ready\n\n"
            f"**{n_hashes:,}** hash entries\n\n"
            f"**{n_songs or '?'}** songs\n\n"
            f"Backend: **{backend}**"
        )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Audio → STFT spectrogram\n"
        "2. Find local-max peaks → *constellation*\n"
        "3. Pair peaks → compact hashes\n"
        "4. Binary search in sorted DB\n"
        "5. Offset histogram → matched song ✨"
    )
    st.caption(f"ℹ️ First **{MAX_AUDIO_SECS}s** of audio used for speed.")


# ── main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🎵 Sonic Fingerprinter")
st.markdown(
    "Upload an audio clip and watch it get identified — "
    "just like Shazam, but built from scratch with DSP."
)

if db_obj is None:
    st.warning(
        "No fingerprint database found. "
        "Run `python build_npz_db.py` locally, then push the "
        "`fingerprint/db_*.npy` and `fingerprint/db_songs.txt` files to GitHub."
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

        try:
            with st.spinner("🎵 Fingerprinting clip …"):
                t0 = time.time()
                S_db, freqs, times, peaks = fingerprint_clip(tmp_path)
                t_fp = time.time() - t0

            with st.spinner(f"🔍 Matching {len(peaks):,} peaks against database …"):
                t1 = time.time()
                offsets = run_match(peaks, db_obj, db_type)
                winner, score, scores = best_match(offsets)
                t_match = time.time() - t1

            elapsed = time.time() - t0

            # ── result card ──
            if winner:
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.9rem;color:#a0a0b0;">Matched Song</div>
                    <div class="match-title">{winner}</div>
                    <span class="score-badge">🏆 confidence: {score}</span>
                    <span class="score-badge" style="margin-left:8px;">🎧 {len(peaks):,} peaks</span>
                    <span class="score-badge" style="margin-left:8px;">⏱ {elapsed:.1f}s total</span>
                    <span class="score-badge" style="margin-left:8px;">🔎 match: {t_match:.1f}s</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("No match found. The clip may not be in the database.")

            st.divider()

            # ── visuals ──
            col1, col2 = st.columns(2)

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
                n = len(top_songs)
                fig, axes = dark_fig(n, 1, figsize=(12, 2.5 * n))
                if n == 1:
                    axes = [axes]
                for idx, song_id in enumerate(top_songs):
                    offs = offsets.get(song_id, np.array([]))
                    color = "#00e5ff" if song_id == winner else "#546e7a"
                    if len(offs):
                        lo, hi = int(offs.min()), int(offs.max())
                        bins = max(10, hi - lo + 1)
                        axes[idx].hist(offs, bins=bins, color=color, edgecolor="none")
                    label = (f"★ {song_id}  (score={scores[song_id]})"
                             if song_id == winner else
                             f"{song_id}  (score={scores[song_id]})")
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
                colors = ["#00e5ff" if n == winner else "#546e7a" for n in names]
                ax.barh(names[::-1], vals[::-1], color=colors[::-1])
                ax.set_xlabel("Confidence Score")
                ax.set_title("Top-20 Candidate Songs")
                ax.grid(alpha=0.3, axis="x")
                plt.tight_layout()
                st.image(fig_to_bytes(fig), use_container_width=True)
                plt.close(fig)

        finally:
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
        progress = st.progress(0, text="Processing clips …")

        for i, uf in enumerate(uploaded_files):
            progress.progress(i / len(uploaded_files),
                              text=f"Processing {uf.name} …")
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uf.name)[-1], delete=False
            ) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name

            try:
                _, _, _, peaks = fingerprint_clip(tmp_path)
                offsets = run_match(peaks, db_obj, db_type)
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
                "filename":   os.path.splitext(uf.name)[0],
                "prediction": prediction,
            })

        progress.progress(1.0, text="Done!")

        st.markdown("### Results")
        st.dataframe(results, use_container_width=True)

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
