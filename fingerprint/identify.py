"""
identify.py  –  Match a query clip against the fingerprint database.

Matching strategy: stream through pkl chunk files one at a time so that
peak RAM stays under ~30 MB regardless of database size.  No SQLite,
no monolithic dict – works correctly on every reboot of an ephemeral
cloud container.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable

from .hashes import generate_hashes


# ── chunk-streaming matching ─────────────────────────────────────────────────
def match_query(query_peaks: List[Tuple[int, int]],
                chunks_dir: str,
                chunk_prefix: str = "db_chunk_",
                progress_cb: Optional[Callable[[float, str], None]] = None,
                **hash_kwargs) -> Dict[str, np.ndarray]:
    """
    Compare query peaks against the database by streaming pkl chunk files.

    Each chunk is loaded, searched, then immediately discarded – peak RAM
    usage is ~25 MB per chunk regardless of total database size.

    Parameters
    ----------
    query_peaks : list of (freq_bin, time_frame) tuples
    chunks_dir  : directory that contains db_chunk_*.pkl files
    chunk_prefix: prefix of chunk filenames (default 'db_chunk_')
    progress_cb : optional callable(fraction: float, label: str) for UI updates

    Returns
    -------
    offsets : {song_id: np.ndarray of time offsets}
    """
    hashes = list(generate_hashes(query_peaks, **hash_kwargs))
    if not hashes:
        return {}

    # Build a fast lookup: hash_key -> [t_query, ...]
    query_map: Dict[tuple, list] = defaultdict(list)
    for h, t_q in hashes:
        query_map[h].append(t_q)
    query_set = set(query_map.keys())

    offsets: Dict[str, list] = defaultdict(list)

    chunks = sorted(
        f for f in os.listdir(chunks_dir)
        if f.startswith(chunk_prefix) and f.endswith(".pkl")
    )
    if not chunks:
        return {}

    for i, chunk_name in enumerate(chunks):
        if progress_cb:
            progress_cb(i / len(chunks),
                        f"Searching chunk {i+1}/{len(chunks)} …")

        path = os.path.join(chunks_dir, chunk_name)
        with open(path, "rb") as fh:
            chunk_db: dict = pickle.load(fh)

        # Only look up keys that actually appear in the query
        for h in query_set:
            if h not in chunk_db:
                continue
            for song_id, t_db in chunk_db[h]:
                for t_q in query_map[h]:
                    offsets[song_id].append(t_db - t_q)

        del chunk_db  # release RAM immediately

    if progress_cb:
        progress_cb(1.0, "Search complete ✓")

    return {sid: np.array(vals) for sid, vals in offsets.items()}


# ── single-peak fallback ─────────────────────────────────────────────────────
def match_single_peaks(query_peaks: List[Tuple[int, int]],
                       db_single: Dict[tuple, List[Tuple[str, int]]]):
    """Match using single peaks (freq_bin only) – weaker baseline."""
    offsets: Dict[str, list] = defaultdict(list)
    for f, t_q in query_peaks:
        key = (f,)
        if key not in db_single:
            continue
        for song_id, t_db in db_single[key]:
            offsets[song_id].append(t_db - t_q)
    return {sid: np.array(vals) for sid, vals in offsets.items()}


def build_single_peak_db(songs_peaks: Dict[str, List[Tuple[int, int]]]):
    """Build a single-peak database (weaker baseline)."""
    db: Dict[tuple, list] = defaultdict(list)
    for song_id, peaks in songs_peaks.items():
        for f, t in peaks:
            db[(f,)].append((song_id, t))
    return dict(db)


# ── scoring ──────────────────────────────────────────────────────────────────
def best_match(offsets: Dict[str, np.ndarray],
               bin_size: int = 1) -> Tuple[Optional[str], int, dict]:
    """
    Find the song with the most hashes aligned at a single time offset.

    Returns
    -------
    (winner_song_id, score, scores_dict)
    """
    scores = {}
    for song_id, offs in offsets.items():
        if len(offs) == 0:
            scores[song_id] = 0
            continue
        lo, hi = offs.min(), offs.max()
        bins = max(1, (hi - lo) // bin_size + 1)
        counts, _ = np.histogram(offs, bins=int(bins))
        scores[song_id] = int(counts.max())

    if not scores:
        return None, 0, {}

    winner = max(scores, key=scores.get)
    return winner, scores[winner], scores


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_offset_histogram(offsets: Dict[str, np.ndarray],
                          winner: str,
                          top_n: int = 5,
                          bin_size: int = 1,
                          ax=None):
    """Plot the offset histogram for the top-N candidate songs."""
    _, _, scores = best_match(offsets, bin_size=bin_size)
    top_songs = sorted(scores, key=scores.get, reverse=True)[:top_n]

    created = ax is None
    if created:
        fig, axes = plt.subplots(len(top_songs), 1,
                                 figsize=(12, 2.5 * len(top_songs)))
        if len(top_songs) == 1:
            axes = [axes]
    else:
        axes = [ax]

    for idx, song_id in enumerate(top_songs):
        a = axes[idx] if created else ax
        offs = offsets.get(song_id, np.array([]))
        if len(offs):
            lo, hi = offs.min(), offs.max()
            bins = max(10, (hi - lo) // bin_size + 1)
            color = "#00e5ff" if song_id == winner else "#546e7a"
            a.hist(offs, bins=int(bins), color=color, edgecolor="none")
        label = f"★ {song_id}" if song_id == winner else song_id
        a.set_title(label, fontsize=9)
        a.set_xlabel("Time offset (frames)")
        a.set_ylabel("Hash count")
        a.grid(alpha=0.3)

    if created:
        plt.tight_layout()
    return axes
