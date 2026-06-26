"""
identify.py  –  Match a query clip against the fingerprint database.

Supports both dict-based (legacy) and NpyDB (NumPy sorted-array) matching.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

from .hashes import generate_hashes, hash_to_int64, NpyDB


# ── NpyDB matching (primary, memory-efficient + fast) ────────────────────────
def match_query_npy(query_peaks: List[Tuple[int, int]],
                    npy_db: NpyDB) -> Dict[str, np.ndarray]:
    """
    Match query peaks against the NpyDB using binary search.

    Returns
    -------
    offsets : {song_id: array_of_time_offsets}
    """
    # Build list of (hash_int64, t_query) for all query hashes
    q_hashes = []   # int64 values
    q_times  = []   # t_query for each
    for (f1, f2, dt), t_query in generate_hashes(query_peaks):
        q_hashes.append(hash_to_int64(f1, f2, dt))
        q_times.append(t_query)

    if not q_hashes:
        return {}

    q_hashes = np.array(q_hashes, dtype=np.int64)
    q_times  = np.array(q_times,  dtype=np.int32)

    # Sort query hashes for efficient binary search
    sort_idx = np.argsort(q_hashes)
    q_hashes_sorted = q_hashes[sort_idx]
    q_times_sorted  = q_times[sort_idx]

    offsets: Dict[str, list] = defaultdict(list)

    # Use NpyDB binary search
    hk = npy_db.hash_keys
    si = npy_db.song_idxs
    ta = npy_db.t_anchors
    songs = npy_db.songs

    # Process unique hashes in batch
    unique_hashes, inv = np.unique(q_hashes_sorted, return_inverse=True)

    for ui, uh in enumerate(unique_hashes):
        lo = int(np.searchsorted(hk, uh, side="left"))
        hi = int(np.searchsorted(hk, uh, side="right"))
        if lo >= hi:
            continue

        # All t_query values that produced this hash
        mask = inv == ui
        tq_vals = q_times_sorted[mask]

        for idx in range(lo, hi):
            song_name = songs[si[idx]]
            t_db = int(ta[idx])
            for tq in tq_vals:
                offsets[song_name].append(t_db - int(tq))

    return {sid: np.array(vals, dtype=np.int32) for sid, vals in offsets.items()}


# ── dict-based matching (legacy fallback) ────────────────────────────────────
def match_query(query_peaks: List[Tuple[int, int]],
                db: Dict[tuple, List[Tuple[str, int]]],
                **hash_kwargs) -> Dict[str, np.ndarray]:
    """Compare query peaks against the dict-based database."""
    offsets: Dict[str, list] = defaultdict(list)
    for h, t_query in generate_hashes(query_peaks, **hash_kwargs):
        if h not in db:
            continue
        for song_id, t_db in db[h]:
            offsets[song_id].append(t_db - t_query)
    return {sid: np.array(vals) for sid, vals in offsets.items()}


def match_single_peaks(query_peaks, db_single):
    """Match using single peaks (weaker baseline)."""
    offsets: Dict[str, list] = defaultdict(list)
    for f, t_q in query_peaks:
        key = (f,)
        if key not in db_single:
            continue
        for song_id, t_db in db_single[key]:
            offsets[song_id].append(t_db - t_q)
    return {sid: np.array(vals) for sid, vals in offsets.items()}


def build_single_peak_db(songs_peaks):
    """Build a single-peak database (weaker baseline)."""
    db = defaultdict(list)
    for song_id, peaks in songs_peaks.items():
        for f, t in peaks:
            db[(f,)].append((song_id, t))
    return dict(db)


# ── scoring ──────────────────────────────────────────────────────────────────
def best_match(offsets: Dict[str, np.ndarray],
               bin_size: int = 1) -> Tuple[Optional[str], int, dict]:
    """
    Find the song with the most hashes aligned at a single time offset.

    Returns (winner_song_id, score, scores_dict).
    """
    scores = {}
    for song_id, offs in offsets.items():
        if len(offs) == 0:
            scores[song_id] = 0
            continue
        lo, hi = int(offs.min()), int(offs.max())
        bins = max(1, (hi - lo) // bin_size + 1)
        counts, _ = np.histogram(offs, bins=int(bins))
        scores[song_id] = int(counts.max())

    if not scores:
        return None, 0, {}

    winner = max(scores, key=scores.get)
    return winner, scores[winner], scores


# ── plotting ─────────────────────────────────────────────────────────────────
def plot_offset_histogram(offsets, winner, top_n=5, bin_size=1, ax=None):
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
            lo, hi = int(offs.min()), int(offs.max())
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
