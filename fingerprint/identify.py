"""
identify.py  –  Match a query clip against the fingerprint database.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .hashes import generate_hashes


# ── paired-hash matching ─────────────────────────────────────────────────────
def match_query(query_peaks: List[Tuple[int, int]],
                db: Dict[tuple, List[Tuple[str, int]]],
                **hash_kwargs) -> Dict[str, np.ndarray]:
    """
    Compare query peaks against the database using paired hashes.

    Returns
    -------
    offsets : {song_id: array_of_time_offsets}
              offset = t_db_anchor - t_query_anchor
    """
    offsets: Dict[str, list] = defaultdict(list)

    for h, t_query in generate_hashes(query_peaks, **hash_kwargs):
        if h not in db:
            continue
        for song_id, t_db in db[h]:
            offsets[song_id].append(t_db - t_query)

    return {sid: np.array(vals) for sid, vals in offsets.items()}


def match_single_peaks(query_peaks: List[Tuple[int, int]],
                       db_single: Dict[tuple, List[Tuple[str, int]]]):
    """
    Match using single peaks (freq_bin, time_frame) only.
    Much weaker than paired hashes.

    db_single : {(freq_bin,): [(song_id, t_anchor), ...]}
    """
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
    where score = height of the tallest bin in the offset histogram.
    """
    scores = {}
    for song_id, offs in offsets.items():
        if len(offs) == 0:
            scores[song_id] = 0
            continue
        # build histogram with given bin width
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
    # rank by max-bin height
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
