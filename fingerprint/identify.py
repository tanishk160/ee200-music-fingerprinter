"""
identify.py  -  Match a query clip against the fingerprint database.

Architecture
------------
load_index()      :  Loads fingerprint/db_numpy.npz (pre-built offline).
                     Takes <5 seconds, ~200 MB RAM.  Zero computation.

match_query()     :  Binary-search the sorted numpy arrays.
                     Typical query time: < 1 second.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from .hashes import generate_hashes

_FREQ_BINS = 2049
_DT_RANGE  = 201
_MULT      = _FREQ_BINS * _DT_RANGE   # compound key multiplier


# ── index loader ──────────────────────────────────────────────────────────────
def load_index(npz_path: str, song_ids_path: str):
    """
    Load the pre-built numpy index from disk.

    Parameters
    ----------
    npz_path       : path to db_numpy.npz
    song_ids_path  : path to song_ids.pkl

    Returns
    -------
    (keys, sid_arr, ta_arr, song_ids)
    keys     : int32 ndarray, sorted compound hash keys
    sid_arr  : int8  ndarray, index into song_ids list
    ta_arr   : uint16 ndarray, anchor time frame
    song_ids : list[str]
    """
    data = np.load(npz_path)
    keys    = data['keys']    # int32, sorted
    sid_arr = data['sid']     # int8
    ta_arr  = data['ta']      # uint16

    with open(song_ids_path, 'rb') as f:
        song_ids = pickle.load(f)

    return keys, sid_arr, ta_arr, song_ids


# ── numpy binary-search matching ──────────────────────────────────────────────
def match_query(query_peaks: List[Tuple[int, int]],
                keys: np.ndarray,
                sid_arr: np.ndarray,
                ta_arr: np.ndarray,
                song_ids: List[str],
                **hash_kwargs) -> Dict[str, np.ndarray]:
    """
    Match query peaks against the pre-built numpy index.

    Uses numpy.searchsorted for O(log N) lookup.
    Typical time for a 15-second clip: < 1 second.

    Parameters
    ----------
    query_peaks              : list of (freq_bin, time_frame)
    keys, sid_arr, ta_arr    : from load_index()
    song_ids                 : list of song ID strings

    Returns
    -------
    offsets : {song_id: np.ndarray of int time-offsets}
    """
    offsets: Dict[str, list] = defaultdict(list)

    for (fa, fo, dt), t_q in generate_hashes(query_peaks, **hash_kwargs):
        qkey = int(fa) * _MULT + int(fo) * _DT_RANGE + int(dt)
        lo = int(np.searchsorted(keys, qkey, side='left'))
        hi = int(np.searchsorted(keys, qkey, side='right'))
        if lo == hi:
            continue
        diffs   = ta_arr[lo:hi].astype(np.int64) - t_q
        matched = sid_arr[lo:hi]
        for k in range(hi - lo):
            offsets[song_ids[int(matched[k])]].append(int(diffs[k]))

    return {sid: np.array(vals) for sid, vals in offsets.items()}


# ── single-peak fallback ──────────────────────────────────────────────────────
def match_single_peaks(query_peaks, db_single):
    offsets: Dict[str, list] = defaultdict(list)
    for f, t_q in query_peaks:
        key = (f,)
        if key not in db_single:
            continue
        for song_id, t_db in db_single[key]:
            offsets[song_id].append(t_db - t_q)
    return {sid: np.array(vals) for sid, vals in offsets.items()}


def build_single_peak_db(songs_peaks):
    db: Dict[tuple, list] = defaultdict(list)
    for song_id, peaks in songs_peaks.items():
        for f, t in peaks:
            db[(f,)].append((song_id, t))
    return dict(db)


# ── scoring ───────────────────────────────────────────────────────────────────
def best_match(offsets: Dict[str, np.ndarray],
               bin_size: int = 1) -> Tuple[Optional[str], int, dict]:
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


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_offset_histogram(offsets, winner, top_n=5, bin_size=1, ax=None):
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
        label = f"WINNER: {song_id}" if song_id == winner else song_id
        a.set_title(label, fontsize=9)
        a.set_xlabel("Time offset (frames)")
        a.set_ylabel("Hash count")
        a.grid(alpha=0.3)
    if created:
        plt.tight_layout()
    return axes
