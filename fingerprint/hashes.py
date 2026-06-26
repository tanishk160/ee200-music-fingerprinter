"""
hashes.py  –  Generate paired-peak hashes and manage the fingerprint DB.

Hash format: (freq_anchor, freq_target, delta_time)
             → list of (song_id, t_anchor)

Supports:
  - dict-based in-memory DB (legacy pickle)
  - NumPy sorted-array DB (new, compact, fast binary search)
"""
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


# ── pairing parameters ──────────────────────────────────────────────────────
FAN_VALUE      = 5      # reduced from 15 → keeps DB files under GitHub 100 MB limit
TIME_DELTA_MIN = 1      # minimum frame gap between anchor and target
TIME_DELTA_MAX = 200    # maximum frame gap between anchor and target
FREQ_DELTA_MAX = 200    # maximum frequency-bin distance between paired peaks


# ── hash packing ─────────────────────────────────────────────────────────────
def hash_to_int64(f1: int, f2: int, dt: int) -> int:
    """Pack (f1, f2, dt) into a single int64 for sorted-array lookup."""
    return (int(f1) << 24) | (int(f2) << 8) | int(dt & 0xFF)


# ── hash generation ─────────────────────────────────────────────────────────
def generate_hashes(peaks: List[Tuple[int, int]],
                    fan_value: int = FAN_VALUE,
                    time_delta_min: int = TIME_DELTA_MIN,
                    time_delta_max: int = TIME_DELTA_MAX,
                    freq_delta_max: int = FREQ_DELTA_MAX):
    """
    Pair each anchor peak with up to `fan_value` nearby target peaks.

    Parameters
    ----------
    peaks : list of (freq_bin, time_frame)

    Yields
    ------
    (hash_key, t_anchor)  where hash_key is a compact int tuple
    """
    sorted_peaks = sorted(peaks, key=lambda p: p[1])

    for i, (f1, t1) in enumerate(sorted_peaks):
        count = 0
        for j in range(i + 1, len(sorted_peaks)):
            f2, t2 = sorted_peaks[j]
            dt = t2 - t1
            if dt < time_delta_min:
                continue
            if dt > time_delta_max:
                break
            if abs(f2 - f1) > freq_delta_max:
                continue
            yield (f1, f2, dt), t1
            count += 1
            if count >= fan_value:
                break


# ── legacy dict-based DB ────────────────────────────────────────────────────
def build_database(songs: Dict[str, List[Tuple[int, int]]],
                   **hash_kwargs) -> Dict[tuple, List[Tuple[str, int]]]:
    db: Dict[tuple, List] = defaultdict(list)
    for song_id, peaks in songs.items():
        for h, t in generate_hashes(peaks, **hash_kwargs):
            db[h].append((song_id, t))
    return dict(db)


def save_database(db: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_database(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── NumPy sorted-array DB (preferred) ────────────────────────────────────────
class NpyDB:
    """
    Compact fingerprint database stored as three sorted NumPy arrays.

    Files (all in fingerprint/):
        db_hashes.npy    – int64 hash keys, sorted ascending
        db_song_idx.npy  – uint16 song index parallel to hashes
        db_t_anchor.npy  – uint32 t_anchor parallel to hashes
        db_songs.txt     – one song name per line (index → name)

    Lookup is O(log N) via np.searchsorted — no full DB scan needed.
    Total RAM: ~60-80 MB for 50 songs.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.hash_keys  = None
        self.song_idxs  = None
        self.t_anchors  = None
        self.songs      = None   # list[str]

    def load(self):
        d = self.directory
        self.hash_keys  = np.load(os.path.join(d, "db_hashes.npy"),   mmap_mode="r")
        self.song_idxs  = np.load(os.path.join(d, "db_song_idx.npy"), mmap_mode="r")
        self.t_anchors  = np.load(os.path.join(d, "db_t_anchor.npy"), mmap_mode="r")
        with open(os.path.join(d, "db_songs.txt"), encoding="utf-8") as f:
            self.songs = [line.strip() for line in f if line.strip()]
        return self

    def is_ready(self) -> bool:
        return self.hash_keys is not None

    def query(self, hash_int_array: np.ndarray):
        """
        Given a sorted int64 array of query hash keys, return all matching
        (song_name, t_anchor) pairs via binary search.

        Parameters
        ----------
        hash_int_array : 1-D int64 array of packed query hashes (need NOT be unique)

        Yields
        ------
        (song_name: str, t_anchor: int)
        """
        hk = self.hash_keys
        si = self.song_idxs
        ta = self.t_anchors
        songs = self.songs

        for h in hash_int_array:
            lo = np.searchsorted(hk, h, side="left")
            hi = np.searchsorted(hk, h, side="right")
            for idx in range(lo, hi):
                yield songs[si[idx]], int(ta[idx])

    def n_hashes(self) -> int:
        return len(self.hash_keys) if self.hash_keys is not None else 0

    def n_songs(self) -> int:
        return len(self.songs) if self.songs is not None else 0


def load_npy_db(directory: str):
    """Load the NumPy sorted-array database. Returns None if files missing."""
    needed = ["db_hashes.npy", "db_song_idx.npy", "db_t_anchor.npy", "db_songs.txt"]
    if not all(os.path.exists(os.path.join(directory, f)) for f in needed):
        return None
    db = NpyDB(directory)
    db.load()
    return db
