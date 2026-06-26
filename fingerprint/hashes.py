"""
hashes.py  –  Generate paired-peak hashes and manage the fingerprint DB.

Hash format: (freq_anchor, freq_target, delta_time)
             → list of (song_id, t_anchor)
"""
import pickle
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ── pairing parameters ──────────────────────────────────────────────────────
FAN_VALUE      = 15     # max number of target peaks paired with each anchor
TIME_DELTA_MIN = 1      # minimum frame gap between anchor and target
TIME_DELTA_MAX = 200    # maximum frame gap between anchor and target
FREQ_DELTA_MAX = 200    # maximum frequency-bin distance between paired peaks


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
    # sort by time for efficient windowing
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
            # hash = (freq_anchor, freq_target, time_delta)
            yield (f1, f2, dt), t1
            count += 1
            if count >= fan_value:
                break


# ── database ────────────────────────────────────────────────────────────────
def build_database(songs: Dict[str, List[Tuple[int, int]]],
                   **hash_kwargs) -> Dict[tuple, List[Tuple[str, int]]]:
    """
    Build fingerprint DB from a dict of {song_id: peaks}.

    Returns
    -------
    db : {hash_key: [(song_id, t_anchor), ...]}
    """
    db: Dict[tuple, List] = defaultdict(list)
    for song_id, peaks in songs.items():
        for h, t in generate_hashes(peaks, **hash_kwargs):
            db[h].append((song_id, t))
    return dict(db)


def save_database(db: dict, path: str):
    """Persist the fingerprint database to disk."""
    with open(path, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_database(path: str) -> dict:
    """Load a persisted fingerprint database."""
    with open(path, "rb") as f:
        return pickle.load(f)
