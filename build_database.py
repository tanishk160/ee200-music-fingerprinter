"""
building_database.py
-----------------
Indexing all songs in the provided song library and saving the fingerprint
database to fingerprint/database.pkl.gz.
"""
import os
import pickle
import glob
import time
import gzip 
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np

from fingerprint.audio import load_audio, compute_spectrogram, SR, HOP_DEFAULT
from fingerprint.peaks import find_peaks
from fingerprint.hashes import generate_hashes, build_database 

SONGS_DIR   = "."          
DB_PATH     = "fingerprint/database.pkl.gz"
PEAKS_PATH  = "fingerprint/peaks_cache.pkl.gz"   
N_FFT       = 4096
HOP         = HOP_DEFAULT
NEIGHBORHOOD = (20, 20)
MIN_AMP_DB  = -50.0

EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.ogg")


def index_song(path: str):
    """Return peaks list for a single song file."""
    y, sr = load_audio(path, sr=SR)
    S_db, _, _ = compute_spectrogram(y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    peaks = find_peaks(S_db, neighborhood=NEIGHBORHOOD,
                       min_amplitude_db=MIN_AMP_DB)
    return peaks


def main():
    song_files = []
    for ext in EXTENSIONS:
        song_files.extend(glob.glob(os.path.join(SONGS_DIR, ext)))
    song_files = sorted(set(song_files))

    clips_dir = os.path.join(SONGS_DIR, "query_clips")
    if os.path.isdir(clips_dir):
        print(f"  (found query_clips/ directory — not indexing query files)")

    print(f"Found {len(song_files)} song(s) to index.\n")

    if os.path.exists(PEAKS_PATH):
        print(f"Loading peaks cache from {PEAKS_PATH} …")
        # Changed to gzip.open
        with gzip.open(PEAKS_PATH, "rb") as f:
            peaks_cache = pickle.load(f)
    else:
        peaks_cache = {}

    songs_peaks = {}
    t_start = time.time()
    for i, fpath in enumerate(song_files, 1):
        song_id = Path(fpath).stem          
        if song_id in peaks_cache:
            songs_peaks[song_id] = peaks_cache[song_id]
            print(f"  [{i:3d}/{len(song_files)}] (cached) {song_id}")
            continue

        print(f"  [{i:3d}/{len(song_files)}] Indexing: {song_id} …", end="", flush=True)
        t0 = time.time()
        try:
            peaks = index_song(fpath)
            songs_peaks[song_id] = peaks
            peaks_cache[song_id] = peaks
            print(f"  {len(peaks):,} peaks  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")

    os.makedirs("fingerprint", exist_ok=True)
    # Changed to gzip.open
    with gzip.open(PEAKS_PATH, "wb") as f:
        pickle.dump(peaks_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nPeaks cached to {PEAKS_PATH}")

    print("Building hash database …", end="", flush=True)
    db = build_database(songs_peaks)
    
    # Changed to direct gzip save instead of save_database()
    with gzip.open(DB_PATH, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"  done.  {len(db):,} unique hashes, {len(songs_peaks)} songs.")
    print(f"Database saved to {DB_PATH}  (total time: {time.time()-t_start:.1f}s)")

    single_peak_path = "fingerprint/songs_peaks.pkl.gz"
    with gzip.open(single_peak_path, "wb") as f:
        pickle.dump(songs_peaks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Songs-peaks map saved to {single_peak_path}")


if __name__ == "__main__":
    main()