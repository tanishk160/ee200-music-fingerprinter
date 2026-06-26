# EE200 Music Fingerprinter — Q3B

A Shazam-like music fingerprinting app built from scratch using DSP.

## Features
- **Single-Clip Mode**: Upload a query clip → see spectrogram, constellation of peaks, offset histogram, and matched song
- **Batch Mode**: Upload multiple clips → download `results.csv`

## Run Locally

```bash
pip install -r requirements.txt
python build_database.py   # index all songs (run once)
streamlit run app.py
```

## Project Structure

```
├── fingerprint/
│   ├── audio.py       # STFT spectrogram
│   ├── peaks.py       # Constellation extraction
│   ├── hashes.py      # Hash generation & DB
│   ├── identify.py    # Matching & offset histogram
│   ├── database.pkl   # Pre-built fingerprint DB
│   └── songs_peaks.pkl
├── app.py             # Streamlit app (Q3B)
├── build_database.py  # Index songs → DB
├── Q3A_notebook.ipynb # Analysis notebook (Q3A)
└── requirements.txt
```

## Deployment

Deployed on Streamlit Community Cloud.
