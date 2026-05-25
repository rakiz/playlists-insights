# /// script
# requires-python = ">=3.12"
# dependencies = ["librosa", "pandas", "numpy", "requests", "python-dotenv"]
# ///
"""
Script 1b — Analyse audio des previews Spotify (30s)
Extrait BPM, énergie, tonalité via librosa. Tout est caché par track_id.
Met à jour data/tracks_spotify.csv et data/tracks.csv (si existant).

Usage :
  python 01b_audio.py              → analyse les morceaux sans cache
  python 01b_audio.py --refresh    → reforce l'analyse de tous les morceaux
"""

import pandas as pd
import numpy as np
import requests
import librosa
import tempfile
import os
import json
import time
import argparse
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

logging.getLogger("spotipy").setLevel(logging.CRITICAL)

os.makedirs("data/audio_cache", exist_ok=True)

SCOPE = "playlist-read-private playlist-read-collaborative"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE,
))

parser = argparse.ArgumentParser()
parser.add_argument("--refresh",     action="store_true", help="Refaire l'analyse même si déjà en cache")
parser.add_argument("--retry-empty", action="store_true", help="Retenter uniquement les morceaux sans résultat audio")
args = parser.parse_args()

# ─── Analyse audio ────────────────────────────────────────────────────────────
KEY_NAMES     = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def estimate_key_mode(chroma_mean):
    best_key, best_mode, best_corr = 0, "major", -2.0
    for i in range(12):
        for profile, mode in [(MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")]:
            corr = float(np.corrcoef(np.roll(profile, i), chroma_mean)[0, 1])
            if corr > best_corr:
                best_key, best_mode, best_corr = i, mode, corr
    return KEY_NAMES[best_key], best_mode

def analyze_preview(track_id, preview_url):
    cache = f"data/audio_cache/{track_id}.json"
    if not args.refresh and not args.retry_empty and os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    if args.retry_empty and os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
        if data.get("tempo") is not None:
            return data

    result = {"tempo": None, "energy": None, "brightness": None, "key": None, "mode": None}

    if not preview_url:
        with open(cache, "w") as f:
            json.dump(result, f)
        return result

    try:
        resp = requests.get(preview_url, timeout=15)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=22050, mono=True)

            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result["tempo"]      = round(float(np.atleast_1d(tempo)[0]), 1)
            result["energy"]     = round(float(np.mean(librosa.feature.rms(y=y))), 4)
            result["brightness"] = round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 1)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key, mode = estimate_key_mode(chroma.mean(axis=1))
            result["key"]  = key
            result["mode"] = mode
        finally:
            os.unlink(tmp_path)

    except Exception:
        pass

    with open(cache, "w") as f:
        json.dump(result, f)
    time.sleep(0.1)
    return result

# ─── Chargement ───────────────────────────────────────────────────────────────
if not os.path.exists("data/tracks_spotify.csv"):
    print("⚠ data/tracks_spotify.csv introuvable — lance d'abord 01_extract.py")
    exit(1)

df = pd.read_csv("data/tracks_spotify.csv")

# ─── Résolution des preview_url manquantes via l'API Spotify ─────────────────
if "preview_url" not in df.columns:
    df["preview_url"] = None

missing = df[df["preview_url"].isna()]
if len(missing) > 0:
    print(f"Récupération des preview URLs ({len(missing)} morceaux sans URL)...")
    for i in range(0, len(missing), 50):
        batch_ids = missing.iloc[i:i+50]["track_id"].tolist()
        try:
            info = sp.tracks(batch_ids)["tracks"]
            for t in info:
                if t:
                    df.loc[df["track_id"] == t["id"], "preview_url"] = t.get("preview_url")
        except Exception:
            pass
        time.sleep(0.2)

# ─── Détermination des morceaux à analyser ───────────────────────────────────
def cache_is_empty(tid):
    path = f"data/audio_cache/{tid}.json"
    if not os.path.exists(path):
        return True
    with open(path) as f:
        return json.load(f).get("tempo") is None

if args.refresh:
    to_analyze = df
elif args.retry_empty:
    to_analyze = df[df["track_id"].apply(cache_is_empty)]
else:
    to_analyze = df[~df["track_id"].apply(
        lambda tid: os.path.exists(f"data/audio_cache/{tid}.json")
    )]

has_preview  = to_analyze["preview_url"].notna().sum()
no_preview   = len(to_analyze) - has_preview
already_done = len(df) - len(to_analyze)

print(f"\nAnalyse audio :")
print(f"  {already_done} morceaux déjà en cache")
print(f"  {has_preview} previews à analyser")
print(f"  {no_preview} sans preview URL (skippés)")

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

N_WORKERS = 8
counter = [0]
lock = threading.Lock()

def process_row(row):
    url = row["preview_url"] if pd.notna(row.get("preview_url")) else None
    result = analyze_preview(row["track_id"], url)
    with lock:
        counter[0] += 1
        print(f"  [{counter[0]}/{len(to_analyze)}] {str(row['track_name'])[:40]:<40}", end="\r")
    return result

rows_to_process = [row for _, row in to_analyze.iterrows()]
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    list(executor.map(process_row, rows_to_process))

print()

# ─── Rechargement du cache et merge ──────────────────────────────────────────
audio_rows = []
for _, row in df.iterrows():
    cache = f"data/audio_cache/{row['track_id']}.json"
    if os.path.exists(cache):
        with open(cache) as f:
            audio_rows.append({"track_id": row["track_id"], **json.load(f)})
    else:
        audio_rows.append({"track_id": row["track_id"],
                           "tempo": None, "energy": None,
                           "brightness": None, "key": None, "mode": None})

audio_df = pd.DataFrame(audio_rows)

def merge_audio(target_df, audio_df):
    drop = [c for c in ["tempo","energy","brightness","key","mode"] if c in target_df.columns]
    return target_df.drop(columns=drop).merge(audio_df, on="track_id", how="left")

df = merge_audio(df, audio_df)
df.to_csv("data/tracks_spotify.csv", index=False)
print(f"→ data/tracks_spotify.csv mis à jour")

if os.path.exists("data/tracks.csv"):
    tracks_full = pd.read_csv("data/tracks.csv")
    tracks_full = merge_audio(tracks_full, audio_df)
    tracks_full.to_csv("data/tracks.csv", index=False)
    print(f"→ data/tracks.csv mis à jour")

analyzed = audio_df["tempo"].notna().sum()
print(f"\nTerminé : {analyzed}/{len(df)} morceaux avec données audio.")
