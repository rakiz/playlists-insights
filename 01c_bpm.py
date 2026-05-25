# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "requests", "python-dotenv"]
# ///
"""
Script 1c — Récupération BPM + tonalité + danceability via GetSong API
Limite : 3000 req/heure → 3700 tracks en ~1h30.
Cache par track_id dans data/bpm_cache/.
Met à jour data/tracks_spotify.csv et data/tracks.csv.

Usage :
  python 01c_bpm.py              → fetch tous les morceaux sans cache
  python 01c_bpm.py --status     → état du cache sans fetch
  python 01c_bpm.py --retry-empty → retente les morceaux sans résultat
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from config import GETSONGBPM_API_KEY

os.makedirs("data/bpm_cache", exist_ok=True)

BASE_URL   = "https://api.getsong.co/search/"
MAX_PER_HR = 2800  # marge de sécurité sous les 3000/heure

parser = argparse.ArgumentParser()
parser.add_argument("--status",      action="store_true", help="Afficher l'état du cache et quitter")
parser.add_argument("--retry-empty", action="store_true", help="Retenter les morceaux sans résultat")
args = parser.parse_args()

if not GETSONGBPM_API_KEY:
    print("⚠ GETSONGBPM_API_KEY manquant dans .env")
    exit(1)

# ─── Cache ────────────────────────────────────────────────────────────────────
def cache_path(track_id):
    return f"data/bpm_cache/{track_id}.json"

def is_cached(track_id):
    return os.path.exists(cache_path(track_id))

def has_result(track_id):
    p = cache_path(track_id)
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return json.load(f).get("bpm") is not None

# ─── Nettoyage du titre ──────────────────────────────────────────────────────
import re as _re
import unicodedata

def ascii_normalize(text):
    """Convertit les caractères accentués/spéciaux en ASCII."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").strip()

def clean_title(title):
    t = _re.sub(r"\s*[\(\[].*?[\)\]]", "", str(title))  # retire (feat. X), [Radio Edit]...
    t = _re.sub(r"\s*-\s*(Extended|Remix|Edit|Version|Remaster.*|Live.*|Acoustic.*)$", "", t, flags=_re.I)
    return t.strip()

def first_artist(artist):
    """Extrait le premier artiste d'un feat/& composé."""
    return _re.split(r"\s*[&,]\s*|\s+feat\.?\s+|\s+ft\.?\s+", str(artist), flags=_re.I)[0].strip()

# ─── Fetch ────────────────────────────────────────────────────────────────────
# ─── Rate limiter global ─────────────────────────────────────────────────────
_rate_lock  = threading.Lock()
_call_times = []  # timestamps des appels dans la dernière heure

def _rate_limited_get(params):
    """Attend si nécessaire pour rester sous MAX_PER_HR appels/heure."""
    while True:
        with _rate_lock:
            now = time.time()
            cutoff = now - 3600
            _call_times[:] = [t for t in _call_times if t > cutoff]
            if len(_call_times) < MAX_PER_HR:
                _call_times.append(now)
                break
        time.sleep(1)

    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("search", [])

def search_api(lookup, search_type="both", limit=5):
    return _rate_limited_get(
        {"api_key": GETSONGBPM_API_KEY, "type": search_type, "lookup": lookup, "limit": limit}
    )

def extract_result(songs):
    for song in songs:
        if song.get("tempo"):
            return {
                "bpm":          float(song["tempo"]),
                "key":          song.get("key_of"),
                "open_key":     song.get("open_key"),
                "time_sig":     song.get("time_sig"),
                "danceability": song.get("danceability"),
                "acousticness": song.get("acousticness"),
            }
    return None

def fetch_bpm(track_id, title, artist):
    p = cache_path(track_id)

    if args.retry_empty:
        if has_result(track_id):
            with open(p) as f:
                return json.load(f)
    elif is_cached(track_id):
        with open(p) as f:
            return json.load(f)

    result = {"bpm": None, "key": None, "open_key": None,
              "time_sig": None, "danceability": None, "acousticness": None}
    try:
        clean   = clean_title(title)
        artist1 = first_artist(artist)
        a_ascii = ascii_normalize(artist1)
        t_ascii = ascii_normalize(clean)

        attempts = [
            # (lookup, type)
            (f"song:{clean} artist:{artist}",   "both"),   # exact
            (f"song:{clean} artist:{artist1}",  "both"),   # premier artiste
            (f"song:{t_ascii} artist:{a_ascii}", "both"),  # ascii normalisé
            (f"{clean} {artist1}",              "song"),   # type=song
            (f"{t_ascii} {a_ascii}",            "song"),   # ascii + type=song
        ]

        found = None
        for lookup, stype in attempts:
            songs = search_api(lookup, stype)
            found = extract_result(songs)
            if found:
                break

        if found:
            result = found

    except Exception:
        pass

    with open(p, "w") as f:
        json.dump(result, f)
    return result

# ─── Chargement ───────────────────────────────────────────────────────────────
if not os.path.exists("data/tracks_spotify.csv"):
    print("⚠ data/tracks_spotify.csv introuvable — lance d'abord 01_extract.py")
    exit(1)

df = pd.read_csv("data/tracks_spotify.csv")
total     = len(df)
cached    = df["track_id"].apply(is_cached).sum()
with_data = df["track_id"].apply(has_result).sum()
remaining = total - cached

print(f"État du cache BPM :")
print(f"  {with_data}/{total} morceaux avec BPM")
print(f"  {cached - with_data} cachés sans résultat")
print(f"  {remaining} non encore interrogés")

if args.status:
    exit(0)

# ─── Sélection des morceaux à traiter ────────────────────────────────────────
if args.retry_empty:
    to_fetch = df[~df["track_id"].apply(has_result)]
else:
    to_fetch = df[~df["track_id"].apply(is_cached)]

if len(to_fetch) == 0:
    print("\nTout est déjà en cache.")
else:
    est_min = len(to_fetch) * 0.05 / 60 * 4  # estimation avec 4 workers
    print(f"\nFetch BPM : {len(to_fetch)} morceaux (~{est_min:.0f} min estimées)...")

    counter = [0]
    lock = threading.Lock()

    def process(row):
        result = fetch_bpm(row["track_id"], row["track_name"], row["artist"])
        with lock:
            counter[0] += 1
            bpm = f"{result['bpm']:.0f} BPM" if result["bpm"] else "—"
            calls = len(_call_times)
            print(f"  [{counter[0]}/{len(to_fetch)}] {str(row['track_name'])[:30]:<30} {bpm:<10} ({calls} appels/h)", end="\r")
        return result

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process, [row for _, row in to_fetch.iterrows()]))
    print()

# ─── Merge et sauvegarde ─────────────────────────────────────────────────────
BPM_COLS = ["bpm", "key", "open_key", "time_sig", "danceability", "acousticness"]

bpm_rows = []
for _, row in df.iterrows():
    p = cache_path(row["track_id"])
    if os.path.exists(p):
        with open(p) as f:
            bpm_rows.append({"track_id": row["track_id"], **json.load(f)})
    else:
        bpm_rows.append({"track_id": row["track_id"], **{c: None for c in BPM_COLS}})

bpm_df = pd.DataFrame(bpm_rows)

def merge_bpm(target_df, bpm_df):
    drop = [c for c in BPM_COLS if c in target_df.columns]
    return target_df.drop(columns=drop).merge(bpm_df, on="track_id", how="left")

df = merge_bpm(df, bpm_df)
df.to_csv("data/tracks_spotify.csv", index=False)
print(f"→ data/tracks_spotify.csv mis à jour")

if os.path.exists("data/tracks.csv"):
    tracks_full = pd.read_csv("data/tracks.csv")
    tracks_full = merge_bpm(tracks_full, bpm_df)
    tracks_full.to_csv("data/tracks.csv", index=False)
    print(f"→ data/tracks.csv mis à jour")

final_count = bpm_df["bpm"].notna().sum()
print(f"\nTerminé : {final_count}/{total} morceaux avec BPM.")
