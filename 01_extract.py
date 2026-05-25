"""
Script 1 — Extraction des playlists Spotify + enrichissement Last.fm
Récupère playlists, morceaux, genres artiste Spotify, et tags/stats Last.fm.
Résultat : data/tracks.csv
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import pylast
import time
import os
import re
import json
import argparse
import logging
logging.getLogger("spotipy").setLevel(logging.CRITICAL)
from config import (
    SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI,
    LASTFM_API_KEY, LASTFM_API_SECRET,
)

SCOPE = "playlist-read-private playlist-read-collaborative"

os.makedirs("data", exist_ok=True)
os.makedirs("data/lastfm_cache", exist_ok=True)

# ─── Auth ─────────────────────────────────────────────────────────────────────
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE,
))

network = pylast.LastFMNetwork(api_key=LASTFM_API_KEY, api_secret=LASTFM_API_SECRET)

# ─── Spotify ──────────────────────────────────────────────────────────────────
def get_all_playlists():
    playlists, offset = [], 0
    while True:
        batch = sp.current_user_playlists(limit=50, offset=offset)
        playlists.extend(batch["items"])
        if batch["next"] is None:
            break
        offset += 50
    return playlists

def get_tracks(playlist_id, playlist_name=""):
    tracks, offset = [], 0
    while True:
        try:
            batch = sp.playlist_items(playlist_id, limit=100, offset=offset)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 403:
                print(f"     ⚠ Accès refusé pour '{playlist_name}' — ignorée")
            else:
                print(f"     ⚠ Erreur {e.http_status} pour '{playlist_name}' — ignorée")
            return []
        for item in batch["items"]:
            t = item.get("item") or item.get("track")
            if t and t.get("id"):
                tracks.append(t)
        if batch["next"] is None:
            break
        offset += 100
    return tracks

def get_artist_genres_batch(artist_ids):
    genres = {}
    unique_ids = list(set(artist_ids))
    for i in range(0, len(unique_ids), 50):
        try:
            batch = sp.artists(unique_ids[i:i+50])
            for a in batch["artists"]:
                if a:
                    genres[a["id"]] = a.get("genres", [])
            time.sleep(0.1)
        except Exception:
            pass
    return genres

# ─── Last.fm ──────────────────────────────────────────────────────────────────
def _cache_path(artist, title):
    safe = re.sub(r"[^\w\s-]", "", f"{artist}_{title}").strip()[:80]
    return f"data/lastfm_cache/{safe}.json"

def get_lastfm_data(artist, title):
    cache = _cache_path(artist, title)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    result = {"tags": "", "listeners": None, "playcount": None}
    try:
        track = network.get_track(artist, title)
        top_tags = track.get_top_tags(limit=10)
        result["tags"] = ", ".join(t.item.get_name().lower() for t in top_tags)
        result["listeners"] = track.get_listener_count()
        result["playcount"]  = track.get_playcount()
    except Exception:
        pass

    # Fallback : tags de l'artiste si le track n'en a pas
    if not result["tags"]:
        try:
            artist_tags = network.get_artist(artist).get_top_tags(limit=10)
            result["tags"] = ", ".join(t.item.get_name().lower() for t in artist_tags)
        except Exception:
            pass

    with open(cache, "w") as f:
        json.dump(result, f)
    time.sleep(0.2)
    return result

# ─── Sélection des playlists ──────────────────────────────────────────────────
def select_playlists(playlists):
    print("\nPlaylists disponibles :")
    print(f"  {'N°':<4} {'Nom':<45} {'Titres':>6}")
    print("  " + "-" * 58)
    for i, pl in enumerate(playlists):
        n = pl.get("tracks", {}).get("total", "?") if pl.get("tracks") else "?"
        print(f"  {i:<4} {pl['name'][:44]:<45} {str(n):>6}")
    print("\nSélection : 'all' / '0 2 5' / '0-4' / '0-3 7'")
    raw = input("\nTon choix : ").strip()
    if raw.lower() == "all":
        return playlists
    selected = set()
    for part in raw.split():
        if "-" in part:
            a, b = part.split("-")
            selected.update(range(int(a), int(b) + 1))
        else:
            selected.add(int(part))
    chosen = [playlists[i] for i in sorted(selected) if i < len(playlists)]
    print(f"\n  → {len(chosen)} playlist(s) sélectionnée(s)")
    return chosen

# ─── Parsing CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--all",             action="store_true", help="Toutes les playlists")
group.add_argument("--select",          type=str,            help="Numéros ex: '0 2 5' ou '0-4 7'")
group.add_argument("--list",            action="store_true", help="Afficher la liste et quitter")
parser.add_argument("--refresh-spotify", action="store_true", help="Forcer une nouvelle extraction Spotify")
args = parser.parse_args()

SPOTIFY_CACHE = "data/tracks_spotify.csv"

# ─── Phase 1 : Extraction Spotify ────────────────────────────────────────────
if not args.refresh_spotify and os.path.exists(SPOTIFY_CACHE):
    print(f"Spotify déjà extrait ({SPOTIFY_CACHE}) — passage direct à Last.fm.")
    print("  (--refresh-spotify pour forcer une nouvelle extraction)")
    df = pd.read_csv(SPOTIFY_CACHE)
else:
    print("Récupération des playlists...")
    playlists = get_all_playlists()
    print(f"  {len(playlists)} playlists trouvées")

    if args.list:
        print(f"\n  {'N°':<4} {'Nom':<45} {'Titres':>6}")
        print("  " + "-" * 58)
        for i, pl in enumerate(playlists):
            n = pl.get("tracks", {}).get("total", "?") if pl.get("tracks") else "?"
            print(f"  {i:<4} {pl['name'][:44]:<45} {str(n):>6}")
        exit(0)
    elif args.all:
        print("  → toutes les playlists sélectionnées")
    elif args.select:
        selected = set()
        for part in args.select.split():
            if "-" in part:
                a, b = part.split("-")
                selected.update(range(int(a), int(b) + 1))
            else:
                selected.add(int(part))
        playlists = [playlists[i] for i in sorted(selected) if i < len(playlists)]
        print(f"  → {len(playlists)} playlist(s) sélectionnée(s)")
    else:
        playlists = select_playlists(playlists)

    rows = []
    for pl in playlists:
        n = pl.get("tracks", {}).get("total", "?") if pl.get("tracks") else "?"
        print(f"\n  → {pl['name']} ({n} morceaux)")
        tracks = get_tracks(pl["id"], pl["name"])
        if not tracks:
            continue

        artist_ids = [t["artists"][0]["id"] for t in tracks if t.get("artists")]
        genres_map = get_artist_genres_batch(artist_ids)

        for t in tracks:
            artist_id = t["artists"][0]["id"] if t.get("artists") else None
            spotify_genres = ", ".join(genres_map.get(artist_id, []))
            release = t["album"].get("release_date", "")
            release_year = int(release[:4]) if release and len(release) >= 4 else None

            rows.append({
                "playlist_id":    pl["id"],
                "playlist_name":  pl["name"],
                "track_id":       t["id"],
                "track_name":     t["name"],
                "artist":         t["artists"][0]["name"],
                "artists_all":    ", ".join(a["name"] for a in t["artists"]),
                "album":          t["album"]["name"],
                "duration_ms":    t.get("duration_ms"),
                "popularity":     t.get("popularity"),
                "release_year":   release_year,
                "preview_url":    t.get("preview_url"),
                "spotify_genres": spotify_genres,
                "lastfm_tags":       "",
                "lastfm_listeners":  None,
                "lastfm_playcount":  None,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset="track_id")
    df.to_csv(SPOTIFY_CACHE, index=False)
    print(f"\n  {len(df)} morceaux uniques sauvegardés → {SPOTIFY_CACHE}")

# ─── Phase 2 : Enrichissement Last.fm (parallèle) ────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

N_WORKERS = 4
print(f"\nEnrichissement Last.fm ({len(df)} morceaux, {N_WORKERS} workers)...")

results = {}
lock = threading.Lock()
counter = [0]

def fetch_row(args):
    idx, row = args
    data = get_lastfm_data(row["artist"], row["track_name"])
    with lock:
        counter[0] += 1
        print(f"  [{counter[0]}/{len(df)}] {str(row['track_name'])[:45]:<45}", end="\r")
    return idx, data

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {executor.submit(fetch_row, (idx, row)): idx for idx, row in df.iterrows()}
    for future in as_completed(futures):
        idx, data = future.result()
        results[idx] = data

tags_col      = {idx: d["tags"]      for idx, d in results.items()}
listeners_col = {idx: d["listeners"] for idx, d in results.items()}
playcount_col = {idx: d["playcount"] for idx, d in results.items()}

df["lastfm_tags"]      = pd.Series(tags_col,      dtype=object)
df["lastfm_listeners"] = pd.Series(listeners_col, dtype=object)
df["lastfm_playcount"] = pd.Series(playcount_col, dtype=object)

df.to_csv("data/tracks.csv", index=False)
print(f"\n\nExtraction terminée : {len(df)} morceaux uniques → data/tracks.csv")
