"""
Script 1 — Extraction des playlists Spotify
Récupère toutes tes playlists, les morceaux, et les audio features.
Résultat : data/tracks.csv
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time
import os
import argparse
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

SCOPE = "playlist-read-private playlist-read-collaborative"

os.makedirs("data", exist_ok=True)

# ─── Auth ─────────────────────────────────────────────────────────────────────
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SCOPE
))

# ─── Récupération des playlists ───────────────────────────────────────────────
def get_all_playlists():
    playlists, offset = [], 0
    while True:
        batch = sp.current_user_playlists(limit=50, offset=offset)
        playlists.extend(batch["items"])
        if batch["next"] is None:
            break
        offset += 50
    return playlists

# ─── Récupération des morceaux d'une playlist ─────────────────────────────────
def get_tracks(playlist_id, playlist_name=""):
    tracks, offset = [], 0
    while True:
        try:
            batch = sp.playlist_items(playlist_id, limit=100, offset=offset,
)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 403:
                print(f"     ⚠ Accès refusé pour '{playlist_name}' (playlist privée ou éditoriale) — ignorée")
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

# ─── Audio features par batch de 100 ─────────────────────────────────────────
def get_audio_features(track_ids):
    features = []
    for i in range(0, len(track_ids), 100):
        batch = sp.audio_features(track_ids[i:i+100])
        features.extend([f for f in batch if f])
        time.sleep(0.1)
    return {f["id"]: f for f in features}

# ─── Sélection des playlists ──────────────────────────────────────────────────
def select_playlists(playlists):
    """Affiche la liste et laisse l'utilisateur choisir."""
    print("\nPlaylists disponibles :")
    print(f"  {'N°':<4} {'Nom':<45} {'Titres':>6}")
    print("  " + "-" * 58)
    for i, pl in enumerate(playlists):
        n = pl.get('tracks', {}).get('total', '?') if pl.get('tracks') else '?'
        print(f"  {i:<4} {pl['name'][:44]:<45} {str(n):>6}")

    print("\nSélection (exemples) :")
    print("  'all'       → toutes les playlists")
    print("  '0 2 5'     → playlists n° 0, 2 et 5")
    print("  '0-4'       → playlists 0 à 4 inclus")
    print("  '0-3 7 10'  → combinaison intervalle + numéros")

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
    print(f"\n  → {len(chosen)} playlist(s) sélectionnée(s) :")
    for pl in chosen:
        n = pl.get('tracks', {}).get('total', '?') if pl.get('tracks') else '?'
        print(f"     • {pl['name']} ({n} titres)")
    return chosen


# ─── Parsing CLI ──────────────────────────────────────────────────────────────
# Usage :
#   python 01_extract.py                     → sélection interactive
#   python 01_extract.py --all               → toutes les playlists
#   python 01_extract.py --select "0 2 5"    → playlists n° 0, 2, 5
#   python 01_extract.py --select "0-4 7"    → intervalle + numéros
#   python 01_extract.py --list              → liste sans extraire

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--all",    action="store_true", help="Toutes les playlists")
group.add_argument("--select", type=str,            help="Numéros ex: '0 2 5' ou '0-4 7'")
group.add_argument("--list",   action="store_true", help="Afficher la liste et quitter")
args = parser.parse_args()

# ─── Main ─────────────────────────────────────────────────────────────────────
print("Récupération des playlists...")
playlists = get_all_playlists()
print(f"  {len(playlists)} playlists trouvées")

if args.list:
    print(f"\n  {'N°':<4} {'Nom':<45} {'Titres':>6}")
    print("  " + "-" * 58)
    for i, pl in enumerate(playlists):
        n = pl.get('tracks', {}).get('total', '?') if pl.get('tracks') else '?'
        print(f"  {i:<4} {pl['name'][:44]:<45} {str(n):>6}")
    exit(0)
elif args.all:
    print(f"  → toutes les playlists sélectionnées")
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
    n = pl.get('tracks', {}).get('total', '?') if pl.get('tracks') else '?'
    print(f"  → {pl['name']} ({n} morceaux)")
    tracks = get_tracks(pl["id"], pl["name"])

    if not tracks:
        continue

    track_ids = [t["id"] for t in tracks]
    af_map = get_audio_features(track_ids)

    for t in tracks:
        af = af_map.get(t["id"], {})
        rows.append({
            "playlist_id":   pl["id"],
            "playlist_name": pl["name"],
            "track_id":      t["id"],
            "track_name":    t["name"],
            "artist":        t["artists"][0]["name"],
            "artists_all":   ", ".join(a["name"] for a in t["artists"]),
            "album":         t["album"]["name"],
            "duration_ms":   t["duration_ms"],
            "popularity":    t["popularity"],
            # Audio features
            "danceability":  af.get("danceability"),
            "energy":        af.get("energy"),
            "valence":       af.get("valence"),       # positivité émotionnelle
            "tempo":         af.get("tempo"),          # BPM
            "acousticness":  af.get("acousticness"),
            "instrumentalness": af.get("instrumentalness"),
            "liveness":      af.get("liveness"),
            "speechiness":   af.get("speechiness"),
            "loudness":      af.get("loudness"),
            "key":           af.get("key"),            # 0=C, 1=C#, 2=D...
            "mode":          af.get("mode"),           # 0=mineur, 1=majeur
            "time_signature": af.get("time_signature"),
        })

df = pd.DataFrame(rows).drop_duplicates(subset="track_id")
df.to_csv("data/tracks.csv", index=False)
print(f"\nExtraction terminée : {len(df)} morceaux uniques → data/tracks.csv")
