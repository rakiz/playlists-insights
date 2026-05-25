# /// script
# requires-python = ">=3.12"
# dependencies = ["spotipy", "pandas", "numpy", "python-dotenv"]
# ///
"""
Script 6 — Création des playlists Spotify par cluster
Crée une playlist privée par groupe, triée par BPM.
Résultat : playlists créées dans ton compte Spotify

Usage :
  uv run 06_playlists.py              → crée ou met à jour toutes les playlists
  uv run 06_playlists.py --dry-run    → affiche ce qui serait créé sans toucher Spotify
  uv run 06_playlists.py --cluster 2  → un seul cluster
"""

import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import argparse
import os
import json
import time
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI

# ─── Auth ─────────────────────────────────────────────────────────────────────
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="playlist-modify-private playlist-read-private",
    cache_path=".cache",
))

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run",  action="store_true", help="Afficher sans créer")
parser.add_argument("--cluster",  type=int, default=None, help="Cluster spécifique")
args = parser.parse_args()

# ─── Données ──────────────────────────────────────────────────────────────────
tracks  = pd.read_csv("data/tracks_clustered.csv")
summary = pd.read_csv("data/cluster_summary.csv")
tracks  = tracks.dropna(subset=["cluster"])
tracks["cluster"] = tracks["cluster"].astype(int)

# tracks_clustered.csv est déjà dédupliqué par 02_clustering.py
tracks = tracks.drop_duplicates(subset=["track_id", "cluster"])

# ─── Nommage des playlists ────────────────────────────────────────────────────
SKIP_NAME_TAGS = {
    "pop", "rock", "indie", "alternative", "american", "british", "french", "german",
    "swedish", "female vocalists", "male vocalists", "female vocalist", "male vocalist",
    "seen live", "favourites", "favorite", "beautiful", "amazing", "all",
    "90s", "80s", "70s", "60s", "00s", "10s", "2000s", "2010s", "oldies",
}

def playlist_name(row):
    c     = int(row["cluster"])
    genre = row.get("dominant_genre", "Divers")
    bpm   = row.get("bpm_median") or row.get("bpm_mean")
    bpm_s = f" · {float(bpm):.0f} BPM" if bpm and not (isinstance(bpm, float) and np.isnan(bpm)) else ""
    tags  = [t.strip() for t in str(row.get("top_tags", "")).split(",")
             if t.strip() and t.strip().lower() not in SKIP_NAME_TAGS]
    tags_s = ", ".join(tags[:3])
    return f"Insights {c+1} — {genre}{bpm_s} · {tags_s}"

def playlist_desc(row):
    artists = row.get("top_artists", "")
    year    = row.get("release_year_mean")
    year_s  = f"Année moy. {float(year):.0f} · " if year and not np.isnan(float(year)) else ""
    n       = int(row.get("n_tracks", 0))
    return f"{year_s}{n} morceaux · Top artistes : {artists}"

# ─── Cache local des playlist_id créées ───────────────────────────────────────
CACHE_FILE = "data/playlists_created.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ─── Helpers Spotify ──────────────────────────────────────────────────────────
def get_user_id():
    return sp.current_user()["id"]

def create_or_update_playlist(user_id, name, description, track_ids, existing_id=None):
    if existing_id:
        sp.playlist_change_details(existing_id, name=name, description=description)
        pl_id = existing_id
        print(f"  Mise à jour : {name}")
    else:
        # /me/playlists fonctionne pour les nouvelles apps ; /users/{id}/playlists renvoie 403
        pl = sp._post("me/playlists", payload={"name": name, "public": False, "description": description})
        pl_id = pl["id"]
        print(f"  Créée : {name}")

    # Remplace les tracks (par batches de 100)
    uris = [f"spotify:track:{tid}" for tid in track_ids]
    sp.playlist_replace_items(pl_id, [])          # vide d'abord
    for i in range(0, len(uris), 100):
        sp.playlist_add_items(pl_id, uris[i:i+100])
        time.sleep(0.1)

    return pl_id

# ─── Boucle principale ────────────────────────────────────────────────────────
cache    = load_cache()
user_id  = None if args.dry_run else get_user_id()  # gardé pour unfollow/mise à jour

clusters = [args.cluster] if args.cluster is not None else sorted(tracks["cluster"].unique())

print(f"\n{'DRY RUN — ' if args.dry_run else ''}Création des playlists Spotify\n")

for c in clusters:
    row = summary[summary["cluster"] == c].iloc[0]
    sub = tracks[tracks["cluster"] == c].copy()

    # Tri par BPM pour une écoute fluide (BPM inconnu en fin de liste)
    sub["_bpm"] = pd.to_numeric(sub.get("bpm", pd.Series(dtype=float)), errors="coerce")
    sub = sub.sort_values("_bpm", na_position="last")

    name  = playlist_name(row)
    desc  = playlist_desc(row)
    tids  = sub["track_id"].tolist()

    print(f"Groupe {c+1} — {len(tids)} titres")
    print(f"  Nom  : {name}")
    print(f"  Desc : {desc}")

    if args.dry_run:
        print(f"  Premiers titres : {', '.join(sub['track_name'].head(5).tolist())}")
        print()
        continue

    existing_id = cache.get(str(c))
    pl_id = create_or_update_playlist(user_id, name, desc, tids, existing_id)
    cache[str(c)] = pl_id
    save_cache(cache)
    print(f"  → spotify:playlist:{pl_id}")
    print()

if not args.dry_run:
    print(f"Playlists créées/mises à jour → ouvre Spotify !")
    print(f"IDs sauvegardés dans {CACHE_FILE}")
