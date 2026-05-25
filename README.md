# Playlists Insights

Analyse complète de tes playlists Spotify : clustering par style musical, analyse des paroles, et génération de profils pour créer de meilleurs prompts [Suno](https://suno.com).

## Objectifs

1. **Comprendre ses goûts** — grouper les morceaux par similarité (genre, style, BPM, tonalité) et identifier les patterns dans les paroles
2. **Créer de meilleures playlists** — exporter des groupes cohérents directement dans Spotify
3. **Générer des prompts Suno** — transformer chaque groupe en prompt prêt à l'emploi pour la génération musicale IA

## Pipeline

```
01_extract.py          → Spotify + Last.fm       → data/tracks.csv
01b_audio.py           → previews audio (optionnel, souvent vide)
01c_bpm.py             → GetSongBPM API          → BPM, tonalité, danceability
02_clustering.py       → clustering KMeans/UMAP  → data/tracks_clustered.csv
03_analyse_musicale.py → genres, tags, stats     → data/cluster_summary.csv
04_analyse_paroles.py  → Genius + NLP            → data/lyrics_analysis.csv
05_rapport.py          → tableau de bord HTML    → rapport.html
```

### Lancer le pipeline complet

```bash
uv run 01_extract.py --all        # ou --select 0 2 5 pour choisir des playlists
uv run 01c_bpm.py                 # ~1h30 pour 3700 titres (limite 3000 req/heure)
uv run 02_clustering.py
uv run 03_analyse_musicale.py
uv run 04_analyse_paroles.py      # tourne en boucle, Ctrl+C pour interrompre
uv run 05_rapport.py && open rapport.html
```

## Installation

```bash
brew install uv
uv sync
```

Copie `.env.example` en `.env` et remplis tes clés API.

## Configuration

| Variable | Source | Obligatoire |
|----------|--------|-------------|
| `SPOTIFY_CLIENT_ID` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) | ✓ |
| `SPOTIFY_CLIENT_SECRET` | idem | ✓ |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8888/callback` (défaut) | |
| `LASTFM_API_KEY` | [last.fm/api/account/create](https://www.last.fm/api/account/create) | ✓ |
| `LASTFM_API_SECRET` | idem | ✓ |
| `GENIUS_ACCESS_TOKEN` | [genius.com/api-clients](https://genius.com/api-clients) | |
| `GETSONGBPM_API_KEY` | [getsongbpm.com/api](https://getsongbpm.com/api) | |

## Scripts en détail

### `01_extract.py` — Extraction Spotify + Last.fm

Récupère toutes tes playlists et les enrichit avec les tags Last.fm (genres, styles, humeurs).

```bash
uv run 01_extract.py --list           # liste les playlists disponibles
uv run 01_extract.py --all            # extrait tout
uv run 01_extract.py --select 0 2 5   # sélectionne par numéro
uv run 01_extract.py --refresh-spotify  # force une nouvelle extraction Spotify
```

- Checkpoint Spotify : `data/tracks_spotify.csv` (évite de re-fetcher si déjà extrait)
- Cache Last.fm : `data/lastfm_cache/` (JSON par artiste+titre, jamais re-fetchés)
- 4 workers parallèles pour Last.fm, fallback sur les tags artiste si le titre n'est pas trouvé

### `01b_audio.py` — Analyse audio locale (optionnel)

Télécharge les previews Spotify 30s et extrait BPM/énergie/tonalité via librosa.
**Note :** les previews sont à 0% pour les apps Spotify créées après novembre 2024 — ce script est rarement utile.

### `01c_bpm.py` — BPM et données audio via GetSongBPM

Récupère BPM, tonalité, danceability et acousticness via l'API [GetSongBPM](https://getsongbpm.com).
Limite : 3000 req/heure (tier gratuit). Le script gère le rate-limiting automatiquement.

```bash
uv run 01c_bpm.py                  # fetch tout
uv run 01c_bpm.py --status         # état du cache sans fetch
uv run 01c_bpm.py --retry-empty    # retente les titres sans résultat
```

- Cache : `data/bpm_cache/` (JSON par track_id)
- Stratégie de recherche : 5 tentatives par titre (nettoyage parenthèses, normalisation ASCII, premier artiste)
- Taux de réussite typique : ~60% (les titres peu connus ne sont pas dans la base)

### `02_clustering.py` — Clustering musical

Groupe les morceaux par similarité en combinant :
- **Tags stylistiques** (genres Spotify + tags Last.fm) — réduits via TruncatedSVD (LSA) pour gérer l'espace creux
- **BPM, danceability, acousticness** (si disponibles via GetSongBPM)
- **Tonalité** encodée sur le cercle des quintes (sin + cos)
- **Année de sortie**

Le nombre de clusters est choisi automatiquement par score silhouette (2–20). Les micro-clusters < 20 titres sont fusionnés au plus proche.

Génère :
- `data/tracks_clustered.csv` — tous les morceaux avec leur cluster + coordonnées UMAP
- `figures/umap_clusters.png` — carte 2D (clusters + audience niche/mainstream)
- `figures/elbow.png` — courbe inertie/silhouette pour justifier le choix de k

### `03_analyse_musicale.py` — Analyse des genres et stats

Pour chaque cluster : genre dominant, top tags, audience Last.fm (niche → mainstream), BPM moyen, danceability, acousticness, top artistes.

Les tags descriptifs (nationalité, "seen live", décennies...) sont exclus du calcul de genre dominant. Les tags stylistiques sont classés dans 9 familles : Électronique, Rock/Metal, Pop, Hip-hop/Rap, Jazz/Soul/Funk, Folk/Acoustique, Chanson Française, Classique/Ambient, Musique du Monde.

Génère :
- `data/cluster_summary.csv`
- `figures/cluster_genres.png` — top tags par groupe
- `figures/genre_distribution.png` — répartition des familles de genres

### `04_analyse_paroles.py` — Analyse des paroles via Genius

Récupère les paroles et calcule : sentiment (TextBlob), richesse du vocabulaire, répétitivité, nombre de mots, langue détectée (langdetect), structure (couplets/refrains).

Fonctionne en **boucle automatique par batches** de 30 titres par cluster — interruptible à tout moment avec Ctrl+C, reprend là où elle s'est arrêtée.

```bash
uv run 04_analyse_paroles.py   # tourne jusqu'à tout analyser
```

- Cache : `data/lyrics_cache/` (JSON par track_id)
- 2 workers parallèles pour les appels Genius
- Append au CSV existant à chaque batch — pas de perte en cas d'interruption

Génère :
- `data/lyrics_analysis.csv`
- `figures/lyrics_metrics.png` — boîtes à moustaches des métriques par groupe
- `figures/lyrics_wordclouds.png` — nuages de mots par groupe

### `05_rapport.py` — Rapport HTML

Génère `rapport.html` — tableau de bord complet avec les cartes par groupe (genre, audience, BPM, sentiment, top artistes) et toutes les figures avec leurs explications.

Peut être relancé à tout moment pour intégrer les dernières données de paroles.

## Structure des données

```
data/
  tracks_spotify.csv      # checkpoint après extraction Spotify
  tracks.csv              # + tags Last.fm
  tracks_clustered.csv    # + cluster, coordonnées UMAP
  cluster_summary.csv     # résumé par cluster
  lyrics_analysis.csv     # métriques paroles
  lastfm_cache/           # cache JSON par artiste+titre
  bpm_cache/              # cache JSON par track_id
  lyrics_cache/           # cache JSON par track_id
figures/
  umap_clusters.png
  elbow.png
  cluster_genres.png
  genre_distribution.png
  lyrics_metrics.png
  lyrics_wordclouds.png
```

## Notes techniques

- L'endpoint Spotify `/audio-features` est déprécié pour les apps créées après novembre 2024. Ce projet utilise Last.fm et GetSongBPM en remplacement.
- La popularité Spotify n'est plus retournée par l'API pour les nouvelles apps — remplacée par `lastfm_listeners` comme proxy d'audience.
- Le clustering utilise TruncatedSVD (LSA) pour réduire la matrice de tags creuse avant KMeans — sans ça, KMeans produit 2 clusters déséquilibrés sur 900+ dimensions.
- BPM data powered by [GetSongBPM](https://getsongbpm.com)
