# Playlists Insights

Analyse complète de tes playlists Spotify : clustering par style, analyse des paroles, et génération de profils musicaux pour créer de meilleurs prompts [Suno](https://suno.com).

## Ce que ça fait

1. **Extraction** — récupère tes playlists et morceaux depuis Spotify
2. **Enrichissement** — tags et genres via [Last.fm](https://www.last.fm), BPM/tonalité via [GetSongBPM](https://getsongbpm.com)
3. **Clustering** — groupe les morceaux par similarité musicale (genre, style, tempo)
4. **Analyse des paroles** — sentiment, richesse vocabulaire, thèmes via [Genius](https://genius.com)
5. **Rapport HTML** — tableau de bord visuel avec profils par groupe

## Pipeline

```
01_extract.py      → Spotify + Last.fm  → data/tracks.csv
01b_audio.py       → previews audio     → features locales (si disponibles)
01c_bpm.py         → GetSongBPM         → BPM, tonalité, danceability
02_clustering.py   → clustering         → data/tracks_clustered.csv
03_analyse_musicale.py → analyse genres → data/cluster_summary.csv
04_analyse_paroles.py  → analyse lyrics → data/lyrics_analysis.csv
05_rapport.py      → rapport HTML       → rapport.html
```

## Installation

```bash
brew install uv
uv sync
```

Copie `.env.example` en `.env` et remplis tes clés API.

## Configuration

| Variable | Source |
|----------|--------|
| `SPOTIFY_CLIENT_ID` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) |
| `SPOTIFY_CLIENT_SECRET` | idem |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8888/callback` (défaut) |
| `LASTFM_API_KEY` | [last.fm/api/account/create](https://www.last.fm/api/account/create) |
| `LASTFM_API_SECRET` | idem |
| `GENIUS_ACCESS_TOKEN` | [genius.com/api-clients](https://genius.com/api-clients) |
| `GETSONGBPM_API_KEY` | [getsongbpm.com/api](https://getsongbpm.com/api) |

## Notes

- L'endpoint Spotify `/audio-features` est déprécié pour les apps créées après novembre 2024. Ce projet utilise Last.fm et GetSongBPM en remplacement.
- BPM data powered by [GetSongBPM](https://getsongbpm.com)
