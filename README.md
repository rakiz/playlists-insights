# Playlists Insights

Analyse complète de tes playlists Spotify : clustering par style musical, analyse des paroles, génération de prompts [Suno](https://suno.com), et création automatique des playlists dans ton compte.

## Ce que ça fait

1. **Extrait** toutes tes playlists Spotify et les enrichit avec les données Last.fm (tags de genre, audience) et GetSongBPM (BPM, tonalité, danceability)
2. **Groupe** les morceaux par similarité stylistique en 20–30 clusters (KMeans + TF-IDF sur les tags + features audio)
3. **Analyse** les paroles via Genius (sentiment, richesse du vocabulaire, thèmes récurrents)
4. **Génère** un rapport HTML interactif, des prompts Suno par groupe, et crée les playlists directement dans Spotify

## Installation

```bash
brew install uv
git clone ...
cd playlists-insights
```

Copie `.env.example` en `.env` et remplis tes clés API (voir section Configuration).

## Configuration

| Variable | Source | Obligatoire |
|----------|--------|-------------|
| `SPOTIFY_CLIENT_ID` | [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) | ✓ |
| `SPOTIFY_CLIENT_SECRET` | idem | ✓ |
| `SPOTIFY_REDIRECT_URI` | `http://127.0.0.1:8888/callback` (défaut) | |
| `LASTFM_API_KEY` | [last.fm/api/account/create](https://www.last.fm/api/account/create) | ✓ |
| `LASTFM_API_SECRET` | idem | ✓ |
| `GENIUS_ACCESS_TOKEN` | [genius.com/api-clients](https://genius.com/api-clients) | pour l'analyse des paroles |
| `GETSONGBPM_API_KEY` | [getsongbpm.com/api](https://getsongbpm.com/api) | pour le BPM |

## Pipeline complet

```
01_extract.py                   Spotify + Last.fm        → data/tracks.csv
01b_audio.py                    previews audio (optionnel, souvent vide post-2024)
01c_bpm.py                      GetSongBPM API           → BPM, tonalité, danceability
02_clustering.py                KMeans + UMAP            → data/tracks_clustered.csv
03_analyse_musicale.py          genres, tags, stats      → data/cluster_summary.csv
04_analyse_paroles.py           Genius + NLP             → data/lyrics_analysis.csv
05_rapport.py                   tableau de bord HTML     → rapport.html
06_playlists.py                 création Spotify         → playlists dans ton compte
07_suno_prompts.py              prompts IA               → data/suno_prompts.csv
08_analyses_supplementaires.py  perles cachées, évolution, carte émotionnelle
```

### Lancement

```bash
# Étape 1 — extraction (une seule fois, ou quand tu ajoutes des playlists)
uv run 01_extract.py --all
uv run 01c_bpm.py                    # ~1h30 pour 3500 titres

# Étape 2 — clustering et analyse
uv run 02_clustering.py
uv run 03_analyse_musicale.py
uv run 04_analyse_paroles.py         # tourne en boucle, Ctrl+C pour interrompre

# Étape 3 — résultats
uv run 05_rapport.py && open rapport.html
uv run 06_playlists.py               # crée les playlists dans Spotify
uv run 07_suno_prompts.py            # affiche les prompts Suno
uv run 08_analyses_supplementaires.py
```

---

## Scripts en détail

### `01_extract.py` — Extraction Spotify + Last.fm

Récupère toutes tes playlists Spotify et enrichit chaque morceau avec les tags Last.fm (genres, styles, humeurs, audience).

```bash
uv run 01_extract.py --list             # liste les playlists disponibles
uv run 01_extract.py --all              # extrait tout
uv run 01_extract.py --select 0 2 5    # sélectionne des playlists par numéro
uv run 01_extract.py --refresh-spotify # force une nouvelle extraction Spotify
```

**Checkpoints :** `data/tracks_spotify.csv` évite de re-fetcher Spotify à chaque run. Le cache Last.fm (`data/lastfm_cache/`) stocke les résultats par artiste+titre et n'est jamais re-fetchés.

**Parallélisme :** 4 workers pour Last.fm. Fallback sur les tags artiste si le titre n'est pas trouvé.

---

### `01c_bpm.py` — BPM et données audio via GetSongBPM

Récupère BPM, tonalité musicale, danceability et acousticness via l'API [GetSongBPM](https://getsongbpm.com). Ces features sont utilisées par le clustering pour différencier les groupes par énergie et tempo.

```bash
uv run 01c_bpm.py                # fetch tout
uv run 01c_bpm.py --status       # état du cache sans fetch
uv run 01c_bpm.py --retry-empty  # retente les titres sans résultat
```

- Limite : 3000 req/heure (tier gratuit), gérée automatiquement
- Cache : `data/bpm_cache/` (JSON par track_id)
- Stratégie : 5 tentatives par titre (nettoyage progressif du titre, normalisation ASCII, premier artiste uniquement)
- Couverture typique : ~60% (les titres peu connus ne sont pas dans la base)

> **Note :** La tonalité retournée utilise les symboles unicode ♯/♭ — normalisés automatiquement via `utils.py`.

---

### `02_clustering.py` — Clustering musical

Cœur du pipeline. Groupe les morceaux par similarité en combinant plusieurs types de features :

- **Tags stylistiques** (genres Last.fm) — pondérés par TF-IDF pour downweighter les tags ultra-génériques (`pop`, `rock`, `electronic`) et valoriser les tags rares (`shoegaze`, `trip-hop`, `chanson française`) — réduits ensuite par TruncatedSVD (LSA) de ~860 à 100 dimensions
- **BPM, danceability, acousticness** — pondérés ×5 pour compenser les 100 dimensions de tags
- **Tonalité** — encodée sur le cercle des quintes (sin + cos) pour respecter la continuité C → G → D...
- **Année de sortie** et **audience** (proxy : `lastfm_listeners`)

Le nombre de clusters est choisi automatiquement par score silhouette (MIN_K=15). Les micro-clusters < 20 titres sont fusionnés au plus proche. Les clusters > 300 titres (trop hétérogènes pour une playlist) sont redécoupés par décennie de sortie.

**Déduplication :** les doublons (versions Remastered, Deluxe, feat. différents) sont supprimés dès cette étape via `utils.normalize_title()` — le titre avec le plus d'auditeurs Last.fm est conservé.

Produit :
- `data/tracks_clustered.csv` — tous les morceaux avec leur numéro de cluster et coordonnées UMAP
- `figures/umap_clusters.png` — carte 2D des clusters (gauche : par groupe, droite : par audience)
- `figures/elbow.png` — courbe inertie/silhouette pour visualiser le choix de k

---

### `03_analyse_musicale.py` — Analyse des genres et statistiques

Pour chaque cluster, calcule le genre dominant, les tags les plus fréquents, l'audience moyenne, le BPM médian, la danceability, l'acousticness et les top artistes.

Les tags descriptifs (nationalité, "seen live", décennies, humeurs génériques) sont exclus du calcul de genre. Les tags restants sont classés dans 9 familles : Électronique, Rock/Metal, Pop, Hip-hop/Rap, Jazz/Soul/Funk, Folk/Acoustique, Chanson Française, Classique/Ambient, Musique du Monde.

Produit :
- `data/cluster_summary.csv`
- `figures/cluster_genres.png` — top tags par groupe
- `figures/genre_distribution.png` — répartition des familles de genres par groupe

---

### `04_analyse_paroles.py` — Analyse des paroles via Genius

Récupère les paroles de chaque morceau et calcule : sentiment positif/négatif (TextBlob), richesse du vocabulaire, taux de répétition (mesure l'accroche des refrains), longueur, langue détectée, structure couplets/refrains.

Fonctionne en **boucle automatique** par batches de 30 titres par cluster, en rotation sur tous les groupes — interruptible à tout moment avec Ctrl+C, reprend là où elle s'est arrêtée.

```bash
uv run 04_analyse_paroles.py   # tourne jusqu'à tout analyser, ou Ctrl+C
```

- Cache : `data/lyrics_cache/` (JSON par track_id) — les titres déjà analysés ne sont jamais re-fetchés
- 2 workers parallèles pour les appels Genius
- Append au CSV existant à chaque batch (pas de perte en cas d'interruption)

Produit :
- `data/lyrics_analysis.csv`
- `figures/lyrics_metrics.png` — boîtes à moustaches (sentiment, richesse vocab, répétitivité...)
- `figures/lyrics_wordclouds.png` — nuages de mots par groupe

---

### `05_rapport.py` — Rapport HTML

Génère `rapport.html` : tableau de bord complet avec une carte par groupe (genre, audience, BPM, sentiment, top artistes) et toutes les figures avec leurs explications et conseils d'utilisation.

```bash
uv run 05_rapport.py && open rapport.html
```

Peut être relancé à tout moment pour intégrer les dernières données.

---

### `06_playlists.py` — Création des playlists Spotify

Crée ou met à jour une playlist Spotify par cluster (minimum 30 titres), triée par BPM pour une écoute fluide. Les IDs de playlists sont mis en cache dans `data/playlists_created.json` pour mettre à jour les playlists existantes plutôt que d'en créer de nouvelles à chaque run.

```bash
uv run 06_playlists.py              # crée ou met à jour toutes les playlists
uv run 06_playlists.py --dry-run    # affiche ce qui serait créé sans toucher Spotify
uv run 06_playlists.py --cluster 2  # un seul cluster
```

> **Note Spotify :** l'endpoint `/users/{id}/playlists` retourne 403 pour les nouvelles apps (post-nov. 2024). Ce script utilise `/me/playlists` en contournement.

---

### `07_suno_prompts.py` — Génération de prompts Suno

Pour chaque cluster, génère un prompt prêt à coller dans [Suno](https://suno.com) en combinant : genre dominant, tags stylistiques filtrés, texture acoustique, énergie/danceability, tempo BPM médian, et thèmes lyriques issus de l'analyse des paroles.

```bash
uv run 07_suno_prompts.py
```

Affiche les prompts dans le terminal et les sauvegarde dans `data/suno_prompts.csv`.

Exemple de prompt généré :
```
Folk / Acoustique, folk, singer-songwriter, acoustic, chanson francaise,
semi-acoustic, groovy, rhythmic, upbeat tempo (110 BPM),
lyrical themes: love, would, away, well
```

---

### `08_analyses_supplementaires.py` — Analyses complémentaires

Génère 4 visualisations supplémentaires intégrées dans le rapport HTML :

- **Perles cachées** — les 40 titres avec le moins d'auditeurs Last.fm : tes découvertes les plus nichées (`figures/perles_cachees.png`, `data/perles_cachees.csv`)
- **Évolution temporelle** — répartition des genres par décennie, de 1960 à aujourd'hui (`figures/evolution_temporelle.png`)
- **Carte émotionnelle** — chaque groupe positionné selon son acousticness (axe X) et le sentiment de ses paroles (axe Y), avec 4 quadrants : Festif électronique, Festif acoustique, Sombre électronique, Mélancolique acoustique (`figures/carte_emotionnelle.png`)
- **Artistes pivot** — artistes présents dans 3 groupes ou plus, transversaux à ta collection (`figures/artistes_pivot.png`)

```bash
uv run 08_analyses_supplementaires.py
```

---

### `utils.py` — Fonctions partagées

Module interne utilisé par les scripts du pipeline :

- `normalize_title()` / `normalize_artist()` — supprime les suffixes d'édition (Remastered, feat., Deluxe, Live...) pour la déduplication
- `ascii_normalize()` — convertit les caractères unicode en ASCII pour les API (RÜFÜS → RUFUS)
- `KEY_ALIASES` — normalise les tonalités musicales (C♯ → Db, G♯ → Ab, unicode ♭/♯ inclus)

---

## Structure des données

```
data/
  tracks_spotify.csv         checkpoint extraction Spotify
  tracks.csv                 + tags Last.fm
  tracks_clustered.csv       + cluster, coordonnées UMAP
  cluster_summary.csv        résumé par cluster (genre, BPM médian, top artistes...)
  lyrics_analysis.csv        métriques paroles par morceau
  suno_prompts.csv           prompts Suno par cluster
  perles_cachees.csv         top 40 titres les plus nichés
  playlists_created.json     cache des IDs Spotify par cluster
  lastfm_cache/              JSON par artiste+titre
  bpm_cache/                 JSON par track_id
  lyrics_cache/              JSON par track_id
figures/
  umap_clusters.png
  elbow.png
  cluster_genres.png
  genre_distribution.png
  lyrics_metrics.png
  lyrics_wordclouds.png
  perles_cachees.png
  evolution_temporelle.png
  carte_emotionnelle.png
  artistes_pivot.png
```

---

## Notes techniques

- **Spotify post-nov. 2024 :** `/audio-features` et la popularité ne sont plus retournés pour les nouvelles apps. Ce projet utilise Last.fm (tags, audience) et GetSongBPM (BPM, tonalité) en remplacement.
- **TF-IDF sur les tags :** sans pondération, les tags `pop`/`rock`/`electronic` dominent le clustering et créent un mega-cluster fourre-tout. TF-IDF downweighte les tags trop communs pour faire émerger les styles de niche.
- **TruncatedSVD (LSA) :** KMeans performe mal sur des matrices creuses à haute dimension (~860 tags). SVD réduit à 100 dimensions denses avant le clustering.
- **Double-tempo BPM :** les algos de détection confondent parfois la pulsation avec le double (folk à 73 BPM détecté à 146). On utilise la médiane plutôt que la moyenne pour limiter l'impact des outliers.
- BPM data powered by [GetSongBPM](https://getsongbpm.com)
