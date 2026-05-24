# Analyse de playlists Spotify

Pipeline complet : extraction → clustering → analyse musicale → analyse des paroles → rapport HTML.

## Installation

```bash
pip install spotipy pandas numpy scikit-learn matplotlib umap-learn \
            lyricsgenius nltk textblob wordcloud langdetect

python -m nltk.downloader punkt stopwords vader_lexicon averaged_perceptron_tagger
```

## Configuration

### 1. Spotify Developer
1. Va sur https://developer.spotify.com/dashboard
2. Crée une application
3. Copie **Client ID** et **Client Secret**
4. Dans les settings de l'app, ajoute `http://localhost:8888/callback` comme Redirect URI
5. Renseigne ces valeurs dans `01_extract.py`

### 2. Genius API (pour les paroles)
1. Va sur https://genius.com/api-clients
2. Crée une app → copie **Client Access Token**
3. Renseigne-le dans `04_analyse_paroles.py`

## Utilisation

Lance les scripts dans l'ordre :

```bash
cd spotify_analysis

# 1. Extraire les playlists (crée data/tracks.csv)
python 01_extract.py

# 2. Clustering par similarité (crée data/tracks_clustered.csv)
python 02_clustering.py

# 3. Analyse musicale des groupes
python 03_analyse_musicale.py

# 4. Analyse des paroles (peut être lent selon le nombre de morceaux)
python 04_analyse_paroles.py

# 5. Rapport HTML final
python 05_rapport.py
# → ouvre rapport.html dans le navigateur
```

## Fichiers produits

```
data/
  tracks.csv              — tous les morceaux + audio features
  tracks_clustered.csv    — avec numéro de cluster
  cluster_summary.csv     — profil moyen par cluster
  lyrics_analysis.csv     — métriques des paroles
  lyrics_cache/           — cache des paroles (évite les re-téléchargements)

figures/
  elbow.png               — choix du nombre de clusters
  umap_clusters.png       — carte 2D des morceaux
  cluster_profiles.png    — radars par groupe
  feature_distributions.png
  lyrics_metrics.png
  lyrics_wordclouds.png

rapport.html              — tableau de bord complet
```

## Audio features Spotify (valeurs entre 0 et 1 sauf indication)

| Feature | Description |
|---------|-------------|
| `energy` | Intensité perçue (0 = calme, 1 = intense) |
| `danceability` | Aptitude à danser |
| `valence` | Positivité émotionnelle (0 = sombre, 1 = joyeux) |
| `acousticness` | Probabilité que le son soit acoustique |
| `instrumentalness` | Absence de voix |
| `speechiness` | Présence de paroles parlées/rap |
| `tempo` | BPM |
| `key` | Tonalité (0=C, 1=C#, ..., 11=B) |
| `mode` | 0 = mineur, 1 = majeur |
