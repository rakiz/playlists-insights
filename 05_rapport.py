"""
Script 5 — Rapport de synthèse
Combine toutes les analyses en un tableau de bord HTML.
Résultat : rapport.html
"""

import pandas as pd
import numpy as np
import base64
import os

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

# ─── Données ──────────────────────────────────────────────────────────────────
tracks  = pd.read_csv("data/tracks_clustered.csv")
summary = pd.read_csv("data/cluster_summary.csv")

try:
    lyrics = pd.read_csv("data/lyrics_analysis.csv")
    has_lyrics = True
except FileNotFoundError:
    has_lyrics = False

# ─── Construction du HTML ────────────────────────────────────────────────────
cluster_cards = ""
for _, row in summary.iterrows():
    c = int(row["cluster"])
    n = int(row["n_tracks"])

    if has_lyrics:
        lsub = lyrics[lyrics["cluster"] == c]
        sent    = lsub["sentiment_polarity"].mean()
        vocab   = lsub["vocab_richness"].mean()
        repeat  = lsub["repetition_rate"].mean()
        sent_str   = f"{sent:+.2f}"
        vocab_str  = f"{vocab:.2f}"
        repeat_str = f"{repeat:.2f}"
    else:
        sent_str = vocab_str = repeat_str = "N/A"

    cluster_cards += f"""
    <div class="card">
      <div class="card-header" style="background: hsl({(c * 360 // int(summary['cluster'].max()+1))}, 55%, 55%);">
        Groupe {c+1}
        <span class="badge">{n} titres</span>
      </div>
      <div class="card-body">
        <table>
          <tr><th>Tonalité</th><td>{row['dominant_key']} {row['dominant_mode']}</td></tr>
          <tr><th>Tempo</th><td>{row['tempo_mean']:.0f} BPM</td></tr>
          <tr><th>Énergie</th><td>{row['energy_mean']:.2f}</td></tr>
          <tr><th>Dansabilité</th><td>{row['danceability_mean']:.2f}</td></tr>
          <tr><th>Positivité</th><td>{row['valence_mean']:.2f}</td></tr>
          <tr><th>Acoustique</th><td>{row['acousticness_mean']:.2f}</td></tr>
          <tr><th>Popularité</th><td>{row['popularity_mean']:.1f}/100</td></tr>
          <tr><th colspan="2" style="padding-top:8px;color:#666">Paroles</th></tr>
          <tr><th>Sentiment</th><td>{sent_str}</td></tr>
          <tr><th>Richesse vocab.</th><td>{vocab_str}</td></tr>
          <tr><th>Répétitivité</th><td>{repeat_str}</td></tr>
          <tr><th>Top artistes</th><td style="font-size:12px">{row['top_artists']}</td></tr>
        </table>
      </div>
    </div>
    """

figures_html = ""
for title, path in [
    ("Carte des clusters",       "figures/umap_clusters.png"),
    ("Profils musicaux (radar)", "figures/cluster_profiles.png"),
    ("Distributions des features","figures/feature_distributions.png"),
    ("Métriques des paroles",    "figures/lyrics_metrics.png"),
    ("Champs lexicaux",          "figures/lyrics_wordclouds.png"),
]:
    b64 = img_to_b64(path)
    if b64:
        figures_html += f"""
        <div class="figure-block">
          <h3>{title}</h3>
          <img src="{b64}" style="max-width:100%;border-radius:8px;">
        </div>"""

html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Analyse Spotify — Rapport</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 20px; margin: 40px 0 16px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
  h3 {{ font-size: 16px; margin: 0 0 8px; }}
  .meta {{ color: #888; font-size: 14px; margin-bottom: 32px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; width: 220px; }}
  .card-header {{ color: white; font-weight: 600; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }}
  .badge {{ background: rgba(255,255,255,0.25); border-radius: 20px; padding: 2px 10px; font-size: 12px; font-weight: 400; }}
  .card-body {{ padding: 12px 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #666; font-weight: 500; padding: 3px 0; width: 45%; }}
  td {{ text-align: right; padding: 3px 0; }}
  .figure-block {{ margin: 24px 0; }}
</style>
</head>
<body>
  <h1>Analyse de tes playlists Spotify</h1>
  <p class="meta">
    {len(tracks)} morceaux uniques · {summary['cluster'].nunique()} groupes · 
    {int(tracks['playlist_name'].nunique())} playlists
  </p>

  <h2>Résumé par groupe</h2>
  <div class="cards">{cluster_cards}</div>

  <h2>Visualisations</h2>
  {figures_html if figures_html else "<p><em>Lance les scripts 02, 03 et 04 pour générer les figures.</em></p>"}

</body>
</html>"""

with open("rapport.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Rapport généré → rapport.html")
print("Ouvre-le dans ton navigateur.")
