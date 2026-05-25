# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "matplotlib"]
# ///
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
    hue = c * 360 // int(summary["cluster"].max() + 1)

    if has_lyrics:
        lsub = lyrics[lyrics["cluster"] == c]
        def _lmean(col):
            v = lsub[col].mean() if not lsub.empty else float("nan")
            return f"{v:+.2f}" if col == "sentiment_polarity" else f"{v:.2f}" if not (v != v) else "N/A"
        sent_str   = _lmean("sentiment_polarity")
        vocab_str  = _lmean("vocab_richness")
        repeat_str = _lmean("repetition_rate")
    else:
        sent_str = vocab_str = repeat_str = "N/A"

    def fmt(val, fmt_str, fallback="N/A"):
        try:
            v = float(val)
            return "N/A" if np.isnan(v) else format(v, fmt_str)
        except (TypeError, ValueError):
            return fallback

    year_str = fmt(row.get("release_year_mean"), ".0f")
    bpm_str  = fmt(row.get("bpm_mean"), ".0f")
    dance_str = fmt(row.get("danceability_mean"), ".0f")
    acou_str  = fmt(row.get("acousticness_mean"), ".0f")

    listeners = row.get("listeners_mean")
    try:
        l = float(listeners)
        if np.isnan(l):
            audience_str = "N/A"
        elif l >= 10_000_000:
            audience_str = f"{l/1e6:.1f}M — mainstream"
        elif l >= 1_000_000:
            audience_str = f"{l/1e6:.1f}M — populaire"
        elif l >= 100_000:
            audience_str = f"{l/1e3:.0f}K — indie"
        else:
            audience_str = f"{l/1e3:.0f}K — niche"
    except (TypeError, ValueError):
        audience_str = "N/A"

    cluster_cards += f"""
    <div class="card">
      <div class="card-header" style="background: hsl({hue}, 55%, 55%);">
        Groupe {c+1}
        <span class="badge">{n} titres</span>
      </div>
      <div class="card-body">
        <table>
          <tr><th>Genre dominant</th><td>{row.get('dominant_genre', 'N/A')}</td></tr>
          <tr><th>Top tags</th><td style="font-size:11px">{str(row.get('top_tags', ''))[:60]}</td></tr>
          <tr><th>Audience</th><td style="font-size:11px">{audience_str}</td></tr>
          <tr><th>Année moy.</th><td>{year_str}</td></tr>
          <tr><th>BPM moy.</th><td>{bpm_str}</td></tr>
          <tr><th>Danceability</th><td>{dance_str}/100</td></tr>
          <tr><th>Acousticness</th><td>{acou_str}/100</td></tr>
          <tr><th colspan="2" style="padding-top:8px;color:#666">Paroles</th></tr>
          <tr><th>Sentiment</th><td>{sent_str}</td></tr>
          <tr><th>Richesse vocab.</th><td>{vocab_str}</td></tr>
          <tr><th>Répétitivité</th><td>{repeat_str}</td></tr>
          <tr><th>Top artistes</th><td style="font-size:11px">{row['top_artists']}</td></tr>
        </table>
      </div>
    </div>
    """

FIGURE_DESCRIPTIONS = {
    "figures/umap_clusters.png": (
        "Carte des clusters",
        "Chaque point est un morceau. Les points proches partagent des tags et un style similaires — "
        "le graphe de gauche colorie par groupe, celui de droite par popularité Last.fm. "
        "<strong>Ce qu'on peut en faire :</strong> si deux groupes sont visuellement proches, leurs styles sont compatibles "
        "pour construire une playlist de transition. Un point isolé loin de son groupe est un titre atypique de ta collection."
    ),
    "figures/cluster_genres.png": (
        "Top tags par groupe",
        "Les tags les plus fréquents de chaque groupe, triés par nombre d'occurrences. "
        "Ce sont les mots-clés qui décrivent le mieux chaque style. "
        "<strong>Ce qu'on peut en faire :</strong> utilise ces tags directement dans un prompt Suno — "
        "ex. <em>\"electronic, synthwave, dark, 140 BPM\"</em> — pour générer de la musique dans cet univers."
    ),
    "figures/genre_distribution.png": (
        "Distribution des genres par groupe",
        "Part relative de chaque grande famille de genres dans chaque groupe (barres groupées). "
        "Un groupe avec 80 % Électronique est homogène ; un groupe avec beaucoup de catégories mélangées "
        "est un \"fourre-tout\" de ta collection. "
        "<strong>Ce qu'on peut en faire :</strong> identifier les groupes homogènes (bons candidats pour une playlist cohérente) "
        "vs. les groupes hétérogènes (à re-découper ou à explorer)."
    ),
    "figures/lyrics_metrics.png": (
        "Métriques des paroles par groupe",
        "Boîtes à moustaches comparant 5 métriques sur les paroles analysées. "
        "<strong>Sentiment</strong> (-1 négatif → +1 positif), "
        "<strong>Richesse vocab.</strong> (ratio mots uniques / total), "
        "<strong>Répétitivité</strong> (part de mots répétés — élevé = refrain accrocheur), "
        "<strong>Mots/phrase</strong> (complexité syntaxique), "
        "<strong>Nb mots</strong> (longueur des textes). "
        "<strong>Ce qu'on peut en faire :</strong> un groupe à sentiment positif + répétitivité élevée = "
        "musique festive/pop ; sentiment négatif + richesse vocab. élevée = musique introspective/poétique."
    ),
    "figures/lyrics_wordclouds.png": (
        "Champs lexicaux par groupe",
        "Les mots les plus fréquents dans les paroles de chaque groupe (stopwords retirés). "
        "La taille d'un mot est proportionnelle à sa fréquence. "
        "<strong>Ce qu'on peut en faire :</strong> ces thèmes récurrents révèlent l'univers émotionnel du groupe — "
        "amour, nuit, liberté, colère... Intègre-les dans ton prompt Suno pour ancrer l'ambiance des paroles."
    ),
    "figures/perles_cachees.png": (
        "Perles cachées",
        "Les 40 titres de ta collection avec le moins d'auditeurs Last.fm — tes découvertes les plus nichées. "
        "Plus une barre est courte, plus l'artiste est confidentiel. "
        "<strong>Ce qu'on peut en faire :</strong> une playlist \"underground\" de tes titres les plus rares, "
        "ou un point de départ pour explorer des artistes que peu de gens connaissent."
    ),
    "figures/evolution_temporelle.png": (
        "Évolution de tes goûts par décennie",
        "Répartition des genres dans ta collection selon l'année de sortie des titres. "
        "Chaque barre représente une décennie, les couleurs les familles de genres. "
        "<strong>Ce qu'on peut en faire :</strong> identifier à quelle époque tu écoutes le plus de musique, "
        "et voir si tes goûts ont évolué (ex. plus d'électronique dans les années 2010 ?)."
    ),
    "figures/carte_emotionnelle.png": (
        "Carte émotionnelle des groupes",
        "Chaque point est un groupe, positionné selon son acousticness moyen (axe horizontal) "
        "et le sentiment médian de ses paroles (axe vertical). La taille ∝ au nombre de titres. "
        "<strong>Quadrants :</strong> Festif acoustique (haut-droite), Festif électronique (haut-gauche), "
        "Mélancolique acoustique (bas-droite), Sombre électronique (bas-gauche). "
        "<strong>Ce qu'on peut en faire :</strong> choisir une playlist selon ton humeur du moment."
    ),
    "figures/artistes_pivot.png": (
        "Artistes pivot",
        "Artistes présents dans 3 groupes ou plus — ils traversent plusieurs univers de ta collection. "
        "Ce sont tes artistes les plus \"transversaux\", dont le style est difficile à catégoriser. "
        "<strong>Ce qu'on peut en faire :</strong> ces artistes sont de bons points d'entrée pour explorer "
        "des groupes voisins, ou pour construire des playlists de transition entre styles."
    ),
}

figures_html = ""
for path, (title, description) in FIGURE_DESCRIPTIONS.items():
    b64 = img_to_b64(path)
    if b64:
        figures_html += f"""
        <div class="figure-block">
          <h3>{title}</h3>
          <img src="{b64}" style="max-width:100%;border-radius:8px;">
          <p class="figure-caption">{description}</p>
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
  .card {{ border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; width: 230px; }}
  .card-header {{ color: white; font-weight: 600; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }}
  .badge {{ background: rgba(255,255,255,0.25); border-radius: 20px; padding: 2px 10px; font-size: 12px; font-weight: 400; }}
  .card-body {{ padding: 12px 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; color: #666; font-weight: 500; padding: 3px 0; width: 45%; }}
  td {{ text-align: right; padding: 3px 0; }}
  .figure-block {{ margin: 32px 0; }}
  .figure-caption {{ margin-top: 10px; font-size: 13px; color: #555; line-height: 1.6; max-width: 900px; background: #f8f8f8; border-left: 3px solid #ddd; padding: 10px 14px; border-radius: 0 6px 6px 0; }}
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
