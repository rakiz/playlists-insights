# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "matplotlib"]
# ///
"""
Script 8 — Analyses supplémentaires
Génère 4 visualisations complémentaires :
  1. Perles cachées     — tes artistes niche les plus précieux
  2. Évolution temporelle — tes goûts par décennie
  3. Carte émotionnelle — clusters par sentiment × acousticness
  4. Artistes pivot     — artistes présents dans plusieurs groupes
Résultat : figures/perles_cachees.png, figures/evolution_temporelle.png,
           figures/carte_emotionnelle.png, figures/artistes_pivot.png
           + data/perles_cachees.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

tracks  = pd.read_csv("data/tracks_clustered.csv")
summary = pd.read_csv("data/cluster_summary.csv")
tracks["cluster"]          = pd.to_numeric(tracks["cluster"], errors="coerce")
tracks["lastfm_listeners"] = pd.to_numeric(tracks["lastfm_listeners"], errors="coerce")
tracks["release_year"]     = pd.to_numeric(tracks["release_year"], errors="coerce")
tracks["acousticness"]     = pd.to_numeric(tracks.get("acousticness", pd.Series(dtype=float)), errors="coerce")

try:
    lyrics = pd.read_csv("data/lyrics_analysis.csv")
    has_lyrics = True
except FileNotFoundError:
    has_lyrics = False

# ─── 1. Perles cachées ────────────────────────────────────────────────────────
print("1. Perles cachées...")

gems = tracks[tracks["lastfm_listeners"].notna()].sort_values("lastfm_listeners").head(40)

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(range(len(gems)), gems["lastfm_listeners"] / 1000, color="steelblue", alpha=0.75)
ax.set_yticks(range(len(gems)))
ax.set_yticklabels(
    [f"{r['artist']} — {str(r['track_name'])[:35]}" for _, r in gems.iterrows()],
    fontsize=8
)
ax.set_xlabel("Auditeurs Last.fm (milliers)")
ax.set_title("Perles cachées — tes 40 titres les plus nichés", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("figures/perles_cachees.png", dpi=150)
plt.close()

gems[["artist", "track_name", "lastfm_listeners", "cluster"]].to_csv("data/perles_cachees.csv", index=False)
print(f"  → figures/perles_cachees.png")

# ─── 2. Évolution temporelle ──────────────────────────────────────────────────
print("2. Évolution temporelle...")

GENRE_KEYWORDS = {
    "Rock / Metal":        ["rock", "metal", "punk", "grunge", "alternative rock", "hard rock"],
    "Électronique":        ["electronic", "electronica", "house", "techno", "edm", "synthpop", "ambient"],
    "Hip-hop / Rap":       ["hip-hop", "hip hop", "rap", "r&b"],
    "Folk / Acoustique":   ["folk", "acoustic", "singer-songwriter", "country", "indie folk"],
    "Jazz / Soul / Funk":  ["jazz", "soul", "funk", "blues"],
    "Chanson Française":   ["chanson francaise", "chanson française", "french pop"],
}

def classify(tags_str):
    if not isinstance(tags_str, str):
        return "Autres"
    tags = {t.strip().lower() for t in tags_str.split(",")}
    for genre, kws in GENRE_KEYWORDS.items():
        if tags & set(kws):
            return genre
    return "Autres"

tracks["decade"]    = (tracks["release_year"] // 10 * 10).astype("Int64")
tracks["genre_cat"] = tracks["lastfm_tags"].map(classify)

decade_genre = (
    tracks[tracks["decade"] >= 1960]
    .groupby(["decade", "genre_cat"]).size()
    .unstack(fill_value=0)
)

COLORS = {
    "Rock / Metal":       "#e74c3c",
    "Électronique":       "#3498db",
    "Hip-hop / Rap":      "#9b59b6",
    "Folk / Acoustique":  "#27ae60",
    "Jazz / Soul / Funk": "#f39c12",
    "Chanson Française":  "#e67e22",
    "Autres":             "#bdc3c7",
}
cols   = [c for c in COLORS if c in decade_genre.columns]
colors = [COLORS[c] for c in cols]

fig, ax = plt.subplots(figsize=(12, 6))
decade_genre[cols].plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.8)
ax.set_xlabel("Décennie")
ax.set_ylabel("Nombre de titres")
ax.set_title("Tes goûts musicaux par décennie", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.set_xticklabels([str(int(d)) for d in decade_genre.index], rotation=45)
plt.tight_layout()
plt.savefig("figures/evolution_temporelle.png", dpi=150)
plt.close()
print("  → figures/evolution_temporelle.png")

# ─── 3. Carte émotionnelle ────────────────────────────────────────────────────
print("3. Carte émotionnelle...")

if has_lyrics:
    lyric_agg = lyrics.groupby("cluster")["sentiment_polarity"].median().reset_index()
    lyric_agg.columns = ["cluster", "sentiment"]
    summary_emo = summary.merge(lyric_agg, on="cluster", how="left")
else:
    summary_emo = summary.copy()
    summary_emo["sentiment"] = np.nan

summary_emo["acousticness_mean"] = pd.to_numeric(summary_emo["acousticness_mean"], errors="coerce")

fig, ax = plt.subplots(figsize=(11, 8))

ax.axhline(0, color="#ccc", linewidth=1)
ax.axvline(50, color="#ccc", linewidth=1)
for (x0, x1, y0, y1, color, label) in [
    (0,  50,  0,  1,  "royalblue", "Festif électronique"),
    (50, 100, 0,  1,  "seagreen",  "Festif acoustique"),
    (0,  50, -1,  0,  "mediumpurple","Sombre électronique"),
    (50, 100,-1,  0,  "tomato",    "Mélancolique acoustique"),
]:
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], alpha=0.06, color=color)
    ax.text((x0+x1)/2, y1*0.85 if y1 > 0 else y0*0.85, label,
            ha="center", color=color, fontsize=9, alpha=0.8, fontweight="bold")

cmap = plt.cm.tab20
plotted = 0
for _, row in summary_emo.iterrows():
    x = row.get("acousticness_mean")
    y = row.get("sentiment")
    n = int(row.get("n_tracks", 20))
    c = int(row["cluster"])
    if pd.isna(x) or pd.isna(y):
        continue
    ax.scatter(x, y, s=n * 0.35 + 40, color=cmap(c % 20 / 20), alpha=0.85, zorder=3)
    ax.annotate(f"G{c+1}", (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    plotted += 1

ax.set_xlim(0, 100)
ax.set_ylim(-1, 1)
ax.set_xlabel("Acousticness  (0 = électronique → 100 = acoustique)")
ax.set_ylabel("Sentiment des paroles  (−1 négatif → +1 positif)")
ax.set_title("Carte émotionnelle des groupes  (taille ∝ nombre de titres)", fontsize=13)
plt.tight_layout()
plt.savefig("figures/carte_emotionnelle.png", dpi=150)
plt.close()
print(f"  → figures/carte_emotionnelle.png ({plotted} groupes placés)")

# ─── 4. Artistes pivot ────────────────────────────────────────────────────────
print("4. Artistes pivot...")

artist_clusters = (
    tracks.dropna(subset=["cluster"])
    .groupby("artist")["cluster"].nunique()
)
pivots = artist_clusters[artist_clusters >= 3].sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(pivots)), pivots.values, color="coral", alpha=0.8)
ax.set_yticks(range(len(pivots)))
ax.set_yticklabels(pivots.index, fontsize=9)
ax.set_xlabel("Nombre de groupes différents")
ax.set_title("Artistes pivot — présents dans 3 groupes ou plus", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("figures/artistes_pivot.png", dpi=150)
plt.close()
print(f"  → figures/artistes_pivot.png ({len(pivots)} artistes)")

print("\nAnalyses supplémentaires terminées.")
