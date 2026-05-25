"""
Script 3 — Analyse des genres et tags par cluster
Résultat : data/cluster_summary.csv + figures/cluster_genres.png + figures/genre_distribution.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/tracks_clustered.csv")
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)
n_clusters = df["cluster"].nunique()
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

# ─── Catégories de genres ────────────────────────────────────────────────────
GENRE_CATEGORIES = {
    "Électronique":  ["electronic", "dance", "edm", "techno", "house", "trance",
                      "dubstep", "electro", "synth", "ambient", "idm"],
    "Rock / Metal":  ["rock", "metal", "punk", "grunge", "hardcore", "alternative",
                      "indie rock", "post-rock"],
    "Pop":           ["pop", "indie pop", "synth-pop", "electropop", "dream pop", "k-pop"],
    "Hip-hop / Rap": ["hip-hop", "hip hop", "rap", "trap", "rnb", "r&b", "grime"],
    "Jazz / Soul":   ["jazz", "soul", "blues", "funk", "gospel", "neo soul", "bossa nova"],
    "Folk / Acoustique": ["folk", "acoustic", "country", "singer-songwriter", "americana"],
    "Classique":     ["classical", "orchestral", "opera", "chamber", "new age"],
}

def parse_all_tags(row):
    tags = []
    for col in ["spotify_genres", "lastfm_tags"]:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip():
            tags.extend(t.strip().lower() for t in val.split(",") if t.strip())
    return tags

def classify_tags(tags):
    cat_counts = Counter()
    for tag in tags:
        matched = False
        for cat, keywords in GENRE_CATEGORIES.items():
            if any(kw in tag or tag in kw for kw in keywords):
                cat_counts[cat] += 1
                matched = True
                break
        if not matched:
            cat_counts["Autres"] += 1
    return cat_counts

# ─── Résumé par cluster ──────────────────────────────────────────────────────
summary_rows = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]

    all_tags = []
    for _, row in sub.iterrows():
        all_tags.extend(parse_all_tags(row))

    tag_counts   = Counter(all_tags)
    cat_totals   = classify_tags(all_tags)
    dominant_cat = cat_totals.most_common(1)[0][0] if cat_totals else "Autres"

    year = pd.to_numeric(sub.get("release_year", pd.Series(dtype=float)), errors="coerce").mean()

    tempo_mean = pd.to_numeric(sub.get("tempo", pd.Series(dtype=float)), errors="coerce").mean() \
        if "tempo" in sub.columns else None
    energy_mean = pd.to_numeric(sub.get("energy", pd.Series(dtype=float)), errors="coerce").mean() \
        if "energy" in sub.columns else None

    summary_rows.append({
        "cluster":           c,
        "n_tracks":          len(sub),
        "top_tags":          ", ".join(t for t, _ in tag_counts.most_common(10)),
        "dominant_genre":    dominant_cat,
        "popularity_mean":   sub["popularity"].mean(),
        "release_year_mean": year,
        "tempo_mean":        tempo_mean,
        "energy_mean":       energy_mean,
        "top_artists":       ", ".join(sub["artist"].value_counts().head(3).index.tolist()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv("data/cluster_summary.csv", index=False)

# ─── Figure 1 : Top tags par cluster ─────────────────────────────────────────
fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5))
if n_clusters == 1:
    axes = [axes]

for ax, c in zip(axes, sorted(df["cluster"].unique())):
    sub = df[df["cluster"] == c]
    all_tags = []
    for _, row in sub.iterrows():
        all_tags.extend(parse_all_tags(row))

    tag_counts = Counter(all_tags)
    top = tag_counts.most_common(12)
    if not top:
        ax.text(0.5, 0.5, "Pas de tags", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        continue

    labels, counts = zip(*top)
    y_pos = range(len(labels) - 1, -1, -1)
    ax.barh(list(y_pos), counts, color=colors[c], alpha=0.75)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    n = int(summary.loc[summary["cluster"] == c, "n_tracks"].values[0])
    ax.set_title(f"Groupe {c+1}  ({n} titres)", fontsize=11)
    ax.set_xlabel("Occurrences")

plt.suptitle("Top tags par groupe", fontsize=14)
plt.tight_layout()
plt.savefig("figures/cluster_genres.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Figure 2 : Distribution des catégories de genres ────────────────────────
cats = list(GENRE_CATEGORIES.keys()) + ["Autres"]
cat_matrix = np.zeros((n_clusters, len(cats)))

for i, c in enumerate(sorted(df["cluster"].unique())):
    sub = df[df["cluster"] == c]
    all_tags = []
    for _, row in sub.iterrows():
        all_tags.extend(parse_all_tags(row))
    totals = classify_tags(all_tags)
    total = sum(totals.values()) or 1
    for j, cat in enumerate(cats):
        cat_matrix[i, j] = totals.get(cat, 0) / total

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(cats))
width = 0.8 / n_clusters

for i, c in enumerate(sorted(df["cluster"].unique())):
    offset = (i - n_clusters / 2 + 0.5) * width
    ax.bar(x + offset, cat_matrix[i], width, label=f"Groupe {c+1}", color=colors[i], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Part relative")
ax.set_title("Distribution des genres par groupe")
ax.legend()
plt.tight_layout()
plt.savefig("figures/genre_distribution.png", dpi=150)
plt.close()

# ─── Rapport console ─────────────────────────────────────────────────────────
print("=" * 60)
print("ANALYSE DES GROUPES")
print("=" * 60)
for _, row in summary.iterrows():
    c = int(row["cluster"])
    print(f"\n── Groupe {c+1} ({int(row['n_tracks'])} titres) ──────────────────")
    print(f"  Genre dominant : {row['dominant_genre']}")
    print(f"  Top tags       : {row['top_tags']}")
    print(f"  Popularité moy.: {row['popularity_mean']:.1f}/100")
    year = row.get("release_year_mean")
    if year and not np.isnan(float(year)):
        print(f"  Année moyenne  : {float(year):.0f}")
    tempo = row.get("tempo_mean")
    if tempo and not np.isnan(float(tempo)):
        print(f"  Tempo moyen    : {float(tempo):.0f} BPM")
    energy = row.get("energy_mean")
    if energy and not np.isnan(float(energy)):
        print(f"  Énergie moy.   : {float(energy):.4f}")
    print(f"  Top artistes   : {row['top_artists']}")

print("\nFigures → figures/cluster_genres.png")
print("         → figures/genre_distribution.png")
