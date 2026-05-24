"""
Script 3 — Analyse musicale des clusters
Décrit chaque groupe : tempo, énergie, tonalité, structure...
Résultat : figures/cluster_profiles.png + data/cluster_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/tracks_clustered.csv")
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)
n_clusters = df["cluster"].nunique()

KEY_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODE_NAMES = {0: "Mineur", 1: "Majeur"}

FEATURES = {
    "danceability":      "Dansabilité",
    "energy":            "Énergie",
    "valence":           "Positivité",
    "acousticness":      "Acoustique",
    "instrumentalness":  "Instrumentale",
    "speechiness":       "Discours/rap",
    "liveness":          "Live",
}

# ─── Résumé par cluster ───────────────────────────────────────────────────────
summary_rows = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    row = {"cluster": c, "n_tracks": len(sub)}

    for feat in FEATURES:
        row[f"{feat}_mean"] = sub[feat].mean()

    row["tempo_mean"]    = sub["tempo"].mean()
    row["loudness_mean"] = sub["loudness"].mean()
    row["popularity_mean"] = sub["popularity"].mean()

    # Tonalité dominante
    top_key  = sub["key"].dropna().mode()
    top_mode = sub["mode"].dropna().mode()
    row["dominant_key"]  = KEY_NAMES[int(top_key.iloc[0])]  if len(top_key)  else "?"
    row["dominant_mode"] = MODE_NAMES.get(int(top_mode.iloc[0]), "?") if len(top_mode) else "?"

    # Top artistes
    row["top_artists"] = ", ".join(sub["artist"].value_counts().head(3).index.tolist())

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv("data/cluster_summary.csv", index=False)

# ─── Radar par cluster ────────────────────────────────────────────────────────
radar_features = list(FEATURES.keys())
N = len(radar_features)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

fig = plt.figure(figsize=(5 * n_clusters, 5))
gs  = GridSpec(1, n_clusters, figure=fig)

for i, c in enumerate(sorted(df["cluster"].unique())):
    sub = df[df["cluster"] == c]
    vals = [sub[f].mean() for f in radar_features]
    vals += vals[:1]

    ax = fig.add_subplot(gs[0, i], polar=True)
    ax.fill(angles, vals, color=colors[i], alpha=0.25)
    ax.plot(angles, vals, color=colors[i], linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([FEATURES[f] for f in radar_features], size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.5", "0.75"], size=7)
    key  = summary.loc[summary["cluster"] == c, "dominant_key"].values[0]
    mode = summary.loc[summary["cluster"] == c, "dominant_mode"].values[0]
    tempo = summary.loc[summary["cluster"] == c, "tempo_mean"].values[0]
    n    = summary.loc[summary["cluster"] == c, "n_tracks"].values[0]
    ax.set_title(f"Groupe {c+1}  ({n} titres)\n{key} {mode} · {tempo:.0f} BPM",
                 size=10, pad=14)

plt.suptitle("Profils musicaux par cluster", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Distribution des features ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, feat in enumerate(list(FEATURES.keys()) + ["tempo"]):
    ax = axes[idx]
    for i, c in enumerate(sorted(df["cluster"].unique())):
        sub = df[df["cluster"] == c][feat].dropna()
        ax.hist(sub, bins=20, alpha=0.5, color=colors[i], label=f"G{c+1}", density=True)
    ax.set_title(feat)
    ax.set_xlabel("")
    if idx == 0:
        ax.legend(fontsize=8)

axes[-1].set_visible(False)
plt.suptitle("Distribution des features par groupe", fontsize=14)
plt.tight_layout()
plt.savefig("figures/feature_distributions.png", dpi=150)
plt.close()

# ─── Rapport console ─────────────────────────────────────────────────────────
print("=" * 60)
print("ANALYSE DES GROUPES MUSICAUX")
print("=" * 60)
for _, row in summary.iterrows():
    c = int(row["cluster"])
    print(f"\n── Groupe {c+1} ({int(row['n_tracks'])} titres) ──────────────────")
    print(f"  Tonalité   : {row['dominant_key']} {row['dominant_mode']}")
    print(f"  Tempo      : {row['tempo_mean']:.1f} BPM")
    print(f"  Énergie    : {row['energy_mean']:.2f}")
    print(f"  Dansabilité: {row['danceability_mean']:.2f}")
    print(f"  Positivité : {row['valence_mean']:.2f}")
    print(f"  Acoustique : {row['acousticness_mean']:.2f}")
    print(f"  Popularité : {row['popularity_mean']:.1f}/100")
    print(f"  Top artistes: {row['top_artists']}")

print("\nFigures → figures/cluster_profiles.png")
print("         → figures/feature_distributions.png")
