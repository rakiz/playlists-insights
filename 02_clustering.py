"""
Script 2 — Clustering des morceaux
Groupe les morceaux par similarité musicale via K-Means + UMAP.
Résultat : data/tracks_clustered.csv + figures/umap_clusters.png
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    HAS_UMAP = False
    print("umap-learn non installé → PCA utilisé à la place (pip install umap-learn)")

os.makedirs("figures", exist_ok=True)

# ─── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/tracks.csv")

FEATURES = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "loudness", "mode"
]

df_clean = df.dropna(subset=FEATURES).copy()
X = df_clean[FEATURES].values

# ─── Normalisation ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Choix du nombre de clusters (méthode elbow + silhouette) ─────────────────
print("Recherche du nombre optimal de clusters...")
inertias, silhouettes = [], []
K_range = range(2, min(12, len(df_clean) // 10 + 2))

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# Meilleur k selon silhouette
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"  Meilleur k : {best_k} (silhouette = {max(silhouettes):.3f})")

# ─── Clustering final ─────────────────────────────────────────────────────────
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster"] = km.fit_predict(X_scaled)

# ─── Réduction dimensionnelle pour visualisation ─────────────────────────────
print("Réduction dimensionnelle...")
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42)
else:
    reducer = PCA(n_components=2)

coords = reducer.fit_transform(X_scaled)
df_clean["x"] = coords[:, 0]
df_clean["y"] = coords[:, 1]

# ─── Visualisation ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
method_name = "UMAP" if HAS_UMAP else "PCA"

# Plot 1 : clusters
colors = cm.tab10(np.linspace(0, 1, best_k))
ax = axes[0]
for c in range(best_k):
    mask = df_clean["cluster"] == c
    ax.scatter(df_clean.loc[mask, "x"], df_clean.loc[mask, "y"],
               c=[colors[c]], label=f"Groupe {c+1}", alpha=0.6, s=20)
ax.set_title(f"Clusters musicaux ({method_name})", fontsize=13)
ax.legend(markerscale=2)
ax.set_xlabel(f"{method_name} dim 1")
ax.set_ylabel(f"{method_name} dim 2")

# Plot 2 : coloré par énergie
sc = axes[1].scatter(df_clean["x"], df_clean["y"],
                     c=df_clean["energy"], cmap="plasma", alpha=0.6, s=20)
plt.colorbar(sc, ax=axes[1], label="Énergie")
axes[1].set_title("Carte d'énergie", fontsize=13)
axes[1].set_xlabel(f"{method_name} dim 1")

plt.tight_layout()
plt.savefig("figures/umap_clusters.png", dpi=150)
plt.close()

# ─── Elbow curve ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax2 = ax.twinx()
ax.plot(K_range, inertias, "o-", color="steelblue", label="Inertie")
ax2.plot(K_range, silhouettes, "s--", color="coral", label="Silhouette")
ax.axvline(best_k, color="gray", linestyle=":", alpha=0.7)
ax.set_xlabel("Nombre de clusters k")
ax.set_ylabel("Inertie", color="steelblue")
ax2.set_ylabel("Score silhouette", color="coral")
ax.set_title("Choix du nombre de clusters")
plt.tight_layout()
plt.savefig("figures/elbow.png", dpi=150)
plt.close()

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
df_merged = df.merge(df_clean[["track_id", "cluster", "x", "y"]], on="track_id", how="left")
df_merged.to_csv("data/tracks_clustered.csv", index=False)

print(f"\nClustering terminé → data/tracks_clustered.csv")
print(f"Figures → figures/umap_clusters.png, figures/elbow.png")
print(f"\nRépartition par cluster :")
print(df_clean["cluster"].value_counts().sort_index().to_string())
