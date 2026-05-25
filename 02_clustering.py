"""
Script 2 — Clustering des morceaux
Groupe les morceaux par similarité : genres Spotify + tags Last.fm + popularité + année.
Résultat : data/tracks_clustered.csv + figures/umap_clusters.png + figures/elbow.png
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
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
    print("umap-learn non installé → PCA utilisé (pip install umap-learn)")

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/tracks.csv")

# ─── Construction des features de tags ───────────────────────────────────────
def parse_tags(row):
    tags = set()
    for col in ["spotify_genres", "lastfm_tags"]:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip():
            for t in val.split(","):
                t = t.strip().lower()
                if t:
                    tags.add(t)
    return list(tags)

df["_all_tags"] = df.apply(parse_tags, axis=1)

# Garder uniquement les tags présents dans au moins 3 morceaux
tag_counts = Counter(t for tags in df["_all_tags"] for t in tags)
valid_tags = {t for t, n in tag_counts.items() if n >= 3}
df["_all_tags"] = df["_all_tags"].apply(lambda tags: [t for t in tags if t in valid_tags])

mlb = MultiLabelBinarizer()
X_tags = mlb.fit_transform(df["_all_tags"]).astype(float)

# ─── Features numériques ─────────────────────────────────────────────────────
num_data = {}
num_data["popularity"] = df["popularity"].fillna(df["popularity"].median()).values

listeners = pd.to_numeric(df.get("lastfm_listeners", pd.Series(dtype=float)), errors="coerce").fillna(0)
num_data["log_listeners"] = np.log1p(listeners.values)

year = pd.to_numeric(df.get("release_year", pd.Series(dtype=float)), errors="coerce")
num_data["release_year"] = year.fillna(year.median()).values

# Features audio (disponibles si 01b_audio.py a tourné)
has_audio = False
if "tempo" in df.columns and df["tempo"].notna().sum() > len(df) * 0.3:
    tempo = pd.to_numeric(df["tempo"], errors="coerce")
    num_data["tempo"] = tempo.fillna(tempo.median()).values
    has_audio = True

if "energy" in df.columns and df["energy"].notna().sum() > len(df) * 0.3:
    energy = pd.to_numeric(df["energy"], errors="coerce")
    num_data["energy"] = energy.fillna(energy.median()).values

if has_audio:
    print("  Features audio (tempo, énergie) incluses dans le clustering.")
else:
    print("  Pas de features audio — lance 01b_audio.py pour les ajouter.")

num_matrix = np.column_stack(list(num_data.values()))
scaler = StandardScaler()
X_num = scaler.fit_transform(num_matrix) * 2.0  # poids relatif

X = np.hstack([X_tags, X_num])

# Garder uniquement les morceaux avec au moins 1 tag valide
has_tags = np.array([len(t) > 0 for t in df["_all_tags"]])
df_clean = df[has_tags].copy().reset_index(drop=True)
X_clean = X[has_tags]

if len(df_clean) < 10:
    print("⚠ Pas assez de morceaux avec des tags pour clusterer.")
    exit(1)

print(f"{len(df_clean)}/{len(df)} morceaux ont des tags exploitables.")

# ─── Choix du nombre de clusters ─────────────────────────────────────────────
print("Recherche du nombre optimal de clusters...")
inertias, silhouettes = [], []
K_range = range(2, min(12, len(df_clean) // 10 + 2))

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clean)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_clean, labels, sample_size=min(1000, len(df_clean))))

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"  Meilleur k : {best_k} (silhouette = {max(silhouettes):.3f})")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster"] = km.fit_predict(X_clean)

# ─── Réduction dimensionnelle ─────────────────────────────────────────────────
print("Réduction dimensionnelle...")
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
else:
    reducer = PCA(n_components=2)

coords = reducer.fit_transform(X_clean)
df_clean["x"] = coords[:, 0]
df_clean["y"] = coords[:, 1]

# ─── Visualisation ───────────────────────────────────────────────────────────
colors = cm.tab10(np.linspace(0, 1, best_k))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
method_name = "UMAP" if HAS_UMAP else "PCA"

ax = axes[0]
for c in range(best_k):
    mask = df_clean["cluster"] == c
    ax.scatter(df_clean.loc[mask, "x"], df_clean.loc[mask, "y"],
               c=[colors[c]], label=f"Groupe {c+1}", alpha=0.6, s=20)
ax.set_title(f"Clusters musicaux ({method_name})", fontsize=13)
ax.legend(markerscale=2)
ax.set_xlabel(f"{method_name} dim 1")
ax.set_ylabel(f"{method_name} dim 2")

sc = axes[1].scatter(df_clean["x"], df_clean["y"],
                     c=df_clean["popularity"], cmap="plasma", alpha=0.6, s=20)
plt.colorbar(sc, ax=axes[1], label="Popularité")
axes[1].set_title("Carte de popularité", fontsize=13)
axes[1].set_xlabel(f"{method_name} dim 1")

plt.tight_layout()
plt.savefig("figures/umap_clusters.png", dpi=150)
plt.close()

# ─── Elbow curve ─────────────────────────────────────────────────────────────
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

# ─── Sauvegarde ──────────────────────────────────────────────────────────────
df_merged = df.merge(df_clean[["track_id", "cluster", "x", "y"]], on="track_id", how="left")
df_merged.to_csv("data/tracks_clustered.csv", index=False)

print(f"\nClustering terminé → data/tracks_clustered.csv")
print(f"Figures → figures/umap_clusters.png, figures/elbow.png")
print(f"\nRépartition par cluster :")
print(df_clean["cluster"].value_counts().sort_index().to_string())
