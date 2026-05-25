# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "scikit-learn", "umap-learn", "matplotlib"]
# ///
"""
Script 2 — Clustering des morceaux
Groupe les morceaux par similarité : genres Spotify + tags Last.fm + popularité + année.
Résultat : data/tracks_clustered.csv + figures/umap_clusters.png + figures/elbow.png
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
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

print("Construction de la matrice de tags...")
df["_all_tags"] = df.apply(parse_tags, axis=1)

# Tags descriptifs à exclure du clustering (conservés dans les données pour l'analyse)
DESCRIPTIVE_TAGS = {
    "american", "british", "french", "german", "swedish", "australian",
    "canadian", "irish", "scottish", "japanese", "korean", "norwegian",
    "female vocalists", "male vocalists", "female vocalist", "male vocalist",
    "usa", "uk", "us", "france", "germany", "europe",
    "seen live", "favourites", "favorite", "love", "beautiful", "amazing",
    "00s", "10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s", "90s",
    "2000s", "2010s", "2020s", "classic", "classics",
}

# Garder uniquement les tags stylistiques présents dans au moins 3 morceaux
tag_counts = Counter(t for tags in df["_all_tags"] for t in tags)
valid_tags = {t for t, n in tag_counts.items() if n >= 3 and t not in DESCRIPTIVE_TAGS}
df["_all_tags"] = df["_all_tags"].apply(lambda tags: [t for t in tags if t in valid_tags])

mlb = MultiLabelBinarizer()
X_tags = mlb.fit_transform(df["_all_tags"]).astype(float)
print(f"  {X_tags.shape[1]} tags uniques, {X_tags.shape[0]} morceaux")

# ─── Réduction LSA des tags (SVD tronquée) ───────────────────────────────────
# KMeans gère mal les espaces creux à haute dimension : on projette dans un
# espace dense plus petit pour faire émerger la structure de genre/style.
n_tags = X_tags.shape[1]
n_svd = min(100, max(1, n_tags - 1))
svd = TruncatedSVD(n_components=n_svd, random_state=42)
X_tags = svd.fit_transform(X_tags)
print(f"  Réduction SVD : {n_svd} composantes (variance expliquée = {svd.explained_variance_ratio_.sum():.2%})")

# ─── Features numériques ─────────────────────────────────────────────────────
def safe_col(series):
    """Retourne la série avec NaN remplacés par la médiane, ou None si tout est NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return None
    return s.fillna(s.median()).values

num_data = {}
for col, key in [("popularity", "popularity"), ("release_year", "release_year")]:
    v = safe_col(df.get(col, pd.Series(dtype=float)))
    if v is not None:
        num_data[key] = v


# Features audio GetSongBPM (disponibles si 01c_bpm.py a tourné)
audio_fields = {
    "bpm":          "BPM",
    "danceability": "Danceability",
    "acousticness": "Acousticness",
}
has_audio = False
for col, label in audio_fields.items():
    v = safe_col(df.get(col, pd.Series(dtype=float)))
    if v is not None and df[col].notna().sum() > len(df) * 0.3:
        num_data[col] = v
        has_audio = True

if has_audio:
    print(f"  Features audio incluses : {[k for k in audio_fields if k in num_data]}")
else:
    print("  Pas de features audio — lance 01c_bpm.py pour les ajouter.")

# Tonalité : encodage circulaire sur le cercle des quintes (sin + cos)
KEY_ORDER = ["C","G","D","A","E","B","F#","Db","Ab","Eb","Bb","F",
             "Cm","Gm","Dm","Am","Em","Bm","F#m","Dbm","Abm","Ebm","Bbm","Fm"]
KEY_TO_IDX = {k: i for i, k in enumerate(KEY_ORDER)}
# Variantes orthographiques fréquentes
KEY_ALIASES = {"C#": "Db", "C#m": "Dbm", "G#": "Ab", "G#m": "Abm",
               "D#": "Eb", "D#m": "Ebm", "A#": "Bb", "A#m": "Bbm",
               "Gb": "F#", "Gbm": "F#m"}

if "key" in df.columns and df["key"].notna().sum() > len(df) * 0.3:
    def key_to_angle(k):
        if not isinstance(k, str):
            return None
        k = KEY_ALIASES.get(k, k)
        idx = KEY_TO_IDX.get(k)
        return (idx / 24) * 2 * np.pi if idx is not None else None

    angles = df["key"].map(key_to_angle)
    valid = angles.notna()
    if valid.sum() > len(df) * 0.3:
        median_angle = angles[valid].median()
        angles = angles.fillna(median_angle)
        num_data["key_sin"] = np.sin(angles.values)
        num_data["key_cos"] = np.cos(angles.values)
        print("  Tonalité incluse (encodage circulaire)")

num_matrix = np.column_stack(list(num_data.values()))
# Remplace les NaN résiduels par la médiane de chaque colonne
col_medians = np.nanmedian(num_matrix, axis=0)
for j in range(num_matrix.shape[1]):
    mask = np.isnan(num_matrix[:, j])
    num_matrix[mask, j] = col_medians[j]

scaler = StandardScaler()
X_num = scaler.fit_transform(num_matrix) * 2.0  # poids relatif

X = np.hstack([X_tags, X_num])

# Garder uniquement les morceaux avec au moins 1 tag valide
has_tags = np.array([len(t) > 0 for t in df["_all_tags"]])
df_clean = df[has_tags].copy().reset_index(drop=True)
X_clean = np.nan_to_num(X[has_tags], nan=0.0)

if len(df_clean) < 10:
    print("⚠ Pas assez de morceaux avec des tags pour clusterer.")
    exit(1)

print(f"{len(df_clean)}/{len(df)} morceaux ont des tags exploitables.")

# ─── Choix du nombre de clusters ─────────────────────────────────────────────
print("Recherche du nombre optimal de clusters...")
inertias, silhouettes = [], []
K_range = range(2, min(20, len(df_clean) // 100 + 2))

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clean)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_clean, labels, sample_size=min(1000, len(df_clean))))

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"  Meilleur k : {best_k} (silhouette = {max(silhouettes):.3f})")

km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean["cluster"] = km.fit_predict(X_clean)

# Fusion des clusters trop petits (< 20 tracks) dans le plus proche
MIN_CLUSTER_SIZE = 20
cluster_sizes = df_clean["cluster"].value_counts()
small_clusters = cluster_sizes[cluster_sizes < MIN_CLUSTER_SIZE].index.tolist()

if small_clusters:
    print(f"  Fusion de {len(small_clusters)} micro-cluster(s) < {MIN_CLUSTER_SIZE} tracks...")
    centers = km.cluster_centers_
    for sc in small_clusters:
        mask = df_clean["cluster"] == sc
        # Trouver le cluster le plus proche (hors lui-même)
        other = [c for c in range(best_k) if c != sc and c not in small_clusters]
        if other:
            dists = np.linalg.norm(centers[other] - centers[sc], axis=1)
            nearest = other[np.argmin(dists)]
            df_clean.loc[mask, "cluster"] = nearest
    # Renumérote proprement
    unique = sorted(df_clean["cluster"].unique())
    remap = {old: new for new, old in enumerate(unique)}
    df_clean["cluster"] = df_clean["cluster"].map(remap)

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

listeners = pd.to_numeric(df_clean.get("lastfm_listeners", pd.Series(dtype=float)), errors="coerce")
listeners_log = np.log10(listeners.fillna(listeners.median()) + 1)
sc = axes[1].scatter(df_clean["x"], df_clean["y"],
                     c=listeners_log, cmap="plasma", alpha=0.6, s=20)
cbar = plt.colorbar(sc, ax=axes[1], label="Auditeurs Last.fm (log)")
ticks = [3, 4, 5, 6, 7]
cbar.set_ticks(ticks)
cbar.set_ticklabels(["1K", "10K", "100K", "1M", "10M"])
axes[1].set_title("Carte d'audience (niche → mainstream)", fontsize=13)
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
