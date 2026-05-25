# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "matplotlib"]
# ///
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
    "Électronique":        ["electronic", "edm", "techno", "house", "trance", "dubstep",
                            "electro", "synth", "idm", "electronica", "synthpop", "electropop",
                            "synthwave", "drum and bass", "dnb", "garage", "ambient", "downtempo",
                            "trip-hop", "trip hop", "chillout", "lounge", "french house",
                            "french touch", "new wave", "darkwave", "industrial", "dance",
                            "dance music", "club", "rave", "breakbeat", "big beat",
                            "lo-fi", "lofi", "indietronica", "minimal", "experimental",
                            "microhouse", "glitch", "noise"],
    "Rock / Metal":        ["rock", "metal", "punk", "grunge", "hardcore", "alternative rock",
                            "indie rock", "post-rock", "hard rock", "classic rock", "progressive rock",
                            "pop punk", "post-punk", "gothic rock", "emo", "metalcore",
                            "psychedelic", "shoegaze", "ska", "surf rock", "garage rock",
                            "math rock", "noise rock", "stoner rock", "doom"],
    "Pop":                 ["pop", "indie pop", "dream pop", "k-pop", "dance pop", "electropop",
                            "easy listening", "ballad", "soft rock", "adult contemporary"],
    "Hip-hop / Rap":       ["hip-hop", "hip hop", "rap", "trap", "rnb", "r&b", "grime",
                            "underground hip-hop", "east coast", "west coast", "boom bap",
                            "conscious hip-hop", "lo-fi hip hop"],
    "Jazz / Soul / Funk":  ["jazz", "soul", "blues", "funk", "gospel", "neo soul", "bossa nova",
                            "rhythm and blues", "motown", "disco", "afrobeat",
                            "swing", "big band", "bebop", "fusion", "smooth jazz"],
    "Folk / Acoustique":   ["folk", "acoustic", "country", "singer-songwriter", "americana",
                            "indie folk", "folk rock", "bluegrass", "celtic"],
    "Chanson Française":   ["chanson", "chanson francaise", "chanson française", "french chanson",
                            "francais", "français", "francophone", "nouvelle scene francaise",
                            "ye-ye", "yé-yé", "variete", "variété"],
    "Classique / Ambient": ["classical", "orchestral", "opera", "chamber", "new age",
                            "neoclassical", "modern classical", "post-classical", "piano",
                            "contemporary classical", "cinematic", "instrumental", "soundtrack",
                            "composer", "film score"],
    "Musique du Monde":    ["latin", "reggae", "world", "afro", "cumbia", "salsa", "flamenco",
                            "bossa", "samba", "tango", "fado", "celtic", "arabic", "indian",
                            "dancehall", "dub", "roots reggae"],
}

# Tags à ignorer pour le calcul du genre dominant (descriptifs, pas stylistiques)
SKIP_FOR_GENRE = {
    "american", "british", "french", "german", "swedish", "australian", "canadian",
    "irish", "scottish", "japanese", "norwegian", "belgian", "dutch", "italian",
    "spanish", "female vocalists", "male vocalists", "female vocalist", "male vocalist",
    "usa", "uk", "us", "france", "seen live", "favourites", "favorite",
    "beautiful", "amazing", "indie", "alternative", "pop", "rock",
    "sexy", "mellow", "party", "love", "sad", "happy", "chill", "all",
    "guitar", "covers", "cover", "spotify", "my top songs", "love at first listen",
    "90s", "80s", "70s", "60s", "00s", "10s", "2000s", "2010s", "2020s", "oldies",
    "new york", "california", "england", "sweden", "canada", "germany", "australia",
    "political", "humour", "scandinavian", "world",
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
        if tag in SKIP_FOR_GENRE:
            continue
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
    real_cats = [(cat, cnt) for cat, cnt in cat_totals.most_common() if cat != "Autres"]
    dominant_cat = real_cats[0][0] if real_cats else "Autres"

    year = pd.to_numeric(sub.get("release_year", pd.Series(dtype=float)), errors="coerce").mean()

    def col_mean(col):
        if col not in sub.columns:
            return None
        v = pd.to_numeric(sub[col], errors="coerce").mean()
        return None if pd.isna(v) else v

    def col_median(col):
        if col not in sub.columns:
            return None
        v = pd.to_numeric(sub[col], errors="coerce").median()
        return None if pd.isna(v) else v

    listeners = pd.to_numeric(sub.get("lastfm_listeners", pd.Series(dtype=float)), errors="coerce")
    listeners_mean = listeners.mean() if listeners.notna().sum() > 0 else None

    summary_rows.append({
        "cluster":            c,
        "n_tracks":           len(sub),
        "top_tags":           ", ".join(t for t, _ in tag_counts.most_common(15)),
        "dominant_genre":     dominant_cat,
        "listeners_mean":     listeners_mean,
        "release_year_mean":  year,
        "bpm_mean":           col_mean("bpm"),
        "bpm_median":         col_median("bpm"),
        "danceability_mean":  col_mean("danceability"),
        "acousticness_mean":  col_mean("acousticness"),
        "top_artists":        ", ".join(sub["artist"].value_counts().head(5).index.tolist()),
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
    listeners = row.get("listeners_mean")
    if listeners and not np.isnan(float(listeners)):
        l = float(listeners)
        if l >= 10_000_000:
            label = "mainstream"
        elif l >= 1_000_000:
            label = "populaire"
        elif l >= 100_000:
            label = "indie/underground"
        else:
            label = "niche"
        print(f"  Audience Last.fm: {l/1_000_000:.1f}M auditeurs → {label}")
    year = row.get("release_year_mean")
    if year and not np.isnan(float(year)):
        print(f"  Année moyenne  : {float(year):.0f}")
    bpm = row.get("bpm_mean")
    if bpm:
        print(f"  BPM moyen      : {float(bpm):.0f}")
    dance = row.get("danceability_mean")
    if dance:
        print(f"  Danceability   : {float(dance):.0f}/100")
    acou = row.get("acousticness_mean")
    if acou:
        print(f"  Acousticness   : {float(acou):.0f}/100")
    print(f"  Top artistes   : {row['top_artists']}")

print("\nFigures → figures/cluster_genres.png")
print("         → figures/genre_distribution.png")
