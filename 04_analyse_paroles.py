# /// script
# requires-python = ">=3.12"
# dependencies = ["lyricsgenius", "pandas", "numpy", "matplotlib", "nltk", "textblob", "wordcloud", "langdetect", "python-dotenv"]
# ///
"""
Script 4 — Analyse des paroles
Récupère les paroles via Genius, analyse thèmes, structure, sentiment.
Résultat : data/lyrics_analysis.csv + figures/lyrics_*.png
"""

import pandas as pd
import numpy as np
import lyricsgenius
import time
import os
import re
import json
from collections import Counter

import nltk
for _res in ["punkt_tab", "stopwords", "vader_lexicon", "averaged_perceptron_tagger_eng"]:
    nltk.download(_res, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import GENIUS_ACCESS_TOKEN

try:
    from langdetect import detect
except ImportError:
    detect = lambda x: "unknown"

os.makedirs("figures", exist_ok=True)
os.makedirs("data/lyrics_cache", exist_ok=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BATCH_SIZE = 30  # tracks par cluster par run

genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, skip_non_songs=True,
                              excluded_terms=["(Remix)", "(Live)"])

# ─── Chargement ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/tracks_clustered.csv")
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)

# ─── Récupération des paroles avec cache ─────────────────────────────────────
def fetch_lyrics(track_id, title, artist):
    cache_file = f"data/lyrics_cache/{track_id}.json"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f).get("lyrics")

    try:
        song = genius.search_song(title, artist)
        lyrics = song.lyrics if song else None
    except Exception:
        lyrics = None

    with open(cache_file, "w") as f:
        json.dump({"lyrics": lyrics}, f)
    time.sleep(0.3)
    return lyrics

# ─── Nettoyage des paroles ────────────────────────────────────────────────────
def clean_lyrics(raw):
    if not raw:
        return ""
    text = re.sub(r"\[.*?\]", " ", raw)        # retire [Couplet 1], [Refrain]...
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return text.strip()

def detect_sections(raw):
    """Détecte les sections structurelles dans les paroles brutes."""
    if not raw:
        return {}
    tags = re.findall(r"\[([^\]]+)\]", raw)
    counts = Counter(t.split(":")[0].lower() for t in tags)
    return dict(counts)

# ─── Analyse d'un texte ───────────────────────────────────────────────────────
def analyze_text(text, lang="en"):
    if not text or len(text) < 50:
        return {}

    blob   = TextBlob(text)
    words  = word_tokenize(text.lower())
    sents  = sent_tokenize(text)

    try:
        sw = set(stopwords.words("french" if lang == "fr" else "english"))
    except Exception:
        sw = set()

    content_words = [w for w in words if w.isalpha() and w not in sw and len(w) > 2]
    vocab_richness = len(set(content_words)) / max(len(content_words), 1)

    # Répétitivité : ratio mots répétés
    freq = Counter(content_words)
    repeated = sum(v for v in freq.values() if v > 2)
    repetition_rate = repeated / max(len(content_words), 1)

    return {
        "word_count":       len(words),
        "sentence_count":   len(sents),
        "avg_words_per_sent": len(words) / max(len(sents), 1),
        "vocab_richness":   round(vocab_richness, 3),
        "repetition_rate":  round(repetition_rate, 3),
        "sentiment_polarity":  round(blob.sentiment.polarity, 3),
        "sentiment_subjectivity": round(blob.sentiment.subjectivity, 3),
        "top_words":        json.dumps([w for w, _ in freq.most_common(10)]),
    }

# ─── Boucle principale ────────────────────────────────────────────────────────
LYRICS_CSV = "data/lyrics_analysis.csv"

# Tracks déjà analysés (pour ne pas les refaire)
if os.path.exists(LYRICS_CSV):
    already_done = set(pd.read_csv(LYRICS_CSV)["track_id"].astype(str))
else:
    already_done = set()

total_in_df   = len(df)
total_done    = len(already_done)
print(f"Paroles : {total_done}/{total_in_df} tracks déjà analysés.")

def process_row(row, cluster_id):
    raw_lyrics = fetch_lyrics(row["track_id"], row["track_name"], row["artist"])

    lang = "en"
    try:
        cleaned = clean_lyrics(raw_lyrics)
        if cleaned:
            lang = detect(cleaned[:500])
    except Exception:
        cleaned = ""

    sections   = detect_sections(raw_lyrics or "")
    analysis   = analyze_text(cleaned, lang)
    has_chorus = sections.get("chorus", 0) + sections.get("refrain", 0) > 0

    return {
        "track_id":   row["track_id"],
        "track_name": row["track_name"],
        "artist":     row["artist"],
        "cluster":    cluster_id,
        "lang":       lang,
        "has_lyrics": bool(cleaned),
        "has_chorus": has_chorus,
        "n_sections": sum(sections.values()),
        **analysis,
    }

round_num = 0
while True:
    round_num += 1
    pending_any = df[~df["track_id"].astype(str).isin(already_done)]
    if pending_any.empty:
        print("\nTous les tracks ont été analysés.")
        break

    print(f"\n{'─'*50}")
    print(f"Round {round_num} — {len(pending_any)} tracks restants")
    print(f"{'─'*50}")

    round_results = []
    for cluster_id in sorted(df["cluster"].unique()):
        pending = df[
            (df["cluster"] == cluster_id) &
            (~df["track_id"].astype(str).isin(already_done))
        ].head(BATCH_SIZE)

        if pending.empty:
            continue

        print(f"\nGroupe {cluster_id+1} — {len(pending)} tracks")
        counter = [0]
        rows = [row for _, row in pending.iterrows()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(process_row, row, cluster_id): row for row in rows}
            for future in as_completed(futures):
                counter[0] += 1
                row = futures[future]
                try:
                    result = future.result()
                    round_results.append(result)
                    already_done.add(str(row["track_id"]))
                    src = "cache" if os.path.exists(f"data/lyrics_cache/{row['track_id']}.json") else "API"
                    print(f"  [{counter[0]}/{len(rows)}] {str(row['track_name'])[:38]:<38} ({src})", end="\r")
                except Exception as e:
                    print(f"  ⚠ {row['track_name'][:40]} — {e}")
                    already_done.add(str(row["track_id"]))  # skip en cas d'erreur
        print()

    # Sauvegarde après chaque round
    if round_results:
        new_df = pd.DataFrame(round_results)
        if os.path.exists(LYRICS_CSV):
            existing = pd.read_csv(LYRICS_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(LYRICS_CSV, index=False)
        print(f"→ {len(round_results)} ajoutés ({len(combined)} au total dans {LYRICS_CSV})")

ldf = pd.read_csv(LYRICS_CSV) if os.path.exists(LYRICS_CSV) else pd.DataFrame()

if ldf.empty or "has_lyrics" not in ldf.columns:
    print("Pas assez de données pour les figures.")
    exit(0)

# ─── Visualisation 1 : métriques par cluster ─────────────────────────────────
metrics = ["sentiment_polarity", "vocab_richness", "repetition_rate",
           "avg_words_per_sent", "word_count"]
labels  = ["Sentiment", "Richesse vocab.", "Répétitivité", "Mots/phrase", "Nb mots"]

ldf_valid = ldf[ldf["has_lyrics"] == True]
colors = plt.cm.tab10(np.linspace(0, 1, df["cluster"].nunique()))

fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
for ax, metric, label in zip(axes, metrics, labels):
    data = [ldf_valid[ldf_valid["cluster"] == c][metric].dropna().values
            for c in sorted(df["cluster"].unique())]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(data)+1))
    ax.set_xticklabels([f"G{c+1}" for c in sorted(df["cluster"].unique())])
    ax.set_title(label, fontsize=11)

plt.suptitle("Analyse des paroles par groupe", fontsize=14)
plt.tight_layout()
plt.savefig("figures/lyrics_metrics.png", dpi=150)
plt.close()

# ─── Visualisation 2 : wordclouds par cluster ─────────────────────────────────
n_clusters = df["cluster"].nunique()
fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 5))
if n_clusters == 1:
    axes = [axes]

for ax, c in zip(axes, sorted(df["cluster"].unique())):
    sub = ldf_valid[ldf_valid["cluster"] == c]
    all_words = []
    for _, row in sub.iterrows():
        try:
            all_words.extend(json.loads(row["top_words"]))
        except Exception:
            pass

    if all_words:
        freq = Counter(all_words)
        wc = WordCloud(width=600, height=400, background_color="white",
                       colormap="plasma", max_words=40)
        wc.generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, "Pas de paroles", ha="center", va="center")
    ax.axis("off")
    ax.set_title(f"Groupe {c+1}", fontsize=12)

plt.suptitle("Champs lexicaux par groupe", fontsize=14)
plt.tight_layout()
plt.savefig("figures/lyrics_wordclouds.png", dpi=150)
plt.close()

print("\nAnalyse des paroles terminée.")
print("→ data/lyrics_analysis.csv")
print("→ figures/lyrics_metrics.png")
print("→ figures/lyrics_wordclouds.png")
