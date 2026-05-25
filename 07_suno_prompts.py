# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy"]
# ///
"""
Script 7 — Générateur de prompts Suno par cluster
Combine genre, BPM, tonalité, danceability, mood paroles → prompt prêt à coller.
Résultat : data/suno_prompts.csv + affichage console
"""

import pandas as pd
import numpy as np
import json
import os

summary = pd.read_csv("data/cluster_summary.csv")
tracks  = pd.read_csv("data/tracks_clustered.csv")
tracks  = tracks.dropna(subset=["cluster"])
tracks["cluster"] = tracks["cluster"].astype(int)

lyrics_path = "data/lyrics_analysis.csv"
if os.path.exists(lyrics_path):
    lyrics = pd.read_csv(lyrics_path)
else:
    lyrics = pd.DataFrame()

# ─── Tags à exclure du prompt (trop génériques ou non stylistiques) ───────────
SKIP_PROMPT_TAGS = {
    "american", "british", "french", "german", "swedish", "australian", "canadian",
    "irish", "scottish", "japanese", "norwegian", "belgian", "dutch",
    "female vocalists", "male vocalists", "female vocalist", "male vocalist",
    "usa", "uk", "us", "france", "seen live", "favourites", "favorite",
    "beautiful", "amazing", "all", "sexy", "mellow", "party", "love", "sad",
    "happy", "chill", "guitar", "covers", "cover", "spotify", "my top songs",
    "love at first listen", "90s", "80s", "70s", "60s", "00s", "10s",
    "2000s", "2010s", "2020s", "oldies", "new york", "california",
    "indie", "alternative", "pop", "rock",  # trop génériques seuls
}

# ─── Mapping BPM → descripteur de tempo ───────────────────────────────────────
def bpm_label(bpm):
    if pd.isna(bpm):       return None
    bpm = float(bpm)
    if bpm < 70:           return "very slow tempo"
    if bpm < 90:           return "slow tempo"
    if bpm < 110:          return "moderate tempo"
    if bpm < 130:          return "upbeat tempo"
    if bpm < 150:          return "fast tempo"
    return "very fast tempo"

# ─── Mapping acousticness → texture ───────────────────────────────────────────
def texture_label(acou):
    if pd.isna(acou):    return None
    acou = float(acou)
    if acou >= 70:       return "acoustic, organic"
    if acou >= 40:       return "semi-acoustic"
    if acou >= 20:       return "mixed electronic-acoustic"
    return "fully electronic, synthesized"

# ─── Mapping danceability → énergie ───────────────────────────────────────────
def energy_label(dance):
    if pd.isna(dance):   return None
    dance = float(dance)
    if dance >= 75:      return "highly danceable, driving beat"
    if dance >= 55:      return "groovy, rhythmic"
    if dance >= 35:      return "moderate groove"
    return "non-danceable, atmospheric"

# ─── Mapping sentiment → mood ─────────────────────────────────────────────────
def mood_label(polarity, subjectivity):
    if pd.isna(polarity): return None
    p, s = float(polarity), float(subjectivity) if not pd.isna(subjectivity) else 0.5
    if p > 0.2 and s > 0.5:   return "emotionally expressive, uplifting"
    if p > 0.2:                return "positive, feel-good"
    if p < -0.2 and s > 0.5:  return "emotionally intense, melancholic"
    if p < -0.2:               return "dark, introspective"
    return "neutral, understated"

# ─── Génération du prompt ─────────────────────────────────────────────────────
def build_prompt(row, lsub):
    parts = []

    # Genre dominant
    genre = row.get("dominant_genre", "")
    if genre and genre != "Autres":
        parts.append(genre)

    # Tags stylistiques (filtrés, dédupliqués, max 6)
    raw_tags = [t.strip() for t in str(row.get("top_tags", "")).split(",")]
    style_tags = [t for t in raw_tags if t and t.lower() not in SKIP_PROMPT_TAGS][:6]
    if style_tags:
        parts.extend(style_tags)

    # Texture acoustique
    tex = texture_label(row.get("acousticness_mean"))
    if tex:
        parts.append(tex)

    # Énergie/danceability
    nrg = energy_label(row.get("danceability_mean"))
    if nrg:
        parts.append(nrg)

    # Tempo (médiane plus robuste aux outliers)
    bpm_val = row.get("bpm_median") or row.get("bpm_mean")
    bpm_l   = bpm_label(bpm_val)
    if bpm_l:
        bpm_s = f"{float(bpm_val):.0f} BPM" if not pd.isna(bpm_val) else ""
        parts.append(f"{bpm_l} ({bpm_s})" if bpm_s else bpm_l)

    # Mood paroles
    if not lsub.empty:
        pol = lsub["sentiment_polarity"].mean()
        sub = lsub["sentiment_subjectivity"].mean()
        mood = mood_label(pol, sub)
        if mood:
            parts.append(mood)

        # Top mots paroles (hors stopwords basiques)
        SKIP_WORDS = {"like", "know", "just", "don", "got", "get", "let", "come",
                      "going", "make", "way", "time", "want", "look", "feel", "go"}
        all_words = []
        for tw in lsub["top_words"].dropna():
            try:
                all_words.extend(json.loads(tw))
            except Exception:
                pass
        word_freq = {}
        for w in all_words:
            if w.lower() not in SKIP_WORDS and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:4]
        if top_words:
            parts.append(f"lyrical themes: {', '.join(top_words)}")

    return ", ".join(parts)

# ─── Boucle principale ────────────────────────────────────────────────────────
results = []
print("=" * 70)
print("PROMPTS SUNO PAR GROUPE")
print("=" * 70)

for _, row in summary.iterrows():
    c = int(row["cluster"])
    lsub = lyrics[lyrics["cluster"] == c] if not lyrics.empty else pd.DataFrame()

    prompt = build_prompt(row, lsub)
    artists = row.get("top_artists", "")
    n       = int(row.get("n_tracks", 0))

    results.append({
        "cluster":     c,
        "n_tracks":    n,
        "top_artists": artists,
        "suno_prompt": prompt,
    })

    print(f"\n── Groupe {c+1} ({n} titres) ─────────────────────────────────────")
    print(f"   Artistes ref : {artists}")
    print(f"   PROMPT       : {prompt}")

out = pd.DataFrame(results)
out.to_csv("data/suno_prompts.csv", index=False)
print(f"\n→ data/suno_prompts.csv")
