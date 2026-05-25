"""
utils.py — fonctions partagées entre les scripts du pipeline
"""

import re
import unicodedata

# ─── Normalisation des titres ─────────────────────────────────────────────────
# Retire les suffixes d'édition/version pour dédupliquer les doublons Spotify
_STRIP_RE = re.compile(
    r"\s*[\(\[]"
    r"(feat\.?|ft\.?|remaster(ed)?|reissue|deluxe|anniversary|edition|version"
    r"|live|radio edit|acoustic|demo|bonus|extended|single|mono|stereo|\d{4})"
    r".*?[\)\]]"
    r"|\s*[\(\[]\s*with .*?[\)\]]"
    r"|\s+feat\.?\s+.*$"
    r"|\s*-\s*(remaster(ed)?|live|acoustic|radio edit|extended|demo).*$",
    re.IGNORECASE,
)

def normalize_title(title: str) -> str:
    """Normalise un titre pour la déduplication (minuscules, sans suffixes d'édition)."""
    return _STRIP_RE.sub("", str(title)).strip().lower()

def normalize_artist(artist: str) -> str:
    """Normalise un nom d'artiste (minuscules, sans espaces superflus)."""
    return str(artist).strip().lower()

# ─── Normalisation unicode pour les API ASCII-only ────────────────────────────
def ascii_normalize(text: str) -> str:
    """Convertit les caractères accentués/spéciaux en ASCII (RÜFÜS → RUFUS)."""
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").strip()

# ─── Normalisation des tonalités musicales ────────────────────────────────────
# Gère les deux encodages : ASCII (#/b) et unicode (♯/♭)
KEY_ALIASES = {
    # ASCII sharp → bemol enharmonique
    "C#": "Db", "C#m": "Dbm",
    "G#": "Ab", "G#m": "Abm",
    "D#": "Eb", "D#m": "Ebm",
    "A#": "Bb", "A#m": "Bbm",
    "Gb": "F#", "Gbm": "F#m",
    # Unicode ♯ (U+266F) — produit par GetSongBPM
    "C♯": "Db",  "C♯m": "Dbm",
    "G♯": "Ab",  "G♯m": "Abm",
    "D♯": "Eb",  "D♯m": "Ebm",
    "A♯": "Bb",  "A♯m": "Bbm",
    "F♯": "F#",  "F♯m": "F#m",
    # Unicode ♭ (U+266D)
    "D♭": "Db",  "D♭m": "Dbm",
    "G♭": "F#",  "G♭m": "F#m",
    "A♭": "Ab",  "A♭m": "Abm",
    "E♭": "Eb",  "E♭m": "Ebm",
    "B♭": "Bb",  "B♭m": "Bbm",
}
