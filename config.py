"""
config.py — charge les variables d'environnement depuis .env
Importé par tous les scripts : from config import SPOTIFY_CLIENT_ID, ...
"""

from dotenv import load_dotenv
import os

load_dotenv()

SPOTIFY_CLIENT_ID     = os.environ["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

GENIUS_ACCESS_TOKEN   = os.getenv("GENIUS_ACCESS_TOKEN", "")

LASTFM_API_KEY        = os.environ["LASTFM_API_KEY"]
LASTFM_API_SECRET     = os.environ["LASTFM_API_SECRET"]

GETSONGBPM_API_KEY    = os.getenv("GETSONGBPM_API_KEY", "")
