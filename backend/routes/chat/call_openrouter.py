import os
from dotenv import load_dotenv
import requests
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# --- 1. Fonction utilitaire ---
def call_openrouter(payload: dict):
    """Envoie le payload à OpenRouter et retourne la réponse JSON."""
    response = requests.post(BASE_URL, headers=HEADERS, json=payload)
    return response.json()