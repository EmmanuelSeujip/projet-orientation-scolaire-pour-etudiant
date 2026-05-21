from fastapi import APIRouter
from backend.routes.chat.call_openrouter import call_openrouter

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("/")
def chat(payload: dict):
    """Envoie le payload à OpenRouter et retourne la réponse JSON."""
    return call_openrouter(payload)
