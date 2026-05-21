from backend.routes.forest_predict.forest_api import router as forest_predict_router
from backend.routes.chat.chat import router as chat_router
# Liste des routers disponibles
all_routers = [
    forest_predict_router,
    chat_router
]

