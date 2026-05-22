from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
from huggingface_hub import hf_hub_download
from typing import Literal
import pickle
import pandas as pd
import os
from backend.schema.studentInput import StudentInput
from backend.routes.forest_predict.encode_input import encode_input
from backend.models.student_predictor import StudentPredictorWithSGD

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "student_model.pkl")
    
    predictor = StudentPredictorWithSGD().load(local_path)  # ✅ utilise .load()
    app.state.predictor = predictor
    yield

# ── Chargement du modèle ──────────────────────────────────────────
# predictor = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global predictor

#     # local_path = hf_hub_download(
#     #     repo_id="EmmanuelSeujip/oulad-completion",  
#     #     filename="student_model.pkl"
#     # )
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     local_path = os.path.join(base_dir, "student_model.pkl")  # Chemin local vers ton modèle
#     with open(local_path, "rb") as f:
#         raw = pickle.load(f)
#         predictor = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
#         app.state.predictor = predictor
#         print("Modèle chargé ✓")
#     yield


app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost:5173",  # ton frontend en dev
    "http://127.0.0.1:5173",  # parfois utile selon config
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],            # GET, POST, PUT, DELETE...
    allow_headers=["*"],            # tous les headers
)
from backend import all_routers
for router in all_routers:
    app.include_router(router)




@app.post("/predict")
def predict(student: StudentInput):
    predictor_obj = app.state.predictor
    if predictor_obj is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # If the pickled object is a dict that wraps the real model, extract it
    if isinstance(predictor_obj, dict):
        if "model" in predictor_obj:
            predictor_obj = predictor_obj["model"]
        else:
            raise HTTPException(status_code=500, detail="Pickle file does not contain a model under key 'model'")

    known = encode_input(student)

    result_df: pd.DataFrame = predictor_obj.predict(
        known=known,
        decode_labels=True,
        confidence=True,
    )

    # Convertir le DataFrame en JSON propre
    response = {}
    for col, row in result_df.iterrows():
        entry = {"valeur_numerique": round(float(row["prediction"]), 4)}

        if "label" in row and pd.notna(row.get("label")):
            entry["label"] = row["label"]

        if "confiance" in row and pd.notna(row.get("confiance")):
            entry["confiance"] = row["confiance"]

        if "source" in row:
            entry["source"] = row["source"]  # "RF" ou "SGD"

        response[col] = entry

    return {
        "input": {
            "gender": student.gender,
            "disability": student.disability,
            "highest_education": student.highest_education,
        },
        "predictions": response,
    }


@app.get("/health")
def health():
    return {"status": "ok", "modele_charge": predictor is not None}