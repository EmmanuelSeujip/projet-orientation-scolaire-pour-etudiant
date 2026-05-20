from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
from huggingface_hub import hf_hub_download
from typing import Literal
import pickle
import pandas as pd
from backend.schema.studentInput import StudentInput
from backend.routes.forest_predict.encode_input import encode_input



# ── Chargement du modèle ──────────────────────────────────────────
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    # local_path = hf_hub_download(
    #     repo_id="EmmanuelSeujip/oulad-completion",  
    #     filename="student_model.pkl"
    # )
    local_path = "student_model.pkl"  # Chemin local vers ton modèle
    with open(local_path, "rb") as f:
        data = pickle.load(f)

    # Reconstruire l'objet StudentPredictorWithSGD depuis le pickle
    import pickle as pkl
    predictor = pkl.load(open(local_path, "rb"))  
    app.state.predictor = predictor
    print("Modèle chargé ✓")
    yield


app = FastAPI(lifespan=lifespan)

from backend import all_routers
for router in all_routers:
    app.include_router(router)




@app.post("/predict")
def predict(student: StudentInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    known = encode_input(student)

    result_df: pd.DataFrame = predictor.predict(
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