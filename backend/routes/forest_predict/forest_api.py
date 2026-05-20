from fastapi import APIRouter, HTTPException, Request
import pandas as pd
from backend.routes.forest_predict.encode_input import encode_input
from backend.schema.studentInput import StudentInput

router = APIRouter(
    prefix="/forest_predict"
)

@router.post("/")
def predict_forest(student: StudentInput, request: Request):
    predictor = request.app.state.predictor
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

