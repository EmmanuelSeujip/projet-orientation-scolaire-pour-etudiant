from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np
from backend.routes.forest_predict.encode_input import encode_input
from backend.schema.studentInput import StudentInput
from backend.routes.forest_predict.calculer_multiplicateur_risque import calculer_multiplicateur_risque


router = APIRouter(prefix="/forest_predict")

LABELS_FINAL_RESULT = {0: "Withdrawn", 1: "Fail", 2: "Pass", 3: "Distinction"}


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

    # ── Multiplicateur social ─────────────────────────────────────
    multiplicateur = 1.0
    if student.profil_social:
        multiplicateur = calculer_multiplicateur_risque(student.profil_social)

    # ── Construction de la réponse ────────────────────────────────
    response = {}
    for col, row in result_df.iterrows():
        entry = {"valeur_numerique": round(float(row["prediction"]), 4)}

        if "label" in row and pd.notna(row.get("label")):
            entry["label"] = row["label"]

        if "confiance" in row and pd.notna(row.get("confiance")):
            entry["confiance"] = row["confiance"]

        if "source" in row:
            entry["source"] = row["source"]

        # ── Recalcul des probas pour final_result ─────────────────
        if col == "final_result" and multiplicateur != 1.0:
            features = [c for c in predictor.all_cols_ if c != "final_result"]
            x = np.array([known.get(f, 0.0) for f in features]).reshape(1, -1)

            rf_model = predictor.rf_models_["final_result"]
            classes  = rf_model.classes_
            probas   = rf_model.predict_proba(x)[0].copy()

            for i, classe in enumerate(classes):
                probas[i] *= multiplicateur if classe in [0, 1] else 1 / multiplicateur

            probas = probas / probas.sum()

            nouvelle_classe      = int(classes[np.argmax(probas)])
            entry["valeur_numerique"] = nouvelle_classe
            entry["label"]            = LABELS_FINAL_RESULT[nouvelle_classe]
            entry["confiance"]        = f"{max(probas) * 100:.1f}%"
            entry["detail_probas"]    = {
                LABELS_FINAL_RESULT[int(c)]: f"{p * 100:.1f}%"
                for c, p in zip(classes, probas)
            }

        response[col] = entry

    return {
        "input": {
            "gender":            student.gender,
            "disability":        student.disability,
            "highest_education": student.highest_education,
        },
        "multiplicateur_social": round(multiplicateur, 2),
        "predictions": response,
    }