from fastapi import APIRouter, HTTPException, Request
import numpy as np
import pandas as pd
from backend.routes.forest_predict.encode_input import encode_input
from backend.schema.studentInput import StudentInput
from backend.schema.formData import FormData
from backend.utils.normaliseRequest import prepare_send
from backend.routes.forest_predict.calculer_multiplicateur_risque import calculer_multiplicateur_risque
from backend.routes.forest_predict.generer_message import generer_message_etudiant

router = APIRouter(prefix="/forest_predict")

LABELS_FINAL_RESULT = {0: "Withdrawn", 1: "Fail", 2: "Pass", 3: "Distinction"}
BONS_RESULTATS = {"Pass", "Distinction"}


def _build_response(result_df: pd.DataFrame, predictor, known: dict, multiplicateur: float) -> dict:
    """Construit le dict de réponse à partir du DataFrame de prédictions."""
    response = {}
    for col, row in result_df.iterrows():
        entry = {"valeur_numerique": round(float(row["prediction"]), 4)}
        if "label" in row and pd.notna(row.get("label")):
            entry["label"] = row["label"]
        if "confiance" in row and pd.notna(row.get("confiance")):
            entry["confiance"] = row["confiance"]
        if "source" in row:
            entry["source"] = row["source"]

        # Recalcul des probas pour final_result avec multiplicateur social
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
                LABELS_FINAL_RESULT[int(c)]: round(float(p) * 100, 1)
                for c, p in zip(classes, probas)
            }

        response[col] = entry
    return response


@router.post("/")
def predict_forest(raw_data: FormData, request: Request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    payload  = prepare_send(raw_data.model_dump())
    student  = StudentInput(**payload)
    known    = encode_input(student)

    if student.known_extras:
        known.update(student.known_extras)

    # ── Multiplicateur social ─────────────────────────────────────
    multiplicateur = 1.0
    if student.profil_social:
        multiplicateur = calculer_multiplicateur_risque(student.profil_social)

    # ── Première prédiction : situation réelle ────────────────────
    result_df_1      = predictor.predict(known=known, decode_labels=True, confidence=True)
    predictions_1    = _build_response(result_df_1, predictor, known, multiplicateur)

    final_label      = predictions_1.get("final_result", {}).get("label", "")
    est_bon_resultat = final_label in BONS_RESULTATS

    # ── Deuxième prédiction : simulation Pass (si résultat mauvais) ──
    predictions_2 = None
    if not est_bon_resultat:
        known_cible = {**known, "final_result": 2}  # 2 = Pass
        result_df_2   = predictor.predict(known=known_cible, decode_labels=True, confidence=True)
        predictions_2 = _build_response(result_df_2, predictor, known_cible, multiplicateur)

    # ── Message LLM ───────────────────────────────────────────────
    message = generer_message_etudiant(
        profil_input        = payload,
        predictions_actuelles = predictions_1,
        predictions_cible   = predictions_2,
        filiere             = student.filiere,
        est_bon_resultat    = est_bon_resultat,
    )

    return {
        "est_bon_resultat":     est_bon_resultat,
        "multiplicateur_social": round(multiplicateur, 2),
        "predictions_actuelles": predictions_1,
        "predictions_cible":    predictions_2,
        "message":              message,
    }