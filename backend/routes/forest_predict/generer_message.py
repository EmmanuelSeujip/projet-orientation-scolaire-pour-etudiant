# backend/services/llm_service.py

import json, re
from backend.routes.chat.call_openrouter import call_openrouter

NOMS_HUMAINS_RESSOURCES = {
    "resource_oucontent":     "contenu principal du cours",
    "resource_forumng":       "forums de discussion",
    "resource_homepage":      "page d'accueil du module",
    "resource_subpage":       "sous-pages du cours",
    "resource_quiz":          "quiz d'entraînement",
    "resource_externalquiz":  "quiz externes",
    "resource_resource":      "documents PDF et supports de cours",
    "resource_url":           "liens et ressources externes",
    "resource_ouwiki":        "wikis collaboratifs",
    "resource_glossary":      "glossaire du cours",
    "resource_questionnaire": "sondages de feedback",
    "resource_oucollaborate": "sessions de classe virtuelle",
    "exam_tma":               "devoirs corrigés par un tuteur",
    "exam_cma":               "QCM automatiques",
}

METRIQUES_HUMAINES = {
    "note_moyenne":             "Note moyenne",
    "taux_participation":       "Taux de participation",
    "moyenne_clics_par_session":"Activité par session",
    "exam_tma":                 "Devoirs tuteur",
    "exam_cma":                 "QCM automatiques",
    "resource_oucontent":       "Lecture du cours",
    "resource_forumng":         "Participation aux forums",
    "resource_quiz":            "Quiz d'entraînement",
    "resource_homepage":        "Navigation générale",
}


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    text = re.sub(r"(?s)^.*?(?=Bonjour|Cher|Salut|Félicitations)", "", text)
    return text.strip()


def _traduire_ressources(predictions: dict) -> dict:
    """Remplace les noms techniques par des noms humains."""
    return {
        NOMS_HUMAINS_RESSOURCES.get(col, col): round(float(val["valeur_numerique"]), 1)
        for col, val in predictions.items()
        if col.startswith("resource_") or col.startswith("exam_")
        if float(val.get("valeur_numerique", 0)) > 0
    }


def _extraire_comparaison(predictions_1: dict, predictions_2: dict) -> list[dict]:
    """Identifie les écarts entre situation actuelle et cible Pass."""
    comparaison = []
    for col, label in METRIQUES_HUMAINES.items():
        val_actuel = float(predictions_1.get(col, {}).get("valeur_numerique", 0))
        val_cible  = float(predictions_2.get(col, {}).get("valeur_numerique", 0))
        if val_cible > 0:
            ecart_pct = round(((val_cible - val_actuel) / max(val_cible, 0.01)) * 100, 1)
            comparaison.append({
                "metrique": label,
                "actuel":   round(val_actuel, 2),
                "cible":    round(val_cible, 2),
                "effort":   ecart_pct,
            })
    return sorted(comparaison, key=lambda x: x["effort"], reverse=True)[:5]


def generer_message_etudiant(
    profil_input:          dict,
    predictions_actuelles: dict,
    predictions_cible:     dict | None = None,
    filiere:               str | None  = None,
    est_bon_resultat:      bool        = False,
    modele:                str         = "nvidia/nemotron-3-super-120b-a12b:free",
) -> str:

    # ── Prénom ────────────────────────────────────────────────────
    nom_complet   = profil_input.get("nom_complet", "").strip()
    prenom        = nom_complet.split()[0] if nom_complet else None
    filiere_label = profil_input.get("filiere_label", "")

    # ── Données actuelles ─────────────────────────────────────────
    fr1      = predictions_actuelles.get("final_result", {})
    note_1   = round(float(predictions_actuelles.get("note_moyenne", {}).get("valeur_numerique", 0)), 1)
    part_1   = round(float(predictions_actuelles.get("taux_participation", {}).get("valeur_numerique", 0)) * 100, 1)
    res_1    = _traduire_ressources(predictions_actuelles)

    # ── Contexte filière ──────────────────────────────────────────
    ctx_filiere = ""
    if filiere_label:
        ctx_filiere += f"\n- Filière : {filiere_label}"
    if filiere:
        ctx_filiere += f"\n- Catégorie : {filiere.replace('_', ' ').title()}"

    # ── Branche selon résultat ────────────────────────────────────
    if est_bon_resultat:
        contexte_mission = f"""
L'étudiant est sur la bonne voie avec un résultat prédit **{fr1.get("label")}** ({fr1.get("confiance")}).
Ta mission : félicite-le, renforce ses bonnes habitudes et donne 2-3 pistes pour viser la Distinction.
"""
        contexte_comparaison = ""
    else:
        note_2  = round(float(predictions_cible.get("note_moyenne", {}).get("valeur_numerique", 0)), 1)
        part_2  = round(float(predictions_cible.get("taux_participation", {}).get("valeur_numerique", 0)) * 100, 1)
        res_2   = _traduire_ressources(predictions_cible)
        comparaison = _extraire_comparaison(predictions_actuelles, predictions_cible)

        contexte_mission = f"""
Le résultat actuel est **{fr1.get("label")}** ({fr1.get("confiance")}) — c'est préoccupant.
Ta mission : explique l'écart entre sa situation et ce qu'il faut pour réussir (Pass), 
recommande des actions concrètes, reste encourageant.
"""
        contexte_comparaison = f"""
## Comparaison : situation actuelle vs ce qu'il faut pour Pass
- Note moyenne : {note_1}/100 → objectif {note_2}/100
- Participation : {part_1}% → objectif {part_2}%

Principaux efforts à fournir :
{json.dumps(comparaison, indent=2, ensure_ascii=False)}

Ressources actuellement utilisées :
{json.dumps(res_1, indent=2, ensure_ascii=False)}

Ressources d'un profil Pass :
{json.dumps(res_2, indent=2, ensure_ascii=False)}
"""

    prompt = f"""Tu es un conseiller pédagogique dans le contexte de l'enseignement supérieur camerounais.
Tu connais les réalités locales : charge de travail, accès à internet parfois limité, études en parallèle d'un emploi, pression familiale.
Réponds DIRECTEMENT avec le message. Pas d'analyse préalable. Pas de méta-commentaire.
{"Adresse-toi à " + prenom + " par son prénom." if prenom else ""}

## Profil
- Genre : {"Homme" if profil_input.get("gender") == "M" else "Femme"}
- Handicap : {"Oui" if profil_input.get("disability") == "Y" else "Non"}
- Niveau : {profil_input.get("highest_education", "non précisé")}
{ctx_filiere}

## Situation actuelle
- Résultat prédit : {fr1.get("label")} ({fr1.get("confiance")})
- Note estimée : {note_1}/100
- Participation : {part_1}%

{contexte_comparaison}

## Mission
{contexte_mission}

Rédige exactement 150-200 mots en français. Structure :
1. Salutation + accroche personnalisée
2. Explication claire du résultat prédit
3. 2-3 recommandations concrètes avec noms humains des ressources (jamais les noms techniques)
4. Conseil adapté au contexte camerounais{" et à la filière " + filiere_label if filiere_label else ""}
5. Phrase finale encourageante

INTERDIT : mentionner resource_quiz, exam_tma, resource_forumng ou tout autre nom de colonne technique."""

    result = call_openrouter({
        "model": modele,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un conseiller pédagogique. Commence directement par le message sans aucune réflexion préalable."
            },
            {"role": "user", "content": prompt}
        ]
    })

    if "choices" not in result:
        print(f"Erreur OpenRouter : {result}")
        return "Désolé, le conseil personnalisé n'est pas disponible pour le moment."

    return _strip_thinking(result["choices"][0]["message"]["content"])