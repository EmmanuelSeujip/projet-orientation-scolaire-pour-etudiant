# backend/services/llm_service.py

import json
from backend.routes.chat.call_openrouter import call_openrouter

DOCUMENTATION_RESSOURCES = """
- resource_oucontent    : Contenu textuel principal du cours
- resource_forumng      : Forums de discussion (interaction sociale)
- resource_homepage     : Page d'accueil du module
- resource_subpage      : Sous-pages de navigation du cours
- resource_quiz         : Quiz d'entraînement internes
- resource_externalquiz : Quiz externes
- resource_resource     : Ressources statiques (PDF, documents)
- resource_url          : Liens vers des sites externes
- resource_ouwiki       : Wikis collaboratifs
- resource_glossary     : Glossaire des termes
- resource_questionnaire: Sondages / formulaires de feedback
- resource_oucollaborate: Outils de classe virtuelle
- exam_tma              : Évaluations corrigées par un tuteur
- exam_cma              : QCM corrigés automatiquement
- moyenne_clics_par_session : Intensité moyenne d'activité par semaine
- taux_participation    : Régularité de présence sur la plateforme (0 à 1)
"""

def generer_message_etudiant(
    profil_input: dict,
    predictions: dict,
    filiere: str | None = None,
    modele: str = "nvidia/nemotron-3-super-120b-a12b:free"
) -> str:
    """
    Génère un message personnalisé pour l'étudiant à partir des prédictions du modèle.
    L'objectif cible est toujours final_result = Pass.
    """

    # ── Extraire les données clés des prédictions ─────────────────
    final_result = predictions.get("final_result", {})
    note_moyenne = predictions.get("note_moyenne", {})
    taux_participation = predictions.get("taux_participation", {})

    # Ressources prédites (clics)
    ressources = {
        col: round(val["valeur_numerique"], 1)
        for col, val in predictions.items()
        if col.startswith("resource_")
    }

    # Trier par usage prédit (du plus au moins utilisé)
    ressources_triees = sorted(ressources.items(), key=lambda x: x[1], reverse=True)

    contexte_filiere = f"\nLa filière de l'étudiant est : {filiere}." if filiere else ""

    prompt = f"""
        Tu es un conseiller pédagogique bienveillant et expert en apprentissage en ligne.

        Un étudiant a soumis son profil à notre système de prédiction basé sur le dataset OULAD 
        (Open University Learning Analytics Dataset). 

        ## Profil de l'étudiant
        - Genre : {profil_input.get("gender")}
        - Handicap déclaré : {profil_input.get("disability")}
        - Niveau d'éducation : {profil_input.get("highest_education")}
        {contexte_filiere}

        ## Ce que le modèle prédit pour cet étudiant
        - Résultat probable : {final_result.get("label", "?")} (confiance : {final_result.get("confiance", "?")})
        - Note moyenne estimée : {note_moyenne.get("valeur_numerique", "?")}
        - Taux de participation estimé : {taux_participation.get("valeur_numerique", "?")}

        ## Ressources prédites (du plus au moins utilisé)
        {json.dumps(dict(ressources_triees[:8]), indent=2)}

        ## Documentation des ressources disponibles
        {DOCUMENTATION_RESSOURCES}

        ## Ta mission
        L'objectif est que l'étudiant obtienne **Pass** (réussite du module).

        Rédige un message personnalisé et motivant qui :
        1. Commence par une phrase d'accroche basée sur son profil
        2. Explique son résultat prédit et ce que ça signifie concrètement
        3. Recommande qualitativement les **types de ressources** à prioriser 
        (sans mentionner des chiffres de clics - parle plutôt d'habitudes et de stratégies)
        4. Donne 2-3 conseils concrets adaptés à son niveau d'éducation{" et sa filière" if filiere else ""}
        5. Termine par une phrase encourageante
        
        Ne mentionne jamais les noms techniques des colonnes comme "resource_forumng".
    """

    payload = {
        "model": modele,
        "max_tokens": 500,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    result = call_openrouter(payload)
    print(result)
    if "choices" not in result:
        print(f"Erreur OpenRouter : {result}")
        return "Désolé, je n'ai pas pu générer de conseil personnalisé pour le moment. Veuillez réessayer plus tard."
    return result["choices"][0]["message"]["content"]