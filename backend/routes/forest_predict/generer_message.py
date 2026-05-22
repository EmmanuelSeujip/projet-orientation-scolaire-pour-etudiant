# backend/services/llm_service.py

import json
import re
from backend.routes.chat.call_openrouter import call_openrouter

DOCUMENTATION_RESSOURCES = """
- Contenu principal du cours     : Le matériel textuel central du module
- Forums de discussion           : Espaces d'échange avec les autres étudiants
- Page d'accueil du module       : Point d'entrée et navigation générale
- Sous-pages de navigation       : Sections détaillées du cours
- Quiz d'entraînement            : Exercices pratiques pour tester ses connaissances
- Ressources téléchargeables     : Documents PDF et supports de cours
- Liens externes                 : Ressources complémentaires en dehors de la plateforme
- Wikis collaboratifs            : Espaces de co-construction des connaissances
- Glossaire                      : Définitions des termes clés du module
- Outils de classe virtuelle     : Sessions en direct avec l'enseignant
- Évaluations tuteur (TMA)       : Devoirs corrigés manuellement par un tuteur
- QCM automatiques (CMA)         : Tests corrigés automatiquement
"""


def _strip_thinking(text: str) -> str:
    """
    Supprime les blocs de réflexion que certains modèles génèrent
    avant leur réponse finale.
    """
    # Balises <think>...</think> (DeepSeek, Qwen, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Balises <reflection>...</reflection>
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    # Lignes "Réflexion :" ou "Thinking :" en début de bloc
    text = re.sub(r"(?i)^(réflexion|thinking|thought|analysis)\s*:.*?\n\n", "", text, flags=re.DOTALL)
    return text.strip()


def generer_message_etudiant(
    profil_input: dict,
    predictions: dict,
    filiere: str | None = None,
    modele: str = "nvidia/nemotron-3-super-120b-a12b:free"
) -> str:
    """
    Génère un message personnalisé pour l'étudiant à partir des prédictions
    du modèle ML. L'objectif cible est toujours final_result = Pass.

    Paramètres
    ──────────
    profil_input : dict contenant gender, disability, highest_education,
                   nom_complet (optionnel), filiere_label (optionnel)
    predictions  : dict retourné par la route /forest_predict
    filiere      : catégorie de filière (ex: "sciences_exactes")
    modele       : modèle OpenRouter à utiliser
    """

    # ── 1. Extraire le prénom ─────────────────────────────────────
    nom_complet = profil_input.get("nom_complet", "").strip()
    prenom = nom_complet.split()[0] if nom_complet else None

    # ── 2. Extraire la filière précise ────────────────────────────
    filiere_precise = profil_input.get("filiere_label", "").strip()

    # ── 3. Construire le contexte filière ─────────────────────────
    contexte_filiere = ""
    if filiere_precise:
        contexte_filiere += f"\n- Filière visée : {filiere_precise}"
    if filiere:
        contexte_filiere += f"\n- Catégorie : {filiere.replace('_', ' ').title()}"

    # ── 4. Extraire les prédictions clés ──────────────────────────
    final_result      = predictions.get("final_result", {})
    note_moyenne      = predictions.get("note_moyenne", {})
    taux_participation= predictions.get("taux_participation", {})

    resultat_label    = final_result.get("label", "inconnu")
    resultat_confiance= final_result.get("confiance", "?")
    note_val          = round(float(note_moyenne.get("valeur_numerique", 0)), 1)
    participation_val = round(float(taux_participation.get("valeur_numerique", 0)) * 100, 1)

    # ── 5. Extraire et trier les ressources prédites ──────────────
    ressources = {
        col: round(float(val["valeur_numerique"]), 1)
        for col, val in predictions.items()
        if col.startswith("resource_") and float(val.get("valeur_numerique", 0)) > 0
    }
    ressources_triees = sorted(ressources.items(), key=lambda x: x[1], reverse=True)[:6]

    # ── 6. Construire le prompt ───────────────────────────────────
    intro_prenom = f"L'étudiant s'appelle {prenom}. Adresse-toi à lui par son prénom." if prenom else ""

    prompt = f"""Tu es un conseiller pédagogique bienveillant et expert en apprentissage en ligne.
Réponds DIRECTEMENT avec le message final. N'inclus aucune réflexion, analyse préalable, ou introduction méta.
{intro_prenom}

## Profil de l'étudiant
- Genre : {"Homme" if profil_input.get("gender") == "M" else "Femme"}
- Handicap déclaré : {"Oui" if profil_input.get("disability") == "Y" else "Non"}
- Niveau d'éducation : {profil_input.get("highest_education", "non précisé")}
{contexte_filiere}

## Prédictions du modèle ML
- Résultat probable : **{resultat_label}** (confiance : {resultat_confiance})
- Note moyenne estimée : {note_val}/100
- Taux de participation estimé : {participation_val}%

## Ressources les plus utilisées par des profils similaires
{json.dumps(dict(ressources_triees), indent=2, ensure_ascii=False)}

## Documentation des types de ressources
{DOCUMENTATION_RESSOURCES}

## Ta mission
L'objectif est que l'étudiant obtienne **Pass** (réussite du module).

Rédige un message personnalisé (150-250 mots) qui :
1. Commence par saluer l'étudiant{"par son prénom" if prenom else ""} avec une phrase d'accroche
2. Explique clairement ce que prédit le modèle et ce que ça signifie pour lui
3. Recommande 2-3 types de ressources à prioriser en utilisant leur description humaine, jamais leur nom technique
4. Donne des conseils adaptés à{"sa filière " + filiere_precise if filiere_precise else "son profil"}
5. Termine par une phrase encourageante et motivante

Ne mentionne JAMAIS les noms de colonnes techniques (resource_forumng, exam_tma, etc.)."""

    # ── 7. Appel OpenRouter ───────────────────────────────────────
    payload = {
        "model": modele,
        "max_tokens": 600,
        "messages": [
            {
                "role": "system",
                "content": "Tu es un conseiller pédagogique. Réponds uniquement avec le message demandé, sans réflexion préalable."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    result = call_openrouter(payload)

    # ── 8. Vérification et nettoyage ──────────────────────────────
    if "choices" not in result:
        print(f"Erreur OpenRouter : {result}")
        return (
            "Désolé, nous n'avons pas pu générer votre conseil personnalisé pour le moment. "
            "Veuillez réessayer dans quelques instants."
        )

    message_brut = result["choices"][0]["message"]["content"]
    return _strip_thinking(message_brut)