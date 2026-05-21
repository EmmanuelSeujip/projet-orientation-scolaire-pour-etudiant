# backend/config/filieres.py

PROFILS_FILIERES = {
    "sciences_exactes": {
        "label": "Sciences exactes (Maths, Physique, Informatique...)",
        "ressources_prioritaires": [
            "resource_quiz", "resource_externalquiz",
            "resource_dataplus", "resource_resource"
        ],
        "ressources_secondaires": [
            "resource_oucontent", "resource_homepage"
        ],
        "conseil_specifique": (
            "En sciences exactes, la pratique régulière via les quiz "
            "et les exercices est plus décisive que la lecture passive."
        )
    },
    "sciences_humaines": {
        "label": "Sciences humaines (Lettres, Histoire, Sociologie...)",
        "ressources_prioritaires": [
            "resource_forumng", "resource_oucontent",
            "resource_ouwiki", "resource_glossary"
        ],
        "ressources_secondaires": [
            "resource_questionnaire", "resource_subpage"
        ],
        "conseil_specifique": (
            "En sciences humaines, la participation aux forums et "
            "la lecture approfondie du contenu sont des facteurs clés de réussite."
        )
    },
    "gestion_commerce": {
        "label": "Gestion / Commerce / Économie",
        "ressources_prioritaires": [
            "resource_oucontent", "resource_resource",
            "resource_url", "resource_homepage"
        ],
        "ressources_secondaires": [
            "resource_quiz", "resource_forumng"
        ],
        "conseil_specifique": (
            "En gestion, les études de cas et les ressources externes "
            "complètent bien le contenu principal du cours."
        )
    },
    "sante_medical": {
        "label": "Santé / Médecine / Paramédical",
        "ressources_prioritaires": [
            "resource_quiz", "resource_externalquiz",
            "resource_oucontent", "resource_glossary"
        ],
        "ressources_secondaires": [
            "resource_resource", "resource_oucollaborate"
        ],
        "conseil_specifique": (
            "En santé, la mémorisation active via les quiz et "
            "la maîtrise du vocabulaire sont essentiels."
        )
    },
    "arts_design": {
        "label": "Arts / Design / Création",
        "ressources_prioritaires": [
            "resource_oucontent", "resource_ouwiki",
            "resource_url", "resource_oucollaborate"
        ],
        "ressources_secondaires": [
            "resource_forumng", "resource_subpage"
        ],
        "conseil_specifique": (
            "En arts, l'exploration de ressources externes et "
            "les échanges collaboratifs enrichissent la créativité."
        )
    },
}