# backend/utils/prepare_send.py

DIPLOME_MAP = {
    "Aucun diplôme officiel":   "No Formal quals",
    "BEPC / CAP":               "Lower Than A Level",
    "Baccalauréat":             "A Level or Equivalent",
    "Diplôme de l'enseignement supérieur (Licence, DUT, BTS, ou équivalent Bac+2/Bac+3)": "HE Qualification",
    "Diplôme de troisième cycle (Master, Doctorat, DEA, DESS, ou équivalent Bac+5 et plus)": "Post Graduate Qualification",
}

FILIERES_DICTIONNAIRE = {
    "sciences_exactes": ["Mathématiques", "Physique", "Chimie", "Informatique",
                         "Génie civil", "Génie électrique", "Génie mécanique",
                         "Statistiques", "Biologie"],
    "sciences_humaines": ["Droit", "Sociologie", "Psychologie", "Histoire",
                          "Géographie", "Philosophie", "Linguistique",
                          "Sciences politiques", "Communication"],
    "gestion_commerce":  ["Comptabilité", "Finance", "Marketing", "Management",
                          "Commerce international", "Ressources humaines",
                          "Audit", "Banque & Assurance", "Entrepreneuriat"],
    "sante_medical":     ["Médecine générale", "Pharmacie", "Chirurgie dentaire",
                          "Infirmerie", "Sage-femme", "Kinésithérapie",
                          "Biologie médicale", "Santé publique"],
    "arts_design":       ["Architecture", "Design graphique", "Arts plastiques",
                          "Mode & Stylisme", "Audiovisuel", "Musique",
                          "Animation 3D", "Photographie"],
}

# Valeurs Q3 (75e percentile) — niveau "bon usage"
METHODES_APPRENTISSAGE_MAP = {
    "en_ligne":    {"resource_url": 28},
    "pdf":         {"resource_resource": 53.0},
    "cours_classe":{"resource_oucontent": 522.0},
}

#  CORRIGÉ : c'était methodesApprentissage au lieu de methodesExercice
METHODES_EXERCICE_MAP = {
    "epreuves": {"exam_exam": 1.0},   # Q3=0 mais on signal l'intention
    "qcm":      {"exam_cma": 5.0},
    "tuteur":   {"exam_tma": 5.0},
}


def prepare_send(data: dict) -> dict:
    payload = {}

    # ── Genre ─────────────────────────────────────────────────────
    payload["gender"] = "M" if data.get("sexe") == "homme" else "F"

    # ── Handicap ──────────────────────────────────────────────────
    payload["disability"] = "Y" if data.get("handicap") == "oui" else "N"

    # ── Diplôme → highest_education ───────────────────────────────
    payload["highest_education"] = DIPLOME_MAP.get(
        data.get("diplomActuel", ""), "No Formal quals"
    )

    # ── Filière spécifique → catégorie ────────────────────────────
    filiere_souhaitee = data.get("filieresouhaitee", "")
    for categorie, filieres in FILIERES_DICTIONNAIRE.items():
        if filiere_souhaitee in filieres:
            payload["filiere"] = categorie
            break

    # ── Profil social ─────────────────────────────────────────────
    payload["profil_social"] = {
        "age":               int(data.get("age", 0)),
        "situation_logement": data.get("situationLogement"),
        "travail_salarie":   bool(data.get("travailleur", False)),
    }

    # ── Méthodes d'apprentissage → colonnes ressources connues ────
    known_extras = {}
    for method in data.get("methodesApprentissage", []):
        if method in METHODES_APPRENTISSAGE_MAP:
            known_extras.update(METHODES_APPRENTISSAGE_MAP[method])

    # ── Modes d'exercice → colonnes examens connues ───────────────
    for method in data.get("methodesExercice", []):    # ← BUG CORRIGÉ
        if method in METHODES_EXERCICE_MAP:
            known_extras.update(METHODES_EXERCICE_MAP[method])

    if known_extras:
        payload["known_extras"] = known_extras

    payload["nom_complet"]   = data.get("nomComplet", "")
    payload["filiere_label"] = data.get("filieresouhaitee", "") 

    return payload