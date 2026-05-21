def calculer_multiplicateur_risque(profil: ProfilSocial) -> float:
    """
    Calcule un coefficient qui va augmenter (>1) ou réduire (<1)
    les probabilités d'échec et d'abandon générées par le ML.
    """
    multiplicateur = 1.0

    # Résilience due à la maturité
    if profil.age >= 25:
        multiplicateur -= 0.05

    # Environnement de travail / Charge domestique
    if profil.situation_logement == "seul":
        multiplicateur -= 0.10
    elif profil.situation_logement == "famille_dense":
        multiplicateur += 0.10

    # Contraintes de temps (Cumul emploi-études)
    if profil.travail_salarie:
        multiplicateur += 0.20

    # Plafond et plancher de sécurité pour éviter des probabilités aberrantes
    return max(0.5, min(1.5, multiplicateur))