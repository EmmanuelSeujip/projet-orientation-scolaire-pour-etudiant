from backend.schema.studentInput import StudentInput

# Constantes identiques au notebook
ORDINAL_MAPS = {
    'final_result':       {0: 'Withdrawn', 1: 'Fail', 2: 'Pass', 3: 'Distinction'},
    'highest_education':  {
        0: 'No Formal quals',
        1: 'Lower Than A Level',
        2: 'A Level or Equivalent',
        3: 'HE Qualification',
        4: 'Post Graduate Qualification',
    },
}

# Encodage des inputs vers le format attendu par le modèle
def encode_input(data: StudentInput) -> dict:
    education_map = {
        "No Formal quals":              0,
        "Lower Than A Level":           1,
        "A Level or Equivalent":        2,
        "HE Qualification":             3,
        "Post Graduate Qualification":  4,
    }
    return {
        "gender_F":          1 if data.gender == "F" else 0,
        "gender_M":          1 if data.gender == "M" else 0,
        "disability_N":      1 if data.disability == "N" else 0,
        "disability_Y":      1 if data.disability == "Y" else 0,
        "highest_education": education_map[data.highest_education],
    }