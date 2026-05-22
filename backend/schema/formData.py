from pydantic import BaseModel
from typing import Any

class FormData(BaseModel):
    nomComplet: str
    age: str
    sexe: str
    situationLogement: str
    handicap: str
    diplomActuel: str
    filieresouhaitee: str
    methodesApprentissage: list
    methodesExercice: list
    travailleur: bool
