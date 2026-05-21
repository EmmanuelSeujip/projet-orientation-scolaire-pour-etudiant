from pydantic import BaseModel
from typing import Literal  # ← manquant


class ProfilSocial(BaseModel):
    age: int
    situation_logement: Literal["seul", "colocation", "famille_dense"]
    travail_salarie: bool
