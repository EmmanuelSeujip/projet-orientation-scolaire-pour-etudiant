# backend/schema/studentInput.py

from pydantic import BaseModel
from typing import Literal, Optional
from backend.schema.profil_social import ProfilSocial  # ← import correct


class StudentInput(BaseModel):
    gender: Literal["M", "F"]
    disability: Literal["Y", "N"]
    highest_education: Literal[
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification"
    ]
    profil_social: Optional[ProfilSocial] = None


StudentInput.model_rebuild()