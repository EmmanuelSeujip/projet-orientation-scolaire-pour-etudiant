from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from backend.schema.profil_social import ProfilSocial

class StudentInput(BaseModel):
    gender: Literal["M", "F"]
    disability: Literal["Y", "N"]
    highest_education: Literal[
        "No Formal quals", "Lower Than A Level", "A Level or Equivalent",
        "HE Qualification", "Post Graduate Qualification"
    ]
    profil_social: Optional[ProfilSocial] = None
    filiere: Optional[Literal[
        "sciences_exactes", "sciences_humaines",
        "gestion_commerce", "sante_medical", "arts_design"
    ]] = None
    known_extras: Optional[Dict[str, Any]] = None  
    nom_complet: Optional[str] = None
    filiere_label: Optional[str] = None

StudentInput.model_rebuild()