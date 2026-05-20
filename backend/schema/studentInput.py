from pydantic import BaseModel
from typing import Literal

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