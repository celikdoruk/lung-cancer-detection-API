from pydantic import BaseModel, Field
from typing import Literal

# request schema
class Request(BaseModel):
    age: int = Field(..., gt=0, le=120, description="Age of the patient.")
    gender: Literal["male", "female"] = Field(..., description="Gender of the patient.")
    pack_years: float = Field(..., ge=0, description="Daily cigarette usege times year since first smoke.")
    radon_exposure: Literal["low", "medium", "high"] = Field(..., description="Radon Exposure")
    asbestos_exposure: Literal["yes", "no"] = Field(..., description="Asbestos Exposure")
    secondhand_smoke_exposure: Literal["yes", "no"] = Field(..., description="Second hand smoke exposure.")
    copd_diagnosis: Literal["yes", "no"] = Field(..., description="Copd Diagnosis")
    alcohol_consumption: Literal["moderate", "heavy"] = Field(..., description="Alcohol consumption.")
    family_history: Literal["yes", "no"] = Field(..., description="Family history of lung cancer.")

# response schema
class ModelOutput(BaseModel):
    model_prediction: int = Field(..., description="The model prediction.")