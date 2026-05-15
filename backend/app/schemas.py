from pydantic import BaseModel
from typing import Optional, Dict


class StudyOut(BaseModel):
    id: int
    filename: str
    metadata: Dict
    prediction: Optional[str]
    score: Optional[float]
    png_url: Optional[str]
    heatmap_url: Optional[str]

    class Config:
        orm_mode = True


class StudyCreate(BaseModel):
    filename: str
