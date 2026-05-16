from pydantic import BaseModel
from typing import Optional, Dict


class StudyOut(BaseModel):
    id: int
    filename: str
    patient_id: Optional[str]
    metadata: Dict
    prediction: Optional[str]
    score: Optional[float]
    ensemble_score: Optional[float]
    uncertainty: Optional[float]
    review_label: Optional[str]
    png_url: Optional[str]
    heatmap_url: Optional[str]

    class Config:
        orm_mode = True


class StudyCreate(BaseModel):
    filename: str


class FeedbackCreate(BaseModel):
    review_label: str
    comment: Optional[str] = None
