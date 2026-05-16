from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from .session import Base


class Study(Base):
    __tablename__ = "studies"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    patient_id = Column(String, nullable=True, index=True)
    metadata_json = Column(Text)
    prediction = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    ensemble_score = Column(Float, nullable=True)
    uncertainty = Column(Float, nullable=True)
    review_label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
