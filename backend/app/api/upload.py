import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..utils.dicom import load_image, save_png_from_array, read_metadata
from ..ai.predictor import Predictor
from ..db.session import SessionLocal
from ..db import models
from ..schemas import StudyCreate, StudyOut, FeedbackCreate
import json

router = APIRouter()

predictor = Predictor()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


@router.post("/upload", response_model=StudyOut)
def upload_dicom(file: UploadFile = File(...)):
    # Save incoming file
    uid = uuid.uuid4().hex
    filename = f"{uid}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file.file.read())

    try:
        img = load_image(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    metadata = read_metadata(path)
    patient_id = metadata.get("PatientID")

    # Save PNG preview
    png_path = os.path.join(STATIC_DIR, f"{uid}.png")
    save_png_from_array(img, png_path)

    # Run prediction
    result = predictor.predict(path)

    # Optional Grad-CAM visualization
    heatmap_path = None
    try:
        heatmap_path = predictor.gradcam(path, png_path)
    except Exception:
        heatmap_path = None

    # Persist to DB
    session = SessionLocal()
    study = models.Study(
        filename=filename,
        patient_id=patient_id,
        metadata_json=json.dumps(metadata),
        prediction=result.get("label"),
        score=float(result.get("score", 0.0)),
        ensemble_score=float(result.get("ensemble_score", result.get("score", 0.0))),
        uncertainty=float(result.get("uncertainty", 0.0)),
    )
    session.add(study)
    session.commit()
    session.refresh(study)

    out = StudyOut(
        id=study.id,
        filename=study.filename,
        patient_id=study.patient_id,
        metadata=metadata,
        prediction=study.prediction,
        score=study.score,
        ensemble_score=study.ensemble_score,
        uncertainty=study.uncertainty,
        review_label=study.review_label,
        png_url=f"/static/{uid}.png",
        heatmap_url=(f"/static/{os.path.basename(heatmap_path)}" if heatmap_path else None),
    )
    return out


@router.get("/studies", response_model=list[StudyOut])
def list_studies():
    session = SessionLocal()
    studies = session.query(models.Study).order_by(models.Study.created_at.desc()).limit(20).all()
    out = []
    for study in studies:
        metadata = json.loads(study.metadata_json or "{}")
        out.append(StudyOut(
            id=study.id,
            filename=study.filename,
            patient_id=study.patient_id,
            metadata=metadata,
            prediction=study.prediction,
            score=study.score,
            ensemble_score=study.ensemble_score,
            uncertainty=study.uncertainty,
            review_label=study.review_label,
            png_url=f"/static/{study.filename.split('_', 1)[0]}.png",
            heatmap_url=f"/static/{study.filename.split('_', 1)[0]}.heat.png" if study.ensemble_score is not None else None,
        ))
    return out


@router.post("/studies/{study_id}/feedback")
def submit_feedback(study_id: int, feedback: FeedbackCreate):
    session = SessionLocal()
    study = session.query(models.Study).filter(models.Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    study.review_label = feedback.review_label
    session.add(study)
    session.commit()
    return {"status": "success", "study_id": study.id, "review_label": feedback.review_label}
