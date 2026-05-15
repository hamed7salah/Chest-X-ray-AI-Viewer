import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..utils.dicom import load_image, save_png_from_array
from ..ai.predictor import Predictor
from ..db.session import SessionLocal
from ..db import models
from ..schemas import StudyCreate, StudyOut
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
        metadata_json=json.dumps({}),
        prediction=result.get("label"),
        score=float(result.get("score", 0.0)),
    )
    session.add(study)
    session.commit()
    session.refresh(study)

    out = StudyOut(
        id=study.id,
        filename=study.filename,
        metadata={},
        prediction=study.prediction,
        score=study.score,
        png_url=f"/static/{uid}.png",
        heatmap_url=(f"/static/{os.path.basename(heatmap_path)}" if heatmap_path else None),
    )
    return out
