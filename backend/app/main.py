from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from sqlalchemy import inspect, text
from .api import upload
from .db.session import engine
from .db import models

app = FastAPI(title="Chest X-ray AI Viewer - Backend")

STATIC_DIR = os.environ.get("STATIC_DIR", "static")
if not os.path.isdir(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")


def ensure_schema_columns():
    inspector = inspect(engine)
    if "studies" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("studies")}
    additions = []
    if "patient_id" not in existing_columns:
        additions.append("patient_id VARCHAR")
    if "metadata_json" not in existing_columns:
        additions.append("metadata_json TEXT")
    if "ensemble_score" not in existing_columns:
        additions.append("ensemble_score FLOAT")
    if "uncertainty" not in existing_columns:
        additions.append("uncertainty FLOAT")
    if "review_label" not in existing_columns:
        additions.append("review_label VARCHAR")
    if additions:
        sql = f"ALTER TABLE studies ADD COLUMN {', ADD COLUMN '.join(additions)}"
        with engine.begin() as conn:
            conn.execute(text(sql))


@app.on_event("startup")
def on_startup():
    # Create DB tables if they don't exist and add any missing columns.
    models.Base.metadata.create_all(bind=engine)
    ensure_schema_columns()


@app.get("/health")
def health():
    return {"status": "ok"}
