from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import upload
from .db.session import engine
from .db import models

app = FastAPI(title="Chest X-ray AI Viewer - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    # Create DB tables if they don't exist
    models.Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}
