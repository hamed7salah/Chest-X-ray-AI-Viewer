# Chest X-ray AI Viewer — Design & Implementation Choices

This document explains goals, architecture, and major implementation choices for the Chest X-ray AI Viewer project. It consolidates the rationale for technology choices, deployment decisions, and how the pieces fit together.

## Goal

Build a production-style end-to-end radiology AI workflow that supports DICOM uploads, parsing, visualization, inference, Grad-CAM explainability, API backend, Streamlit frontend, persistence, and Dockerized deployment.

## High-level Architecture

- Frontend: Streamlit app for uploading DICOM and visualizing results.
- Backend: FastAPI providing `/api/upload` endpoint, DICOM parsing, inference orchestration, Grad-CAM generation, and DB persistence.
- AI: Modular predictor supporting a baseline heuristic and an optional PyTorch model.
- Storage: Uploaded DICOMs and generated PNGs stored on disk (mounted volumes in Docker), metadata and predictions stored in a relational DB.
- DB: PostgreSQL in production; development defaults to SQLite unless `DATABASE_URL` is provided.
- Deployment: Docker + docker-compose with separate backend, frontend, and DB services.

## Why these choices (brief)

- FastAPI: modern, async-capable, and fast for building REST services. Clear routing and Pydantic support.
- Streamlit: rapid interactive UI suitable for demos and reviewers; easy to deploy alongside backend.
- PyTorch + torchvision: industry standard for model inference and Grad-CAM integration.
- pydicom + SimpleITK: robust DICOM handling options. pydicom suffices for pixel extraction in many CXR cases.
- PostgreSQL: production-grade relational DB; Docker-compose provides Postgres for integration testing.
- Docker-compose: simple local orchestration mirroring production microservices split.

## Key Implementation Notes

- The repo contains a minimal, production-oriented scaffold in `backend/app` and `frontend`.
- The backend uses SQLAlchemy with a `DATABASE_URL` environment variable. In absence of a Postgres URL, the code uses a local SQLite DB at `./dev.db` for quick development.
- The `Predictor` supports two modes controlled by `PREDICTOR_MODE` env var:
  - `baseline`: deterministic heuristic based on mean pixel intensity. Useful as a reproducible baseline and demo without model checkpoints.
  - `pytorch`: uses a ResNet-18 backbone (ImageNet weights) with a single-output head; if a `MODEL_CHECKPOINT` path is provided, the head weights will be loaded.

Rationale: this hybrid approach avoids shipping fragile or proprietary medical model weights while giving a straightforward hook for replacing with a clinical model later.

## Folder Structure (generated)

- `backend/app/` — FastAPI app, routers, DB models, DICOM utils, AI modules.
- `frontend/` — Streamlit app.
- `docker-compose.yml` — orchestrates `db`, `backend`, and `frontend`.
- `requirements.txt` — pinned dependencies for reproducibility.

## Security & Production Considerations

- Do not expose upload directories directly in production; serve static content from the backend with secure access control.
- Configure CORS to trusted origins only (development currently allows `*`).
- Use TLS and reverse proxy (NGINX) in production; let NGINX handle static files, gzip, buffering, and rate limits.
- Use secrets management for DB credentials and model checkpoints (HashiCorp Vault, AWS Secrets Manager, etc.).
- For inference on GPUs, ensure container images include CUDA-enabled PyTorch and schedule GPU workers separately.

## Running Locally (developer-friendly)

1. Create virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run backend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

3. Run Streamlit frontend:

```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

4. Or use Docker Compose (postgres + backend + frontend):

```bash
docker compose up --build
```

## How to plug a real clinical model

1. Train or obtain a classifier head fine-tuned for pneumonia detection (DenseNet/CheXNet-style) and save `state_dict` matching the ResNet-18 head architecture used in `Predictor`.
2. Mount the checkpoint into the backend container and set `MODEL_CHECKPOINT=/path/in/container/model.pt` and `PREDICTOR_MODE=pytorch`.

## Testing

- Basic pytest scaffold is included. Add unit tests for DICOM edge cases, model integration tests, and API contract tests.

## Scaling for Real Healthcare Workloads

- Use asynchronous task queues (Celery, RabbitMQ, or Redis + RQ) for heavy preprocessing and inference.
- Separate inference into dedicated GPU-backed services behind an internal gRPC or HTTP API.
- Store images and heatmaps in object storage (S3) and store only URLs/metadata in the DB.
- Implement RBAC, audit logging, and data retention policies to meet healthcare compliance (HIPAA, GDPR).

## Next Steps (recommended)

1. Add robust DICOM metadata parsing and validation, patient/study association, and UID handling.
2. Implement a production-ready Grad-CAM flow using `pytorch-grad-cam` once a trained model checkpoint is available.
3. Harden the API with authentication (OAuth2), input sanitization, and request size limits.
4. Add unit and integration tests, CI pipeline, and optionally Helm charts for Kubernetes deployment.

---
This file summarizes major design decisions; further phase-by-phase detailed docs and in-file comments are present in source files.
