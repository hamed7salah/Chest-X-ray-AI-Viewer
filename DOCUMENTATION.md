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

## Implementation Log — Interview Format

The following is a chronological, interview-ready log of what was implemented, why each change was made, the problems that were fixed, and the alternatives considered.

1) Project scaffold and environment
- What: Added `.gitignore` and `requirements.txt` to establish a reproducible developer environment.
- Files: `.gitignore`, `requirements.txt`
- Why: Provide a consistent base for dependency installation and avoid committing local artifacts. Chose pinned `requirements.txt` for simplicity and reproducibility during the demo.

2) Backend skeleton with FastAPI
- What: Implemented a FastAPI app with CORS and a health endpoint.
- Files: `backend/app/main.py`
- Problem fixed: No existing API; needed a typed API surface for uploads and health checks.
- Why FastAPI: typed validation (Pydantic), async support, and automatic docs are valuable in interviews and production.

3) Upload API and database model
- What: Added `/api/upload` endpoint to accept file uploads and persist study records.
- Files: `backend/app/api/upload.py`, `backend/app/db/models.py`, `backend/app/db/session.py`
- Problem fixed: Needed persistence for uploads and predictions. Initially used `metadata` as a model attribute, which triggered a SQLAlchemy `InvalidRequestError` (reserved name). Renamed it to `metadata_json` to fix the crash.
- Why SQLAlchemy/Postgres: relational DBs map naturally to study metadata and support realistic production scenarios. Postgres chosen in `docker-compose.yml` to mirror a production-like environment.

4) DICOM and normal image handling utilities
- What: Implemented robust DICOM parsing and PNG generation with fallback support for JPEG/PNG images.
- Files: `backend/app/utils/dicom.py`
- Problems fixed:
  - Handled `PhotometricInterpretation==MONOCHROME1` inversion.
  - Normalized pixel intensity and converted to 3-channel RGB for visualization and models.
  - Added `load_image()` to accept regular images (PNG/JPG) or DICOM.
- Why: Many interviewers will test DICOM knowledge — addressing photometric interpretation, normalization, and multi-format support demonstrates practical experience.

5) Frontend (Streamlit) with safe secrets handling
- What: Built a Streamlit demo app to upload images and display predictions and heatmaps.
- Files: `frontend/streamlit_app.py`
- Problems fixed:
  - Streamlit raised a FileNotFoundError when `secrets.toml` was missing. Fixed by falling back to environment variable `BACKEND_URL`.
  - Pointed frontend to `http://backend:8000` within Docker to use service DNS.
- Why Streamlit: fastest way to demonstrate visual results in an interview without building a full frontend stack.

6) AI Predictor design (modular)
- What: Created `Predictor` class with two modes:
  - `baseline` (heuristic) — runs in every container by default.
  - `pytorch` (optional) — lazy-loads `torch`/`torchvision` and a ResNet-18 backbone when enabled and a checkpoint is provided.
- Files: `backend/app/ai/predictor.py`
- Why design: Avoid shipping heavy ML libs by default; allow the project to demonstrate plug-in model capability without including model weights. ResNet-18 chosen for a compact, transferable backbone that is easy to understand in an interview.
- Tradeoffs: DenseNet / CheXNet variants are more common for CXR tasks but are heavier; use them for production model integration later.

7) Explainability (Grad-CAM fallback)
- What: Implemented a fallback Grad-CAM overlay generator for demos; reserved full `pytorch-grad-cam` integration for when a trained model is available.
- Files: `backend/app/ai/predictor.py`
- Problems fixed: Initially the heatmap over-wrote the preview PNG. Changed outputs so preview is `uid.png` and heatmap is `uid.heat.png`.
- Why: Demonstrates explainability pipeline without depending on a heavy model.

8) Dockerization and compose orchestration
- What: Added `Dockerfile` for backend & frontend and `docker-compose.yml` to orchestrate Postgres, backend, and frontend.
- Files: `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`
- Problems fixed:
  - Initial Dockerfile used incorrect relative COPY paths which failed during build. Updated build contexts to use repository root and point Dockerfiles to subfolders.
  - Some packages caused pip resolution issues in the demo environment (pytorch-related packages). Kept heavy ML packages optional and lazy-loaded them in code.
- Why: Docker Compose demonstrates a realistic multi-service environment, useful for interviewing ops and deployment questions.

9) Runtime fixes discovered during testing
- Name resolution error: frontend could not resolve `backend` because backend crashed on model startup. Fixed by renaming the SQLAlchemy `metadata` field to `metadata_json`.
- Streamlit secrets warning: fixed via environment fallback.
- Static file serving: mounted and mounted directory via `FastAPI StaticFiles` so preview and heatmap images are reachable at `/static/`.

10) Support for normal images and heuristic improvement
- What: Added `image_to_numpy()` and `load_image()` to support PNG/JPG; improved the baseline heuristic to better detect pneumonia opacity.
- Files: `backend/app/utils/dicom.py`, `backend/app/ai/predictor.py`
- Why: Interviewers may not have DICOM files; accepting normal images makes demo easier and shows practical flexibility. Heuristic improved with CLAHE, local variance, lower-lung density, and edge density features rather than a single global intensity metric.

11) Testing & Validation performed
- Actions performed manually during development:
  - `python -m compileall` to verify Python syntax across modified modules.
  - `docker compose up --build` to build services and verify runtime logs.
  - Upload tests via the Streamlit UI and `curl -F` for API contract checks.

12) Documentation added and rationale
- Files: `DOCUMENTATION.md` (this file), `README.md`
- What: Documented architecture choices, tradeoffs, run commands, and next steps for production hardening.

13) Known limitations and production path
- Limitations today:
  - Baseline heuristic is not a substitute for a validated clinical model.
  - No authentication, audit logging, or PHI governance yet.
  - Grad-CAM is demo-level until integrated with a trained model's activations.
- Production path:
  - Integrate a validated transfer-learned CXR model (DenseNet/CheXNet), use `pytorch-grad-cam` for real activation maps, add GPU-backed inference workers, use S3 for images, and add RBAC/audit for compliance.

14) Why these choices (short interview bullets)
- FastAPI: typed APIs and async; widely adopted in modern ML infra.
- Streamlit: rapid visual demo, minimal frontend maintenance.
- PyTorch (optional): research standard for medical imaging and easy grad-cam tooling.
- ResNet-18 as demo backbone: small/fast/easy to load; switch to DenseNet for production performance.

## Appendix — Quick file map of important changes
- `backend/app/main.py`: mounted `/static` and enabled app startup table creation.
- `backend/app/api/upload.py`: accepted multipart uploads, saved file to `uploads/`, used `load_image()`, saved `static/{uid}.png` and produced heatmap `static/{uid}.heat.png`.
- `backend/app/utils/dicom.py`: added `load_image()` and robust normalization.
- `backend/app/ai/predictor.py`: modular predictor with `baseline` heuristic and optional `pytorch` mode (lazy import), improved heuristic.
- `backend/app/db/models.py`: renamed `metadata` → `metadata_json` to avoid SQLAlchemy reserved name conflict.
- `frontend/streamlit_app.py`: updated uploader to accept `png/jpg/jpeg` and use `BACKEND_URL` env var.
- `docker-compose.yml`: configured services, environment variables, and volumes for `uploads` and `static`.

---
This implementation log is written to be presented in an interview for a radiology AI team. If you want, I will commit this updated `DOCUMENTATION.md` and also add a one-page `INTERVIEW.md` that condenses talking points and demo steps into a 2-page handout.
