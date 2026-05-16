import os
from flask import Flask, render_template, request, Response, send_from_directory
import requests

app = Flask(__name__, static_folder="static", template_folder="templates")
BACKEND_INTERNAL_URL = os.environ.get("BACKEND_INTERNAL_URL", "http://backend:8000")
BACKEND_STATIC_DIR = os.environ.get("BACKEND_STATIC_DIR", "/data/backend_static")


@app.route("/")
def index():
    return render_template("index.html")


def proxy_request(method, url, **kwargs):
    resp = requests.request(method, url, **kwargs)
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    ]
    headers = [(name, value) for name, value in resp.headers.items() if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, headers)


@app.route("/api/upload", methods=["POST"])
def upload_proxy():
    files = {name: (file.filename, file.stream, file.content_type) for name, file in request.files.items()}
    data = request.form.to_dict(flat=True)
    url = f"{BACKEND_INTERNAL_URL}/api/upload"
    return proxy_request("POST", url, files=files, data=data)


@app.route("/api/studies", methods=["GET"])
def list_studies_proxy():
    url = f"{BACKEND_INTERNAL_URL}/api/studies"
    return proxy_request("GET", url, params=request.args)


@app.route("/api/studies/<int:study_id>/feedback", methods=["POST"])
def feedback_proxy(study_id):
    url = f"{BACKEND_INTERNAL_URL}/api/studies/{study_id}/feedback"
    return proxy_request("POST", url, json=request.get_json())


@app.route("/backend-static/<path:filename>")
def backend_static(filename):
    return send_from_directory(BACKEND_STATIC_DIR, filename)
