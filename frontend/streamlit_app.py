import streamlit as st
import os
import requests
from PIL import Image
from io import BytesIO

# Resolve backend URL: prefer env var, then Streamlit secrets if available
try:
    backend_secret = None
    try:
        backend_secret = st.secrets.get("backend_url")
    except Exception:
        backend_secret = None
    BACKEND_URL = os.environ.get("BACKEND_URL") or backend_secret or "http://localhost:8000"
except Exception:
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.title("Chest X-ray AI Viewer")

uploaded = st.file_uploader("Upload chest X-ray image", type=["dcm", "dicom", "png", "jpg", "jpeg"])
if uploaded is not None:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    resp = requests.post(f"{BACKEND_URL}/api/upload", files=files)
    if resp.status_code == 200:
        data = resp.json()
        st.success(f"Prediction: {data.get('prediction')} (score={data.get('score')})")
        png_url = f"{BACKEND_URL}{data.get('png_url')}"
        heatmap = data.get('heatmap_url')
        if png_url:
            try:
                r = requests.get(png_url)
                img = Image.open(BytesIO(r.content))
                st.image(img, caption="Preview", use_column_width=True)
            except Exception:
                st.write("Could not load preview image.")
        if heatmap:
            try:
                r = requests.get(f"{BACKEND_URL}{heatmap}")
                img = Image.open(BytesIO(r.content))
                st.image(img, caption="Grad-CAM", use_column_width=True)
            except Exception:
                st.write("Could not load heatmap.")
    else:
        st.error(f"Upload failed: {resp.status_code} {resp.text}")
