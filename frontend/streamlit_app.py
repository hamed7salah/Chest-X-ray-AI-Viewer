import streamlit as st
import os
import requests
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
import numpy as np

st.set_page_config(page_title="Chest X-ray AI Viewer", layout="wide")

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

st.sidebar.header("Viewer Controls")
window_center = st.sidebar.slider("Window Center", min_value=0, max_value=255, value=128)
window_width = st.sidebar.slider("Window Width", min_value=1, max_value=255, value=128)
show_prior = st.sidebar.checkbox("Show prior study comparison", value=True)


def build_plotly_image(image: Image.Image, title: str):
    img_arr = np.asarray(image.convert("RGB"))
    fig = go.Figure(go.Image(z=img_arr))
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def apply_window_level(image: Image.Image, center: int, width: int) -> Image.Image:
    img = np.asarray(image.convert("L")).astype(np.float32)
    low = center - width / 2
    high = center + width / 2
    img = np.clip((img - low) / (high - low), 0, 1) * 255
    return Image.fromarray(img.astype('uint8')).convert("RGB")


def fetch_prior_studies():
    try:
        resp = requests.get(f"{BACKEND_URL}/api/studies")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return []
    return []


prior_studies = fetch_prior_studies() if show_prior else []
prior_options = ["None"] + [f"{item['id']}: {item.get('patient_id','unknown')} ({item['prediction']})" for item in prior_studies]
selected_prior = st.sidebar.selectbox("Select prior study", prior_options)


uploaded = st.file_uploader("Upload chest X-ray image", type=["dcm", "dicom", "png", "jpg", "jpeg"])
current_study = None
if uploaded is not None:
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    resp = requests.post(f"{BACKEND_URL}/api/upload", files=files)
    if resp.status_code == 200:
        current_study = resp.json()
        st.success(f"Prediction: {current_study.get('prediction')} (score={current_study.get('score'):.2f})")
        if current_study.get('ensemble_score') is not None:
            st.info(f"Ensemble: {current_study.get('ensemble_score'):.2f}, Uncertainty: {current_study.get('uncertainty'):.2f}")
    else:
        st.error(f"Upload failed: {resp.status_code} {resp.text}")


if current_study is not None:
    preview_url = f"{BACKEND_URL}{current_study.get('png_url')}"
    heatmap_url = f"{BACKEND_URL}{current_study.get('heatmap_url')}" if current_study.get('heatmap_url') else None

    try:
        r = requests.get(preview_url)
        preview_img = Image.open(BytesIO(r.content))
        preview_img = apply_window_level(preview_img, window_center, window_width)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(build_plotly_image(preview_img, "Current Study"), use_container_width=True)
        with col2:
            if heatmap_url:
                r2 = requests.get(heatmap_url)
                heatmap_img = Image.open(BytesIO(r2.content))
                st.plotly_chart(build_plotly_image(heatmap_img, "Grad-CAM Heatmap"), use_container_width=True)
            else:
                st.write("No Grad-CAM available yet.")
    except Exception:
        st.write("Could not load current preview.")

    st.subheader("Model Feedback")
    feedback = st.radio("Correct prediction if needed", ["No change", "normal", "pneumonia"])
    comment = st.text_area("Review comments (optional)")
    if st.button("Submit feedback"):
        if feedback != "No change":
            payload = {"review_label": feedback, "comment": comment}
            fb_resp = requests.post(f"{BACKEND_URL}/api/studies/{current_study['id']}/feedback", json=payload)
            if fb_resp.status_code == 200:
                st.success("Feedback submitted.")
            else:
                st.error(f"Feedback failed: {fb_resp.status_code} {fb_resp.text}")
        else:
            st.info("No feedback submitted.")

if show_prior and selected_prior != "None":
    prior_id = int(selected_prior.split(":", 1)[0])
    prior = next((item for item in prior_studies if item['id'] == prior_id), None)
    if prior:
        st.subheader("Prior Study Comparison")
        prior_preview_url = f"{BACKEND_URL}{prior.get('png_url')}"
        prior_heatmap_url = f"{BACKEND_URL}{prior.get('heatmap_url')}" if prior.get('heatmap_url') else None
        try:
            r = requests.get(prior_preview_url)
            prior_img = Image.open(BytesIO(r.content))
            prior_img = apply_window_level(prior_img, window_center, window_width)
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.plotly_chart(build_plotly_image(prior_img, "Prior Study"), use_container_width=True)
            with pcol2:
                if prior_heatmap_url:
                    r2 = requests.get(prior_heatmap_url)
                    prior_heat = Image.open(BytesIO(r2.content))
                    st.plotly_chart(build_plotly_image(prior_heat, "Prior Heatmap"), use_container_width=True)
                else:
                    st.write("No prior heatmap available.")
        except Exception:
            st.write("Could not load prior study images.")
