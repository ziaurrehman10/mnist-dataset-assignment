"""
frontend/app.py
Main Streamlit application — MNIST digit classifier interface.
"""
import os
import io

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from dotenv import load_dotenv

load_dotenv()

# Bridge Streamlit Cloud secrets into os.environ so os.getenv() picks them up
if "API_BASE_URL" in st.secrets:
    os.environ["API_BASE_URL"] = st.secrets["API_BASE_URL"]

from utils.api_client import check_health, predict_image, get_model_info  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header banner */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2.4rem; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0; opacity: .7; font-size: 1rem; }

    /* Prediction badge */
    .digit-badge {
        background: #0f3460;
        color: white;
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        border-radius: 16px;
        padding: 1rem;
        border: 3px solid #e94560;
    }
    .confidence-text {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        color: #e94560;
        margin-top: 0.5rem;
    }

    /* Status dots */
    .status-ok  { color: #28a745; font-weight: 600; }
    .status-err { color: #dc3545; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plot_probabilities(probs: list[float]) -> go.Figure:
    digits = list(range(10))
    colors = ["#e94560" if p == max(probs) else "#1a84c4" for p in probs]
    fig = go.Figure(
        go.Bar(
            x=digits,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Class Probabilities",
        xaxis=dict(title="Digit", tickmode="linear", dtick=1),
        yaxis=dict(title="Probability", range=[0, 1.05]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
        height=320,
        margin=dict(t=40, b=20, l=40, r=20),
    )
    return fig


def _pil_from_canvas(canvas_data) -> Image.Image | None:
    """Convert raw canvas RGBA numpy array to greyscale PIL Image."""
    if canvas_data is None or canvas_data.image_data is None:
        return None
    arr = canvas_data.image_data.astype(np.uint8)
    if arr.max() == 0:
        return None
    img = Image.fromarray(arr, mode="RGBA").convert("L")
    return img


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_url = st.text_input("API Base URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))
    os.environ["API_BASE_URL"] = api_url

    st.divider()
    st.markdown("### 🔌 API Status")
    if st.button("Check Connection", use_container_width=True):
        health = check_health()
        if health and health.get("status") == "ok":
            st.markdown('<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
            st.caption(f"Model loaded: {health.get('model_loaded')} · v{health.get('version')}")
        else:
            st.markdown('<span class="status-err">● Unreachable</span>', unsafe_allow_html=True)
            st.caption("Make sure the FastAPI server is running.")

    st.divider()
    st.markdown("### 🧠 Model Info")
    if st.button("Fetch Model Info", use_container_width=True):
        info = get_model_info()
        if info:
            st.json(info)
        else:
            st.error("Could not reach the API.")

    st.divider()
    st.markdown(
        "**MNIST ANN Classifier**\n\nAssignment project — ANN → FastAPI → Streamlit",
        help="Full-stack ML project",
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>✏️ MNIST Digit Classifier</h1>
        <p>Draw a digit or upload an image — powered by your trained ANN</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_draw, tab_upload = st.tabs(["🖊️ Draw a Digit", "📁 Upload Image"])


# ── TAB 1: Draw ───────────────────────────────────────────────────────────────
with tab_draw:
    col_canvas, col_result = st.columns([1, 1], gap="large")

    with col_canvas:
        st.markdown("#### Draw a digit below (0–9)")
        stroke_width = st.slider("Brush size", 8, 30, 18, key="brush")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        predict_btn = st.button("🔍 Classify Digit", use_container_width=True, key="draw_btn")

    with col_result:
        st.markdown("#### Prediction")
        if predict_btn:
            img = _pil_from_canvas(canvas_result)
            if img is None:
                st.warning("Canvas is empty — draw a digit first.")
            else:
                with st.spinner("Sending to API…"):
                    result = predict_image(img)
                if result is None:
                    st.error("Could not reach the API. Is the server running?")
                else:
                    st.markdown(
                        f'<div class="digit-badge">{result["digit"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="confidence-text">Confidence: {result["confidence"]:.1%}</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        _plot_probabilities(result["probabilities"]),
                        use_container_width=True,
                    )
        else:
            st.info("Draw a digit on the canvas, then click **Classify Digit** ______.Note: Your drawing can affect the model's predictions, so make sure to draw clearly.")


# ── TAB 2: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### Upload an image (PNG / JPEG)")
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read()))
            st.image(img, caption="Uploaded image", width=280)
            run_btn = st.button("🔍 Classify Digit", use_container_width=True, key="upload_btn")
        else:
            st.info("Upload a handwritten digit image.")
            run_btn = False

    with col_res:
        st.markdown("#### Prediction")
        if uploaded and run_btn:
            with st.spinner("Sending to API…"):
                result = predict_image(img)
            if result is None:
                st.error("Could not reach the API.")
            else:
                st.markdown(
                    f'<div class="digit-badge">{result["digit"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="confidence-text">Confidence: {result["confidence"]:.1%}</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _plot_probabilities(result["probabilities"]),
                    use_container_width=True,
                )
        else:
            st.info("Upload an image and click **Classify Digit**.")
