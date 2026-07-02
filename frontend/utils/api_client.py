"""
utils/api_client.py
Thin wrapper around the FastAPI backend — handles requests, errors, retries.
"""
from __future__ import annotations

import os
import io
import logging

import requests
from PIL import Image

logger = logging.getLogger(__name__)

_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 15  # seconds


def _url(path: str) -> str:
    return f"{_BASE_URL.rstrip('/')}{path}"


# ── Health ────────────────────────────────────────────────────────────────────

def check_health() -> dict | None:
    try:
        r = requests.get(_url("/health"), timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        return None


# ── Model info ────────────────────────────────────────────────────────────────

def get_model_info() -> dict | None:
    try:
        r = requests.get(_url("/api/v1/model/info"), timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.error("get_model_info failed: %s", exc)
        return None


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_image(pil_image: Image.Image) -> dict | None:
    """Send a PIL Image and return the prediction dict or None on error."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    try:
        r = requests.post(
            _url("/api/v1/predict"),
            files={"file": ("digit.png", buf, "image/png")},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        logger.error("predict_image HTTP error: %s — %s", exc, exc.response.text)
        return None
    except Exception as exc:
        logger.error("predict_image failed: %s", exc)
        return None
