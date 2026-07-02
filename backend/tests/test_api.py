"""
tests/test_api.py
Fast unit tests — model is mocked so no GPU / .h5 file required.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Patch the model service BEFORE importing the app
mock_svc = MagicMock()
mock_svc.is_loaded.return_value = True
mock_svc.predict.return_value = {
    "digit": 3,
    "confidence": 0.97,
    "probabilities": [0.0] * 10,
}
mock_svc.info.return_value = {
    "input_shape": [None, 784],
    "output_shape": [None, 10],
    "total_params": 50890,
    "layers": [],
}

with patch("app.services.model_service._model_service", mock_svc):
    from app.main import app  # noqa: E402

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ── Model info ────────────────────────────────────────────────────────────────

def test_model_info():
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    assert "total_params" in r.json()


# ── Prediction ────────────────────────────────────────────────────────────────

def _make_png_bytes() -> bytes:
    """Create a tiny 28×28 white PNG in memory."""
    from PIL import Image
    import io
    img = Image.fromarray(np.full((28, 28), 255, dtype=np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_success():
    png = _make_png_bytes()
    r = client.post(
        "/api/v1/predict",
        files={"file": ("digit.png", png, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "digit" in body
    assert "confidence" in body
    assert len(body["probabilities"]) == 10


def test_predict_wrong_content_type():
    r = client.post(
        "/api/v1/predict",
        files={"file": ("model.h5", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 415
