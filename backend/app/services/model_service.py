"""
app/services/model_service.py
Loads the Keras model once and exposes predict helpers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from app.config import get_settings

logger = logging.getLogger(__name__)


class ModelService:
    """Singleton wrapper around the saved Keras ANN model."""

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._settings = get_settings()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        path = Path(self._settings.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at '{path.resolve()}'. "
                "Place your model.h5 file there and restart."
            )
        logger.info("Loading model from %s …", path.resolve())
        self._model = tf.keras.models.load_model(str(path))
        logger.info("Model loaded — input shape: %s", self._model.input_shape)

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, pixel_array: np.ndarray) -> dict:
        """
        Accept a (784,) or (28, 28) float32 array (values 0–1).
        Returns {"digit": int, "confidence": float, "probabilities": list[float]}
        """
        self._ensure_loaded()
        x = self._preprocess(pixel_array)               # shape (1, 784)
        probs: np.ndarray = self._model.predict(x, verbose=0)[0]  # (10,)
        digit = int(np.argmax(probs))
        return {
            "digit": digit,
            "confidence": float(probs[digit]),
            "probabilities": probs.tolist(),
        }

    def predict_batch(self, arrays: list[np.ndarray]) -> list[dict]:
        self._ensure_loaded()
        batch = np.stack([self._preprocess(a)[0] for a in arrays])  # (N, 784)
        probs_batch: np.ndarray = self._model.predict(batch, verbose=0)
        results = []
        for probs in probs_batch:
            digit = int(np.argmax(probs))
            results.append({
                "digit": digit,
                "confidence": float(probs[digit]),
                "probabilities": probs.tolist(),
            })
        return results

    # ── Model metadata ───────────────────────────────────────────────────────

    def info(self) -> dict:
        self._ensure_loaded()
        return {
            "input_shape": list(self._model.input_shape),
            "output_shape": list(self._model.output_shape),
            "total_params": self._model.count_params(),
            "layers": [
                {"name": l.name, "type": type(l).__name__, "trainable": l.trainable}
                for l in self._model.layers
            ],
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _preprocess(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = arr.flatten()
        if len(arr) != self._settings.input_shape:
            raise ValueError(
                f"Expected {self._settings.input_shape} pixels, got {len(arr)}."
            )
        return arr.reshape(1, -1)

    def _ensure_loaded(self) -> None:
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() first.")


# Module-level singleton
_model_service = ModelService()


def get_model_service() -> ModelService:
    return _model_service
