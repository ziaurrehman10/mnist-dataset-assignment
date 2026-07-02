"""
app/main.py
FastAPI application factory and startup / shutdown lifecycle.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import prediction, model
from app.schemas.prediction import HealthResponse
from app.services.model_service import get_model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="REST API for handwritten digit classification using an ANN trained on MNIST.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(prediction.router, prefix=settings.api_prefix)
    app.include_router(model.router,      prefix=settings.api_prefix)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        svc = get_model_service()
        svc.load()

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health():
        svc = get_model_service()
        return {
            "status": "ok",
            "model_loaded": svc.is_loaded(),
            "version": settings.app_version,
        }

    return app


app = create_app()
