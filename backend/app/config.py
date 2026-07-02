"""
app/config.py
Central settings loaded from environment / .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "MNIST ANN Classifier"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model
    model_path: str = "D:/New folder/mnist-ann-project/model/model.h5"
    input_shape: int = 784  # 28×28

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["http://localhost:8501"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache()          # singleton — loaded once at startup
def get_settings() -> Settings:
    return Settings()
