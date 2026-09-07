"""Centralized application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Project metadata
    PROJECT_NAME: str = "FinPilot"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Server & API
    API_V1_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # LLM Settings (Phase 2+)
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_MAX_RETRIES: int = 3
    LLM_INITIAL_RETRY_DELAY: float = 0.5
    LLM_BACKOFF_FACTOR: float = 2.0

    # Financial Data Providers (Phase 6+)
    TWELVE_DATA_API_KEY: str = ""
    FINNHUB_API_KEY: str = ""

    # Database & Vector Store (Phase 9+ / Phase 14+)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/finpilot"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
