"""Unit tests for centralized application settings and environment overrides."""

from app.core.config import Settings, get_settings


def test_default_settings(clean_settings):
    """Verify default settings values when no environment variables are set."""
    settings = Settings()
    assert settings.PROJECT_NAME == "FinPilot"
    assert settings.VERSION == "0.1.0"
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is True
    assert settings.LOG_LEVEL == "INFO"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.DATABASE_URL.startswith("postgresql")
    assert settings.CHROMA_PERSIST_DIRECTORY == "./chroma_data"


def test_settings_environment_override(monkeypatch, clean_settings):
    """Verify settings can be overridden via environment variables."""
    monkeypatch.setenv("PROJECT_NAME", "FinPilotTest")
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LLM_PROVIDER", "gemini-custom")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@db:5432/testdb")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma_test")
    monkeypatch.setenv("CORS_ORIGINS", "http://test1.local, http://test2.local")

    settings = Settings()
    assert settings.PROJECT_NAME == "FinPilotTest"
    assert settings.ENVIRONMENT == "testing"
    assert settings.DEBUG is False
    assert settings.LOG_LEVEL == "DEBUG"
    assert settings.LLM_PROVIDER == "gemini-custom"
    assert settings.GEMINI_API_KEY == "test-key-12345"
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@db:5432/testdb"
    assert settings.CHROMA_PERSIST_DIRECTORY == "/tmp/chroma_test"
    assert settings.CORS_ORIGINS == ["http://test1.local", "http://test2.local"]


def test_get_settings_cache(clean_settings):
    """Verify get_settings returns a singleton cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
