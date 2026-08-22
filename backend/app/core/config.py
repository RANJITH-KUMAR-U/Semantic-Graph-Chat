"""
Centralized application configuration.

All settings are loaded from environment variables (or a .env file).
Never scatter os.getenv() calls throughout the codebase — add new
settings here instead.

Usage:
    from app.core.config import settings
    print(settings.router_model)
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    # ── LLM via OpenRouter, Gemini & xAI ─────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_api_key: str = ""
    xai_api_key: str = ""

    # Fast/cheap model for routing classification
    router_model: str = "nvidia/nemotron-3.5-lightning:free"
    # Second router model for the playground toggle
    router_model_alt: str = "google/gemma-4-31b-it:free"
    # High-quality model for response generation
    generator_model: str = "nvidia/nemotron-3.5-lightning:free"
    # Embedding model for semantic retrieval
    embedding_model: str = "liquid/lfm-2.5-embedding-350m:free"

    # ── Summarizer ──────────────────────────────────────────────────────
    # Refresh global_summary every N turns (PRD section 3.3 step 7)
    summarizer_every_n_turns: int = 5

    # In-node bounded memory: max live messages per node before archiving
    # (10 = 5 user+assistant turn pairs; older messages are compressed into local_summary)
    node_keep_last_n: int = 10

    # ── Database (optional for MVP) ─────────────────────────────────────
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/semantic_graph"
    redis_url: str = "redis://localhost:6379/0"

    # Use in-memory LangGraph saver when True (no Postgres needed)
    use_memory_saver: bool = True

    # ── App metadata ────────────────────────────────────────────────────
    app_title: str = "Semantic Graph Chat API"
    app_version: str = "0.1.0"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton — import this everywhere
settings = Settings()
