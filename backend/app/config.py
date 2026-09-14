"""
Central configuration for ML Engineer Academy.

Everything is environment driven (12-factor). The application must start and
be fully usable with *no* AI credentials configured -- in that case AI-backed
features degrade to the deterministic offline engines (see app/ai/fallback.py)
and surface a clear message in the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return default if val is None else val


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)).strip())
    except ValueError:
        return default


def _load_dotenv() -> None:
    """Tiny .env loader (no external dependency, no secret logging)."""
    candidates = [
        Path.cwd() / ".env",
        _repo_root() / ".env",
        _repo_root() / "backend" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue
        break


_load_dotenv()


@dataclass
class Settings:
    # ------------------------------------------------------------------ paths
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(_repo_root() / "data"))))

    # ----------------------------------------------------------------- server
    app_name: str = "ML Engineer Academy"
    app_version: str = "1.0.0"
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))
    serve_frontend_build: bool = field(default_factory=lambda: _env_bool("SERVE_FRONTEND_BUILD", True))

    # ------------------------------------------------------------------- data
    db_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))
    alembic_url: str = field(default_factory=lambda: _env("ALEMBIC_DATABASE_URL", ""))

    # --------------------------------------------------------- vector storage
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "local").lower())
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "mlea_kb"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY", ""))
    chroma_collection: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "mlea_kb"))

    # ------------------------------------------------------------- embedding
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "hashing").lower())
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 512))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "mlea-hashing-v1"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # ---------------------------------------------------------------- chunking
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 1600))
    chunk_min_chars: int = field(default_factory=lambda: _env_int("CHUNK_MIN_CHARS", 320))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 220))

    # ---------------------------------------------------------------- AI layer
    ai_provider: str = field(default_factory=lambda: _env("AI_PROVIDER", "auto").lower())
    ai_timeout_seconds: int = field(default_factory=lambda: _env_int("AI_TIMEOUT_SECONDS", 90))
    ai_max_retries: int = field(default_factory=lambda: _env_int("AI_MAX_RETRIES", 2))

    yandex_api_key: str = field(default_factory=lambda: _env("YANDEX_API_KEY", ""))
    yandex_folder_id: str = field(default_factory=lambda: _env("YANDEX_FOLDER_ID", ""))
    yandex_api_url: str = field(
        default_factory=lambda: _env("YANDEX_API_URL", "https://llm.api.cloud.yandex.net/foundationModels/v1")
    )
    # "cloud"    -> Yandex Cloud Foundation Models API (Api-Key + x-folder-id)
    # "aistudio" -> Yandex AI Studio  (Authorization: Api-Key, OpenAI-ish payloads)
    yandex_api_flavour: str = field(default_factory=lambda: _env("YANDEX_API_FLAVOUR", "cloud").lower())
    yandex_model: str = field(default_factory=lambda: _env("YANDEX_MODEL", "yandexgpt-lite"))
    yandex_model_version: str = field(default_factory=lambda: _env("YANDEX_MODEL_VERSION", "latest"))
    yandex_embedding_model: str = field(
        default_factory=lambda: _env("YANDEX_EMBEDDING_MODEL", "text-embedding-multilang-01")
    )
    yandex_temperature: float = field(default_factory=lambda: _env_float("YANDEX_TEMPERATURE", 0.3))
    yandex_max_tokens: int = field(default_factory=lambda: _env_int("YANDEX_MAX_TOKENS", 1200))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))

    # ------------------------------------------------------- cost control/lim
    max_ai_requests_per_day: int = field(default_factory=lambda: _env_int("MAX_AI_REQUESTS_PER_DAY", 80))
    max_ai_tokens_per_day: int = field(default_factory=lambda: _env_int("MAX_AI_TOKENS_PER_DAY", 120_000))
    max_tokens_per_request: int = field(default_factory=lambda: _env_int("MAX_TOKENS_PER_REQUEST", 1600))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 9000))
    monthly_budget: float = field(default_factory=lambda: _env_float("MONTHLY_AI_BUDGET", 0.0))
    cost_per_1k_input_tokens: float = field(
        default_factory=lambda: _env_float("COST_PER_1K_INPUT_TOKENS", 0.25)
    )
    cost_per_1k_output_tokens: float = field(
        default_factory=lambda: _env_float("COST_PER_1K_OUTPUT_TOKENS", 1.20)
    )
    cost_currency: str = field(default_factory=lambda: _env("COST_CURRENCY", "RUB"))
    ai_cache_enabled: bool = field(default_factory=lambda: _env_bool("AI_CACHE_ENABLED", True))
    ai_cache_ttl_days: int = field(default_factory=lambda: _env_int("AI_CACHE_TTL_DAYS", 90))

    # ------------------------------------------------------------------- code
    code_execution_enabled: bool = field(default_factory=lambda: _env_bool("CODE_EXECUTION_ENABLED", True))
    code_execution_timeout_seconds: int = field(default_factory=lambda: _env_int("CODE_EXECUTION_TIMEOUT_SECONDS", 12))
    code_execution_memory_mb: int = field(default_factory=lambda: _env_int("CODE_EXECUTION_MEMORY_MB", 1024))
    code_execution_max_output_chars: int = field(default_factory=lambda: _env_int("CODE_EXECUTION_MAX_OUTPUT_CHARS", 20000))

    # ------------------------------------------------------------------ web
    web_ingestion_enabled: bool = field(default_factory=lambda: _env_bool("WEB_INGESTION_ENABLED", True))
    web_fetch_timeout_seconds: int = field(default_factory=lambda: _env_int("WEB_FETCH_TIMEOUT_SECONDS", 25))
    web_allow_private_hosts: bool = field(default_factory=lambda: _env_bool("WEB_ALLOW_PRIVATE_HOSTS", False))
    user_agent: str = field(default_factory=lambda: _env("USER_AGENT", "MLEngineerAcademy/1.0 (+personal study tool)"))

    # ---------------------------------------------------------------- search
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 8))
    retrieval_min_score: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SCORE", 0.08))
    lexical_weight: float = field(default_factory=lambda: _env_float("LEXICAL_WEIGHT", 0.55))
    vector_weight: float = field(default_factory=lambda: _env_float("VECTOR_WEIGHT", 0.45))

    # ------------------------------------------------------------ learning
    default_daily_minutes: int = field(default_factory=lambda: _env_int("DEFAULT_DAILY_MINUTES", 60))
    diagnostic_question_count: int = field(default_factory=lambda: _env_int("DIAGNOSTIC_QUESTION_COUNT", 28))
    auto_generate_questions: bool = field(default_factory=lambda: _env_bool("AUTO_GENERATE_QUESTIONS", True))

    # ------------------------------------------------------------ misc / dev
    seed_on_startup: bool = field(default_factory=lambda: _env_bool("SEED_ON_STARTUP", True))
    allow_anonymous_default_user: bool = field(default_factory=lambda: _env_bool("ALLOW_ANONYMOUS_DEFAULT_USER", True))
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 150))
    ingest_in_background: bool = field(default_factory=lambda: _env_bool("INGEST_IN_BACKGROUND", True))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    # ------------------------------------------------------------ derived
    def model_uri(self) -> str:
        if self.yandex_api_flavour == "aistudio":
            return self.yandex_model
        return f"gpt://{self.yandex_folder_id}/{self.yandex_model}/{self.yandex_model_version}"

    def embedding_uri(self) -> str:
        return f"embedding://{self.yandex_folder_id}/{self.yandex_embedding_model}/latest"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sandbox_dir(self) -> Path:
        return self.data_dir / "sandbox"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def resolved_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        return f"sqlite:///{self.data_dir / 'academy.db'}"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.sandbox_dir, self.vector_dir):
            path.mkdir(parents=True, exist_ok=True)

    def has_yandex_credentials(self) -> bool:
        return bool(self.yandex_api_key and (self.yandex_folder_id or self.yandex_api_flavour == "aistudio"))

    def ai_available(self) -> bool:
        provider = self.ai_provider
        if provider in {"", "none", "off", "disabled"}:
            return False
        if provider in {"auto", "yandex"}:
            if self.has_yandex_credentials():
                return True
            if provider == "yandex":
                return False
        if provider in {"auto", "openai"} and self.openai_api_key:
            return True
        if provider in {"auto", "ollama"} and self.ollama_base_url and provider == "ollama":
            return True
        return provider not in {"auto", "none"}

    def to_public_dict(self) -> dict:
        """Settings safe to expose to the frontend. Never includes secrets."""
        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "ai_provider": self.selected_provider_name(),
            "ai_available": self.ai_available(),
            "ai_configured": self.ai_available(),
            "embedding_provider": self.embedding_provider,
            "embedding_dim": self.embedding_dim,
            "vector_backend": self.vector_backend,
            "web_ingestion_enabled": self.web_ingestion_enabled,
            "code_execution_enabled": self.code_execution_enabled,
            "max_ai_requests_per_day": self.max_ai_requests_per_day,
            "max_tokens_per_request": self.max_tokens_per_request,
            "cost_currency": self.cost_currency,
            "cost_per_1k_input_tokens": self.cost_per_1k_input_tokens,
            "cost_per_1k_output_tokens": self.cost_per_1k_output_tokens,
            "monthly_budget": self.monthly_budget,
            "default_daily_minutes": self.default_daily_minutes,
            "yandex_model": self.yandex_model,
            "yandex_api_key_present": bool(self.yandex_api_key),
            "yandex_folder_id_present": bool(self.yandex_folder_id),
            "openai_api_key_present": bool(self.openai_api_key),
            "yandex_folder_id_present": bool(self.yandex_folder_id),
            "yandex_api_key_present": bool(self.yandex_api_key),
        }

    def selected_provider_name(self) -> str:
        if self.ai_provider != "auto":
            return self.ai_provider
        if self.has_yandex_credentials():
            return "yandex"
        if self.openai_api_key:
            return "openai"
        return "none"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
