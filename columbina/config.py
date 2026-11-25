"""Настройки из переменных окружения (.env поддерживается через python-dotenv, если установлен)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # необязательная зависимость
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name).lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on", "да"}


def _env_ids(name: str) -> frozenset[int]:
    raw = _env(name)
    return frozenset(int(x) for x in raw.replace(";", ",").split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # Telegram
    telegram_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    allowed_user_ids: frozenset[int] = field(default_factory=lambda: _env_ids("ALLOWED_USER_IDS"))
    admin_user_ids: frozenset[int] = field(default_factory=lambda: _env_ids("ADMIN_USER_IDS"))
    mood_command: str = field(default_factory=lambda: _env("MOOD_COMMAND", "all").lower())  # all | admin
    debounce_seconds: float = field(default_factory=lambda: _env_float("DEBOUNCE_SECONDS", 2.5))

    # LLM: "openrouter" | "anthropic" (по умолчанию openrouter, если задан OPENROUTER_API_KEY)
    # LLM_API_KEY / LLM_BASE_URL — то же самое для любого OpenAI-совместимого API (Gemini, Groq, Mistral…)
    llm_provider: str = field(default_factory=lambda: _env(
        "LLM_PROVIDER", "openrouter" if (_env("OPENROUTER_API_KEY") or _env("LLM_API_KEY")) else "anthropic").lower())
    openrouter_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY") or _env("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL") or _env(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter_embedding_model: str = field(default_factory=lambda: _env("OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3"))
    main_fallback_models: tuple[str, ...] = field(default_factory=lambda: tuple(
        m.strip() for m in _env("MAIN_FALLBACK_MODELS").split(",") if m.strip()))
    # user — «внутренний контекст» блоком в конце сообщения (кэш истории работает на любой модели);
    # system — отдельным system-сообщением (так делает прямой Anthropic API)
    inner_context_placement: str = field(default_factory=lambda: _env("INNER_CONTEXT_PLACEMENT", "user"))
    # Пусто — по провайдеру: anthropic/claude-opus-5.5 и anthropic/claude-haiku-5.5 (OpenRouter)
    # или claude-opus-5-5 и claude-haiku-5-5 (Anthropic)
    main_model: str = field(default_factory=lambda: _env("MAIN_MODEL"))
    background_model: str = field(default_factory=lambda: _env("BACKGROUND_MODEL"))
    main_effort: str = field(default_factory=lambda: _env("MAIN_EFFORT", "low"))
    main_max_tokens: int = field(default_factory=lambda: _env_int("MAIN_MAX_TOKENS", 2500))
    use_refusal_fallbacks: bool = field(default_factory=lambda: _env_bool("USE_REFUSAL_FALLBACKS", True))

    # Embeddings: "voyage" | "local"
    # Эмбеддинги: "openrouter" (по умолчанию при ключе OpenRouter) | "voyage" | "local"
    embedding_provider: str = field(default_factory=lambda: _env(
        "EMBEDDING_PROVIDER", "openrouter" if _env("OPENROUTER_API_KEY") else "local").lower())
    voyage_api_key: str = field(default_factory=lambda: _env("VOYAGE_API_KEY"))
    voyage_model: str = field(default_factory=lambda: _env("VOYAGE_MODEL", "voyage-3.5-lite"))

    # Speech-to-text: "openrouter" (аудио-модель через тот же ключ) | "whisper" (OpenAI/Groq-совместимый endpoint)
    stt_provider: str = field(default_factory=lambda: _env(
        "STT_PROVIDER", "openrouter" if _env("OPENROUTER_API_KEY") and not _env("STT_API_KEY") else "whisper").lower())
    openrouter_stt_model: str = field(default_factory=lambda: _env("OPENROUTER_STT_MODEL", "google/gemini-3.1-flash-lite"))
    stt_base_url: str = field(default_factory=lambda: _env("STT_BASE_URL", "https://api.openai.com/v1"))
    stt_api_key: str = field(default_factory=lambda: _env("STT_API_KEY"))
    stt_model: str = field(default_factory=lambda: _env("STT_MODEL", "gpt-4o-mini-transcribe"))

    # Хранилище: DATABASE_URL (postgresql://...) важнее DB_PATH (SQLite)
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL"))
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(ROOT / "var" / "columbina.sqlite3"))))
    persona_dir: Path = field(default_factory=lambda: Path(_env("PERSONA_DIR", str(ROOT / "data" / "persona"))))
    media_tmp_dir: Path = field(default_factory=lambda: Path(_env("MEDIA_TMP_DIR", str(ROOT / "var" / "tmp"))))

    # Контекст и память
    history_keep_messages: int = field(default_factory=lambda: _env_int("HISTORY_KEEP_MESSAGES", 24))
    history_summarize_at: int = field(default_factory=lambda: _env_int("HISTORY_SUMMARIZE_AT", 40))
    memory_top_k: int = field(default_factory=lambda: _env_int("MEMORY_TOP_K", 5))
    context_budget_tokens: int = field(default_factory=lambda: _env_int("CONTEXT_BUDGET_TOKENS", 1400))

    # Инициатива
    initiative_tick_seconds: int = field(default_factory=lambda: _env_int("INITIATIVE_TICK_SECONDS", 600))
    default_timezone: str = field(default_factory=lambda: _env("DEFAULT_TIMEZONE", "Europe/Moscow"))

    # Стоимость
    daily_budget_usd: float = field(default_factory=lambda: _env_float("DAILY_BUDGET_USD_PER_USER", 3.0))

    # Ссылки
    link_fetch_enabled: bool = field(default_factory=lambda: _env_bool("LINK_FETCH_ENABLED", True))
    link_ca_bundle: str = field(default_factory=lambda: _env("LINK_CA_BUNDLE"))  # пусто — системные сертификаты


def storage_target(s: "Settings") -> str:
    return s.database_url or str(s.db_path)


def load_settings() -> Settings:
    return Settings()
