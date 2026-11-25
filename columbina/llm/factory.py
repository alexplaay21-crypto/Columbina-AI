"""Выбор провайдера LLM по настройкам. Остальной код видит только LLMProvider."""

from __future__ import annotations

import re

from ..config import Settings
from .base import LLMProvider

DEFAULT_MODELS = {
    "openrouter": ("anthropic/claude-opus-5.5", "anthropic/claude-haiku-5.5"),
    "anthropic": ("claude-opus-5-5", "claude-haiku-5-5"),
}


def to_openrouter_slug(model: str) -> str:
    """claude-opus-5-5 → anthropic/claude-opus-5.5 (если в .env остались имена прямого Anthropic API)."""
    if "/" in model or not model.startswith("claude-"):
        return model
    m = re.match(r"^(claude-[a-z]+)-(\d+)(?:-(\d+))?$", model)
    if not m:
        return "anthropic/" + model
    name, major, minor = m.groups()
    return f"anthropic/{name}-{major}" + (f".{minor}" if minor else "")


def resolved_models(s: Settings) -> tuple[str, str]:
    main_d, bg_d = DEFAULT_MODELS.get(s.llm_provider, DEFAULT_MODELS["anthropic"])
    main, bg = s.main_model or main_d, s.background_model or bg_d
    if s.llm_provider == "openrouter":
        main, bg = to_openrouter_slug(main), to_openrouter_slug(bg)
    return main, bg


def make_llm(s: Settings) -> LLMProvider:
    main, bg = resolved_models(s)
    if s.llm_provider == "openrouter":
        if not s.openrouter_api_key:
            raise SystemExit("LLM_PROVIDER=openrouter, но OPENROUTER_API_KEY не задан")
        from .openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(s.openrouter_api_key, main, bg, s.main_effort, s.openrouter_base_url,
                                  list(s.main_fallback_models), s.inner_context_placement)
    if s.llm_provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(main, bg, s.main_effort, s.use_refusal_fallbacks)
    raise SystemExit(f"неизвестный LLM_PROVIDER={s.llm_provider}")
