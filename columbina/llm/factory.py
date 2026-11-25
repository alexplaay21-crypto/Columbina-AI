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
            raise SystemExit("не задан ключ модели: OPENROUTER_API_KEY или LLM_API_KEY")
        from .openrouter_provider import OpenRouterProvider

        primary = OpenRouterProvider(s.openrouter_api_key, main, bg, s.main_effort, s.openrouter_base_url,
                                     list(s.main_fallback_models), s.inner_context_placement, timeout=s.llm_timeout)
        if s.background_api_key and s.background_base_url:
            bg_provider = OpenRouterProvider(s.background_api_key, bg, bg, "low", s.background_base_url,
                                             [], s.inner_context_placement, timeout=s.llm_timeout)
            return RoutingProvider(primary, bg_provider)
        return primary
    if s.llm_provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(main, bg, s.main_effort, s.use_refusal_fallbacks)
    raise SystemExit(f"неизвестный LLM_PROVIDER={s.llm_provider}")


class RoutingProvider:
    """Основной ответ — один сервис, фоновые задачи (память, резюме) — другой."""

    def __init__(self, main: LLMProvider, background: LLMProvider):
        self.main, self.background = main, background

    async def complete(self, request):
        return await (self.background if request.tier == "background" else self.main).complete(request)


def make_vision(s: Settings):
    """Зрение для основной модели без зрения: VISION_* из .env или, на OpenRouter, Gemini Flash-Lite тем же ключом."""
    from ..perception.vision import VisionAdapter
    from .capabilities import supports_pdf, supports_vision

    main, _ = resolved_models(s)
    has_vision = supports_vision(main, s.main_vision)
    has_pdf = supports_pdf(main, s.main_pdf)
    describer = None
    if not has_vision:
        from .openrouter_provider import OpenRouterProvider

        if s.vision_api_key and s.vision_base_url and s.vision_model:
            describer = OpenRouterProvider(s.vision_api_key, s.vision_model, s.vision_model, "low", s.vision_base_url,
                                           timeout=s.llm_timeout)
        elif s.llm_provider == "openrouter" and "openrouter.ai" in s.openrouter_base_url and s.openrouter_api_key:
            describer = OpenRouterProvider(s.openrouter_api_key, s.vision_model or "google/gemini-3.1-flash-lite",
                                           s.vision_model or "google/gemini-3.1-flash-lite", "low",
                                           s.openrouter_base_url, timeout=s.llm_timeout)
    return VisionAdapter(has_vision, has_pdf, describer)
