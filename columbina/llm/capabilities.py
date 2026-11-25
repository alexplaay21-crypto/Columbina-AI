"""Что умеет модель: видит ли картинки, читает ли PDF. Эвристика по имени + ручная настройка в .env."""

from __future__ import annotations

import re

VISION = re.compile(
    r"claude|gemini|gpt-4o|gpt-4\.1|gpt-5|o3|o4|vision|[-/]vl\b|-vl-|llama-4|pixtral|grok-[2-9]|"
    r"mistral-(?:medium|small-3)|nemotron.*vl|gemma-[34]|qwen.*vl|qwen3\.[5-9]-?omni|omni|kimi-vl|muse",
    re.IGNORECASE,
)
PDF = re.compile(r"claude|gemini", re.IGNORECASE)


def supports_vision(model: str, setting: str = "auto") -> bool:
    if setting in ("on", "true", "1", "yes"):
        return True
    if setting in ("off", "false", "0", "no"):
        return False
    return bool(VISION.search(model or ""))


def supports_pdf(model: str, setting: str = "auto") -> bool:
    if setting in ("on", "true", "1", "yes"):
        return True
    if setting in ("off", "false", "0", "no"):
        return False
    return bool(PDF.search(model or ""))
