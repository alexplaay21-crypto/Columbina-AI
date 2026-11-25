from __future__ import annotations

from columbina.cognition.introspection import admin_error_note, bar, health_check, mood_report
from columbina.llm.base import LLMError
from columbina.perception.model import Percept


def t(x):
    return [Percept("text", True, "text", "", text=x, history=x)]


def test_bar():
    assert bar(0.0) == "▱" * 10 and bar(1.0) == "▰" * 10 and bar(0.34) == "▰▰▰" + "▱" * 7


async def test_mood_report_shows_state_and_relationship(engine, store):
    await engine.respond(1, 1, t("Да отстань ты, бесишь"), display_name="Алекс")
    text = await mood_report(engine, 1)
    assert "Коломбина сейчас" in text and "Эмоции" in text
    assert "обида" in text  # грубость задела
    assert "Стадия 1/5 — знакомство" in text and "доверие" in text
    assert "− грубость" in text  # событие видно в истории отношений
    assert "До «начальное доверие»" in text
    assert await mood_report(engine, 999) == "Мы ещё не разговаривали."


async def test_health_check_reports_failures(engine, fake_llm):
    ok = await health_check(engine)
    assert "✅ модель main" in ok and "✅ эмбеддинги" in ok and "✅ база" in ok

    def boom(req):
        raise LLMError("openrouter 402: Insufficient credits (на балансе OpenRouter нет кредитов)", retryable=False)

    fake_llm.script = boom
    bad = await health_check(engine)
    assert "❌ модель main: openrouter 402" in bad


async def test_admin_error_note(engine, fake_llm):
    def boom(req):
        raise LLMError("openrouter 401: no auth (неверный OPENROUTER_API_KEY)", retryable=False)

    fake_llm.script = boom
    reply = await engine.respond(1, 1, t("привет"))
    note = admin_error_note(reply.debug)
    assert note and "OPENROUTER_API_KEY" in note and "/check" in note
    assert admin_error_note({}) is None
