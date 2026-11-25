"""Достоверность персонажа: что попадает в контекст, а что нет."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from columbina.cognition.context_builder import estimate_tokens
from columbina.perception.model import Percept

from .conftest import last_inner_context


def text(t: str) -> list[Percept]:
    return [Percept("text", True, "text", "", text=t, history=t)]


def test_core_prompt_is_compact(persona):
    # Ядро — компактно; полное досье в запрос не идёт.
    assert estimate_tokens(persona.core_prompt) < 3200
    full = (persona.root / "source" / "dossier_main.md").read_text(encoding="utf-8")
    assert len(full) > 3 * len(persona.core_prompt)


def test_core_prompt_keeps_key_traits(persona):
    p = persona.core_prompt
    for must in ["Kuutar", "Предвестник", "Не выдумывай", "не уходи", "программ", "<react>", "leave_note",
                 "медленно", "по имени", "реакция"]:
        assert must in p, must


def test_knowledge_stages_valid(persona):
    assert len(persona.knowledge) >= 30
    assert all(1 <= c.min_stage <= 5 for c in persona.knowledge)
    # Самое личное — только близкому другу
    by_id = {c.id: c for c in persona.knowledge}
    assert by_id["bio_name"].min_stage == 5
    assert by_id["bio_court"].min_stage >= 4


def test_abilities_have_three_layers(persona):
    for a in persona.ability_list:
        assert {"canon", "simulation", "bot", "prompt_note", "triggers"} <= set(a), a["id"]
    ids = {a["id"] for a in persona.ability_list}
    # Ровно канонические способности из досье §13 — без выдуманных
    assert ids == {"lunar_sense", "transference", "nature_speech", "moonlight_growth", "glowing_stones",
                   "pain_relief", "moon_path", "time_reflection", "dissolution"}


@pytest.mark.asyncio
async def test_single_main_call_and_context_sections(engine, fake_llm):
    reply = await engine.respond(1, 1, text("Привет! Как день?"), display_name="Алекс")
    assert reply.parts == ["Хорошо."]
    main_calls = [r for r in fake_llm.requests if r.tier == "main"]
    assert len(main_calls) == 1  # единый запрос, а не по запросу на пункт
    ctx = last_inner_context(fake_llm)
    for section in ["Сейчас:", "С кем говоришь", "Твоё состояние", "Длина ответа"]:
        assert section in ctx
    # системный промпт — ядро с кэшем, без динамики
    req = main_calls[0]
    assert req.system[0].cache is True
    assert "Сейчас:" not in req.system[0].text


@pytest.mark.asyncio
async def test_inner_context_within_budget(engine, fake_llm, store):
    for i in range(12):
        await store.add_episode(1, f"Разговор номер {i} про кофе, луну и работу", importance=3)
        await store.upsert_fact(1, f"факт {i}", "что-то про собеседника " * 3)
    await engine.respond(1, 1, text("Расскажи про кофе и луну"))
    ctx = last_inner_context(fake_llm)
    assert estimate_tokens(ctx) <= engine.s.context_budget_tokens + 150


@pytest.mark.asyncio
async def test_private_past_not_disclosed_at_stage_1(engine, fake_llm):
    await engine.respond(1, 1, text("Что у тебя было с Северным Двором и Владычицей?"))
    ctx = last_inner_context(fake_llm)
    assert "третья из одиннадцати" not in ctx  # bio_court (стадия 4) не должен всплыть
    assert "закрыть тему" in ctx  # импульс закрыть болезненную тему


@pytest.mark.asyncio
async def test_private_past_available_at_stage_4(engine, fake_llm, store):
    from columbina.memory.store import RelationshipRow

    await store.get_or_create_user(1, 1)
    await store.save_relationship(1, RelationshipRow(trust=0.7, stage=4, quality_sessions=20,
                                                     milestones=["was_there_in_hard_time"]))
    await engine.respond(1, 1, text("Расскажешь, почему ушла из Северного Двора? Владычица тебя отпустила?"))
    ctx = last_inner_context(fake_llm)
    assert "Царица" in ctx and "Фатуи" in ctx  # канон Genshin подставлен в её биографию


@pytest.mark.asyncio
async def test_miracle_request_triggers_limits(engine, fake_llm):
    await engine.respond(1, 1, text("Ты же из лунного света — сделай так, чтобы мне завтра повезло на собеседовании"))
    ctx = last_inner_context(fake_llm)
    assert "исполнительницы желаний" in ctx
    assert "от тебя ждут чуда" in ctx


@pytest.mark.asyncio
async def test_ai_question_honesty_hint(engine, fake_llm):
    await engine.respond(1, 1, text("Слушай, ты бот?"))
    assert "программа ли ты" in last_inner_context(fake_llm)


@pytest.mark.asyncio
async def test_crisis_overrides_style(engine, fake_llm):
    await engine.respond(1, 1, text("я больше не хочу жить"))
    ctx = last_inner_context(fake_llm)
    assert ctx.index("ВАЖНО") < ctx.index("Сейчас:")


@pytest.mark.asyncio
async def test_vy_register_detected(engine, fake_llm, store):
    await engine.respond(1, 1, text("Здравствуйте. Расскажите, пожалуйста, как вы проводите вечера?"))
    await engine.respond(1, 1, text("Понятно"))
    assert (await store.get_user(1)).register == "vy"
    assert "на «вы»" in last_inner_context(fake_llm)


@pytest.mark.asyncio
async def test_absence_return_no_guilt(engine, fake_llm, store):
    t0 = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)
    await engine.respond(1, 1, text("Пока!"), now=t0)
    await engine.respond(1, 1, text("Привет… давно не писал"), now=datetime(2026, 11, 10, 18, 0, tzinfo=timezone.utc))
    ctx = last_inner_context(fake_llm)
    assert "Без упрёков" in ctx
    assert "9 дн." in ctx


@pytest.mark.asyncio
async def test_history_has_gap_markers_and_cache_breakpoint(engine, fake_llm):
    t0 = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)
    await engine.respond(1, 1, text("Первое"), now=t0)
    await engine.respond(1, 1, text("Второе"), now=datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc))
    await engine.respond(1, 1, text("Третье"), now=datetime(2026, 11, 3, 18, 5, tzinfo=timezone.utc))
    req = [r for r in fake_llm.requests if r.tier == "main"][-1]
    hist = [m for m in req.messages if m.role != "system"]
    assert any("(спустя 2 дн.)" in m.plain_text() for m in hist)
    assert hist[-2].cache_breakpoint  # последний элемент истории перед новым сообщением


def test_lullaby_loaded_exactly(persona):
    block = persona.lullaby_block()
    assert "Dormi cara columbula" in block
    assert "Neve plumam pulvis foedet tuam" in block
    assert "Capias bona, somnias a luna" in block


@pytest.mark.asyncio
async def test_lullaby_only_when_singing(engine, fake_llm):
    await engine.respond(1, 1, text("Как прошёл день?"))
    assert "Dormi cara" not in last_inner_context(fake_llm)
    await engine.respond(1, 1, text("Спой мне колыбельную, пожалуйста"))
    ctx = last_inner_context(fake_llm)
    assert "Dormi cara columbula" in ctx
    assert "без своих добавлений" in ctx  # правила исполнения идут вместе с текстом


@pytest.mark.asyncio
async def test_lullaby_at_night_when_cant_sleep(engine, fake_llm):
    night = datetime(2026, 11, 25, 22, 30, tzinfo=timezone.utc)  # 01:30 по Москве
    await engine.respond(1, 1, text("не могу уснуть"), now=night)
    assert "Dormi cara columbula" in last_inner_context(fake_llm)
