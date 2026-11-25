from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from columbina.initiative.engine import InitiativeEngine, in_quiet_hours
from columbina.memory.store import RelationshipRow, UserProfile
from columbina.perception.model import Percept


def t(text):
    return [Percept("text", True, "text", "", text=text, history=text)]


NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)  # 15:00 по Москве


async def _setup(engine, stage=2, days_ago=7):
    await engine.respond(1, 1, t("Привет"), now=NOW - timedelta(days=days_ago))
    await engine.store.save_relationship(1, RelationshipRow(trust=0.3, stage=stage, quality_sessions=5))


def test_quiet_hours():
    u = UserProfile(user_id=1, chat_id=1, timezone="Europe/Moscow", quiet_start=23, quiet_end=9)
    assert in_quiet_hours(u, datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc), "UTC")  # 00:00 MSK
    assert not in_quiet_hours(u, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), "UTC")


@pytest.mark.asyncio
async def test_stage1_never_initiates(engine):
    await _setup(engine, stage=1)
    ie = InitiativeEngine(engine)
    assert await ie.decide(await engine.store.get_user(1), NOW) is None


@pytest.mark.asyncio
async def test_absence_once_then_silence(engine):
    await _setup(engine, stage=2, days_ago=7)
    ie = InitiativeEngine(engine)
    sent = []

    async def send(chat_id, parts):
        sent.append(parts)

    assert await ie.tick(send, NOW) == 1
    # на неё не ответили — больше ничего, даже через несколько дней
    assert await ie.tick(send, NOW + timedelta(days=1)) == 0
    assert await ie.tick(send, NOW + timedelta(days=4)) == 0
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_initiative_off(engine):
    await _setup(engine, stage=3, days_ago=8)
    await engine.store.update_user(1, initiative_mode="off")
    ie = InitiativeEngine(engine)
    assert await ie.decide(await engine.store.get_user(1), NOW) is None


@pytest.mark.asyncio
async def test_note_delivered_even_if_off_but_not_at_night(engine):
    await _setup(engine, stage=1, days_ago=1)
    await engine.store.update_user(1, initiative_mode="off")
    await engine.store.add_note(1, "позвонить маме", (NOW - timedelta(minutes=5)).timestamp())
    ie = InitiativeEngine(engine)
    user = await engine.store.get_user(1)
    night = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)  # 01:00 MSK
    assert await ie.decide(user, night) is None
    c = await ie.decide(user, NOW)
    assert c and c.kind == "note"


@pytest.mark.asyncio
async def test_no_initiative_right_after_chat(engine):
    await _setup(engine, stage=3, days_ago=0)
    await engine.store.add_thread(1, "собеседование", (NOW - timedelta(hours=1)).timestamp())
    ie = InitiativeEngine(engine)
    user = await engine.store.get_user(1)
    user.last_user_msg_at = (NOW - timedelta(hours=2)).timestamp()
    assert await ie.decide(user, NOW) is None
    user.last_user_msg_at = (NOW - timedelta(hours=8)).timestamp()
    c = await ie.decide(user, NOW)
    assert c and c.kind == "thread"


@pytest.mark.asyncio
async def test_initiative_prompt_has_reason_and_is_stored(engine, fake_llm):
    await _setup(engine, stage=2, days_ago=7)
    fake_llm.script = lambda req: "Я сегодня учила духов снежкам. Они опять растаяли." if req.tier == "main" else "{}"
    ie = InitiativeEngine(engine)
    sent = []

    async def send(chat_id, parts):
        sent.append(parts)

    await ie.tick(send, NOW)
    req = [r for r in fake_llm.requests if r.purpose == "initiative"][0]
    assert "Ты сама пишешь первой" in req.messages[-1].plain_text()
    hist = await engine.store.active_messages(1)
    assert hist[-1].kind == "initiative"


@pytest.mark.asyncio
async def test_bad_initiative_is_dropped(engine, fake_llm):
    await _setup(engine, stage=2, days_ago=7)
    fake_llm.script = lambda req: "Почему ты так долго не писал? Мне без тебя плохо." if req.tier == "main" else "{}"
    ie = InitiativeEngine(engine)

    async def send(chat_id, parts):
        raise AssertionError("не должна отправлять манипуляцию")

    assert await ie.tick(send, NOW) == 0
