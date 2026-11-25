"""Внутренняя жизнь: интересы, исследования в бюджете, журнал знаний, дневник, сны, цели, находки в инициативе."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from columbina.config import ROOT
from columbina.initiative.engine import InitiativeEngine
from columbina.inner.life import InnerLife, Limits
from columbina.memory.store import RelationshipRow
from columbina.perception.model import Percept

from .conftest import last_inner_context
from .test_web import make_tools

PLAN = {"angle": "почему Луна удаляется", "queries": ["луна удаляется от земли"]}
SYNTH = {
    "facts": [
        {"text": "Луна удаляется от Земли примерно на 3,8 см в год.", "source_url": "https://example.org/moon",
         "kind": "fact", "confidence": 0.9},
        {"text": "Выдуманный факт без источника.", "source_url": "https://fake.example/none", "kind": "fact",
         "confidence": 0.9},
    ],
    "conclusion": "Луна медленно уходит из-за приливов.", "contradictions": "", "open_questions": ["что будет через миллиард лет"],
    "share_worthy": True, "share_hint": "Представляешь, Луна каждый год отходит на ширину ногтя.",
    "diary": "Прочитала, что Луна отходит от Земли. Вежливо, но уходит.", "continue_later": True,
    "new_interests": ["приливы"],
}


def script(req):
    if req.purpose == "research":
        return PLAN if "Тема:" in req.messages[0].plain_text() and "Уже знала" not in req.messages[0].plain_text() else SYNTH
    if req.purpose == "dream":
        return "Мне снилось, что снег падал вверх, а духи ловили его сачками."
    if req.tier == "main":
        return "Хорошо."
    return "{}"


async def _setup(engine, fake_llm, monkeypatch):
    import columbina.web.reader as rd

    async def ok_resolve(host, port):
        return "93.184.216.34"

    monkeypatch.setattr(rd, "resolve_public", ok_resolve)
    engine.web = make_tools(tavily=False)
    fake_llm.script = script
    life = InnerLife(engine, ROOT / "data" / "persona", Limits(research_per_day=2))
    engine.inner = life
    await engine.store.get_or_create_user(1, 1)
    await engine.store.update_user(1, last_user_msg_at=time.time())
    return life


async def test_interests_seeded_and_learned(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    topic = await life.choose_topic(1, time.time())
    assert topic == "луна и космос"
    await life.learn_topics(1, ["механические часы"])
    assert any(i["topic"] == "механические часы" and i["source"] == "conversation"
               for i in await engine.store.interests(1))


async def test_research_saves_only_sourced_facts_and_writes_diary(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    res = await life.research(1)
    assert res and res["facts"] == 1 and res["continue"]
    know = await engine.store.knowledge(1)
    assert len(know) == 1 and know[0]["source_url"] == "https://example.org/moon" and know[0]["share_worthy"]
    assert "Выдуманный" not in str(know)
    assert (await engine.store.diary(1, "discovery"))[0]["text"].startswith("Прочитала")
    assert (await engine.store.open_research(1))[0]["steps"] == 1
    assert any(g["title"].startswith("разобраться") for g in await engine.store.goals(1))
    assert any(i["topic"] == "приливы" for i in await engine.store.interests(1))
    # повтор: тот же факт не дублируется
    await life.research(1)
    assert len(await engine.store.knowledge(1)) == 1


async def test_research_budget_and_pause(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    now = time.time()
    assert await life.research_due(1, now)
    await life.research(1)
    await life.research(1)
    assert not await life.research_due(1, time.time())  # лимит в день
    await engine.store.set_setting(1, "paused", "1")
    assert not await life.research_due(1, time.time() + 2 * 86400)
    engine.web = None
    await engine.store.set_setting(1, "paused", "0")
    assert not await life.research_due(1, time.time() + 2 * 86400)  # без интернета не исследует


async def test_dream_only_at_night_and_rarely(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    await engine.store.update_user(1, timezone="Europe/Moscow")
    night = datetime.now(timezone.utc).replace(hour=0, minute=30)  # 03:30 по Москве
    day = night.replace(hour=12)
    assert not await life.dream_due(1, day)
    assert await life.dream_due(1, night)
    assert "снег падал вверх" in await life.dream(1)
    assert not await life.dream_due(1, night + timedelta(days=1))
    assert (await engine.store.diary(1, "dream"))


async def test_findings_and_activity_reach_conversation(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    await life.research(1)
    await engine.respond(1, 1, [Percept("text", True, "text", "", text="Чем занималась?", history="x")])
    ctx = last_inner_context(fake_llm)
    assert "Что ты действительно делала" in ctx and "луна и космос" in ctx
    assert "3,8 см" in ctx and "example.org/moon" in ctx


async def test_discovery_initiative_shared_once(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    await life.research(1)
    await engine.store.save_relationship(1, RelationshipRow(trust=0.3, stage=2, quality_sessions=4))
    await engine.store.update_user(1, last_user_msg_at=time.time() - 10 * 3600, quiet_start=0, quiet_end=0)
    ie = InitiativeEngine(engine)
    sent = []

    async def send(chat_id, parts):
        sent.append(parts)

    assert await ie.tick(send, datetime.now(timezone.utc)) == 1
    req = [r for r in fake_llm.requests if r.purpose == "initiative"][-1]
    assert "прочитала кое-что интересное" in req.messages[-1].plain_text()
    assert not await engine.store.unshared_findings(1)  # больше не повторит ту же находку


async def test_inner_tick_runs_research_when_due(engine, fake_llm, monkeypatch):
    life = await _setup(engine, fake_llm, monkeypatch)
    done = await life.tick(1, datetime.now(timezone.utc).replace(hour=10))
    assert done and done[0].startswith("research:")
