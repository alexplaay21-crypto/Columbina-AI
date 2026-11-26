"""Целостность памяти, контроль расходов, защита характера, параллельные сообщения, перезапуск."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import numpy as np

from columbina.cognition.engine import CognitiveEngine
from columbina.llm.base import LLMError
from columbina.llm.fake import FakeProvider
from columbina.memory.embeddings import LocalHashEmbedder
from columbina.memory.maintenance import Maintenance
from columbina.memory.store import MemoryStore
from columbina.perception.model import Percept

from .conftest import last_inner_context


def t(x):
    return [Percept("text", True, "text", "", text=x, history=x)]


def reflection(episode_text="", facts=None, importance=3):
    return {"facts": facts or [], "forget_keys": [], "episode": {"worth_remembering": bool(episode_text),
            "text": episode_text, "importance": importance, "emotion": ""}, "emotion_events": [],
            "relationship_events": [], "open_threads": [], "closed_thread_ids": [], "media_description": "",
            "user_mood": "", "her_life": "", "user_timezone": "", "user_city": "", "topics": []}


async def test_similar_episodes_merge_instead_of_duplicate(engine, fake_llm, store):
    fake_llm.script = lambda r: "Да." if r.tier == "main" else reflection("Рассказал про кота Бублика и ветеринара")
    for _ in range(3):
        await engine.respond(1, 1, t("у меня кот Бублик, завтра к ветеринару"))
        await engine.drain()
    eps = await store.episodes(1, "shared")
    assert len(eps) == 1 and eps[0].recall_count >= 2


async def test_fact_contradiction_logged_and_guess_does_not_override(engine, fake_llm, store):
    seq = iter([reflection(facts=[{"key": "город", "value": "Москва", "confidence": 0.9, "certain": True}]),
                reflection(facts=[{"key": "город", "value": "Казань", "confidence": 0.4, "certain": False}]),
                reflection(facts=[{"key": "город", "value": "Казань", "confidence": 0.9, "certain": True}])])
    fake_llm.script = lambda r: "Да." if r.tier == "main" else next(seq)
    for msg in ["я живу в Москве", "ну, может, в Казани", "переехал в Казань"]:
        await engine.respond(1, 1, t(msg))
        await engine.drain()
        if msg.startswith("ну"):
            assert await store.fact_value(1, "город") == "Москва"
    assert await store.fact_value(1, "город") == "Казань"
    assert any(k == "fact_changed" for k, *_ in await store.recent_events(1))


async def test_maintenance_merges_forgets_prunes(store):
    v = np.ones(8, dtype=np.float32) / np.sqrt(8)
    old = time.time() - 100 * 86400
    a = await store.add_episode(1, "разговор о снеге", 3, embedding=v)
    await store.add_episode(1, "разговор о снеге, подробнее", 2, embedding=v)
    await store.add_episode(1, "мелочь давняя", 1, embedding=np.eye(8, dtype=np.float32)[0], at=old)
    await store.add_episode(1, "важное давнее", 5, embedding=np.eye(8, dtype=np.float32)[1], at=old)
    await store.log_web(1, "web_search", "x", True)
    await store.d.execute("UPDATE web_log SET created_at=?", (old,))
    rep = await Maintenance(store).run(1)
    texts = [e.text for e in await store.episodes(1)]
    assert rep["merged"] == 1 and rep["forgotten"] == 1 and rep["pruned"] >= 1
    assert "мелочь давняя" not in texts and "важное давнее" in texts
    assert any(e.id == a and e.text.endswith("подробнее") for e in await store.episodes(1))


async def test_cost_controller_total_and_calls(engine, fake_llm, store):
    from columbina.llm.base import Usage

    engine.s = dataclasses.replace(engine.s, daily_budget_total=1.0)
    await store.log_usage(99, "chat", "m", Usage(), 1.5)  # другой собеседник потратил весь общий бюджет
    reply = await engine.respond(1, 1, t("привет"))
    assert reply.debug.get("over_budget")
    engine.s = dataclasses.replace(engine.s, daily_budget_total=0, daily_calls_per_user=2)
    await store.log_usage(2, "chat", "m", Usage(), 0)
    await store.log_usage(2, "reflection", "m", Usage(), 0)
    assert (await engine.respond(2, 2, t("привет"))).debug.get("over_budget")


async def test_personality_drift_hint(persona):
    from columbina.cognition.style import StyleGuard

    g = StyleGuard(persona.style)
    hints = g.overused(["Отличный вопрос! Вот несколько идей.", "Надеюсь, это поможет.", "Ну да."])
    assert any("ассистент" in h for h in hints)


async def test_topic_change_noticed(engine, fake_llm):
    await engine.respond(1, 1, t("Сегодня ходил в горы, было очень красиво и ветрено"))
    await engine.respond(1, 1, t("Слушай, а как правильно варить гречку без комков и пригара?"))
    assert "сменил тему" in last_inner_context(fake_llm)


async def test_parallel_messages_are_serialized(engine, fake_llm, store):
    order = []

    async def slow_complete(req):
        if req.tier == "main":
            order.append("start")
            await asyncio.sleep(0.05)
            order.append("end")
        return await FakeProvider().complete(req)

    fake_llm.complete = slow_complete  # type: ignore[method-assign]
    await asyncio.gather(engine.respond(1, 1, t("раз")), engine.respond(1, 1, t("два")))
    assert order[:4] == ["start", "end", "start", "end"]  # второй ход ждёт первый
    roles = [m.role for m in await store.active_messages(1)]
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_state_survives_restart(settings, persona):
    s1 = await MemoryStore(settings.db_path).open()
    e1 = CognitiveEngine(settings, persona, s1, FakeProvider(), LocalHashEmbedder())
    await e1.setup()
    await e1.respond(1, 1, t("Да отстань ты, бесишь"))
    await e1.drain()
    hurt_before = (await s1.get_emotion(1))[0]["hurt"]
    await s1.close()
    s2 = await MemoryStore(settings.db_path).open()  # «перезапуск»
    e2 = CognitiveEngine(settings, persona, s2, FakeProvider(), LocalHashEmbedder())
    await e2.setup()
    row = await s2.get_emotion(1)
    assert row and row[0]["hurt"] == hurt_before > 0
    assert (await s2.get_relationship(1)).trust == 0.0 and len(await s2.active_messages(1)) == 2
    await s2.close()


async def test_api_failure_does_not_lose_user_message(engine, fake_llm, store):
    def boom(req):
        raise LLMError("down", retryable=True)

    fake_llm.script = boom
    await engine.respond(1, 1, t("важное сообщение"))
    assert [m.text for m in await store.active_messages(1)] == ["важное сообщение"]
