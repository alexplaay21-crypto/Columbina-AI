"""Хранилище: одинаковое поведение на SQLite и PostgreSQL."""

from __future__ import annotations

import time

import numpy as np

from columbina.llm.base import Usage
from columbina.memory.db import to_pg

BIG_ID = 7_812_345_678  # реальные Telegram ID не влезают в 32 бита


def test_placeholders():
    assert to_pg("SELECT * FROM t WHERE a=? AND b IN (?, ?)") == "SELECT * FROM t WHERE a=$1 AND b IN ($2, $3)"


async def test_big_ids_and_profile(store):
    u = await store.get_or_create_user(BIG_ID, BIG_ID, "Алекс")
    assert u.user_id == BIG_ID
    await store.update_user(BIG_ID, timezone="Europe/Moscow", lat=55.75, lon=37.6, quiet_start=22)
    u2 = await store.get_user(BIG_ID)
    assert u2.timezone == "Europe/Moscow" and u2.quiet_start == 22 and abs(u2.lat - 55.75) < 1e-9
    assert [x.user_id for x in await store.list_users()] == [BIG_ID]


async def test_fact_upsert_keeps_max_confidence_and_pin(store):
    await store.upsert_fact(1, "Питомец", "кошка", 0.9, np.ones(4, dtype=np.float32), pinned=True)
    await store.upsert_fact(1, "питомец", "кошка Мурка", 0.5, None, pinned=False)
    [f] = await store.facts(1)
    assert f.value == "кошка Мурка" and f.confidence == 0.9 and f.pinned
    assert f.embedding is not None and f.embedding.shape == (4,)  # вектор не затёрт None


async def test_messages_archive_and_summary(store):
    ids = [await store.add_message(1, "user" if i % 2 == 0 else "assistant", f"m{i}") for i in range(6)]
    assert ids == sorted(ids)
    await store.archive_messages_upto(1, ids[3])
    assert [m.text for m in await store.active_messages(1)] == ["m4", "m5"]
    await store.set_summary(1, "резюме", ids[3])
    await store.set_summary(1, "резюме 2", ids[3])
    assert await store.get_summary(1) == "резюме 2"
    assert [m.text for m in await store.last_messages(1, 2, role="user")] == ["m2", "m4"]


async def test_usage_and_costs(store):
    await store.log_usage(1, "chat", "claude-opus-5-5", Usage(100, 50, 2000, 10), 0.0125)
    await store.log_usage(1, "reflection", "claude-haiku-5-5", Usage(1500, 200, 0, 0), 0.0003)
    await store.log_usage(2, "chat", "claude-opus-5-5", Usage(1, 1, 0, 0), 1.0)
    since = time.time() - 60
    assert abs(await store.cost_since(1, since) - 0.0128) < 1e-9
    assert abs(await store.cost_since(None, since) - 1.0128) < 1e-9
    rep = await store.usage_report(since)
    assert rep[0]["purpose"] == "chat" and rep[0]["n"] == 2 and int(rep[0]["inp"]) == 101


async def test_persona_embedding_cache(store):
    v = np.arange(3, dtype=np.float32)
    await store.set_persona_embedding("c1", "h1", v)
    await store.set_persona_embedding("c1", "h2", v * 2)
    assert await store.get_persona_embedding("c1", "h1") is None
    assert (await store.get_persona_embedding("c1", "h2")).tolist() == [0.0, 2.0, 4.0]


async def test_forget_user_wipes_everything(store):
    await store.get_or_create_user(1, 1)
    await store.add_message(1, "user", "секрет")
    await store.add_episode(1, "эпизод")
    await store.upsert_fact(1, "имя", "Алекс")
    await store.add_note(1, "записка", time.time() + 100)
    await store.add_message(2, "user", "чужое")
    await store.forget_user(1)
    assert await store.active_messages(1) == []
    assert await store.episodes(1) == [] and await store.facts(1) == []
    assert await store.pending_notes(1) == []
    assert len(await store.active_messages(2)) == 1
