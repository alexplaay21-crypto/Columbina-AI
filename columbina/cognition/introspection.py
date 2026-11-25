"""/mood — открытый отчёт о её состоянии и отношениях с тем, кто спрашивает. /check — проверка всех API.

По досье (Открытые решения) стадии дружбы скрыты, чтобы дружба не превращалась в игру на очки.
Здесь они показаны по решению автора проекта; модели эти числа по-прежнему не передаются.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..llm.base import LLMError, LLMRequest, Message, SystemBlock

BAR = 10


def bar(value: float) -> str:
    n = max(0, min(BAR, round(value * BAR)))
    return "▰" * n + "▱" * (BAR - n)


def _ago(ts: float, now: float) -> str:
    d = now - ts
    if d < 3600:
        return f"{max(1, int(d // 60))} мин назад"
    if d < 86400:
        return f"{int(d // 3600)} ч назад"
    return f"{int(d // 86400)} дн. назад"


async def mood_report(engine, user_id: int) -> str:
    store = engine.store
    user = await store.get_user(user_id)
    if user is None:
        return "Мы ещё не разговаривали."
    sit = await engine.situation_for(user)
    emo = await engine.load_emotion(user_id, sit)
    rel = await store.get_relationship(user_id)
    now = time.time()
    em_cfg = engine.persona.emotions["emotions"]
    rel_cfg = engine.persona.relationship

    text, _ = engine.emotions.describe(emo, sit.tags)
    mood = "светлый" if emo.mood > 0.35 else "тяжёлый" if emo.mood < -0.35 else "ровный"
    lines = [
        "🕊 Коломбина сейчас",
        f"Состояние: {text}",
        f"Фон дня: {mood}",
        f"Цветы в зале: {engine.emotions.flowers(emo, sit.tags)}",
        f"Луна: {sit.moon.describe()}",
        "",
        "Эмоции",
    ]
    vals = sorted(emo.values.items(), key=lambda kv: kv[1], reverse=True)
    for k, v in vals:
        if v < 0.05:
            continue
        name = em_cfg.get(k, {}).get("name", k)
        lines.append(f"{bar(v)} {name} {round(v * 100)}%")
    if emo.last_cause:
        cause = (engine.persona.emotions.get("event_labels", {}).get(emo.last_cause)
                 or rel_cfg.get("event_labels", {}).get(emo.last_cause) or emo.last_cause)
        lines.append(f"Последнее, что на неё повлияло: {cause}")

    stages = engine.relationship.stages
    st = stages.get(rel.stage, stages[1])
    lines += ["", "Отношения с тобой", f"Стадия {rel.stage}/{max(stages)} — {st['name']}",
              f"{bar(rel.trust)} доверие {round(rel.trust * 100)}%",
              f"Разговоров «по душам»: {rel.quality_sessions}"]
    nxt = stages.get(rel.stage + 1)
    if nxt:
        need = []
        if rel.trust < float(nxt.get("min_trust", 0)):
            need.append(f"доверие {round(float(nxt['min_trust']) * 100)}%")
        if rel.quality_sessions < int(nxt.get("min_sessions", 0)):
            need.append(f"разговоров {nxt['min_sessions']}")
        req = nxt.get("requires_milestone")
        if req and req not in rel.milestones:
            need.append(rel_cfg.get("milestone_labels", {}).get(req, req))
        lines.append(f"До «{nxt['name']}»: " + (", ".join(need) if need else "почти — всё есть, скоро"))
    if rel.milestones:
        labels = rel_cfg.get("milestone_labels", {})
        lines.append("Вехи: " + ", ".join(labels.get(m, m) for m in rel.milestones))
    lines.append(f"Сейчас она: {' '.join(str(st['behavior']).split())}")

    events = await store.recent_events(user_id, 8)
    if events:
        lines += ["", "Что недавно меняло отношение"]
        labels = rel_cfg.get("event_labels", {})
        for kind, _source, _detail, at in events:
            delta = engine.relationship.events.get(kind)
            sign = "+" if (delta or 0) > 0 else "−" if (delta or 0) < 0 else "·"
            lines.append(f"{sign} {labels.get(kind, kind)} ({_ago(at, now)})")

    facts = await store.facts(user_id)
    episodes = await store.episodes(user_id, "shared")
    notes = await store.pending_notes(user_id)
    lines += ["", f"Память о тебе: фактов {len(facts)}, общих воспоминаний {len(episodes)}, записок {len(notes)}"]
    if user.last_user_msg_at:
        tz = ZoneInfo(user.timezone or engine.s.default_timezone)
        lines.append(f"Последний разговор: {datetime.fromtimestamp(user.last_user_msg_at, tz):%d.%m %H:%M}")
    return "\n".join(lines)


async def health_check(engine, stt=None) -> str:
    """Короткие реальные запросы ко всем внешним сервисам. Стоит доли цента."""
    out: list[str] = []
    llm = engine.llm
    for tier in ("main", "background"):
        t = time.time()
        try:
            r = await llm.complete(LLMRequest(
                system=[SystemBlock("Ты проверка связи.")], messages=[Message.text("user", "Ответь одним словом: ок")],
                tier=tier, max_tokens=2000, effort="low", purpose="check",
            ))
            await engine.store.log_usage(None, "check", r.model, r.usage, r.cost_usd)
            out.append(f"✅ модель {tier}: {r.model} — «{r.text[:30]}» ({time.time() - t:.1f} с, ${r.cost_usd:.5f})")
        except LLMError as e:
            out.append(f"❌ модель {tier}: {e}")
        except Exception as e:  # noqa: BLE001
            out.append(f"❌ модель {tier}: {type(e).__name__}: {e}")
    try:
        r = await llm.complete(LLMRequest(
            system=[], messages=[Message.text("user", "Верни JSON с полем ok=true")], tier="background",
            max_tokens=2000, effort="low", purpose="check",
            json_schema={"type": "object", "additionalProperties": False, "required": ["ok"],
                         "properties": {"ok": {"type": "boolean"}}},
        ))
        out.append(f"✅ JSON-ответы (рефлексия памяти): {r.text[:40]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"❌ JSON-ответы (рефлексия памяти): {e}")
    try:
        v = await engine.embedder.embed(["проверка"])
        out.append(f"✅ эмбеддинги: {type(engine.embedder).__name__}, размер {v[0].shape[0]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"❌ эмбеддинги: {e}")
    if stt is not None:
        out.append(("✅" if stt.available else "⚠️") + f" распознавание речи: {type(stt).__name__}"
                   + ("" if stt.available else " — не настроено (нет ключа или ffmpeg)"))
    try:
        await engine.store.d.fetchone("SELECT 1 AS x")
        out.append(f"✅ база: {engine.store.d.dialect}")
    except Exception as e:  # noqa: BLE001
        out.append(f"❌ база: {e}")
    return "\n".join(out)


def admin_error_note(debug: dict[str, Any]) -> str | None:
    err = debug.get("error")
    if not err:
        return None
    return f"⚙️ (видно только админу) Ошибка модели: {str(err)[:500]}\nПроверка всех сервисов: /check"
