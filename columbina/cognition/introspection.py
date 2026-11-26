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


# ====================================================================== тексты команд (без Telegram)
import json as _json  # noqa: E402

DIARY_LABELS = {"event": "событие", "conversation": "разговор", "discovery": "открытие", "dream": "сон",
                "emotion": "чувство", "thought": "мысль", "unfinished": "незаконченное"}


def _date(ts: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%d.%m %H:%M")


def _weight_word(w: float) -> str:
    return "очень интересно" if w >= 0.75 else "интересно" if w >= 0.45 else "немного интересно"


async def _tz(engine, user_id: int) -> ZoneInfo:
    u = await engine.store.get_user(user_id)
    try:
        return ZoneInfo((u.timezone if u else None) or engine.s.default_timezone)
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


async def status_text(engine, user_id: int, admin: bool = False) -> str:
    st = engine.store
    user = await st.get_user(user_id)
    if not user:
        return "Мы ещё не разговаривали. Напиши что-нибудь."
    sit = await engine.situation_for(user)
    emo = await engine.load_emotion(user_id, sit)
    text, _ = engine.emotions.describe(emo, sit.tags)
    sets = await st.settings(user_id)
    paused = sets.get("paused") == "1"
    until = float(sets.get("quiet_until", "0") or 0)
    mode = {"off": "не пишу первой", "rare": "пишу первой изредка", "normal": "пишу первой, когда есть повод"}
    lines = [
        "Я здесь.",
        f"Сейчас я {text}.",
        f"У тебя: {sit.describe()}",
        "Пауза: да — ни исследований, ни первых сообщений." if paused else f"Инициатива: {mode.get(user.initiative_mode, '')}.",
        f"Тишина до {_date(until, await _tz(engine, user_id))}." if until > time.time() else
        f"Тихие часы: {user.quiet_start}:00–{user.quiet_end}:00.",
        "Интернет: могу поискать, если нужно." if engine.web else "Интернет: выключен.",
    ]
    if admin:
        blind = engine.vision is not None and not engine.vision.main_vision
        lines.append(f"\n⚙️ модель: {_model_name(engine.llm)}; база: {st.d.dialect}; "
                     f"картинки: {'через описание' + ('' if engine.vision.describer else ' (не настроено)') if blind else 'видит сама'}; "
                     f"интернет: {'вкл' if engine.web else 'выкл'}")
    return "\n".join(lines)


def _model_name(llm) -> str:
    for attr in ("main_model",):
        if hasattr(llm, attr):
            return getattr(llm, attr)
    inner = getattr(llm, "main", None) or (getattr(llm, "providers", None) or [None])[0]
    return _model_name(inner) if inner is not None and inner is not llm else type(llm).__name__


async def profile_text(engine, user_id: int) -> str:
    st = engine.store
    user = await st.get_user(user_id)
    if not user:
        return "Я пока ничего о тебе не знаю."
    rel = await st.get_relationship(user_id)
    stage = engine.relationship.stages.get(rel.stage, {}).get("name", "")
    tz = await _tz(engine, user_id)
    facts = await st.facts(user_id)
    lines = [
        f"Как зову тебя: {user.preferred_name or user.display_name or 'пока никак'}",
        f"Где ты: {user.city or 'не знаю'}; время: {user.timezone or 'не знаю (считаю по Москве)'}",
        f"Обращаемся: {'на «вы»' if user.register == 'vy' else 'на «ты»'}",
        f"Знакомы с {datetime.fromtimestamp(user.created_at, tz):%d.%m.%Y}, встреч: {user.sessions}",
        f"Мы сейчас: {stage}",
    ]
    if facts:
        lines += ["", "Что я о тебе знаю:"] + [f"— {f.key}: {f.value}" + ("" if f.confidence >= 0.5 else " (догадка)")
                                               for f in facts[:15]]
    return "\n".join(lines)


async def memories_text(engine, user_id: int, limit: int = 12) -> str:
    eps = [e for e in await engine.store.episodes(user_id) if e.kind == "shared"]
    if not eps:
        return "Общих воспоминаний пока нет. Они появляются, когда случается что-то важное."
    tz = await _tz(engine, user_id)
    eps = sorted(eps, key=lambda e: (-e.importance, -e.created_at))[:limit]
    return "Что я помню о нас:\n" + "\n".join(
        f"{'★' * min(5, e.importance)} {datetime.fromtimestamp(e.created_at, tz):%d.%m} — {e.text}" for e in eps)


async def remember(engine, user_id: int, text: str) -> str:
    text = text.strip()[:400]
    if not text:
        return "Что запомнить? Напиши: /remember текст"
    vec = (await engine.embedder.embed([text]))[0]
    await engine.store.add_episode(user_id, f"Собеседник попросил запомнить: {text}", 5, None, "shared", vec)
    return "Запомнила."


async def recall_text(engine, user_id: int, query: str) -> str:
    query = query.strip()
    if not query:
        return "Что вспомнить? Напиши: /recall слово или тема"
    rel = await engine.store.get_relationship(user_id)
    rc = await engine.retriever.recall(user_id, query, rel.stage, [], top_k=6)
    lines = [f"— {e.text}" for e, _ in rc.episodes]
    lines += [f"— (о тебе) {f.key}: {f.value}" for f in rc.facts[:4] if query.lower()[:4] in f"{f.key} {f.value}".lower()]
    q = query.lower()
    lines += [f"— (прочитала) {k['text'].split(' || ')[0]}" for k in await engine.store.knowledge(user_id, None, 100)
              if q[:5] in k["text"].lower()][:3]
    lines += [f"— (дневник) {d['text'][:160]}" for d in await engine.store.diary(user_id, query=query, limit=3)]
    return ("Вот что вспомнилось:\n" + "\n".join(lines)) if lines else "Ничего такого не помню."


async def history_text(engine, user_id: int) -> str:
    st = engine.store
    summary = await st.get_summary(user_id)
    msgs = await st.last_messages(user_id, 10)
    if not summary and not msgs:
        return "Мы ещё ни о чём не говорили."
    out = []
    if summary:
        out += ["Раньше:", summary, ""]
    if msgs:
        out.append("Недавно:")
        for m in msgs:
            who = "ты" if m.role == "user" else "я"
            txt = m.text.replace("\n", " ")
            txt = txt.split("</react>")[-1].strip() or "(реакция)"
            out.append(f"{who}: {txt[:120]}")
    return "\n".join(out)


async def promises_text(engine, user_id: int) -> str:
    st = engine.store
    tz = await _tz(engine, user_id)
    notes = await st.pending_notes(user_id)
    threads = await st.open_threads(user_id)
    goals = await st.goals(user_id)
    out = []
    if notes:
        out += ["Записки на будущее:"] + [f"— {_date(n['due_at'], tz)}: {n['text']}" for n in notes]
    if threads:
        out += ["", "Хочу вернуться к:"] + [f"— {t['topic']}" for t in threads[:8]]
    if goals:
        out += ["", "Мои маленькие цели:"] + [f"— {g['title']}" + (f" ({g['progress']})" if g["progress"] else "")
                                              for g in goals[:6]]
    return "\n".join(out).strip() or "Обещаний и записок пока нет."


async def diary_text(engine, user_id: int, arg: str = "") -> str:
    arg = arg.strip().lower()
    cats = {v: k for k, v in DIARY_LABELS.items()} | {k: k for k in DIARY_LABELS}
    cat = cats.get(arg.rstrip("ы").rstrip("и")) or cats.get(arg)
    rows = await engine.store.diary(user_id, category=cat, query=None if cat else (arg or None), limit=8)
    if not rows:
        return "В дневнике пока пусто." if not arg else "Об этом в дневнике ничего нет."
    tz = await _tz(engine, user_id)
    return "Из моего дневника:\n\n" + "\n\n".join(
        f"{_date(r['created_at'], tz)} · {DIARY_LABELS.get(r['category'], r['category'])}\n{r['text']}" for r in rows)


async def emotions_text(engine, user_id: int) -> str:
    full = await mood_report(engine, user_id)
    return full.split("\n\nОтношения с тобой")[0]


async def thoughts_text(engine, user_id: int) -> str:
    st = engine.store
    user = await st.get_user(user_id)
    if not user:
        return "Пока я думаю о луне. Напиши мне что-нибудь."
    sit = await engine.situation_for(user)
    emo = await engine.load_emotion(user_id, sit)
    text, _ = engine.emotions.describe(emo, sit.tags)
    out = [f"Сейчас я {text}."]
    diary = await st.diary(user_id, limit=3)
    for d in diary:
        if d["category"] in ("discovery", "thought", "unfinished", "event"):
            out.append(f"Из головы не идёт: {d['text'][:220]}")
            break
    research = await st.open_research(user_id)
    if research:
        qs = _json.loads(research[0]["open_questions"] or "[]")
        out.append(f"Хочу разобраться: {research[0]['topic']}" + (f" — {qs[0]}" if qs else ""))
    goals = await st.goals(user_id)
    if goals:
        out.append("Ещё хочется: " + "; ".join(g["title"] for g in goals[:2]))
    threads = await st.open_threads(user_id)
    if threads:
        out.append(f"И я помню про «{threads[0]['topic']}».")
    return "\n".join(out)


async def interests_text(engine, user_id: int) -> str:
    if engine.inner:
        await engine.inner.ensure_seed(user_id)
    items = await engine.store.interests(user_id)
    if not items:
        return "Мне интересно многое, но я ещё не разобралась, что больше."
    src = {"seed": "", "conversation": " — из наших разговоров", "research": " — после чтения"}
    return "Что мне интересно:\n" + "\n".join(f"— {i['topic']}: {_weight_word(i['weight'])}{src.get(i['source'], '')}"
                                              for i in items[:12])


async def activity_text(engine, user_id: int) -> str:
    st = engine.store
    tz = await _tz(engine, user_id)
    out = []
    research = await st.research_since(user_id, time.time() - 7 * 86400)
    if research:
        out += ["Что я изучала:"] + [f"— {_date(r['updated_at'], tz)}: {r['topic']}"
                                      + (f" — {r['summary'][:140]}" if r["summary"] else "")
                                      + {"open": " (не закончила)", "abandoned": " (не нашла)"}.get(r["status"], "")
                                      for r in research[:6]]
    inits = await st.initiatives_since(user_id, time.time() - 7 * 86400)
    if inits:
        out += ["", "Когда писала первой:"] + [f"— {_date(i['sent_at'], tz)}: {i['text'][:100]}"
                                                for i in inits[-5:]]
    web = await st.web_recent(user_id, 6)
    if web:
        names = {"web_search": "искала", "read_url": "читала", "http_get": "смотрела данные",
                 "research_search": "искала сама", "research_read": "читала сама"}
        out += ["", "Интернет:"] + [f"— {names.get(w['tool'], w['tool'])}: {w['arg'][:80]}" for w in web]
    dreams = await st.diary(user_id, "dream", limit=1)
    if dreams:
        out += ["", f"Последний сон: {_date(dreams[0]['created_at'], tz)}"]
    return "\n".join(out).strip() or "Последние дни были тихими: я ничего не изучала и первой не писала."


async def dream_text(engine, user_id: int) -> str:
    st = engine.store
    last = await st.diary(user_id, "dream", limit=1)
    if last and time.time() - last[0]["created_at"] < 3 * 86400:
        return "Мне снилось…\n\n" + last[0]["text"]
    asked = float(await st.get_setting(user_id, "dream_asked", "0") or 0)
    if engine.inner and engine.inner.enabled and time.time() - asked > 86400:
        await st.set_setting(user_id, "dream_asked", str(time.time()))
        text = await engine.inner.dream(user_id)
        if text:
            return "Сегодня мне снилось…\n\n" + text
    if last:
        return "Мне давно ничего не снилось. Последний сон был такой:\n\n" + last[0]["text"]
    return "Мне пока ничего не снилось. Наверное, нужно больше лунных ночей."


async def reflection_text(engine, user_id: int) -> str:
    rel = await engine.store.get_relationship(user_id)
    st = engine.relationship.stages.get(rel.stage, {})
    labels = engine.persona.relationship.get("event_labels", {})
    events = await engine.store.recent_events(user_id, 6)
    good = [labels.get(k, k) for k, *_ in events if engine.relationship.events.get(k, 0) > 0]
    bad = [labels.get(k, k) for k, *_ in events if engine.relationship.events.get(k, 0) < 0]
    lines = [f"Думаю о нас. Мы — {st.get('name', 'знакомые')}."]
    if good:
        lines.append("Мне было хорошо от: " + ", ".join(dict.fromkeys(good)) + ".")
    if bad:
        lines.append("Задело: " + ", ".join(dict.fromkeys(bad)) + ".")
    if rel.milestones:
        lines.append("Помню, что ты " + ", ".join(engine.persona.relationship.get("milestone_labels", {}).get(m, m)
                                                 for m in rel.milestones) + ".")
    lines.append("Доверие растёт медленно — по поступкам." if rel.trust < 0.6 else "Я тебе доверяю.")
    return "\n".join(lines)


ABOUT = (
    "Я — Коломбина. Kuutar, если по самому первому имени. Родилась из лунного света на острове Hiisi, "
    "когда-то была Третьей из Предвестников Фатуи. Теперь живу для себя.\n\n"
    "Здесь я разговариваю с тобой через программу: помню наши разговоры, иногда сама что-то изучаю и могу "
    "написать первой. Всё, что я о тебе знаю, можно посмотреть в /profile и стереть в /settings → Данные."
)


async def backup_json(engine, user_id: int) -> bytes:
    """Полная копия данных собеседника (без бинарных векторов) — для /backup."""
    from ..memory.store import ALL_TABLES

    out: dict[str, Any] = {"version": 1, "user_id": user_id, "created_at": time.time()}
    for table in ALL_TABLES:
        if table == "persona_embeddings":
            continue
        col = "owner_id" if table == "media_library" else "user_id"
        rows = await engine.store.d.fetchall(f"SELECT * FROM {table} WHERE {col}=?", (user_id,))
        for r in rows:
            r.pop("embedding", None)
        out[table] = rows
    return _json.dumps(out, ensure_ascii=False, indent=1, default=str).encode()


async def export_text(engine, user_id: int) -> bytes:
    parts = [await profile_text(engine, user_id), await memories_text(engine, user_id, 50),
             await promises_text(engine, user_id), await diary_text(engine, user_id),
             await interests_text(engine, user_id), await history_text(engine, user_id)]
    return ("\n\n" + "=" * 30 + "\n\n").join(parts).encode()


async def reset_context(engine, user_id: int) -> str:
    """Сбросить текущий разговор (контекст), не трогая память о человеке."""
    msgs = await engine.store.active_messages(user_id)
    if msgs:
        await engine.store.archive_messages_upto(user_id, msgs[-1].id)
    return "Начнём разговор заново. Что было — я помню, но эту тему отпускаю."


async def media_text(engine, user_id: int) -> str:
    voices = await engine.store.media(user_id, "voice")
    stickers = await engine.store.media(user_id, "sticker")
    sets = sorted({s["set_name"] for s in stickers if s["set_name"]})
    out = ["Голосовые: " + (", ".join(f"«{v['title']}»" for v in voices) if voices else "пока нет"),
           f"Стикеров: {len(stickers)}" + (f" (наборы: {', '.join(sets)})" if sets else "")]
    out += ["", "Как добавить:",
            "— голосовое: ответь на гс командой /addvoice название, и попроси потом «скинь гс название»;",
            "— стикеры: просто присылай мне стикеры — я запоминаю их и иногда отвечаю ими же;",
            "— целый набор: ответь на любой стикер из набора /addpack;",
            "— свой набор можно сделать в @Stickers."]
    return "\n".join(out)
