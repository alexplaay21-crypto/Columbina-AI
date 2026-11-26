"""Initiative Engine: решает, есть ли у Коломбины повод написать первой.

Повод должен быть настоящим. Кандидаты:
  note           — её записка сквозь время (напоминание) — срабатывает всегда в срок, кроме тихих часов
  thread         — вернуться к незавершённой теме (собеседование, болезнь) в назначенное время
  full_moon      — полнолуние сегодня (её праздник месяца)
  first_snow     — у собеседника пошёл снег (если известна погода)
  absence        — собеседник молчит 5+ дней (одно сообщение, без упрёков)
  gift           — тайный «подарок»: маленькая история/наблюдение по его интересам (редко, со стадии 3)

Правила против спама (досье-дополнение §6):
  - инициатива выключается командой; режимы: off | rare | normal
  - не в тихие часы собеседника (по умолчанию 23:00–09:00)
  - если на прошлую инициативу не ответили — молчим (кроме записок, о которых просили)
  - не чаще одного раза в сутки (rare — раз в 3 дня); после разговора — пауза 6 часов
  - стадия 1 — первой не пишет (только записки по просьбе)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ..cognition.engine import CognitiveEngine
from ..memory.store import UserProfile

log = logging.getLogger(__name__)

MIN_GAP = {"normal": 20 * 3600, "rare": 3 * 86400}
AFTER_CHAT_PAUSE = 6 * 3600
ABSENCE_DAYS = 5


@dataclass
class Candidate:
    kind: str
    reason: str
    score: float
    ref_id: int | None = None


def in_quiet_hours(user: UserProfile, now: datetime, default_tz: str) -> bool:
    try:
        tz = ZoneInfo(user.timezone or default_tz)
    except Exception:
        tz = ZoneInfo(default_tz)
    h = now.astimezone(tz).hour
    qs, qe = user.quiet_start, user.quiet_end
    if qs == qe:
        return False
    return (h >= qs or h < qe) if qs > qe else (qs <= h < qe)


class InitiativeEngine:
    def __init__(self, engine: CognitiveEngine):
        self.engine = engine
        self.store = engine.store

    async def candidates(self, user: UserProfile, now: datetime) -> list[Candidate]:
        ts = now.timestamp()
        out: list[Candidate] = []
        rel = await self.store.get_relationship(user.user_id)
        recent = await self.store.initiatives_since(user.user_id, ts - 60 * 86400)

        def sent_within(kind: str, days: float) -> bool:
            return any(r["kind"] == kind and ts - r["sent_at"] < days * 86400 for r in recent)

        for n in await self.store.pending_notes(user.user_id):
            if n["due_at"] <= ts:
                out.append(Candidate("note", f"пришло время твоей записки: «{n['text']}». Напомни естественно, своими словами",
                                     10.0, n["id"]))
        if rel.stage >= 2:
            for t in await self.store.open_threads(user.user_id):
                if t["follow_up_at"] and t["follow_up_at"] <= ts and ts - t["follow_up_at"] < 3 * 86400:
                    out.append(Candidate("thread", f"незакрытая тема: {t['topic']}. Спроси, как прошло, — один раз",
                                         6.0, t["id"]))
        if rel.stage >= 2:
            sit = await self.engine.situation_for(user, now)
            if "full_moon" in sit.tags and not sent_within("full_moon", 20):
                out.append(Candidate("full_moon", "сегодня полнолуние — твоё любимое время; поделись этим коротко", 3.0))
            if sit.weather and "snow" in sit.weather.tags and not sent_within("first_snow", 60):
                out.append(Candidate("first_snow", "у собеседника идёт снег (по погоде)", 3.5))
        if rel.stage >= 2 and user.last_user_msg_at:
            days = (ts - user.last_user_msg_at) / 86400
            if days >= ABSENCE_DAYS and not sent_within("absence", 14):
                out.append(Candidate("absence",
                                     f"собеседник молчит {int(days)} дн.; напиши одно лёгкое сообщение о своём, "
                                     "без упрёков и без «почему молчишь»", 2.5 + min(2.0, days / 7)))
        if rel.stage >= 2 and not sent_within("discovery", 2):
            for f in await self.store.unshared_findings(user.user_id):
                fact, _, hint = f["text"].partition(" || Как поделиться: ")
                out.append(Candidate("discovery", (
                    f"ты сама прочитала кое-что интересное ({f.get('source_title') or 'источник'}: {f.get('source_url')}): "
                    f"{fact}. Поделись находкой коротко и по-своему" + (f", например так: {hint}" if hint else "")
                    + ". Можно дать ссылку. Не выдумывай сверх прочитанного"), 3.2, f["id"]))
                break
        if rel.stage >= 3 and not sent_within("dream", 5):
            dream = await self.store.diary(user.user_id, "dream", limit=1)
            if dream and ts - dream[0]["created_at"] < 10 * 3600:
                out.append(Candidate("dream", "тебе этой ночью снился сон: " + dream[0]["text"][:400]
                                     + ". Можно коротко рассказать его, как рассказывают сон утром (это сон)", 2.2))
        if rel.stage >= 3 and user.last_user_msg_at and (ts - user.last_user_msg_at) > 2 * 86400:
            facts = await self.store.facts(user.user_id)
            if facts and not sent_within("gift", 7):
                f = facts[int(ts // 86400) % len(facts)]
                out.append(Candidate("gift", f"тайный подарок: короткое наблюдение или история, связанная с тем, что "
                                             f"собеседнику важно ({f.key}: {f.value}); без вопросов-допросов", 1.5))
        return out

    async def decide(self, user: UserProfile, now: datetime) -> Candidate | None:
        mode = user.initiative_mode or "normal"
        if (await self.store.get_setting(user.user_id, "paused", "0")) == "1":
            mode = "off"  # /pause: только записки, о которых просили
        until = float(await self.store.get_setting(user.user_id, "quiet_until", "0") or 0)
        if until and now.timestamp() < until:
            return None  # /quiet: тишина до /wake или до утра
        cands = await self.candidates(user, now)
        if not cands:
            return None
        ts = now.timestamp()
        notes = [c for c in cands if c.kind == "note"]
        if in_quiet_hours(user, now, self.engine.s.default_timezone):
            return None
        if notes:
            return notes[0]  # записки, о которых просили, работают даже при выключенной инициативе
        if mode == "off":
            return None
        last = await self.store.last_initiative(user.user_id)
        if last and not last["answered"]:
            return None
        if last and ts - last["sent_at"] < MIN_GAP.get(mode, MIN_GAP["normal"]):
            return None
        if user.last_user_msg_at and ts - user.last_user_msg_at < AFTER_CHAT_PAUSE:
            return None
        best = max(cands, key=lambda c: c.score)
        threshold = 2.0 if mode == "normal" else 3.0
        return best if best.score >= threshold else None

    async def tick(self, send, now: datetime | None = None) -> int:
        """Один проход планировщика. `send(chat_id, parts)` — корутина отправки. Возвращает число отправленных."""
        now = now or datetime.now(timezone.utc)
        sent = 0
        for user in await self.store.list_users():
            if user.chat_id != user.user_id:
                continue  # первой пишет только в личку, никогда в группу
            try:
                cand = await self.decide(user, now)
                if not cand:
                    continue
                reply = await self.engine.compose_initiative(user, cand.reason, cand.kind, now)
                if not reply:
                    continue
                await send(user.chat_id, reply.parts)
                sent += 1
                if cand.kind == "note" and cand.ref_id:
                    await self.store.set_note_status(cand.ref_id, "done")
                if cand.kind == "thread" and cand.ref_id:
                    await self.store.close_thread(cand.ref_id, "asked")
                if cand.kind == "discovery" and cand.ref_id:
                    await self.store.mark_knowledge(cand.ref_id, shared=True)
            except Exception:
                log.exception("initiative tick failed for %s", user.user_id)
        return sent
