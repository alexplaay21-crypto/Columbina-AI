"""Cognitive Engine — промежуточный слой между Telegram и AI API.

Один ход = один основной запрос к модели (+ редкий второй — если нужен инструмент или стиль сломан)
и один дешёвый фоновый запрос рефлексии после ответа.

Порядок хода:
  1. Ситуация: время у собеседника, луна, погода.
  2. Состояние: затухание эмоций с прошлого раза, ритм суток.
  3. Быстрая оценка входящего (appraisal) → события эмоций и доверия, условия.
  4. Память: релевантные эпизоды, факты, фрагменты её биографии по стадии дружбы.
  5. Способности: какие актуальны сейчас.
  6. Восприятие: что именно дошло.
  7. Внутренний контекст (единый блок) + история + новое сообщение → модель.
  8. Проверка стиля, разбор реакции/частей, сохранение, фоновая рефлексия.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..abilities.engine import AbilitiesEngine, AbilityContext
from ..abilities.weather import WeatherService
from ..config import Settings
from ..emotion.model import EmotionModel, EmotionState
from ..llm.base import (
    ImagePart, LLMError, LLMProvider, LLMRequest, LLMResponse, Message, SystemBlock, TextPart, ToolResult, ToolSpec,
)
from ..memory.embeddings import Embedder
from ..memory.retrieval import MemoryRetriever
from ..memory.store import MemoryStore, UserProfile
from ..perception.model import Percept, build_user_parts, render_perception
from ..persona.loader import Persona
from ..relationship.model import RelationshipModel
from .appraisal import appraise
from .context_builder import TurnContext, gap_marker, history_to_text_messages, render_inner_context
from .reflection import Reflector
from .situation import Situation, build_situation
from .style import ParsedReply, StyleGuard, parse_reply

log = logging.getLogger(__name__)

SESSION_GAP = 3 * 3600  # новая «встреча», если пауза больше 3 часов

PALE_REPLIES = [
    "Я сегодня какая-то бледная… Напиши чуть позже, хорошо?",
    "Свет сегодня плохо держится. Вернись немного позже — я дослушаю.",
    "Хм. Слова сейчас не складываются. Давай чуть позже.",
]

LEAVE_NOTE_TOOL = ToolSpec(
    name="leave_note",
    description=(
        "Оставить записку сквозь время: напоминание собеседнику или себе вернуться к теме в указанный момент. "
        "Используй, когда собеседник просит напомнить или ты сама обещаешь спросить позже в конкретное время. "
        "Время — локальное время собеседника."
    ),
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "when_local"],
        "properties": {
            "text": {"type": "string", "description": "Что напомнить, от первого лица Коломбины, коротко."},
            "when_local": {"type": "string",
                           "description": "Локальная дата и время собеседника в формате YYYY-MM-DD HH:MM."},
        },
    },
)


@dataclass
class Reply:
    parts: list[str]
    reaction: str | None = None
    cost_usd: float = 0.0
    quote: bool = False  # ответить «реплаем» на сообщение собеседника
    debug: dict[str, Any] = field(default_factory=dict)


class CognitiveEngine:
    def __init__(
        self,
        settings: Settings,
        persona: Persona,
        store: MemoryStore,
        llm: LLMProvider,
        embedder: Embedder,
        weather: WeatherService | None = None,
        vision=None,
    ):
        self.s = settings
        self.persona = persona
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.weather = weather
        self.vision = vision  # VisionAdapter: зрение для моделей без зрения
        self.emotions = EmotionModel(persona.emotions)
        self.relationship = RelationshipModel(persona.relationship)
        self.abilities = AbilitiesEngine(persona.abilities)
        self.style = StyleGuard(persona.style)
        self.retriever = MemoryRetriever(store, embedder)
        self.reflector = Reflector(llm, list(self.emotions.events), list(self.relationship.events))
        self.system_blocks = [SystemBlock(persona.core_prompt, cache=True)]
        self._locks: dict[int, asyncio.Lock] = {}
        self._background: set[asyncio.Task] = set()

    async def setup(self) -> None:
        await self.retriever.index_persona(self.persona.knowledge)

    # ------------------------------------------------------------------ helpers
    def lock_for(self, user_id: int) -> asyncio.Lock:
        return self._locks.setdefault(user_id, asyncio.Lock())

    def spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def drain(self, timeout: float = 60.0) -> None:
        """Дождаться фоновых задач (рефлексия, резюме). Не крутится на уже завершённых и не висит вечно."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._background = {t for t in self._background if not t.done()}
            remaining = deadline - loop.time()
            if not self._background or remaining <= 0:
                return
            await asyncio.wait(self._background, timeout=remaining)

    async def _record_usage(self, user_id: int | None, purpose: str, resp: LLMResponse) -> None:
        await self.store.log_usage(user_id, purpose, resp.model, resp.usage, resp.cost_usd)

    async def situation_for(self, user: UserProfile, now: datetime | None = None) -> Situation:
        weather = None
        if self.weather and user.lat is not None and user.lon is not None:
            try:
                weather = await self.weather.current(user.lat, user.lon)
            except Exception:
                weather = None
        return build_situation(user.timezone, self.s.default_timezone, weather, now)

    async def load_emotion(self, user_id: int, sit: Situation) -> EmotionState:
        row = await self.store.get_emotion(user_id)
        now = sit.now_utc.timestamp()
        if row is None:
            return self.emotions.initial(now, sit.tags)
        vals, mood, cause, at = row
        return self.emotions.decay(EmotionState(vals, mood, cause, at), now, sit.tags)

    async def over_budget(self, user_id: int) -> bool:
        if self.s.daily_budget_usd <= 0:
            return False
        start = time.time() - 86400
        return await self.store.cost_since(user_id, start) >= self.s.daily_budget_usd

    def conditions(self, sit: Situation, percepts: list[Percept], emo: EmotionState, appr_conditions: set[str]) -> set[str]:
        c = set(appr_conditions)
        if sit.part == "night":
            c.add("night")
        if sit.part == "day":
            c.add("day")
        if "full_moon" in sit.tags:
            c.add("full_moon")
        if "new_moon" in sit.tags:
            c.add("new_moon")
        if sit.weather:
            c.add("has_weather")
        if self.emotions.intensity(emo, "tiredness") > 0.6:
            c.add("weak")
        return c

    # ------------------------------------------------------------------ main turn
    async def respond(self, user_id: int, chat_id: int, percepts: list[Percept],
                      display_name: str | None = None, now: datetime | None = None) -> Reply:
        async with self.lock_for(user_id):
            return await self._respond(user_id, chat_id, percepts, display_name, now)

    async def _respond(self, user_id: int, chat_id: int, percepts: list[Percept],
                       display_name: str | None, now: datetime | None) -> Reply:
        user = await self.store.get_or_create_user(user_id, chat_id, display_name)
        sit = await self.situation_for(user, now)
        ts = sit.now_utc.timestamp()

        if self.vision is not None:
            percepts = await self.vision.adapt(percepts)
            for p in percepts:
                desc = p.meta.get("vision_description")
                if desc and "[фото]" in p.history:
                    p.history = p.history.replace("[фото]", f"[фото: {desc[:400]}]", 1)
                elif desc:
                    p.history += f" (на кадрах: {desc[:400]})"
        parts, history_text = build_user_parts(percepts)
        user_text = "\n".join(p.text for p in parts if isinstance(p, TextPart))

        if await self.over_budget(user_id):
            await self.store.add_message(user_id, "user", history_text, at=ts)
            return Reply(parts=[random.choice(PALE_REPLIES)], debug={"over_budget": True})

        # --- сессии и отсутствие ---
        absence_days = None
        new_session = user.last_user_msg_at is None or ts - user.last_user_msg_at > SESSION_GAP
        if user.last_user_msg_at and ts - user.last_user_msg_at > 2 * 86400:
            absence_days = (ts - user.last_user_msg_at) / 86400

        # --- состояние и оценка ---
        emo = await self.load_emotion(user_id, sit)
        rel = await self.store.get_relationship(user_id)
        recent_user = await self.store.last_messages(user_id, 3, role="user")
        appr = appraise(user_text, [len(m.text) for m in recent_user])
        if absence_days:
            emo = self.emotions.apply(emo, "user_absent_returned", cause="собеседник вернулся")
            rel = self.relationship.apply(rel, "returned_after_absence")
        if new_session and sit.weather:
            for tag in sit.weather.tags:
                if tag in ("snow", "rain", "thunder", "heat"):
                    emo = self.emotions.apply(emo, tag, weight=0.5, cause=f"погода: {sit.weather.text}")
        for ev in appr.emotion_events:
            emo = self.emotions.apply(emo, ev, weight=0.8, cause=ev)
        for ev in appr.relationship_events:
            rel = self.relationship.apply(rel, ev)
            await self.store.log_event(user_id, ev, "appraisal", user_text[:120])
        if appr.register and appr.register != user.register:
            user.register = appr.register
        extra_obs: list[str] = []

        # Она писала первой, а ответа долго не было — немного задета (без драмы).
        last_init = await self.store.last_initiative(user_id)
        if last_init and not last_init["answered"]:
            hours = (ts - last_init["sent_at"]) / 3600
            if hours >= 6:
                emo = self.emotions.apply(emo, "ignored_initiative", weight=min(1.5, 0.5 + hours / 24),
                                          cause="написала первой — ответа долго не было")
                extra_obs.append(f"ты писала первой {int(hours)} ч назад, а ответа не было — тебе немного обидно; "
                                 "можешь показать это одной сухой фразой или иронией и дальше говорить нормально")

        # Флирт: медленная романтика — по стадии дружбы.
        if "flirt" in appr.conditions:
            close = rel.stage >= 4
            emo = self.emotions.apply(emo, "flirt_close" if close else "flirt_early", cause="флирт")
            extra_obs.append(
                "собеседник флиртует; вы правда близки — можешь ответить тепло и честно о своих чувствах, "
                "но без зависимости и ревности" if close else
                "собеседник флиртует, а вы ещё мало знакомы — смутись или ответь игриво, всерьёз не подыгрывай "
                "и не отталкивай грубо")

        # Долгий разговор утомляет.
        recent_count = sum(1 for m in await self.store.last_messages(user_id, 80) if ts - m.created_at < SESSION_GAP)
        if recent_count and recent_count % 30 == 0:
            emo = self.emotions.apply(emo, "long_conversation", cause="долгий разговор")

        if not user.timezone and not user.city:
            extra_obs.append("ты не знаешь, где собеседник и сколько у него времени (время выше — по Москве); "
                             "если к месту, можешь спросить город")
        if any(p.kind == "reply" for p in percepts):
            extra_obs.append("собеседник отвечает на конкретное сообщение — отвечай прежде всего на него")

        # --- память ---
        history = await self.store.active_messages(user_id)
        last_reply = next((m.text for m in reversed(history) if m.role == "assistant"), "")
        query = user_text + ("\n" + last_reply[-300:] if len(user_text) < 80 else "")
        recall = await self.retriever.recall(user_id, query, rel.stage, self.persona.knowledge,
                                             top_k=self.s.memory_top_k, now=ts)
        summary = await self.store.get_summary(user_id)
        threads = await self.store.open_threads(user_id)
        notes = await self.store.pending_notes(user_id)

        # --- способности и восприятие ---
        conds = self.conditions(sit, percepts, emo, appr.conditions)
        active = self.abilities.active(AbilityContext(user_text, conds, rel.stage))
        perception_text = render_perception(percepts)

        emo_text, manners = self.emotions.describe(emo, sit.tags)
        her_recent = [m.text for m in history if m.role == "assistant"][-6:]
        ctx = TurnContext(
            situation=sit, user=user, stage=rel.stage, stage_text=self.relationship.describe(rel),
            emotion_text=emo_text, emotion_manners=manners, recall=recall, summary=summary,
            appraisal=appr, abilities=active, perception_text=perception_text, open_threads=threads,
            pending_notes=notes, style_hints=self.style.overused(her_recent, user.preferred_name or user.display_name),
            absence_days=absence_days, extra_observations=extra_obs,
            lullaby=self.persona.lullaby_block() if self._wants_lullaby(appr.conditions, sit) else "",
        )
        inner = render_inner_context(ctx, self.s.context_budget_tokens)

        messages = self._history_messages(history)
        # Метка паузы для текущего сообщения — та же, что история посчитает потом (кэш не ломается).
        marker = gap_marker(history[-1].created_at if history else None, ts)
        if marker:
            for p in parts:
                if isinstance(p, TextPart):
                    p.text = marker + p.text
                    break
        messages.append(Message(role="user", parts=parts))
        messages.append(Message.text("system", inner))

        try:
            reply_text, cost, tool_notes = await self._generate(user_id, user, sit, messages)
        except LLMError as e:
            log.warning("LLM failure for %s: %s", user_id, e)
            await self.store.add_message(user_id, "user", history_text, at=ts)
            return Reply(parts=[random.choice(PALE_REPLIES)], debug={"error": str(e)})

        parsed = parse_reply(reply_text)
        if not parsed.parts and not parsed.reaction:
            parsed = ParsedReply(parts=["…"], reaction=None, raw=reply_text)
        if not parsed.reaction:
            parsed.reaction = await self._event_reaction(user_id, appr.emotion_events)

        # --- сохранение ---
        user_msg_id = await self.store.add_message(user_id, "user", history_text, at=ts)
        # Реакция хранится в нашем же формате <react>, чтобы модель не училась писать «реакция» текстом.
        stored_reply = "\n\n---\n\n".join(parsed.parts)
        if parsed.reaction:
            stored_reply = f"<react>{parsed.reaction}</react>" + ("\n" + stored_reply if stored_reply else "")
        await self.store.add_message(user_id, "assistant", stored_reply, at=ts + 0.001)
        if new_session:
            user.sessions += 1
        await self.store.update_user(
            user_id, last_user_msg_at=ts, last_bot_msg_at=ts, register=user.register, sessions=user.sessions,
            display_name=display_name or user.display_name,
        )
        await self.store.mark_initiatives_answered(user_id)
        await self.store.save_emotion(user_id, emo.values, emo.mood, emo.last_cause, ts)
        await self.store.save_relationship(user_id, rel)

        images = [p for pc in percepts for p in pc.parts if isinstance(p, ImagePart)]
        self.spawn(self._after_turn(user_id, user_msg_id, history_text, user_text, stored_reply,
                                    appr.emotion_events + appr.relationship_events, images, ts))
        return Reply(parts=parsed.parts, reaction=parsed.reaction, cost_usd=cost,
                     quote=any(p.kind == "reply" for p in percepts),
                     debug={"inner_context": inner, "tool_notes": tool_notes, "stage": rel.stage, "emotion": emo_text})

    EVENT_REACTIONS = {
        "gift_or_food": "🤗", "user_shares_good_news": "🔥", "user_surprising_news": "👀", "snow": "☃",
        "thanks": "🕊", "user_shares_sad_news": "❤", "joke_shared": "😁", "compliment": "🙈", "flirt_early": "🙈",
        "thunder": "🤯", "mechanism_or_animal": "👀", "sarcasm": "🌚",
    }

    async def _event_reaction(self, user_id: int, events: list[str]) -> str | None:
        """Реакция, когда в сообщении что-то происходит, — но не на каждое сообщение подряд."""
        emoji = next((self.EVENT_REACTIONS[e] for e in events if e in self.EVENT_REACTIONS), None)
        if not emoji:
            return None
        recent = await self.store.last_messages(user_id, 4, role="assistant")
        if any(m.text.startswith("<react>") for m in recent[-2:]):
            return None
        return emoji

    @staticmethod
    def _wants_lullaby(conditions: set[str], sit: Situation) -> bool:
        return "sing_request" in conditions or ("user_cant_sleep" in conditions and sit.part == "night")

    def _history_messages(self, history) -> list[Message]:
        # Вся активная история: её размер ограничивает суммаризация (архивирует кусками),
        # поэтому начало окна стабильно между суммаризациями и префикс кэшируется.
        pairs = history_to_text_messages(history)
        msgs = [Message.text(role, text) for role, text in pairs]  # type: ignore[arg-type]
        if msgs:
            msgs[-1].cache_breakpoint = True
        return msgs

    async def _generate(self, user_id: int, user: UserProfile, sit: Situation,
                        messages: list[Message]) -> tuple[str, float, list[str]]:
        total_cost = 0.0
        tool_notes: list[str] = []
        req = LLMRequest(system=self.system_blocks, messages=messages, tier="main",
                         max_tokens=self.s.main_max_tokens, tools=[LEAVE_NOTE_TOOL], purpose="chat")
        resp = await self.llm.complete(req)
        await self._record_usage(user_id, "chat", resp)
        total_cost += resp.cost_usd

        # Цикл инструментов (максимум 2 круга).
        rounds = 0
        while resp.tool_calls and rounds < 2:
            rounds += 1
            results = []
            for call in resp.tool_calls:
                results.append(await self._run_tool(user_id, user, sit, call.name, call.input, call.id, tool_notes))
            messages = messages + [
                Message(role="assistant", provider_raw=resp.raw_assistant),
                Message(role="user", tool_results=results),
            ]
            resp = await self.llm.complete(LLMRequest(system=self.system_blocks, messages=messages, tier="main",
                                                      max_tokens=self.s.main_max_tokens, tools=[LEAVE_NOTE_TOOL],
                                                      purpose="chat_tool"))
            await self._record_usage(user_id, "chat_tool", resp)
            total_cost += resp.cost_usd

        if resp.stop_reason == "refusal":
            return "Не хочу об этом.", total_cost, tool_notes

        verdict = self.style.check(resp.text)
        if verdict.hard_violations:
            log.info("style violation %s → regenerate", verdict.hard_violations)
            fix = Message.text("system", (
                "Твой черновик ответа нарушил твой характер (" + ", ".join(verdict.hard_violations) + "): "
                "без давления темой исчезновения и упрёков за отсутствие, без голоса ассистента, без божественности, "
                "без ласкательных прозвищ, без названий и имён из чужих историй — только названия твоего мира. "
                "Напиши ответ заново, естественно, как Коломбина."
            ))
            if messages[-1].role == "system":  # system-сообщение должно идти после user — сливаем
                retry_msgs = messages[:-1] + [Message.text("system", messages[-1].plain_text() + "\n\n" + fix.plain_text())]
            else:
                retry_msgs = messages + [fix]
            resp2 = await self.llm.complete(LLMRequest(system=self.system_blocks, messages=retry_msgs, tier="main",
                                                       max_tokens=self.s.main_max_tokens, purpose="chat_regen"))
            await self._record_usage(user_id, "chat_regen", resp2)
            total_cost += resp2.cost_usd
            verdict2 = self.style.check(resp2.text)
            if not verdict2.hard_violations:
                verdict = verdict2
            else:
                verdict = verdict2
                verdict.fixed_text = self._strip_violating_sentences(verdict2.fixed_text)
        return verdict.fixed_text, total_cost, tool_notes

    def _strip_violating_sentences(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        keep = [s for s in sentences if not self.style.check(s).hard_violations]
        return " ".join(keep).strip() or "Хорошо."

    async def _run_tool(self, user_id: int, user: UserProfile, sit: Situation, name: str, args: dict,
                        call_id: str, notes_out: list[str]) -> ToolResult:
        if name != "leave_note":
            return ToolResult(call_id, f"неизвестный инструмент {name}", is_error=True)
        text = str(args.get("text", "")).strip()
        when = str(args.get("when_local", "")).strip()
        try:
            tz = ZoneInfo(sit.tz_name)
            local = datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        except Exception:
            return ToolResult(call_id, "не удалось понять время; нужен формат YYYY-MM-DD HH:MM", is_error=True)
        due = local.astimezone(timezone.utc)
        if due <= sit.now_utc + timedelta(minutes=1):
            return ToolResult(call_id, "это время уже прошло", is_error=True)
        if due > sit.now_utc + timedelta(days=366):
            return ToolResult(call_id, "слишком далеко — больше года", is_error=True)
        if not text:
            return ToolResult(call_id, "пустая записка", is_error=True)
        await self.store.add_note(user_id, text, due.timestamp())
        notes_out.append(f"{when}: {text}")
        return ToolResult(call_id, f"записка оставлена на {when} (локальное время собеседника)")

    # ------------------------------------------------------------------ background
    async def _after_turn(self, user_id: int, user_msg_id: int, history_text: str, user_text: str, reply: str,
                          applied: list[str], images: list[ImagePart], ts: float) -> None:
        try:
            await self._reflect(user_id, user_msg_id, history_text, user_text, reply, applied, images, ts)
        except Exception:
            log.exception("reflection failed")
        try:
            await self.maybe_summarize(user_id)
        except Exception:
            log.exception("summarization failed")

    async def _reflect(self, user_id: int, user_msg_id: int, history_text: str, user_text: str, reply: str,
                       applied: list[str], images: list[ImagePart], ts: float) -> None:
        facts = await self.store.facts(user_id)
        threads = await self.store.open_threads(user_id)
        result, resp = await self.reflector.reflect(
            history_text if not user_text else user_text, reply,
            [f"{f.key}: {f.value}" for f in facts[:20]], threads, applied, images,
        )
        await self._record_usage(user_id, "reflection", resp)
        async with self.lock_for(user_id):  # запись памяти не пересекается с ходом разговора
            await self._apply_reflection(user_id, user_msg_id, history_text, user_text, applied, threads, result, ts)

    async def _apply_reflection(self, user_id, user_msg_id, history_text, user_text, applied, threads, result, ts):
        await self._apply_location(user_id, result)
        # Эмоции и доверие (фоновые события — после ответа, на следующий ход)
        row = await self.store.get_emotion(user_id)
        if row:
            vals, mood, cause, at = row
            emo = EmotionState(vals, mood, cause, at)
            for ev in result.emotion_events:
                if ev not in applied:
                    emo = self.emotions.apply(emo, ev, cause=ev)
            await self.store.save_emotion(user_id, emo.values, emo.mood, emo.last_cause, at)
        rel = await self.store.get_relationship(user_id)
        rel_events = [e for e in result.relationship_events if e not in applied]
        if not any(self.relationship.events.get(e, 0) < 0 for e in rel_events + applied) and len(user_text) > 40:
            rel_events.append("quality_conversation")
        for ev in rel_events:
            rel = self.relationship.apply(rel, ev)
            await self.store.log_event(user_id, ev, "reflection", user_text[:120])
        await self.store.save_relationship(user_id, rel)

        # Факты
        for key in result.forget_keys:
            await self.store.delete_fact(user_id, key)
        if result.facts:
            vecs = await self.embedder.embed([f"{f['key']}: {f['value']}" for f in result.facts])
            for f, v in zip(result.facts, vecs):
                conf = max(0.0, min(1.0, float(f.get("confidence", 0.7))))
                if f.get("certain") is False:  # догадка, а не сказанное прямо
                    conf = min(conf, 0.45)
                pinned = f["key"].strip().lower() in {"имя", "как называть"}
                await self.store.upsert_fact(user_id, f["key"], f["value"], conf, v, pinned=pinned)

        # Эпизод
        if result.episode:
            ep = result.episode
            vec = (await self.embedder.embed([ep["text"]]))[0]
            await self.store.add_episode(user_id, ep["text"], max(1, min(5, int(ep.get("importance", 2)))),
                                         ep.get("emotion") or None, "shared", vec, at=ts)

        if result.her_life:
            vec = (await self.embedder.embed([result.her_life]))[0]
            await self.store.add_episode(user_id, result.her_life, 1, None, "her_life", vec, at=ts)

        # Темы
        for tid in result.closed_thread_ids:
            if any(t["id"] == tid for t in threads):
                await self.store.close_thread(tid)
        for t in result.open_threads:
            hours = int(t.get("follow_up_in_hours", 0))
            due = ts + hours * 3600 if hours > 0 else None
            await self.store.add_thread(user_id, t["topic"], due)

        # Описание медиа → в историю, чтобы потом помнить, что было на фото
        if result.media_description and ("[фото" in history_text or "[видео" in history_text):
            new_text = history_text.replace("[фото]", f"[фото: {result.media_description}]", 1)
            if new_text == history_text:
                new_text = history_text + f" (на кадрах: {result.media_description})"
            await self.store.update_message_text(user_msg_id, new_text)

    async def _apply_location(self, user_id: int, result) -> None:
        """Часовой пояс собеседника из разговора («я в Новосибирске», «у меня 3 ночи»)."""
        user = await self.store.get_user(user_id)
        if user is None or user.timezone:
            return
        tz = (result.user_timezone or "").strip()
        if tz:
            try:
                ZoneInfo(tz)
                await self.store.update_user(user_id, timezone=tz)
                return
            except Exception:
                pass
        city = (result.user_city or "").strip()
        if city and self.weather:
            try:
                geo = await self.weather.geocode(city)
            except Exception:
                geo = None
            if geo:
                lat, lon, name, gtz = geo
                await self.store.update_user(user_id, city=name, lat=lat, lon=lon,
                                             **({"timezone": gtz} if gtz else {}))

    async def maybe_summarize(self, user_id: int) -> None:
        msgs = await self.store.active_messages(user_id)
        if len(msgs) < self.s.history_summarize_at:
            return
        cut = msgs[: len(msgs) - self.s.history_keep_messages]
        if not cut:
            return
        transcript = "\n".join(("Собеседник: " if m.role == "user" else "Коломбина: ") + m.text for m in cut)
        prev = await self.store.get_summary(user_id)
        text, resp = await self.reflector.summarize(prev, transcript)
        await self._record_usage(user_id, "summary", resp)
        if text:
            await self.store.set_summary(user_id, text, cut[-1].id)
            await self.store.archive_messages_upto(user_id, cut[-1].id)

    # ------------------------------------------------------------------ initiative generation
    async def compose_initiative(self, user: UserProfile, reason: str, kind: str,
                                 now: datetime | None = None) -> Reply | None:
        """Сгенерировать сообщение, которое она пишет первой. Решение «писать ли» — в InitiativeEngine."""
        async with self.lock_for(user.user_id):
            sit = await self.situation_for(user, now)
            ts = sit.now_utc.timestamp()
            if await self.over_budget(user.user_id):
                return None
            emo = await self.load_emotion(user.user_id, sit)
            rel = await self.store.get_relationship(user.user_id)
            history = await self.store.active_messages(user.user_id)
            recall = await self.retriever.recall(user.user_id, reason, rel.stage, self.persona.knowledge,
                                                 top_k=3, now=ts)
            emo_text, manners = self.emotions.describe(emo, sit.tags)
            ctx = TurnContext(
                situation=sit, user=user, stage=rel.stage, stage_text=self.relationship.describe(rel),
                emotion_text=emo_text, emotion_manners=manners, recall=recall,
                summary=await self.store.get_summary(user.user_id), appraisal=appraise(""), abilities=[],
                perception_text="", open_threads=[], pending_notes=[],
                style_hints=self.style.overused([m.text for m in history if m.role == "assistant"][-6:]),
                initiative_reason=(
                    "Ты сама пишешь первой. Повод: " + reason + ". Одно короткое сообщение (1–3 фразы), "
                    "лёгкое, без давления; на него не обязательно отвечать. Не извиняйся за беспокойство, "
                    "не упоминай, что давно не говорили, если повод не в этом. Не задавай больше одного вопроса."
                ),
            )
            inner = render_inner_context(ctx, self.s.context_budget_tokens)
            messages = self._history_messages(history)
            messages.append(Message.text("user", "(собеседник молчит; ты решаешь написать сама)"))
            messages.append(Message.text("system", inner))
            try:
                resp = await self.llm.complete(LLMRequest(system=self.system_blocks, messages=messages, tier="main",
                                                          max_tokens=self.s.main_max_tokens, purpose="initiative"))
            except LLMError:
                return None
            await self._record_usage(user.user_id, "initiative", resp)
            verdict = self.style.check(resp.text)
            if verdict.hard_violations or not verdict.fixed_text.strip():
                return None  # лучше промолчать, чем написать не по-своему
            parsed = parse_reply(verdict.fixed_text)
            if not parsed.parts:
                return None
            text = "\n\n".join(parsed.parts)
            await self.store.add_message(user.user_id, "assistant", text, kind="initiative", at=ts)
            await self.store.update_user(user.user_id, last_bot_msg_at=ts)
            await self.store.log_initiative(user.user_id, kind, reason, text)
            return Reply(parts=parsed.parts, cost_usd=resp.cost_usd)
