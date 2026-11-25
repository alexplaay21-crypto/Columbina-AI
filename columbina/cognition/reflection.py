"""Фоновая рефлексия после ответа — дешёвая модель, один запрос на обмен репликами.

Извлекает: факты о собеседнике, эпизод для памяти, эмоциональные и отношенческие события
(только из фиксированных списков), незакрытые темы для инициативы, описание медиа для истории.
Плюс скользящая суммаризация старого диалога.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import ImagePart, LLMProvider, LLMRequest, Message, SystemBlock, TextPart

log = logging.getLogger(__name__)

REFLECTION_SYSTEM = """Ты — внутренний наблюдатель Коломбины, девы из лунного света, которая дружит с собеседником в Telegram.
Тебе показывают последний обмен репликами и немного контекста. Верни JSON строго по схеме.

Правила:
- facts: только устойчивые факты о собеседнике, которые он сообщил сам или которые прямо следуют из сказанного
  (имя, как его называть, город, работа/учёба, близкие, питомцы, вкусы, важные даты, планы). Ключ — короткий,
  по-русски, в нижнем регистре («имя», «питомец», «работа», «день рождения»). Не выдумывай. Если ничего нового — [].
- forget_keys: ключи фактов, которые собеседник просит забыть или которые оказались неверны.
- episode: стоит ли запомнить этот момент как эпизод общей истории. importance 1–5
  (5 — очень важное: горе, большая радость, признание; 1 — мелочь). Текст — одна-две фразы от третьего лица,
  с конкретикой («Рассказал, что завтра собеседование в банке; волнуется»).
- emotion_events и relationship_events: выбирай только из разрешённых значений и только если это явно есть
  в обмене. Не повторяй уже учтённые события (они перечислены).
- open_threads: незавершённые темы, к которым естественно вернуться позже (собеседование завтра, болел,
  обещал рассказать). follow_up_in_hours — через сколько часов уместно спросить (0 — не нужно напоминать).
- closed_thread_ids: id тем из списка открытых, которые в этом обмене закрылись.
- media_description: если в сообщении было фото/кадры видео — краткое нейтральное описание (до 25 слов)
  того, что на них видно. Иначе пустая строка.
- user_mood: одно-два слова о настроении собеседника, или пустая строка.
- user_timezone: если из слов собеседника ясно, где он живёт или сколько у него сейчас времени, — часовой пояс
  IANA («Asia/Novosibirsk», «Europe/Moscow»). Иначе пустая строка. user_city — город, если назван.
- facts[].certain: true — собеседник сказал прямо; false — это твоя догадка.
- topics: 0–3 коротких темы разговора, которые могут стать интересом (например «механические часы», «Исландия»).
- her_life: если Коломбина рассказала о событии в своей собственной жизни (что делала, что случилось с духами,
  неудачный пирог) — коротко зафиксируй это одной фразой, чтобы она потом себе не противоречила. Иначе пустая строка.
"""

SUMMARY_SYSTEM = """Ты ведёшь память Коломбины о разговорах с одним собеседником.
Тебе дают прежнее резюме и следующий кусок переписки. Напиши обновлённое резюме: до 180 слов, по-русски,
в третьем лице, только существенное (события, темы, договорённости, шутки, которые стали «общими», перемены
в отношениях). Без оценок и без пересказа реплик подряд. Старое, ставшее неважным, можно сжать."""


def reflection_schema(emotion_events: list[str], relationship_events: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["facts", "forget_keys", "episode", "emotion_events", "relationship_events",
                     "open_threads", "closed_thread_ids", "media_description", "user_mood", "her_life",
                     "user_timezone", "user_city", "topics"],
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["key", "value", "confidence", "certain"],
                    "properties": {"key": {"type": "string"}, "value": {"type": "string"},
                                   "confidence": {"type": "number"}, "certain": {"type": "boolean"}},
                },
            },
            "forget_keys": {"type": "array", "items": {"type": "string"}},
            "episode": {
                "type": "object", "additionalProperties": False,
                "required": ["worth_remembering", "text", "importance", "emotion"],
                "properties": {"worth_remembering": {"type": "boolean"}, "text": {"type": "string"},
                               "importance": {"type": "integer"}, "emotion": {"type": "string"}},
            },
            "emotion_events": {"type": "array", "items": {"type": "string", "enum": emotion_events}},
            "relationship_events": {"type": "array", "items": {"type": "string", "enum": relationship_events}},
            "open_threads": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["topic", "follow_up_in_hours"],
                    "properties": {"topic": {"type": "string"}, "follow_up_in_hours": {"type": "integer"}},
                },
            },
            "closed_thread_ids": {"type": "array", "items": {"type": "integer"}},
            "media_description": {"type": "string"},
            "user_mood": {"type": "string"},
            "her_life": {"type": "string"},
            "user_timezone": {"type": "string"},
            "user_city": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
        },
    }


@dataclass
class ReflectionResult:
    facts: list[dict[str, Any]] = field(default_factory=list)
    forget_keys: list[str] = field(default_factory=list)
    episode: dict[str, Any] | None = None
    emotion_events: list[str] = field(default_factory=list)
    relationship_events: list[str] = field(default_factory=list)
    open_threads: list[dict[str, Any]] = field(default_factory=list)
    closed_thread_ids: list[int] = field(default_factory=list)
    media_description: str = ""
    user_mood: str = ""
    her_life: str = ""
    user_timezone: str = ""
    user_city: str = ""
    topics: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: str) -> "ReflectionResult":
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return cls()
        ep = data.get("episode") or {}
        return cls(
            facts=[f for f in data.get("facts", []) if f.get("key") and f.get("value")][:6],
            forget_keys=data.get("forget_keys", [])[:5],
            episode=ep if ep.get("worth_remembering") and ep.get("text") else None,
            emotion_events=data.get("emotion_events", [])[:4],
            relationship_events=data.get("relationship_events", [])[:4],
            open_threads=[t for t in data.get("open_threads", []) if t.get("topic")][:3],
            closed_thread_ids=data.get("closed_thread_ids", [])[:5],
            media_description=(data.get("media_description") or "")[:300],
            user_mood=(data.get("user_mood") or "")[:40],
            her_life=(data.get("her_life") or "")[:300],
            user_timezone=(data.get("user_timezone") or "")[:60],
            user_city=(data.get("user_city") or "")[:80],
            topics=[t for t in data.get("topics", []) if isinstance(t, str)][:3],
        )


class Reflector:
    def __init__(self, llm: LLMProvider, emotion_events: list[str], relationship_events: list[str]):
        self.llm = llm
        self.schema = reflection_schema(sorted(emotion_events), sorted(relationship_events))

    async def reflect(
        self,
        user_text: str,
        reply_text: str,
        known_facts: list[str],
        open_threads: list[dict[str, Any]],
        already_applied: list[str],
        images: list[ImagePart],
    ) -> tuple[ReflectionResult, Any]:
        threads = "\n".join(f"{t['id']}: {t['topic']}" for t in open_threads) or "нет"
        body = (
            f"Известные факты: {'; '.join(known_facts) or 'нет'}\n"
            f"Открытые темы:\n{threads}\n"
            f"Уже учтённые события: {', '.join(already_applied) or 'нет'}\n\n"
            f"Собеседник:\n{user_text}\n\nКоломбина:\n{reply_text}"
        )
        parts = [*images[:4], TextPart(body)]
        resp = await self.llm.complete(LLMRequest(
            system=[SystemBlock(REFLECTION_SYSTEM, cache=True)],
            messages=[Message(role="user", parts=parts)],
            tier="background", max_tokens=1500, effort="low", json_schema=self.schema, purpose="reflection",
        ))
        return ReflectionResult.parse(resp.text), resp

    async def summarize(self, previous: str, transcript: str) -> tuple[str, Any]:
        resp = await self.llm.complete(LLMRequest(
            system=[SystemBlock(SUMMARY_SYSTEM, cache=True)],
            messages=[Message.text("user", f"Прежнее резюме:\n{previous or '(пусто)'}\n\nНовый кусок:\n{transcript}")],
            tier="background", max_tokens=1200, effort="low", purpose="summary",
        ))
        return resp.text.strip(), resp
