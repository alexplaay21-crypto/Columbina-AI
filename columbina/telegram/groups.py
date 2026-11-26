"""Групповые чаты: у группы своя память и характер общения, Коломбина различает, кто говорит.

Режимы (настраивают админы группы):
  mention — отвечает, только когда к ней обращаются (@бот, «Коломбина», ответ на её сообщение). По умолчанию.
  active  — ещё иногда вступает сама: на вопрос в чат по её интересам, не чаще раза в 15 минут.
  all     — отвечает на всё (для маленьких групп).
Всё, что пишут в группе, она «слышит» и помнит как контекст (без запросов к модели), а отвечает в лимите.
Личное из бесед один на один в группе не раскрывает: у группы отдельная память.
"""

from __future__ import annotations

import time

from aiogram.types import Message

from ..memory.embeddings import stems
from ..perception.model import Percept

MODES = ("mention", "active", "all")
NAME_WORDS = ("коломбин", "kuutar", "куутар", "columbina")


def speaker(m: Message) -> str:
    u = m.from_user
    if not u:
        return "кто-то"
    return (u.first_name or u.username or "кто-то")[:40]


class GroupLogic:
    def __init__(self, engine, max_replies_10min: int = 12):
        self.engine = engine
        self.store = engine.store
        self.max_replies = max_replies_10min

    async def mode(self, chat_id: int) -> str:
        m = await self.store.get_setting(chat_id, "group_mode", "mention")
        return m if m in MODES else "mention"

    async def addressed(self, m: Message, bot_id: int, bot_username: str | None) -> bool:
        if m.reply_to_message and m.reply_to_message.from_user and m.reply_to_message.from_user.id == bot_id:
            return True
        text = (m.text or m.caption or "").lower()
        if bot_username and f"@{bot_username.lower()}" in text:
            return True
        return any(w in text for w in NAME_WORDS)

    async def _her_recent(self, chat_id: int, seconds: float) -> int:
        msgs = await self.store.last_messages(chat_id, 30, role="assistant")
        return sum(1 for x in msgs if time.time() - x.created_at < seconds)

    async def should_reply(self, m: Message, bot_id: int, bot_username: str | None) -> bool:
        if await self._her_recent(m.chat.id, 600) >= self.max_replies:
            return False  # лимит ответов в группе — и токены целы, и чат не заспамлен
        if await self.addressed(m, bot_id, bot_username):
            return True
        mode = await self.mode(m.chat.id)
        if mode == "all":
            return True
        if mode == "active":
            text = (m.text or "").lower()
            if "?" not in text or await self._her_recent(m.chat.id, 900) > 0:
                return False
            return any(k in text for k in await self._interest_roots(m.chat.id))
        return False

    BASE_ROOTS = ("лун", "косм", "звёзд", "звезд", "снег", "ноч", "живот", "кошк", "кот ", "собак", "ягод", "хлеб",
                  "истори", "наук", "колыбел", "сон ", "сны", "мечт", "чай")

    async def _interest_roots(self, chat_id: int) -> set[str]:
        roots = set(self.BASE_ROOTS)
        for i in await self.store.interests(chat_id):
            roots |= {w[:5] for w in stems(i["topic"]) if len(w) > 3}
        return roots

    async def remember_context(self, m: Message) -> None:
        """То, что в группе сказали не ей, — тоже часть разговора (сохраняем без вызова модели)."""
        text = m.text or m.caption or ""
        if not text and (m.photo or m.sticker or m.voice or m.video):
            text = "[" + ("фото" if m.photo else "стикер" if m.sticker else "голосовое" if m.voice else "видео") + "]"
        if text:
            await self.store.get_or_create_user(m.chat.id, m.chat.id, m.chat.title)
            await self.store.add_message(m.chat.id, "user", f"[{speaker(m)}]: {text[:600]}")

    @staticmethod
    def frame(percepts: list[Percept], m: Message) -> list[Percept]:
        """Подписывает, кто говорит, и добавляет заметку о группе для внутреннего контекста."""
        who = speaker(m)
        for p in percepts:
            if p.text:
                p.text = f"[{who}]: {p.text}"
            p.history = f"[{who}]: {p.history}" if p.history else p.history
        note = (f"групповой чат «{m.chat.title or 'группа'}»; сейчас к тебе обращается {who}. Тут несколько людей — "
                "отвечай тому, кто писал, держись чуть сдержаннее, чем наедине, не раскрывай личного из бесед один на один")
        return [Percept("group", True, "text", note, history="")] + percepts

    @staticmethod
    def greeting(title: str | None) -> str:
        return ("Здравствуйте. Я Коломбина. Буду тихо слушать — позовите по имени или ответьте на моё сообщение, "
                "если захотите поговорить.")

    @staticmethod
    def welcome(names: list[str]) -> str:
        return f"{', '.join(names)}, здравствуй{'те' if len(names) > 1 else ''}. Я Коломбина. Не пугайтесь, если я иногда молчу."
