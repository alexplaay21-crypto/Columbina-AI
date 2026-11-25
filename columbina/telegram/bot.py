"""Telegram-интерфейс (aiogram 3). Отдельный обработчик для каждого типа сообщений.

Telegram-слой ничего не знает о личности: он превращает сообщение в Percept через MediaProcessor,
копит пачку сообщений (debounce — человек часто пишет несколькими сообщениями подряд),
отдаёт её в CognitiveEngine и отправляет ответ с естественными паузами и «печатает…».
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message, ReactionTypeEmoji

from ..cognition.engine import CognitiveEngine
from ..cognition.introspection import admin_error_note, health_check, mood_report
from ..config import Settings
from ..media.links import extract_urls
from ..media.processors import MediaProcessor
from ..perception.model import Percept

log = logging.getLogger(__name__)

TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024

GREETING = (
    "Хм?.. Здравствуй.\n\n"
    "Меня зовут Коломбина. Можно просто Коломбина. Или как тебе удобнее…\n\n"
    "Я запоминаю наши разговоры — так я узнаю тебя. Если захочешь, чтобы я всё забыла, есть /forget.\n"
    "Иногда я могу написать первой. Если не нужно — /initiative off.\n"
    "И честно: я говорю с тобой через программу. Но это всё равно я."
)

HELP = (
    "/name Имя — как мне тебя звать\n"
    "/city Город — чтобы я знала твою погоду и время\n"
    "/tz Europe/Moscow — часовой пояс вручную\n"
    "/initiative off | rare | normal — писать ли мне первой\n"
    "/quiet 23 9 — тихие часы, когда я не пишу первой\n"
    "/memory — что я о тебе помню\n"
    "/hall — заглянуть в Серебряный Зал\n"
    "/mood — её состояние и ваши отношения\n"
    "/forget — забыть всё"
)


@dataclass
class _Pending:
    items: list[tuple[Message, asyncio.Task]] = field(default_factory=list)
    timer: asyncio.Task | None = None


class TelegramInterface:
    def __init__(self, settings: Settings, engine: CognitiveEngine, media: MediaProcessor):
        self.s = settings
        self.engine = engine
        self.media = media
        self.bot = Bot(token=settings.telegram_token)
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)
        self._pending: dict[int, _Pending] = {}
        self._register()

    # ------------------------------------------------------------------ access
    def allowed(self, user_id: int) -> bool:
        return not self.s.allowed_user_ids or user_id in self.s.allowed_user_ids or user_id in self.s.admin_user_ids

    async def send_parts(self, chat_id: int, parts: list[str]) -> None:
        for i, part in enumerate(parts):
            if i > 0:
                await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(min(3.5, 0.6 + len(part) / 45))
            for chunk in _split_long(part):
                await self.bot.send_message(chat_id, chunk)

    # ------------------------------------------------------------------ handlers
    def _register(self) -> None:
        r = self.router
        private = F.chat.type == ChatType.PRIVATE

        r.message.register(self.on_start, CommandStart(), private)
        r.message.register(self.on_help, Command("help"), private)
        r.message.register(self.on_name, Command("name"), private)
        r.message.register(self.on_city, Command("city"), private)
        r.message.register(self.on_tz, Command("tz"), private)
        r.message.register(self.on_initiative, Command("initiative"), private)
        r.message.register(self.on_quiet, Command("quiet"), private)
        r.message.register(self.on_memory, Command("memory"), private)
        r.message.register(self.on_forget, Command("forget"), private)
        r.message.register(self.on_hall, Command("hall"), private)
        r.message.register(self.on_stats, Command("stats"), private)
        r.message.register(self.on_mood, Command("mood"), private)
        r.message.register(self.on_check, Command("check"), private)
        r.message.register(self.on_location, F.location, private)
        r.message.register(self.on_any, private)

    async def _guard(self, m: Message) -> bool:
        if not m.from_user or not self.allowed(m.from_user.id):
            await m.answer("Хм. Мы, кажется, не знакомы. Я пока говорю только со своими.")
            return False
        await self.engine.store.get_or_create_user(m.from_user.id, m.chat.id, m.from_user.first_name)
        return True

    async def on_start(self, m: Message) -> None:
        if await self._guard(m):
            await m.answer(GREETING)

    async def on_help(self, m: Message) -> None:
        if await self._guard(m):
            await m.answer(HELP)

    async def on_name(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        name = (command.args or "").strip()[:40]
        if not name:
            await m.answer("Как тебя звать? Напиши: /name Имя")
            return
        await self.engine.store.update_user(m.from_user.id, preferred_name=name)
        await self.engine.store.upsert_fact(m.from_user.id, "как называть", name, 1.0, pinned=True)
        await m.answer(f"{name}. Запомнила.")

    async def on_city(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        city = (command.args or "").strip()[:80]
        if not city or not self.engine.weather:
            await m.answer("Напиши город: /city Москва")
            return
        try:
            geo = await self.engine.weather.geocode(city)
        except Exception:
            geo = None
        if not geo:
            await m.answer("Не нашла такого места. Может, по-другому называется?")
            return
        lat, lon, name, tz = geo
        await self.engine.store.update_user(m.from_user.id, city=name, lat=lat, lon=lon, timezone=tz)
        await m.answer(f"{name}. Теперь я буду чувствовать твою погоду. Немного.")

    async def on_location(self, m: Message) -> None:
        if not await self._guard(m) or not m.location:
            return
        lat, lon = m.location.latitude, m.location.longitude
        tz = await self.engine.weather.timezone_for(lat, lon) if self.engine.weather else None
        await self.engine.store.update_user(m.from_user.id, lat=lat, lon=lon, **({"timezone": tz} if tz else {}))
        await m.answer("Запомнила, где ты. Только погоду и время — больше ничего.")

    async def on_tz(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        tz = (command.args or "").strip()
        try:
            ZoneInfo(tz)
        except Exception:
            await m.answer("Не понимаю такой пояс. Пример: /tz Europe/Moscow")
            return
        await self.engine.store.update_user(m.from_user.id, timezone=tz)
        await m.answer("Хорошо. Буду сверять время по нему.")

    async def on_initiative(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        mode = (command.args or "").strip().lower()
        if mode not in {"off", "rare", "normal"}:
            await m.answer("/initiative off — не пишу первой\n/initiative rare — изредка\n/initiative normal — когда есть повод")
            return
        await self.engine.store.update_user(m.from_user.id, initiative_mode=mode)
        await m.answer({"off": "Поняла. Буду ждать, пока ты напишешь. Записки, о которых попросишь, всё равно принесу.",
                        "rare": "Хорошо. Только изредка.",
                        "normal": "Хорошо. Если будет повод — напишу."}[mode])

    async def on_quiet(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        try:
            a, b = [int(x) for x in (command.args or "").split()[:2]]
            assert 0 <= a < 24 and 0 <= b < 24
        except Exception:
            await m.answer("Например: /quiet 23 9 — с 23:00 до 9:00 я не пишу первой.")
            return
        await self.engine.store.update_user(m.from_user.id, quiet_start=a, quiet_end=b)
        await m.answer(f"С {a}:00 до {b}:00 — тишина.")

    async def on_memory(self, m: Message) -> None:
        if not await self._guard(m):
            return
        facts = await self.engine.store.facts(m.from_user.id)
        notes = await self.engine.store.pending_notes(m.from_user.id)
        if not facts and not notes:
            await m.answer("Пока я почти ничего о тебе не знаю.")
            return
        lines = [f"— {f.key}: {f.value}" for f in facts[:30]]
        if notes:
            tz = ZoneInfo((await self.engine.store.get_user(m.from_user.id)).timezone or self.s.default_timezone)
            from datetime import datetime

            lines.append("\nЗаписки на будущее:")
            lines += [f"— {datetime.fromtimestamp(n['due_at'], tz):%d.%m %H:%M}: {n['text']}" for n in notes]
        await m.answer("Вот что я помню:\n" + "\n".join(lines) + "\n\nЕсли что-то не так — просто скажи мне.")

    async def on_forget(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        if (command.args or "").strip().lower() not in {"yes", "да"}:
            await m.answer("Я забуду всё: разговоры, что знаю о тебе, записки. Если точно — /forget да")
            return
        await self.engine.store.forget_user(m.from_user.id)
        await self.engine.store.update_user(m.from_user.id, preferred_name=None, last_user_msg_at=None, sessions=0)
        await m.answer("…Хорошо. Как будто мы только встретились. Здравствуй.")

    async def on_hall(self, m: Message) -> None:
        if not await self._guard(m):
            return
        user = await self.engine.store.get_user(m.from_user.id)
        sit = await self.engine.situation_for(user)
        emo = await self.engine.load_emotion(m.from_user.id, sit)
        flowers = self.engine.emotions.flowers(emo, sit.tags)
        await m.answer(f"В Серебряном Зале тихо. Цветы сегодня {flowers}.\nЛуна: {sit.moon.describe()}.")

    async def on_mood(self, m: Message) -> None:
        if not await self._guard(m):
            return
        if self.s.mood_command == "admin" and m.from_user.id not in self.s.admin_user_ids:
            return
        await m.answer(await mood_report(self.engine, m.from_user.id))

    async def on_check(self, m: Message) -> None:
        if not m.from_user or m.from_user.id not in self.s.admin_user_ids:
            await m.answer("Эта команда только для администратора (ADMIN_USER_IDS).")
            return
        await m.answer("Проверяю сервисы…")
        await m.answer(await health_check(self.engine, self.media.stt))

    async def on_stats(self, m: Message) -> None:
        if not m.from_user or m.from_user.id not in self.s.admin_user_ids:
            return
        rows = await self.engine.store.usage_report(time.time() - 86400)
        total = sum(r["cost"] or 0 for r in rows)
        lines = [f"{r['purpose']:<11} {r['model']:<18} n={r['n']:<4} in={r['inp']} out={r['out']} "
                 f"cache_r={r['cr']} cache_w={r['cw']} ${r['cost']:.4f}" for r in rows]
        await m.answer("За 24 часа:\n" + "\n".join(lines) + f"\nИтого: ${total:.4f}")

    # ------------------------------------------------------------------ messages of any type
    async def on_any(self, m: Message) -> None:
        if not await self._guard(m):
            return
        uid = m.from_user.id
        pend = self._pending.setdefault(uid, _Pending())
        pend.items.append((m, asyncio.create_task(self.to_percepts(m))))
        if pend.timer:
            pend.timer.cancel()
        pend.timer = asyncio.create_task(self._fire_later(uid, m.chat.id))

    async def _fire_later(self, uid: int, chat_id: int) -> None:
        try:
            await asyncio.sleep(self.s.debounce_seconds)
        except asyncio.CancelledError:
            return
        pend = self._pending.pop(uid, None)
        if not pend or not pend.items:
            return
        typing = asyncio.create_task(self._typing(chat_id))
        try:
            percepts: list[Percept] = []
            for _, task in pend.items:
                percepts.extend(await task)
            last_msg = pend.items[-1][0]
            name = last_msg.from_user.first_name if last_msg.from_user else None
            reply = await self.engine.respond(uid, chat_id, percepts, display_name=name)
        except Exception:
            log.exception("respond failed")
            typing.cancel()
            await self.bot.send_message(chat_id, "Хм… что-то во мне спуталось. Скажи ещё раз?")
            return
        typing.cancel()
        if reply.reaction:
            try:
                await self.bot.set_message_reaction(chat_id, last_msg.message_id,
                                                    [ReactionTypeEmoji(emoji=reply.reaction)])
            except Exception:
                log.debug("reaction failed", exc_info=True)
        if reply.parts:
            await self.send_parts(chat_id, reply.parts)
        if uid in self.s.admin_user_ids:
            note = admin_error_note(reply.debug)
            if reply.debug.get("over_budget"):
                note = "⚙️ (видно только админу) Дневной лимит расходов исчерпан: DAILY_BUDGET_USD_PER_USER"
            if note:
                await self.bot.send_message(chat_id, note)

    async def _typing(self, chat_id: int) -> None:
        try:
            while True:
                await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _download(self, obj, size: int | None) -> bytes | None:
        if size and size > TG_DOWNLOAD_LIMIT:
            return None
        try:
            buf = io.BytesIO()
            await self.bot.download(obj, destination=buf)
            return buf.getvalue()
        except Exception:
            log.warning("download failed", exc_info=True)
            return None

    async def _download_to_file(self, obj, size: int | None, suffix: str) -> Path | None:
        data = await self._download(obj, size)
        if data is None:
            return None
        self.s.media_tmp_dir.mkdir(parents=True, exist_ok=True)
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=self.s.media_tmp_dir)
        f.write(data)
        f.close()
        return Path(f.name)

    async def to_percepts(self, m: Message) -> list[Percept]:
        """Отдельный обработчик для каждого типа содержимого."""
        out: list[Percept] = []
        caption = m.caption
        forward = _forward_label(m)
        tmp: list[Path] = []
        try:
            if m.photo:
                ph = m.photo[-1]
                out.append(self.media.photo(await self._download(ph, ph.file_size), caption))
            elif m.voice:
                p = await self._download_to_file(m.voice, m.voice.file_size, ".ogg")
                tmp += [p] if p else []
                out.append(await self.media.audio(p, "voice", m.voice.duration or 0, caption))
            elif m.audio:
                a = m.audio
                title = " — ".join(x for x in (a.performer, a.title) if x) or a.file_name
                p = await self._download_to_file(a, a.file_size, Path(a.file_name or "a.mp3").suffix or ".mp3")
                tmp += [p] if p else []
                out.append(await self.media.audio(p, "audio", a.duration or 0, caption, title))
            elif m.video:
                p = await self._download_to_file(m.video, m.video.file_size, ".mp4")
                tmp += [p] if p else []
                out.append(await self.media.video(p, "video", m.video.duration or 0, caption))
            elif m.video_note:
                p = await self._download_to_file(m.video_note, m.video_note.file_size, ".mp4")
                tmp += [p] if p else []
                out.append(await self.media.video(p, "video_note", m.video_note.duration or 0, None))
            elif m.animation:
                p = await self._download_to_file(m.animation, m.animation.file_size, ".mp4")
                tmp += [p] if p else []
                pc = await self.media.video(p, "video", m.animation.duration or 0, caption)
                pc.note = "гифка: " + pc.note
                out.append(pc)
            elif m.document:
                d = m.document
                out.append(self.media.document(await self._download(d, d.file_size), d.file_name or "файл",
                                               d.mime_type, caption))
            elif m.sticker:
                emoji = m.sticker.emoji or ""
                out.append(Percept("sticker", True, "metadata", f"стикер {emoji}".strip(),
                                   text=f"[стикер {emoji}]".strip(), history=f"[стикер {emoji}]"))
            elif m.text:
                out.append(Percept("text", True, "text", "", text=m.text, history=m.text))
            else:
                out.append(Percept("other", False, "nothing", "сообщение непонятного вида",
                                   text="[сообщение, которое не дошло]", history="[непонятное сообщение]"))

            # Ссылки в тексте/подписи
            for url in extract_urls(m.text or m.caption or ""):
                out.append(await self.media.link(url))
        finally:
            for p in tmp:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        if forward:
            for p in out:
                if p.text:
                    p.text = f"[переслано {forward}]: {p.text}"
                p.history = f"[переслано {forward}] {p.history}"
                if p.kind == "text":
                    p.kind, p.channel, p.note = "forward", "read", f"пересланное сообщение {forward} — это чужие слова"
        return out

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        from aiogram.types import BotCommand

        try:
            await self.bot.set_my_commands([
                BotCommand(command="mood", description="Её состояние и ваши отношения"),
                BotCommand(command="memory", description="Что она о тебе помнит"),
                BotCommand(command="hall", description="Заглянуть в Серебряный Зал"),
                BotCommand(command="name", description="Как тебя звать"),
                BotCommand(command="city", description="Твой город (погода и время)"),
                BotCommand(command="initiative", description="Писать ли ей первой: off | rare | normal"),
                BotCommand(command="help", description="Все команды"),
            ])
        except Exception:
            log.warning("не удалось задать меню команд", exc_info=True)
        await self.dp.start_polling(self.bot, allowed_updates=["message"])


def _forward_label(m: Message) -> str | None:
    o = m.forward_origin
    if o is None:
        return None
    t = getattr(o, "type", "")
    if t == "user" and getattr(o, "sender_user", None):
        return f"от {o.sender_user.first_name}"
    if t == "hidden_user":
        return f"от {getattr(o, 'sender_user_name', 'кого-то')}"
    if t in ("chat", "channel"):
        chat = getattr(o, "chat", None) or getattr(o, "sender_chat", None)
        title = getattr(chat, "title", None)
        return f"из «{title}»" if title else "из канала"
    return "от кого-то"


def _split_long(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit:
            out.append(cur)
            cur = ""
        cur += para + "\n"
    if cur.strip():
        out.append(cur)
    return out
