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
from ..cognition import introspection as I
from ..cognition.introspection import admin_error_note, health_check, mood_report
from ..config import Settings
from ..media.links import extract_urls
from ..media.processors import MediaProcessor
from ..perception.model import Percept

log = logging.getLogger(__name__)

TG_DOWNLOAD_LIMIT = 20 * 1024 * 1024

GREETING = (
    "Здравствуй.\n\n"
    "Я — Коломбина. Можно просто Коломбина. Или как тебе удобнее…\n\n"
    "Я запоминаю наши разговоры, иногда сама что-то изучаю и могу написать первой. "
    "Всё это настраивается в /settings, а забыть меня можно попросить там же.\n\n"
    "Скажи, в каком ты городе? Так я буду знать твоё время и небо. И честно: я говорю с тобой через программу. "
    "Но это всё равно я."
)

# Единый реестр команд: (команда, метод, описание, в основном меню, только админ, раздел справки)
COMMANDS: list[tuple[str, str, str, bool, bool, str]] = [
    ("start", "on_start", "Начать", True, False, "Основное"),
    ("help", "on_help", "Все команды", True, False, "Основное"),
    ("status", "on_status", "Как я и что включено", True, False, "Основное"),
    ("memory", "on_memory", "Что я о тебе помню", True, False, "Память"),
    ("diary", "on_diary", "Мой дневник (можно /diary слово)", True, False, "Внутренний мир"),
    ("profile", "on_profile", "Твой профиль в моих глазах", True, False, "Основное"),
    ("thoughts", "on_thoughts", "О чём я сейчас думаю", True, False, "Внутренний мир"),
    ("settings", "on_settings", "Меню и настройки", True, False, "Основное"),
    ("about", "on_about", "Кто я", False, False, "Основное"),
    ("mood", "on_mood", "Моё состояние и наши отношения", False, False, "Внутренний мир"),
    ("emotions", "on_emotions", "Мои эмоции сейчас", False, False, "Внутренний мир"),
    ("memories", "on_memories", "Наши общие воспоминания", False, False, "Память"),
    ("remember", "on_remember", "Попросить запомнить: /remember текст", False, False, "Память"),
    ("forget", "on_forget", "Забыть: /forget слово или всё", False, False, "Память"),
    ("recall", "on_recall", "Вспомнить: /recall тема", False, False, "Память"),
    ("history", "on_history", "О чём мы говорили", False, False, "Память"),
    ("promises", "on_promises", "Записки, обещания, мои цели", False, False, "Память"),
    ("dream", "on_dream", "Что мне снилось", False, False, "Внутренний мир"),
    ("reflection", "on_reflection", "Что я думаю о нас", False, False, "Внутренний мир"),
    ("interests", "on_interests", "Что мне интересно", False, False, "Внутренний мир"),
    ("activity", "on_activity", "Чем я занималась", False, False, "Внутренний мир"),
    ("hall", "on_hall", "Заглянуть в мой зал", False, False, "Внутренний мир"),
    ("quiet", "on_quiet", "Тишина до утра или /quiet 23 9", False, False, "Управление"),
    ("wake", "on_wake", "Снять тишину", False, False, "Управление"),
    ("pause", "on_pause", "Пауза: не пишу первой и не исследую", False, False, "Управление"),
    ("resume", "on_resume", "Снять паузу", False, False, "Управление"),
    ("initiative", "on_initiative", "Писать ли мне первой: off | rare | normal", False, False, "Управление"),
    ("name", "on_name", "Как тебя звать: /name Имя", False, False, "Управление"),
    ("city", "on_city", "Твой город: /city Город", False, False, "Управление"),
    ("tz", "on_tz", "Часовой пояс: /tz Europe/Moscow", False, False, "Управление"),
    ("addvoice", "on_addvoice", "Сохранить голосовое: ответь на гс /addvoice название", False, False, "Голос и стикеры"),
    ("voices", "on_voices", "Мои сохранённые голосовые", False, False, "Голос и стикеры"),
    ("delvoice", "on_delvoice", "Удалить голосовое: /delvoice название", False, False, "Голос и стикеры"),
    ("addpack", "on_addpack", "Выучить набор стикеров: ответь на стикер /addpack", False, False, "Голос и стикеры"),
    ("stickers", "on_stickers", "Какие стикеры у меня есть", False, False, "Голос и стикеры"),
    ("delpack", "on_delpack", "Забыть набор: /delpack имя", False, False, "Голос и стикеры"),
    ("backup", "on_backup", "Резервная копия твоих данных", False, False, "Данные"),
    ("export", "on_export", "Экспорт в текст", False, False, "Данные"),
    ("reset", "on_reset", "Сбросить текущий разговор", False, False, "Данные"),
    ("stats", "on_stats", "Статистика", False, False, "Данные"),
    ("check", "on_check", "Проверка сервисов", False, True, "Система"),
    ("groupmode", "on_groupmode", "В группе: когда отвечать — mention | active | all", False, False, "Группа"),
    ("greet", "on_greet", "В группе: здороваться с новенькими — on | off", False, False, "Группа"),
]


def help_text(admin: bool) -> str:
    out, section = [], None
    for name, _m, desc, _main, adm, sec in COMMANDS:
        if adm and not admin:
            continue
        if sec != section:
            out.append(f"\n{sec}")
            section = sec
        out.append(f"/{name} — {desc}")
    return "\n".join(out).strip() + "\n\nВсё это есть и в меню: /settings"


@dataclass
class _Pending:
    items: list[tuple[Message, asyncio.Task]] = field(default_factory=list)
    timer: asyncio.Task | None = None


class TelegramInterface:
    def __init__(self, settings: Settings, engine: CognitiveEngine, media: MediaProcessor, tts=None):
        self.s = settings
        self.engine = engine
        self.media = media
        self.tts = tts  # GeminiTTS или None
        self._last_was_circle: dict[int, bool] = {}
        self.stickers: dict[str, str] = {}  # эмодзи → file_id из STICKER_SET
        engine.voice_available = bool(tts and tts.available)
        self.giphy = None
        if settings.giphy_api_key:
            from ..media.gif import GiphySearch

            self.giphy = GiphySearch(settings.giphy_api_key)
            engine.gifs_available = True
        self.bot = Bot(token=settings.telegram_token)
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)
        self._pending: dict[int, _Pending] = {}
        from .menu import MenuController

        self.menus = MenuController(engine, settings.admin_user_ids)
        from .groups import GroupLogic

        self.groups = GroupLogic(engine, settings.group_max_replies)
        self._register()

    # ------------------------------------------------------------------ access
    def allowed(self, user_id: int) -> bool:
        return not self.s.allowed_user_ids or user_id in self.s.allowed_user_ids or user_id in self.s.admin_user_ids

    async def send_parts(self, chat_id: int, parts: list[str], reply_to: int | None = None) -> None:
        from aiogram.types import ReplyParameters

        first = True
        for i, part in enumerate(parts):
            if i > 0:
                await self.bot.send_chat_action(chat_id, ChatAction.TYPING)
                await asyncio.sleep(min(3.5, 0.6 + len(part) / 45))
            for chunk in _split_long(part):
                kwargs = {}
                if first and reply_to:
                    kwargs["reply_parameters"] = ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)
                await self.bot.send_message(chat_id, chunk, **kwargs)
                first = False

    # ------------------------------------------------------------------ handlers
    def _register(self) -> None:
        r = self.router
        # Группы фильтруются внутри обработчиков (GROUPS_ENABLED проверяется на каждом сообщении).
        chat_ok = F.chat.type.in_({ChatType.PRIVATE, ChatType.GROUP, ChatType.SUPERGROUP})
        r.message.register(self.on_start, CommandStart(), chat_ok)
        for name, method, *_ in COMMANDS:
            if name == "start":
                continue
            r.message.register(getattr(self, method), Command(name), chat_ok)
        r.callback_query.register(self.on_callback, F.data.regexp(r"^[macxtf]:"))
        r.my_chat_member.register(self.on_my_chat_member)
        r.message.register(self.on_location, F.location, chat_ok)
        r.message.register(self.on_any, chat_ok)

    GROUP_OK_COMMANDS = {"start", "help", "about", "groupmode", "greet"}

    async def _guard(self, m: Message) -> bool:
        if m.chat.type != ChatType.PRIVATE:
            if not self.s.groups_enabled:
                return False
            cmd = (m.text or m.caption or "").split()[0].lstrip("/").split("@")[0].lower() if (m.text or m.caption) else ""
            if cmd and cmd not in self.GROUP_OK_COMMANDS:
                await m.answer("Это личное — напиши мне в личку.")  # память и данные не показываю при всех
                return False
        if not m.from_user or not self.allowed(m.from_user.id):
            await m.answer("Хм. Мы, кажется, не знакомы. Я пока говорю только со своими.")
            return False
        # chat_id личного чата = id пользователя: инициатива уходит в личку, даже если познакомились в группе
        await self.engine.store.get_or_create_user(m.from_user.id, m.from_user.id, m.from_user.first_name)
        return True

    async def on_start(self, m: Message) -> None:
        if await self._guard(m):
            await m.answer(GREETING if m.chat.type == ChatType.PRIVATE else self.groups.greeting(m.chat.title))

    async def on_help(self, m: Message) -> None:
        if await self._guard(m):
            await m.answer(help_text(m.from_user.id in self.s.admin_user_ids))

    # ---------------- текстовые команды поверх introspection ----------------
    async def _reply_text(self, m: Message, fn, *args) -> None:
        if not await self._guard(m):
            return
        try:
            text = await fn(self.engine, m.from_user.id, *args)
        except Exception:
            log.exception("command failed")
            text = "Что-то во мне спуталось. Попробуй ещё раз чуть позже."
        for chunk in _split_long(text):
            await m.answer(chunk)

    async def on_status(self, m: Message) -> None:
        await self._reply_text(m, lambda e, u: I.status_text(e, u, u in self.s.admin_user_ids))

    async def on_profile(self, m: Message) -> None:
        await self._reply_text(m, I.profile_text)

    async def on_about(self, m: Message) -> None:
        await self._reply_text(m, lambda e, u: _const(I.ABOUT))

    async def on_thoughts(self, m: Message) -> None:
        await self._reply_text(m, I.thoughts_text)

    async def on_emotions(self, m: Message) -> None:
        await self._reply_text(m, I.emotions_text)

    async def on_memories(self, m: Message) -> None:
        await self._reply_text(m, I.memories_text)

    async def on_history(self, m: Message) -> None:
        await self._reply_text(m, I.history_text)

    async def on_promises(self, m: Message) -> None:
        await self._reply_text(m, I.promises_text)

    async def on_dream(self, m: Message) -> None:
        await self._reply_text(m, I.dream_text)

    async def on_reflection(self, m: Message) -> None:
        await self._reply_text(m, I.reflection_text)

    async def on_interests(self, m: Message) -> None:
        await self._reply_text(m, I.interests_text)

    async def on_activity(self, m: Message) -> None:
        await self._reply_text(m, I.activity_text)

    async def on_diary(self, m: Message, command: CommandObject) -> None:
        await self._reply_text(m, I.diary_text, command.args or "")

    async def on_recall(self, m: Message, command: CommandObject) -> None:
        await self._reply_text(m, I.recall_text, command.args or "")

    async def on_remember(self, m: Message, command: CommandObject) -> None:
        await self._reply_text(m, I.remember, command.args or "")

    # ---------------- медиатека: голосовые и стикеры ----------------
    def _owner(self, uid: int) -> int:
        return 0 if uid in self.s.admin_user_ids else uid  # админ пополняет общую медиатеку

    async def on_addvoice(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        src = m if (m.voice or m.audio) else m.reply_to_message
        title = (command.args or "").strip()[:80]
        if not src or not (src.voice or src.audio):
            await m.answer("Ответь на голосовое (или аудио) командой /addvoice название — например "
                           "/addvoice колыбельная. Можно и подписью к самому голосовому.")
            return
        if not title:
            await m.answer("Как его назвать? /addvoice название — по названию я его потом и пришлю.")
            return
        kind_file = src.voice.file_id if src.voice else src.audio.file_id
        await self.engine.store.add_media(self._owner(m.from_user.id), "voice", kind_file, title=title,
                                          set_name="audio" if src.audio else None)
        await m.answer(f"Сохранила «{title}». Попросишь — пришлю.")

    async def on_voices(self, m: Message) -> None:
        if not await self._guard(m):
            return
        items = await self.engine.store.media(m.from_user.id, "voice")
        if not items:
            await m.answer("Сохранённых голосовых нет. Ответь на голосовое командой /addvoice название.")
            return
        await m.answer("Мои голосовые:\n" + "\n".join(f"— {i['title']}" + (" (общее)" if i["owner_id"] == 0 else "")
                                                       for i in items) + "\n\nПопроси: «скинь гс …» или «спой».")

    async def on_delvoice(self, m: Message, command: CommandObject) -> None:
        if await self._guard(m):
            n = await self.engine.store.delete_media(m.from_user.id, "voice", (command.args or "").strip(),
                                                     admin=m.from_user.id in self.s.admin_user_ids)
            await m.answer("Удалила." if n else "Такого голосового нет. Список: /voices")

    async def on_addpack(self, m: Message, command: CommandObject) -> None:
        if not await self._guard(m):
            return
        st = m.reply_to_message.sticker if m.reply_to_message and m.reply_to_message.sticker else None
        name = (command.args or "").strip() or (st.set_name if st else "")
        if not name:
            await m.answer("Пришли мне любой стикер из набора и ответь на него /addpack — я выучу весь набор. "
                           "Или /addpack имя_набора (из ссылки t.me/addstickers/ИМЯ).")
            return
        n = await self.learn_pack(self._owner(m.from_user.id), name)
        await m.answer(f"Выучила набор «{name}»: {n} стикеров. Буду иногда ими отвечать." if n else
                       "Не нашла такой набор.")

    async def learn_pack(self, owner: int, set_name: str) -> int:
        try:
            sset = await self.bot.get_sticker_set(set_name)
        except Exception:
            log.info("sticker set %s not found", set_name, exc_info=True)
            return 0
        for stk in sset.stickers:
            await self.engine.store.add_media(owner, "sticker", stk.file_id, (stk.emoji or "").replace("\ufe0f", ""),
                                              set_name=set_name)
        return len(sset.stickers)

    async def on_stickers(self, m: Message) -> None:
        if not await self._guard(m):
            return
        items = await self.engine.store.media(m.from_user.id, "sticker")
        sets: dict[str, int] = {}
        for i in items:
            sets[i["set_name"] or "по одному"] = sets.get(i["set_name"] or "по одному", 0) + 1
        sets.update({f"{self.s.sticker_set} (из настроек)": len(self.stickers)} if self.stickers else {})
        if not sets:
            await m.answer("Стикеров у меня пока нет. Пришли мне стикер — я его запомню. Ответь на стикер /addpack — "
                           "выучу весь набор. Свой набор можно сделать в @Stickers.")
            return
        emojis = " ".join(dict.fromkeys(i["emoji"] for i in items if i["emoji"]))
        await m.answer("Мои стикеры:\n" + "\n".join(f"— {k}: {v}" for k, v in sets.items()) +
                       (f"\n\nЭмоции: {emojis[:300]}" if emojis else ""))

    async def on_delpack(self, m: Message, command: CommandObject) -> None:
        if await self._guard(m):
            n = await self.engine.store.delete_media(m.from_user.id, "sticker", (command.args or "").strip(),
                                                     admin=m.from_user.id in self.s.admin_user_ids)
            await m.answer(f"Забыла {n} стикеров." if n else "Такого набора нет. Список: /stickers")

    async def _send_sticker(self, chat_id: int, uid: int, emoji: str) -> None:
        items = [i for i in await self.engine.store.media(uid, "sticker") if i["emoji"] == emoji]
        if items:
            pick = min(items, key=lambda i: i["uses"])  # не один и тот же стикер подряд
            await self.engine.store.media_used(pick["id"])
            await self.bot.send_sticker(chat_id, pick["file_id"])
        elif emoji in self.stickers:
            await self.bot.send_sticker(chat_id, self.stickers[emoji])

    async def _send_clip(self, chat_id: int, uid: int, title: str, reply_to: int | None) -> bool:
        from aiogram.types import ReplyParameters

        items = [i for i in await self.engine.store.media(uid, "voice") if (i["title"] or "").lower() == title.lower()]
        if not items:
            return False
        kw = {"reply_parameters": ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)} if reply_to else {}
        clip = items[0]
        await self.engine.store.media_used(clip["id"])
        if clip["set_name"] == "audio":
            await self.bot.send_audio(chat_id, clip["file_id"], **kw)
        else:
            await self.bot.send_voice(chat_id, clip["file_id"], **kw)
        return True

    async def on_wake(self, m: Message) -> None:
        if await self._guard(m):
            await self.engine.store.set_setting(m.from_user.id, "quiet_until", "0")
            await m.answer("Я снова здесь.")

    async def on_pause(self, m: Message) -> None:
        if await self._guard(m):
            await self.engine.store.set_setting(m.from_user.id, "paused", "1")
            await m.answer("Хорошо, затихну: первой не пишу и сама ничего не изучаю. Записки, о которых просил, "
                           "принесу. Вернуть — /resume.")

    async def on_resume(self, m: Message) -> None:
        if await self._guard(m):
            await self.engine.store.set_setting(m.from_user.id, "paused", "0")
            await m.answer("Снова живу как обычно.")

    async def on_backup(self, m: Message) -> None:
        if await self._guard(m):
            await self._send_screen(m.chat.id, await self.menus.file("backup", m.from_user.id))

    async def on_export(self, m: Message) -> None:
        if await self._guard(m):
            await self._send_screen(m.chat.id, await self.menus.file("export", m.from_user.id))

    async def on_reset(self, m: Message) -> None:
        if await self._guard(m):
            await self._send_screen(m.chat.id, self.menus.confirm("reset"))

    async def on_settings(self, m: Message) -> None:
        if await self._guard(m):
            await self._send_screen(m.chat.id, await self.menus.menu("root", m.from_user.id))

    async def _send_screen(self, chat_id: int, screen) -> None:
        from aiogram.types import BufferedInputFile

        if screen.file:
            name, data = screen.file
            await self.bot.send_document(chat_id, BufferedInputFile(data, filename=name), caption=screen.text)
        else:
            await self.bot.send_message(chat_id, screen.text, reply_markup=screen.markup)

    async def on_callback(self, cq) -> None:
        from aiogram.exceptions import TelegramBadRequest

        uid = cq.from_user.id
        if not self.allowed(uid):
            await cq.answer("Мы не знакомы.", show_alert=False)
            return
        data = cq.data or ""
        if self.menus.debounce(uid, data):
            await cq.answer()
            return
        kind, _, arg = data.partition(":")
        try:
            if kind == "m":
                screen = await self.menus.menu(arg, uid)
            elif kind == "a":
                screen = await self.menus.action(arg, uid)
            elif kind == "t":
                screen = await self.menus.toggle(arg, uid)
            elif kind == "c":
                screen = self.menus.confirm(arg)
            elif kind == "x":
                screen = await self.menus.execute(arg, uid)
            elif kind == "f":
                await cq.answer("Собираю…")
                await self._send_screen(cq.message.chat.id, await self.menus.file(arg, uid))
                return
            else:
                await cq.answer()
                return
        except Exception:
            log.exception("menu failed")
            await cq.answer("Не получилось. Попробуй ещё раз.", show_alert=True)
            return
        await cq.answer()
        try:
            await cq.message.edit_text(screen.text, reply_markup=screen.markup)
        except TelegramBadRequest as e:
            if "not modified" not in str(e):
                await self.bot.send_message(cq.message.chat.id, screen.text, reply_markup=screen.markup)

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
        if not (command.args or "").strip():
            from .menu import quiet_until_morning

            until = await quiet_until_morning(self.engine, m.from_user.id)
            await self.engine.store.set_setting(m.from_user.id, "quiet_until", str(until))
            await m.answer("Тише воды. До утра первой не напишу. Разбудить — /wake.\n"
                           "Тихие часы насовсем: /quiet 23 9")
            return
        try:
            a, b = [int(x) for x in (command.args or "").split()[:2]]
            assert 0 <= a < 24 and 0 <= b < 24
        except Exception:
            await m.answer("Например: /quiet 23 9 — с 23:00 до 9:00 я не пишу первой. Или просто /quiet — тишина до утра.")
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
        arg = (command.args or "").strip()
        if not arg or arg.lower() in {"всё", "все", "all"}:
            await self._send_screen(m.chat.id, self.menus.confirm("forget"))
            return
        if arg.lower() not in {"yes", "да"}:
            n = await self.engine.store.forget_matching(m.from_user.id, arg)
            await m.answer(f"Забыла то, что касалось «{arg}» ({n})." if n else f"Про «{arg}» я ничего не помню.")
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
        if not m.from_user:
            return
        if m.from_user.id not in self.s.admin_user_ids:
            if await self._guard(m):
                st, uid = self.engine.store, m.from_user.id
                msgs = await st.d.fetchone("SELECT COUNT(*) AS n FROM messages WHERE user_id=?", (uid,))
                await m.answer(f"Сообщений: {msgs['n']}\nВоспоминаний: {len(await st.episodes(uid, 'shared'))}\n"
                               f"Фактов о тебе: {len(await st.facts(uid))}\nЗаписей в дневнике: "
                               f"{len(await st.diary(uid, limit=1000))}")
            return
        rows = await self.engine.store.usage_report(time.time() - 86400)
        total = sum(r["cost"] or 0 for r in rows)
        lines = [f"{r['purpose']:<11} {r['model']:<18} n={r['n']:<4} in={r['inp']} out={r['out']} "
                 f"cache_r={r['cr']} cache_w={r['cw']} ${r['cost']:.4f}" for r in rows]
        await m.answer("За 24 часа:\n" + "\n".join(lines) + f"\nИтого: ${total:.4f}")

    # ------------------------------------------------------------------ messages of any type
    async def on_any(self, m: Message) -> None:
        if m.chat.type != ChatType.PRIVATE:
            await self.on_group_message(m)
            return
        if not await self._guard(m):
            return
        uid = m.from_user.id
        pend = self._pending.setdefault(uid, _Pending())
        pend.items.append((m, asyncio.create_task(self.to_percepts(m))))
        if pend.timer:
            pend.timer.cancel()
        pend.timer = asyncio.create_task(self._fire_later(uid, m.chat.id))

    # ---------------- группы ----------------
    def group_allowed(self, m: Message) -> bool:
        if m.chat.id in self.s.allowed_group_ids:
            return True
        return bool(m.from_user) and self.allowed(m.from_user.id)

    async def on_group_message(self, m: Message) -> None:
        if not self.s.groups_enabled or not m.from_user or m.from_user.is_bot:
            return
        if m.new_chat_members:
            if (await self.engine.store.get_setting(m.chat.id, "greet", "on")) != "off":
                names = [u.first_name for u in m.new_chat_members if not u.is_bot]
                if names:
                    await m.answer(self.groups.welcome(names))
            return
        if not self.group_allowed(m):
            return
        me = await self.bot.me()
        if not await self.groups.should_reply(m, self.bot.id, me.username):
            await self.groups.remember_context(m)  # слышит, но молчит
            return
        conv = m.chat.id
        await self.engine.store.get_or_create_user(conv, conv, m.chat.title)

        async def framed():
            return self.groups.frame(await self.to_percepts(m), m)

        pend = self._pending.setdefault(conv, _Pending())
        pend.items.append((m, asyncio.create_task(framed())))
        if pend.timer:
            pend.timer.cancel()
        pend.timer = asyncio.create_task(self._fire_later(conv, conv))

    async def _is_group_admin(self, chat_id: int, user_id: int) -> bool:
        if user_id in self.s.admin_user_ids:
            return True
        try:
            member = await self.bot.get_chat_member(chat_id, user_id)
            return member.status in ("administrator", "creator")
        except Exception:
            return False

    async def on_groupmode(self, m: Message, command: CommandObject) -> None:
        if m.chat.type == ChatType.PRIVATE:
            await m.answer("Это команда для группы: /groupmode mention | active | all")
            return
        if not await self._is_group_admin(m.chat.id, m.from_user.id):
            await m.answer("Режим меняют только админы группы.")
            return
        mode = (command.args or "").strip().lower()
        from .groups import MODES

        if mode not in MODES:
            cur = await self.groups.mode(m.chat.id)
            await m.answer(f"Сейчас: {cur}.\nmention — отвечаю, когда зовут; active — иногда вступаю сама "
                           "(на вопросы о том, что мне интересно); all — отвечаю на всё.")
            return
        await self.engine.store.set_setting(m.chat.id, "group_mode", mode)
        await m.answer({"mention": "Хорошо. Буду говорить, только когда позовёте.",
                        "active": "Хорошо. Иногда буду вступать сама.", "all": "Хорошо. Буду отвечать на всё."}[mode])

    async def on_greet(self, m: Message, command: CommandObject) -> None:
        if m.chat.type == ChatType.PRIVATE or not await self._is_group_admin(m.chat.id, m.from_user.id):
            await m.answer("Это команда для админов группы: /greet on | off")
            return
        val = "off" if (command.args or "").strip().lower() in ("off", "выкл", "нет") else "on"
        await self.engine.store.set_setting(m.chat.id, "greet", val)
        await m.answer("Буду здороваться с новенькими." if val == "on" else "Не буду здороваться с новенькими.")

    async def on_my_chat_member(self, update) -> None:
        if not self.s.groups_enabled or update.chat.type == ChatType.PRIVATE:
            return
        if update.new_chat_member.status in ("member", "administrator") and \
                update.old_chat_member.status in ("left", "kicked"):
            await self.bot.send_message(update.chat.id, self.groups.greeting(update.chat.title))

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
        if reply.silent and not reply.reaction:
            return  # она решила промолчать — так тоже бывает
        if reply.reaction:
            try:
                await self.bot.set_message_reaction(chat_id, last_msg.message_id,
                                                    [ReactionTypeEmoji(emoji=reply.reaction)])
            except Exception:
                log.debug("reaction failed", exc_info=True)
        if reply.parts:
            # Если собеседник отвечал на конкретное сообщение — отвечаем «реплаем» на его сообщение.
            quoted = next((m for m, _ in reversed(pend.items) if m.reply_to_message), None) if reply.quote else None
            if uid < 0:  # в группе всегда отвечает «реплаем» — видно, кому
                quoted = pend.items[-1][0]
            heard = any(m.voice or m.video_note for m, _ in pend.items)
            self._last_was_circle[uid] = any(m.video_note for m, _ in pend.items)
            if not await self._maybe_voice(uid, chat_id, reply, heard, quoted.message_id if quoted else None):
                await self.send_parts(chat_id, reply.parts, reply_to=quoted.message_id if quoted else None)
        if reply.gif and self.giphy:
            url = await self.giphy.find(reply.gif)
            if url:
                try:
                    await self.bot.send_animation(chat_id, url)
                except Exception:
                    log.debug("gif failed", exc_info=True)
        if reply.clip:
            try:
                await self._send_clip(chat_id, uid, reply.clip, None)
            except Exception:
                log.debug("clip failed", exc_info=True)
        if reply.sticker:
            try:
                await self._send_sticker(chat_id, uid, reply.sticker)
            except Exception:
                log.debug("sticker failed", exc_info=True)
        if reply.sources:
            from aiogram.types import LinkPreviewOptions

            await self.bot.send_message(chat_id, reply.sources, link_preview_options=LinkPreviewOptions(is_disabled=True))
        if uid in self.s.admin_user_ids:
            note = admin_error_note(reply.debug)
            if reply.debug.get("over_budget"):
                note = "⚙️ (видно только админу) Дневной лимит расходов исчерпан: DAILY_BUDGET_USD_PER_USER"
            if note:
                await self.bot.send_message(chat_id, note)

    async def _maybe_voice(self, uid: int, chat_id: int, reply, heard: bool, reply_to: int | None) -> bool:
        """Ответ голосом: когда она сама захотела (<voice>), в ответ на голосовое (режим mirror) или всегда."""
        if not (self.tts and self.tts.available):
            return False
        mode = await self.engine.store.get_setting(uid, "voice", "mirror")
        want = mode != "off" and (reply.voice or mode == "always" or (mode == "mirror" and heard) or
                                  (reply.singing and mode != "self"))
        text = " ".join(reply.parts)
        if not want or len(text) > self.tts.max_chars:
            return False
        from aiogram.enums import ChatAction as CA
        from aiogram.types import BufferedInputFile, ReplyParameters

        await self.bot.send_chat_action(chat_id, CA.RECORD_VOICE)
        audio = await self.tts.synthesize(text, singing=reply.singing)
        if not audio:
            return False
        kwargs = {"reply_parameters": ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)} \
            if reply_to else {}
        circle = await self.engine.store.get_setting(uid, "circle", "mirror")
        avatar = Path(self.s.avatar_image) if self.s.avatar_image else None
        if avatar and avatar.exists() and circle != "off" and (circle == "always" or self._last_was_circle.get(uid)):
            from ..media.gif import video_note

            await self.bot.send_chat_action(chat_id, CA.RECORD_VIDEO_NOTE)
            vid = await video_note(avatar, audio, self.s.media_tmp_dir)
            if vid:
                await self.bot.send_video_note(chat_id, BufferedInputFile(vid, filename="note.mp4"), **kwargs)
                return True
        await self.bot.send_voice(chat_id, BufferedInputFile(audio, filename="voice.ogg"), **kwargs)
        return True

    async def load_stickers(self) -> None:
        if not self.s.sticker_set:
            return
        try:
            sset = await self.bot.get_sticker_set(self.s.sticker_set)
            for st in sset.stickers:
                if st.emoji and st.emoji not in self.stickers:
                    self.stickers[st.emoji.replace("\ufe0f", "")] = st.file_id
            self.engine.sticker_emojis = list(self.stickers)
            log.info("stickers: %d из набора %s", len(self.stickers), self.s.sticker_set)
        except Exception:
            log.warning("не удалось загрузить набор стикеров %s", self.s.sticker_set, exc_info=True)

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
        reply_p = self._reply_percept(m)
        if reply_p:
            out.append(reply_p)
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
                if (await self.engine.store.get_setting(m.from_user.id, "learn_stickers", "on")) != "off":
                    # «скопировать»: запоминает стикеры, которые ей присылают, и может ответить ими же
                    await self.engine.store.add_media(m.from_user.id, "sticker", m.sticker.file_id,
                                                      emoji.replace("\ufe0f", ""), set_name=m.sticker.set_name)
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

    def _reply_percept(self, m: Message) -> Percept | None:
        """Собеседник ответил на конкретное сообщение (свайп/«Ответить») — это важный контекст."""
        r = m.reply_to_message
        if r is None:
            return None
        mine = bool(r.from_user and r.from_user.is_bot and r.from_user.id == self.bot.id)
        who = "твоё" if mine else "своё" if (r.from_user and m.from_user and r.from_user.id == m.from_user.id) else "чужое"
        quote = getattr(m, "quote", None)
        snippet = (quote.text if quote and quote.text else None) or r.text or r.caption or ""
        if not snippet:
            snippet = ("фото" if r.photo else "голосовое" if r.voice else "видео" if (r.video or r.video_note)
                       else "стикер" if r.sticker else "файл" if r.document else "сообщение")
            snippet = f"[{snippet}]"
        snippet = " ".join(snippet.split())[:400]
        text = f"[в ответ на {who} сообщение: «{snippet}»]"
        return Percept("reply", True, "text", f"собеседник отвечает на {who} сообщение: «{snippet[:120]}»",
                       text=text, history=text)

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        from aiogram.types import BotCommand

        try:
            await self.bot.set_my_commands([BotCommand(command=n, description=d[:60])
                                            for n, _m, d, main, _a, _s in COMMANDS if main])
        except Exception:
            log.warning("не удалось задать меню команд", exc_info=True)
        await self.load_stickers()
        await self.dp.start_polling(self.bot, allowed_updates=["message", "callback_query", "my_chat_member"])


async def _const(text: str) -> str:
    return text


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
