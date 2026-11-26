"""Сквозной прогон Telegram-обработчиков через настоящий Dispatcher aiogram с подменённой сетью."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from aiogram.client.session.base import BaseSession
from aiogram.methods import EditMessageText, GetMe, SendDocument, SendMessage, TelegramMethod
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from columbina.media.links import SafeFetcher
from columbina.media.processors import MediaProcessor, SpeechToText
from columbina.telegram.bot import TelegramInterface


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls: list[TelegramMethod] = []

    async def close(self):
        pass

    async def stream_content(self, *a, **k):  # pragma: no cover
        yield b""

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        chat = Chat(id=1, type="private")
        now = dt.datetime.now()
        if isinstance(method, (SendMessage, EditMessageText)):
            return Message(message_id=100 + len(self.calls), date=now, chat=chat, text=method.text)
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Коломбина", username="columbina_bot")
        if isinstance(method, SendDocument):
            return Message(message_id=100 + len(self.calls), date=now, chat=chat)
        return True


def _msg(text: str, mid: int = 1) -> Update:
    return Update(update_id=mid, message=Message(message_id=mid, date=dt.datetime.now(), text=text,
                                                 chat=Chat(id=1, type="private"),
                                                 from_user=User(id=1, is_bot=False, first_name="Алекс")))


def _cb(data: str, uid: int = 7) -> Update:
    msg = Message(message_id=50, date=dt.datetime.now(), text="меню", chat=Chat(id=1, type="private"))
    return Update(update_id=uid, callback_query=CallbackQuery(id=str(uid), from_user=User(id=1, is_bot=False,
                  first_name="Алекс"), chat_instance="x", data=data, message=msg))


@pytest.fixture
async def tg(engine, settings, fake_llm):
    s = dataclasses.replace(settings, telegram_token="123456:ABCdefGHIjklMNO", admin_user_ids=frozenset({1}))
    fake_llm.script = lambda r: "Хорошо." if r.tier == "main" else "{}"
    iface = TelegramInterface(s, engine, MediaProcessor(SpeechToText("http://x", "", "m"), SafeFetcher(), s.media_tmp_dir))
    iface.bot.session = FakeSession()
    yield iface


def _texts(iface) -> list[str]:
    return [getattr(c, "text", "") or getattr(c, "caption", "") or "" for c in iface.bot.session.calls]


async def test_commands_answer(tg):
    for i, cmd in enumerate(["/start", "/help", "/status", "/profile", "/thoughts", "/memories", "/emotions",
                             "/interests", "/activity", "/promises", "/history", "/reflection", "/diary",
                             "/remember мой кот Бублик", "/recall кот", "/pause", "/resume", "/quiet", "/wake",
                             "/stats", "/check"], 1):
        await tg.dp.feed_update(tg.bot, _msg(cmd, i))
    texts = _texts(tg)
    assert not any("спуталось" in t for t in texts), texts
    joined = "\n".join(texts)
    for marker in ["Я — Коломбина", "/settings", "Я здесь", "Знакомы с", "Сейчас я", "Запомнила", "Бублик",
                   "затихну", "Снова живу", "Тише воды", "снова здесь", "модель main"]:
        assert marker in joined, marker


async def test_settings_navigation_and_confirm(tg):
    await tg.dp.feed_update(tg.bot, _msg("/settings"))
    for i, data in enumerate(["m:activity", "t:pause", "m:data", "c:reset", "x:reset", "m:root", "f:backup"], 10):
        await tg.dp.feed_update(tg.bot, _cb(data, i))
    kinds = [type(c).__name__ for c in tg.bot.session.calls]
    assert kinds.count("EditMessageText") >= 5 and "SendDocument" in kinds and "AnswerCallbackQuery" in kinds
    assert await tg.engine.store.get_setting(1, "paused") == "1"
    edits = [c.text for c in tg.bot.session.calls if isinstance(c, EditMessageText)]
    assert any("Начнём разговор заново" in e for e in edits)
