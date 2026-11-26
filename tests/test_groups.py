"""Группы: когда отвечать, память группы, кто говорит, лимиты, приватность, админские настройки."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import time

from aiogram.methods import GetChatMember, SendMessage
from aiogram.types import Chat, ChatMemberMember, ChatMemberOwner, Message, Update, User

from columbina.telegram.groups import GroupLogic

from .test_telegram_flow import FakeSession, tg  # noqa: F401

GID = -1001234


def gmsg(i, text, uid=1, name="Алекс", reply_to_bot=False, bot_id=123456):
    reply = Message(message_id=900, date=dt.datetime.now(), chat=Chat(id=GID, type="supergroup"), text="я тут",
                    from_user=User(id=bot_id, is_bot=True, first_name="Коломбина")) if reply_to_bot else None
    return Update(update_id=i, message=Message(message_id=i, date=dt.datetime.now(), text=text,
                  chat=Chat(id=GID, type="supergroup", title="Лунный клуб"),
                  from_user=User(id=uid, is_bot=False, first_name=name), reply_to_message=reply))


async def _wait(tg_, cond, n=60):  # noqa: F811
    for _ in range(n):
        if cond():
            return True
        await asyncio.sleep(0.05)
    return False


def _setup(tg_, admin_ids=frozenset({1})):  # noqa: F811
    tg_.s = dataclasses.replace(tg_.s, groups_enabled=True, debounce_seconds=0, admin_user_ids=admin_ids)
    tg_.engine.llm.script = lambda r: "Слушаю." if r.tier == "main" else "{}"


async def test_group_listens_and_answers_when_called(tg):  # noqa: F811
    _setup(tg)
    await tg.dp.feed_update(tg.bot, gmsg(1, "кто идёт гулять вечером?", uid=2, name="Мира"))
    hist = await tg.engine.store.active_messages(GID)
    assert hist and hist[0].text == "[Мира]: кто идёт гулять вечером?"  # слышит, но молчит
    assert not any(isinstance(c, SendMessage) for c in tg.bot.session.calls)
    await tg.dp.feed_update(tg.bot, gmsg(2, "Коломбина, а ты пойдёшь?"))
    assert await _wait(tg, lambda: any(isinstance(c, SendMessage) for c in tg.bot.session.calls))
    sent = [c for c in tg.bot.session.calls if isinstance(c, SendMessage)][0]
    assert sent.text == "Слушаю." and sent.reply_parameters.message_id == 2  # отвечает тому, кто звал
    req = [r for r in tg.engine.llm.requests if r.tier == "main"][-1]
    users = "\n".join(m.plain_text() for m in req.messages if m.role == "user")
    assert "[Мира]: кто идёт гулять" in users and "[Алекс]: Коломбина, а ты пойдёшь?" in users
    assert "групповой чат «Лунный клуб»" in req.messages[-1].plain_text()


async def test_reply_to_her_message_counts_as_addressed(tg):  # noqa: F811
    _setup(tg)
    await tg.dp.feed_update(tg.bot, gmsg(3, "согласен", reply_to_bot=True, bot_id=tg.bot.id))
    assert await _wait(tg, lambda: any(isinstance(c, SendMessage) for c in tg.bot.session.calls))


async def test_personal_commands_are_private_in_group(tg):  # noqa: F811
    _setup(tg)
    await tg.dp.feed_update(tg.bot, gmsg(4, "/memory"))
    texts = [c.text for c in tg.bot.session.calls if isinstance(c, SendMessage)]
    assert texts == ["Это личное — напиши мне в личку."]


async def test_groupmode_only_for_group_admins(tg):  # noqa: F811
    _setup(tg, admin_ids=frozenset())
    session: FakeSession = tg.bot.session
    orig = session.make_request

    async def mr(bot, method, timeout=None):
        if isinstance(method, GetChatMember):
            session.calls.append(method)
            u = User(id=method.user_id, is_bot=False, first_name="x")
            return ChatMemberOwner(user=u, is_anonymous=False) if method.user_id == 7 else ChatMemberMember(user=u)
        return await orig(bot, method, timeout)

    session.make_request = mr
    await tg.dp.feed_update(tg.bot, gmsg(5, "/groupmode active", uid=2))
    assert await tg.engine.store.get_setting(GID, "group_mode", "mention") == "mention"
    await tg.dp.feed_update(tg.bot, gmsg(6, "/groupmode active", uid=7))
    assert await tg.engine.store.get_setting(GID, "group_mode") == "active"


async def test_active_mode_and_rate_limit(engine, store):
    gl = GroupLogic(engine, max_replies_10min=2)
    await store.set_setting(GID, "group_mode", "active")

    def m(text):
        return Message(message_id=1, date=dt.datetime.now(), text=text, chat=Chat(id=GID, type="supergroup"),
                       from_user=User(id=2, is_bot=False, first_name="Мира"))

    assert await gl.should_reply(m("а кто видел полнолуние вчера?"), 1, "bot")  # вопрос по её интересам
    assert not await gl.should_reply(m("сколько стоит бензин?"), 1, "bot")
    for _ in range(2):
        await store.add_message(GID, "assistant", "x", at=time.time())
    assert not await gl.should_reply(m("Коломбина, ты тут?"), 1, "bot")  # лимит ответов — молчит даже на зов


async def test_group_never_gets_initiative(engine, store):
    from columbina.initiative.engine import InitiativeEngine
    from columbina.memory.store import RelationshipRow

    await store.get_or_create_user(GID, GID, "Лунный клуб")
    await store.update_user(GID, last_user_msg_at=time.time() - 9 * 86400, quiet_start=0, quiet_end=0)
    await store.save_relationship(GID, RelationshipRow(trust=0.5, stage=3, quality_sessions=10))
    sent = []

    async def send(chat_id, parts):
        sent.append(chat_id)

    await InitiativeEngine(engine).tick(send)
    assert GID not in sent
