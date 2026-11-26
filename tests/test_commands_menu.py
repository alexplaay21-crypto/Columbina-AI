"""Команды и inline-меню: реестр, каждое меню и действие, «Назад», переключатели, подтверждения, файлы."""

from __future__ import annotations

import json
import time

import pytest

from columbina.cognition import introspection as I
from columbina.perception.model import Percept
from columbina.telegram.bot import COMMANDS, TelegramInterface, help_text
from columbina.telegram.menu import ACTIONS, MenuController

MAIN = ["start", "help", "status", "memory", "diary", "profile", "thoughts", "settings"]
EXTRA = ["mood", "emotions", "memories", "remember", "forget", "recall", "history", "promises", "dream", "reflection",
         "interests", "activity", "quiet", "wake", "pause", "resume", "backup", "export", "reset"]


def t(x):
    return [Percept("text", True, "text", "", text=x, history=x)]


def test_registry_main_and_extra():
    names = [c[0] for c in COMMANDS]
    assert len(names) == len(set(names))  # без дублей
    assert [c[0] for c in COMMANDS if c[3]] == MAIN  # в меню Telegram ровно 8 основных
    assert set(EXTRA) <= set(names)
    for _n, method, *_ in COMMANDS:
        assert hasattr(TelegramInterface, method), method
    assert "/check" not in help_text(False) and "/check" in help_text(True)


async def _talk(engine, fake_llm):
    fake_llm.script = lambda r: "Хорошо." if r.tier == "main" else "{}"
    await engine.respond(1, 1, t("Привет, у меня кот Бублик"), display_name="Алекс")


async def test_every_menu_renders_with_back(engine, fake_llm):
    await _talk(engine, fake_llm)
    mc = MenuController(engine, frozenset({1}))
    root = await mc.menu("root", 1)
    cats = [b.callback_data for row in root.markup.inline_keyboard for b in row]
    assert cats == ["m:memory", "m:diary", "m:emotions", "m:inner", "m:persona", "m:activity", "m:data", "m:system"]
    for c in cats:
        sc = await mc.menu(c[2:], 1)
        datas = [b.callback_data for row in sc.markup.inline_keyboard for b in row]
        assert "m:root" in datas, c  # кнопка «Назад»
        for d in datas:  # каждое действие подменю отвечает текстом и «Назад»
            if d.startswith("a:"):
                res = await mc.action(d[2:], 1)
                assert res.text and res.markup.inline_keyboard[-1][0].text == "← Назад", d


async def test_admin_actions_protected(engine, fake_llm):
    await _talk(engine, fake_llm)
    mc = MenuController(engine, frozenset())
    assert "администратора" in (await mc.action("check", 1)).text
    system = await mc.menu("system", 1)
    assert "a:check" not in str(system.markup)


async def test_toggles_persist(engine, fake_llm, store):
    await _talk(engine, fake_llm)
    mc = MenuController(engine, frozenset())
    await mc.toggle("initiative", 1)
    assert (await store.get_user(1)).initiative_mode == "rare"
    await mc.toggle("research", 1)
    assert await store.get_setting(1, "research") == "off"
    await mc.toggle("pause", 1)
    assert await store.get_setting(1, "paused") == "1"
    await mc.toggle("quiet", 1)
    assert float(await store.get_setting(1, "quiet_until")) > time.time()
    sc = await mc.toggle("register", 1)
    assert (await store.get_user(1)).register == "vy" and "на «вы»" in str(sc.markup)
    # после «перезапуска» (новый контроллер) настройки на месте
    again = await MenuController(engine, frozenset()).menu("activity", 1)
    assert "изредка" in str(again.markup) and "выкл" in str(again.markup) and "Снять паузу" in str(again.markup)


async def test_confirmations_and_separate_reset_vs_forget(engine, fake_llm, store):
    await _talk(engine, fake_llm)
    await engine.drain()
    await store.upsert_fact(1, "питомец", "кот Бублик")
    mc = MenuController(engine, frozenset())
    conf = mc.confirm("reset")
    assert [b.callback_data for row in conf.markup.inline_keyboard for b in row] == ["x:reset", "m:data"]
    await mc.execute("reset", 1)
    assert await store.active_messages(1) == [] and await store.facts(1)  # контекст ушёл, память осталась
    await mc.execute("forget", 1)
    assert await store.facts(1) == []


async def test_files_backup_and_export(engine, fake_llm, store):
    await _talk(engine, fake_llm)
    mc = MenuController(engine, frozenset())
    b = await mc.file("backup", 1)
    data = json.loads(b.file[1])
    assert data["user_id"] == 1 and data["messages"] and "embedding" not in json.dumps(data["episodes"])
    e = await mc.file("export", 1)
    assert e.file[0].endswith(".txt") and "Как зову тебя" in e.file[1].decode()


def test_debounce():
    mc = MenuController(None, frozenset())
    assert not mc.debounce(1, "m:root")
    assert mc.debounce(1, "m:root")
    assert not mc.debounce(1, "m:data")


async def test_text_commands(engine, fake_llm, store):
    await _talk(engine, fake_llm)
    assert "Я здесь" in await I.status_text(engine, 1)
    assert "модель" in await I.status_text(engine, 1, admin=True)
    assert "Знакомы с" in await I.profile_text(engine, 1)
    assert await I.remember(engine, 1, "мой день рождения 3 мая") == "Запомнила."
    assert "3 мая" in await I.memories_text(engine, 1)
    assert "3 мая" in await I.recall_text(engine, 1, "день рождения")
    assert "Недавно" in await I.history_text(engine, 1)
    await store.add_note(1, "спросить про врача", time.time() + 3600)
    assert "врача" in await I.promises_text(engine, 1)
    await store.add_diary(1, "discovery", "Луна отходит на 3,8 см в год.")
    assert "открытие" in await I.diary_text(engine, 1)
    assert "3,8" in await I.diary_text(engine, 1, "луна")
    assert "Сейчас я" in await I.thoughts_text(engine, 1)
    assert "Эмоции" in await I.emotions_text(engine, 1) and "Отношения" not in await I.emotions_text(engine, 1)
    assert "Думаю о нас" in await I.reflection_text(engine, 1)
    assert "тихими" in await I.activity_text(engine, 1)
    assert await store.forget_matching(1, "3 мая") >= 1
    assert "3 мая" not in await I.memories_text(engine, 1)


async def test_dream_command_rate_limited(engine, fake_llm, store):
    from columbina.config import ROOT
    from columbina.inner.life import InnerLife

    await _talk(engine, fake_llm)
    engine.inner = InnerLife(engine, ROOT / "data" / "persona")
    fake_llm.script = lambda r: "Мне снились киты в облаках." if r.purpose == "dream" else "{}"
    assert "киты" in await I.dream_text(engine, 1)
    calls = len([r for r in fake_llm.requests if r.purpose == "dream"])
    await I.dream_text(engine, 1)
    assert len([r for r in fake_llm.requests if r.purpose == "dream"]) == calls  # не генерирует заново


@pytest.mark.parametrize("action", sorted(ACTIONS))
def test_actions_have_parent_menu(action):
    assert ACTIONS[action][1] in {"memory", "diary", "emotions", "inner", "persona", "system"}
