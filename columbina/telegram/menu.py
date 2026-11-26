"""Inline-меню /settings: категории → подменю → действия. Навигация редактирует одно и то же сообщение.

callback_data:
  m:<menu>     — открыть меню
  a:<action>   — показать результат действия (текст + «Назад»)
  t:<toggle>   — переключить настройку и перерисовать меню
  c:<action>   — экран подтверждения опасного действия
  x:<action>   — выполнить подтверждённое опасное действие
  f:<action>   — отправить файл (резервная копия, экспорт)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from ..cognition import introspection as I

TG_LIMIT = 3900

CATEGORIES = [
    [("🧠 Память", "m:memory"), ("📔 Дневник", "m:diary")],
    [("🌙 Эмоции", "m:emotions"), ("✨ Внутренний мир", "m:inner")],
    [("🎭 Личность", "m:persona"), ("🔔 Активность", "m:activity")],
    [("💾 Данные", "m:data"), ("⚙️ Система", "m:system")],
]

TITLES = {
    "root": "Настройки и мой мир. Выбери, куда заглянуть:",
    "memory": "🧠 Память — что я помню о тебе и о нас.",
    "diary": "📔 Мой дневник.",
    "emotions": "🌙 Как я сейчас и как мы с тобой.",
    "inner": "✨ Что у меня внутри.",
    "persona": "🎭 Обо мне и о том, как мы говорим.",
    "activity": "🔔 Когда и как я проявляю себя сама.",
    "data": "💾 Твои данные. Опасные действия попросят подтверждения.",
    "system": "⚙️ Состояние.",
}

CONFIRM = {
    "reset": ("Сбросить текущий разговор? Память о тебе останется — уйдёт только контекст последних сообщений.",
              "Да, сбросить разговор"),
    "forget": ("Забыть ВСЁ: разговоры, факты о тебе, воспоминания, дневник, записки. Это не отменить.",
               "Да, забыть всё"),
}


def _kb(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in row]
                                                 for row in rows])


def back(to: str = "root") -> list[tuple[str, str]]:
    return [("← Назад", f"m:{to}")]


@dataclass
class Screen:
    text: str
    markup: InlineKeyboardMarkup | None = None
    file: tuple[str, bytes] | None = None  # (имя, содержимое) для отправки документом


# Действие → (функция, меню для «Назад», нужен ли админ)
ACTIONS = {
    "facts": (I.profile_text, "memory", False),
    "memories": (I.memories_text, "memory", False),
    "promises": (I.promises_text, "memory", False),
    "history": (I.history_text, "memory", False),
    "diary_all": (lambda e, u: I.diary_text(e, u, ""), "diary", False),
    "diary_discovery": (lambda e, u: I.diary_text(e, u, "discovery"), "diary", False),
    "diary_dream": (lambda e, u: I.diary_text(e, u, "dream"), "diary", False),
    "diary_event": (lambda e, u: I.diary_text(e, u, "event"), "diary", False),
    "emotions": (I.emotions_text, "emotions", False),
    "mood": (I.mood_report, "emotions", False),
    "reflection": (I.reflection_text, "emotions", False),
    "thoughts": (I.thoughts_text, "inner", False),
    "interests": (I.interests_text, "inner", False),
    "dream": (I.dream_text, "inner", False),
    "activity_log": (I.activity_text, "inner", False),
    "about": (lambda e, u: _const(I.ABOUT), "persona", False),
    "profile": (I.profile_text, "persona", False),
    "status": (lambda e, u: I.status_text(e, u, False), "system", False),
    "status_admin": (lambda e, u: I.status_text(e, u, True), "system", True),
    "check": (lambda e, u: I.health_check(e), "system", True),
}


async def _const(text: str) -> str:
    return text


class MenuController:
    def __init__(self, engine, admin_ids: frozenset[int]):
        self.engine = engine
        self.admin_ids = admin_ids
        self._last: dict[tuple[int, str], float] = {}

    def debounce(self, user_id: int, data: str, window: float = 1.2) -> bool:
        """True — нажатие повторное и его нужно проигнорировать."""
        now = time.monotonic()
        key = (user_id, data)
        if now - self._last.get(key, 0) < window:
            return True
        self._last[key] = now
        if len(self._last) > 5000:  # без утечки памяти
            self._last = {k: v for k, v in self._last.items() if now - v < 60}
        return False

    async def menu(self, name: str, user_id: int) -> Screen:
        st = self.engine.store
        if name == "root":
            return Screen(TITLES["root"], _kb(CATEGORIES))
        if name == "memory":
            rows = [[("Что я о тебе знаю", "a:facts"), ("Наши воспоминания", "a:memories")],
                    [("Записки и обещания", "a:promises"), ("О чём говорили", "a:history")], back()]
            return Screen(TITLES["memory"] + "\nНайти воспоминание: /recall слово. Попросить запомнить: /remember текст.",
                          _kb(rows))
        if name == "diary":
            rows = [[("Последние записи", "a:diary_all"), ("Открытия", "a:diary_discovery")],
                    [("Сны", "a:diary_dream"), ("События", "a:diary_event")], back()]
            return Screen(TITLES["diary"] + "\nПоиск: /diary слово.", _kb(rows))
        if name == "emotions":
            rows = [[("Как я сейчас", "a:emotions"), ("Подробно и отношения", "a:mood")],
                    [("Что я думаю о нас", "a:reflection")], back()]
            return Screen(TITLES["emotions"], _kb(rows))
        if name == "inner":
            rows = [[("Мысли", "a:thoughts"), ("Интересы", "a:interests")],
                    [("Сон", "a:dream"), ("Чем я занималась", "a:activity_log")], back()]
            return Screen(TITLES["inner"], _kb(rows))
        if name == "persona":
            user = await st.get_user(user_id)
            reg = "на «вы»" if user and user.register == "vy" else "на «ты»"
            rows = [[("О Коломбине", "a:about"), ("Твой профиль", "a:profile")],
                    [(f"Обращение: {reg} ⇄", "t:register")], back()]
            return Screen(TITLES["persona"] + "\nКак тебя звать: /name Имя. Город: /city Город.", _kb(rows))
        if name == "activity":
            user = await st.get_user(user_id)
            sets = await st.settings(user_id)
            mode = {"off": "не пишу первой", "rare": "изредка", "normal": "когда есть повод"}.get(
                user.initiative_mode if user else "normal", "когда есть повод")
            onoff = lambda key, default="on": "вкл" if sets.get(key, default) != "off" else "выкл"  # noqa: E731
            paused = sets.get("paused") == "1"
            quiet = float(sets.get("quiet_until", "0") or 0) > time.time()
            voice = {"mirror": "в ответ на голос", "always": "всегда", "self": "когда сама захочу",
                     "off": "выкл"}.get(sets.get("voice", "mirror"), "в ответ на голос")
            rows = [[(f"Пишу первой: {mode} ⇄", "t:initiative")],
                    [(f"Голосом: {voice} ⇄", "t:voice"),
                     ("Кружки: " + {"mirror": "на кружок", "always": "всегда", "off": "выкл"}.get(
                         sets.get("circle", "mirror"), "на кружок") + " ⇄", "t:circle")],
                    [(f"Исследования: {onoff('research')} ⇄", "t:research"), (f"Сны: {onoff('dreams')} ⇄", "t:dreams")],
                    [("▶️ Снять паузу" if paused else "⏸ Пауза", "t:pause"),
                     ("🔔 Разбудить" if quiet else "🤫 Тишина до утра", "t:quiet")],
                    back()]
            return Screen(TITLES["activity"], _kb(rows))
        if name == "data":
            rows = [[("📦 Резервная копия", "f:backup"), ("📄 Экспорт", "f:export")],
                    [("↺ Сбросить разговор", "c:reset")], [("🗑 Забыть всё", "c:forget")], back()]
            return Screen(TITLES["data"], _kb(rows))
        if name == "system":
            rows = [[("Статус", "a:status")]]
            if user_id in self.admin_ids:
                rows.append([("Статус (тех.)", "a:status_admin"), ("Проверка сервисов", "a:check")])
            rows.append(back())
            return Screen(TITLES["system"], _kb(rows))
        return Screen("Такого раздела нет.", _kb([back()]))

    async def action(self, action: str, user_id: int) -> Screen:
        spec = ACTIONS.get(action)
        if not spec:
            return Screen("Не поняла, что открыть.", _kb([back()]))
        fn, parent, admin = spec
        if admin and user_id not in self.admin_ids:
            return Screen("Это только для администратора.", _kb([back(parent)]))
        text = await fn(self.engine, user_id)
        return Screen(text[:TG_LIMIT], _kb([back(parent)]))

    async def toggle(self, what: str, user_id: int) -> Screen:
        st = self.engine.store
        user = await st.get_user(user_id)
        if what == "initiative" and user:
            nxt = {"normal": "rare", "rare": "off", "off": "normal"}[user.initiative_mode or "normal"]
            await st.update_user(user_id, initiative_mode=nxt)
            return await self.menu("activity", user_id)
        if what in ("research", "dreams"):
            cur = await st.get_setting(user_id, what, "on")
            await st.set_setting(user_id, what, "off" if cur != "off" else "on")
            return await self.menu("activity", user_id)
        if what == "voice":
            cur = await st.get_setting(user_id, "voice", "mirror")
            nxt = {"mirror": "always", "always": "self", "self": "off", "off": "mirror"}[cur if cur in
                                                                                         {"mirror", "always", "self", "off"} else "mirror"]
            await st.set_setting(user_id, "voice", nxt)
            return await self.menu("activity", user_id)
        if what == "circle":
            cur = await st.get_setting(user_id, "circle", "mirror")
            await st.set_setting(user_id, "circle", {"mirror": "always", "always": "off"}.get(cur, "mirror"))
            return await self.menu("activity", user_id)
        if what == "pause":
            cur = await st.get_setting(user_id, "paused", "0")
            await st.set_setting(user_id, "paused", "0" if cur == "1" else "1")
            return await self.menu("activity", user_id)
        if what == "quiet":
            until = float(await st.get_setting(user_id, "quiet_until", "0") or 0)
            await st.set_setting(user_id, "quiet_until", "0" if until > time.time() else str(await quiet_until_morning(
                self.engine, user_id)))
            return await self.menu("activity", user_id)
        if what == "register" and user:
            await st.update_user(user_id, register="ty" if user.register == "vy" else "vy")
            return await self.menu("persona", user_id)
        return await self.menu("root", user_id)

    def confirm(self, action: str) -> Screen:
        text, yes = CONFIRM[action]
        return Screen(text, _kb([[(yes, f"x:{action}")], [("Отмена", "m:data")]]))

    async def execute(self, action: str, user_id: int) -> Screen:
        if action == "reset":
            return Screen(await I.reset_context(self.engine, user_id), _kb([back("data")]))
        if action == "forget":
            await self.engine.store.forget_user(user_id)
            await self.engine.store.update_user(user_id, preferred_name=None, last_user_msg_at=None, sessions=0)
            return Screen("…Хорошо. Как будто мы только встретились. Здравствуй.", None)
        return Screen("Не поняла.", _kb([back()]))

    async def file(self, what: str, user_id: int) -> Screen:
        if what == "backup":
            return Screen("Резервная копия всего, что я о тебе храню.", None,
                          (f"columbina_backup_{user_id}.json", await I.backup_json(self.engine, user_id)))
        return Screen("Экспорт: профиль, воспоминания, записки, дневник, интересы и история.", None,
                      (f"columbina_export_{user_id}.txt", await I.export_text(self.engine, user_id)))


async def quiet_until_morning(engine, user_id: int) -> float:
    """Тишина до ближайшего конца тихих часов (или до 9 утра) по времени собеседника."""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    user = await engine.store.get_user(user_id)
    try:
        tz = ZoneInfo((user.timezone if user else None) or engine.s.default_timezone)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    end_h = user.quiet_end if user else 9
    local = datetime.now(timezone.utc).astimezone(tz)
    target = local.replace(hour=end_h, minute=0, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.timestamp()
