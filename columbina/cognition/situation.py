"""Текущая ситуация: время у собеседника, часть суток, луна, погода."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ..abilities.moon import MoonInfo, moon_info
from ..abilities.weather import Weather

WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


def part_of_day(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "day"
    if 17 <= hour < 22:
        return "evening"
    return "night"


PART_RU = {"morning": "утро", "day": "день", "evening": "вечер", "night": "ночь"}


@dataclass
class Situation:
    now_utc: datetime
    local: datetime
    tz_name: str
    part: str
    moon: MoonInfo
    weather: Weather | None = None
    tags: list[str] = field(default_factory=list)  # ритм для эмоций: morning/day/evening/night/full_moon/...

    def describe(self) -> str:
        d = self.local
        s = (f"{WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]}, {d:%H:%M} у собеседника — "
             f"{PART_RU[self.part]}. Луна: {self.moon.describe()}.")
        if self.weather:
            s += f" Погода у него: {self.weather.describe()}."
        else:
            s += " Погоды у него ты не знаешь."
        return s


def build_situation(tz_name: str | None, default_tz: str, weather: Weather | None = None,
                    now: datetime | None = None) -> Situation:
    now = now or datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(tz_name or default_tz)
        name = tz_name or default_tz
    except Exception:
        tz = ZoneInfo(default_tz)
        name = default_tz
    local = now.astimezone(tz)
    part = part_of_day(local.hour)
    moon = moon_info(now, tz)
    tags = [part]
    if moon.is_full_today:
        tags.append("full_moon")
    if moon.is_new_today:
        tags.append("new_moon")
    if weather and "heat" in weather.tags:
        tags.append("heat")
    return Situation(now, local, name, part, moon, weather, tags)
