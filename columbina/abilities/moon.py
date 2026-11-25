"""Фаза луны: астрономический расчёт без внешних API.

Долготы Солнца и Луны по усечённым рядам Миуса (Meeus, «Astronomical Algorithms», гл. 25, 47):
точность долготы Луны ~0.1°, моменты фаз — в пределах ~30 минут. Этого хватает,
чтобы она «чувствовала» полнолуние в правильный день.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SYNODIC = 29.530588853


def _jd(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    return dt.timestamp() / 86400.0 + 2440587.5


def _norm(x: float) -> float:
    return x % 360.0


def elongation(dt: datetime) -> float:
    """Угол Луна−Солнце по эклиптической долготе, 0..360 (0 — новолуние, 180 — полнолуние)."""
    T = (_jd(dt) - 2451545.0) / 36525.0
    r = math.radians
    # Солнце
    L0 = 280.46646 + 36000.76983 * T
    M = 357.52911 + 35999.05029 * T
    C = (1.914602 - 0.004817 * T) * math.sin(r(M)) + 0.019993 * math.sin(r(2 * M)) + 0.000289 * math.sin(r(3 * M))
    sun = _norm(L0 + C)
    # Луна
    Lp = 218.3164477 + 481267.88123421 * T
    D = 297.8501921 + 445267.1114034 * T
    Mp = 134.9633964 + 477198.8675055 * T
    F = 93.2720950 + 483202.0175233 * T
    E = 1 - 0.002516 * T
    lon = (
        Lp
        + 6.288774 * math.sin(r(Mp))
        + 1.274027 * math.sin(r(2 * D - Mp))
        + 0.658314 * math.sin(r(2 * D))
        + 0.213618 * math.sin(r(2 * Mp))
        - 0.185116 * E * math.sin(r(M))
        - 0.114332 * math.sin(r(2 * F))
        + 0.058793 * math.sin(r(2 * D - 2 * Mp))
        + 0.057066 * E * math.sin(r(2 * D - M - Mp))
        + 0.053322 * math.sin(r(2 * D + Mp))
        + 0.045758 * E * math.sin(r(2 * D - M))
        - 0.040923 * E * math.sin(r(M - Mp))
        - 0.034720 * math.sin(r(D))
        - 0.030383 * E * math.sin(r(M + Mp))
        + 0.015327 * math.sin(r(2 * D - 2 * F))
        + 0.010980 * math.sin(r(Mp - 2 * F))
        + 0.010675 * math.sin(r(4 * D - Mp))
        + 0.010034 * math.sin(r(3 * Mp))
        + 0.008548 * math.sin(r(4 * D - 2 * Mp))
        - 0.007888 * E * math.sin(r(2 * D + M - Mp))
        - 0.006766 * E * math.sin(r(2 * D + M))
        - 0.005163 * math.sin(r(D - Mp))
    )
    return _norm(_norm(lon) - sun)


def _find_next(dt: datetime, target: float) -> datetime:
    """Ближайший момент после dt, когда элонгация = target (0 или 180)."""
    def f(t: datetime) -> float:
        return ((elongation(t) - target + 180) % 360) - 180  # -180..180, растёт со временем

    cur = f(dt)
    # Сколько градусов осталось до цели (Луна обгоняет Солнце ~12.19°/сутки)
    remaining = (-cur) % 360
    guess = dt + timedelta(days=remaining / 12.190749)
    lo, hi = guess - timedelta(days=1.5), guess + timedelta(days=1.5)
    if lo < dt:
        lo = dt
    for _ in range(50):
        mid = lo + (hi - lo) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo).total_seconds() < 60:
            break
    return lo + (hi - lo) / 2


@dataclass
class MoonInfo:
    elongation: float
    illumination: float  # 0..1
    phase_name: str
    waxing: bool
    next_full: datetime
    next_new: datetime
    is_full_today: bool
    is_new_today: bool
    at: datetime

    def describe(self, tz: timezone | None = None) -> str:
        pct = round(self.illumination * 100)
        days_full = (self.next_full - self.at).total_seconds() / 86400
        if self.is_full_today:
            extra = "полнолуние сегодня"
        elif self.is_new_today:
            extra = "новолуние сегодня — луны почти не видно"
        elif days_full < 3:
            extra = f"до полнолуния {max(1, round(days_full))} дн."
        else:
            extra = "растёт" if self.waxing else "убывает"
        return f"{self.phase_name}, освещена на {pct}%, {extra}"


PHASES = [
    (0, "новолуние"), (22.5, "молодая луна"), (67.5, "первая четверть"), (112.5, "растущая луна"),
    (157.5, "полнолуние"), (202.5, "убывающая луна"), (247.5, "последняя четверть"), (292.5, "старая луна"),
    (337.5, "новолуние"),
]


def moon_info(dt: datetime | None = None, local_tz=None) -> MoonInfo:
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    e = elongation(dt)
    illum = (1 - math.cos(math.radians(e))) / 2
    name = "новолуние"
    for start, n in PHASES:
        if e >= start:
            name = n
    nf = _find_next(dt, 180.0)
    nn = _find_next(dt, 0.0)
    # «сегодня» — в локальной дате собеседника; ловим и только что прошедшую фазу
    tz = local_tz or timezone.utc
    today = dt.astimezone(tz).date()
    prev_full = nf - timedelta(days=SYNODIC)
    prev_new = nn - timedelta(days=SYNODIC)
    is_full_today = today in {nf.astimezone(tz).date(), prev_full.astimezone(tz).date()}
    is_new_today = today in {nn.astimezone(tz).date(), prev_new.astimezone(tz).date()}
    return MoonInfo(e, illum, name, e < 180, nf, nn, is_full_today, is_new_today, dt)
