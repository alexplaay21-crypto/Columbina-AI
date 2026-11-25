"""Погода у собеседника через Open-Meteo (бесплатно, без ключа). Только если он сам дал город/геолокацию."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

WMO = {
    0: "ясно", 1: "почти ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь и туман",
    51: "морось", 53: "морось", 55: "сильная морось", 56: "ледяная морось", 57: "ледяная морось",
    61: "небольшой дождь", 63: "дождь", 65: "ливень", 66: "ледяной дождь", 67: "ледяной дождь",
    71: "небольшой снег", 73: "снег", 75: "сильный снег", 77: "снежная крупа",
    80: "ливневый дождь", 81: "ливни", 82: "сильные ливни", 85: "снегопад", 86: "сильный снегопад",
    95: "гроза", 96: "гроза с градом", 99: "гроза с градом",
}


@dataclass
class Weather:
    code: int
    temperature: float
    is_day: bool
    fetched_at: float

    @property
    def text(self) -> str:
        return WMO.get(self.code, "погода неясна")

    @property
    def tags(self) -> list[str]:
        t = []
        if self.code in (71, 73, 75, 77, 85, 86):
            t.append("snow")
        if 51 <= self.code <= 67 or 80 <= self.code <= 82:
            t.append("rain")
        if self.code >= 95:
            t.append("thunder")
        if self.temperature >= 28:
            t.append("heat")
        if self.code in (0, 1):
            t.append("clear")
        return t

    def describe(self) -> str:
        return f"{self.text}, {round(self.temperature):+d}°"


class WeatherService:
    def __init__(self, ttl_seconds: int = 1800):
        self.ttl = ttl_seconds
        self._cache: dict[tuple[float, float], Weather] = {}
        self._client = httpx.AsyncClient(timeout=10.0)

    async def geocode(self, city: str) -> tuple[float, float, str, str | None] | None:
        r = await self._client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru", "format": "json"},
        )
        r.raise_for_status()
        res = r.json().get("results") or []
        if not res:
            return None
        g = res[0]
        return float(g["latitude"]), float(g["longitude"]), g.get("name", city), g.get("timezone")

    async def timezone_for(self, lat: float, lon: float) -> str | None:
        try:
            r = await self._client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current": "is_day", "timezone": "auto"},
            )
            r.raise_for_status()
            return r.json().get("timezone")
        except Exception:
            return None

    async def current(self, lat: float, lon: float) -> Weather | None:
        key = (round(lat, 2), round(lon, 2))
        cached = self._cache.get(key)
        if cached and time.time() - cached.fetched_at < self.ttl:
            return cached
        try:
            r = await self._client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code,is_day"},
            )
            r.raise_for_status()
            cur = r.json()["current"]
            w = Weather(int(cur["weather_code"]), float(cur["temperature_2m"]), bool(cur["is_day"]), time.time())
        except Exception:
            return cached  # пусть лучше старое, чем выдуманное; None если ничего нет
        self._cache[key] = w
        return w
