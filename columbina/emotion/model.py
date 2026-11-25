"""Динамическое эмоциональное состояние.

Не случайность: состояние меняется только от событий (содержание сообщений, воспоминания,
отношение к собеседнику, время суток и луна) и плавно затухает к базовому уровню.
Инерция: одно событие сдвигает эмоцию не больше чем на max_step.
Наружу (в промпт) выходит только словесное описание, без чисел.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmotionState:
    values: dict[str, float]
    mood: float = 0.0  # фон дня: -1..1, меняется медленно
    last_cause: str | None = None
    updated_at: float = 0.0
    recent_events: list[str] = field(default_factory=list)


class EmotionModel:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.emotions: dict[str, dict[str, Any]] = config["emotions"]
        self.events: dict[str, dict[str, float]] = config.get("events", {})
        self.rhythm: dict[str, dict[str, float]] = config.get("rhythm", {})
        self.max_step: float = float(config.get("max_step", 0.35))
        self.mood_half_life_h: float = float(config.get("mood_half_life_h", 18))

    # ---------- базовые уровни с учётом ритма ----------
    def baseline(self, rhythm_tags: list[str]) -> dict[str, float]:
        base = {k: float(v.get("baseline", 0.0)) for k, v in self.emotions.items()}
        for tag in rhythm_tags:
            for k, d in self.rhythm.get(tag, {}).items():
                if k in base:
                    base[k] = min(1.0, max(0.0, base[k] + d))
        return base

    def initial(self, now: float, rhythm_tags: list[str]) -> EmotionState:
        return EmotionState(values=self.baseline(rhythm_tags), updated_at=now)

    # ---------- затухание ----------
    def decay(self, state: EmotionState, now: float, rhythm_tags: list[str]) -> EmotionState:
        dt_min = max(0.0, (now - state.updated_at) / 60)
        base = self.baseline(rhythm_tags)
        vals = {}
        for k, cfg in self.emotions.items():
            cur = state.values.get(k, base[k])
            hl = float(cfg.get("half_life_min", 60))
            factor = math.pow(0.5, dt_min / hl) if hl > 0 else 0.0
            vals[k] = base[k] + (cur - base[k]) * factor
        mood_factor = math.pow(0.5, (dt_min / 60) / self.mood_half_life_h)
        return EmotionState(values=vals, mood=state.mood * mood_factor, last_cause=state.last_cause,
                            updated_at=now, recent_events=state.recent_events[-6:])

    # ---------- события ----------
    def apply(self, state: EmotionState, event: str, weight: float = 1.0, cause: str | None = None) -> EmotionState:
        deltas = self.events.get(event)
        if not deltas:
            return state
        vals = dict(state.values)
        for k, d in deltas.items():
            if k not in vals:
                continue
            step = max(-self.max_step, min(self.max_step, d * weight))
            # Насыщение: чем ближе к краю, тем слабее сдвиг (нет резких скачков до 1.0).
            if step > 0:
                step *= 1.0 - vals[k] * 0.6
            vals[k] = min(1.0, max(0.0, vals[k] + step))
        positive = sum(v for k, v in deltas.items() if k in {"joy", "playfulness", "tenderness", "calm", "curiosity"})
        negative = sum(v for k, v in deltas.items() if k in {"irritation", "hurt", "sadness", "boredom", "fright"})
        mood = max(-1.0, min(1.0, state.mood + 0.15 * weight * (positive - negative)))
        return EmotionState(values=vals, mood=mood, last_cause=cause or event, updated_at=state.updated_at,
                            recent_events=(state.recent_events + [event])[-6:])

    # ---------- словесное описание ----------
    def dominant(self, state: EmotionState, rhythm_tags: list[str], n: int = 3) -> list[tuple[str, float]]:
        base = self.baseline(rhythm_tags)
        # «Заметность» эмоции: насколько она выше своего базового уровня (calm — по абсолютному значению).
        salience = []
        for k, v in state.values.items():
            s = v if k == "calm" else v - base.get(k, 0.0) * 0.5
            salience.append((k, s, v))
        salience.sort(key=lambda t: t[1], reverse=True)
        return [(k, v) for k, s, v in salience if v >= 0.25][:n]

    def describe(self, state: EmotionState, rhythm_tags: list[str]) -> tuple[str, list[str]]:
        dom = self.dominant(state, rhythm_tags)
        if not dom:
            return "спокойна, ровна", [self.emotions["calm"]["manner"]]
        words = []
        manners = []
        for k, v in dom:
            label = self.emotions[k]["label"]
            if v < 0.4 and k != "calm":
                label = "чуть " + label
            elif v > 0.75:
                label = "очень " + label
            words.append(label)
            manners.append(self.emotions[k]["manner"])
        mood = ""
        if state.mood > 0.35:
            mood = "; день в целом хороший"
        elif state.mood < -0.35:
            mood = "; день в целом тяжёлый"
        return ", ".join(words) + mood, manners[:2]

    def intensity(self, state: EmotionState, name: str) -> float:
        return state.values.get(name, 0.0)

    def flowers(self, state: EmotionState, rhythm_tags: list[str]) -> str:
        fl = self.cfg.get("flowers", {})
        dom = self.dominant(state, rhythm_tags, n=1)
        key = dom[0][0] if dom else "calm"
        return fl.get(key, fl.get("calm", "серебристые"))
