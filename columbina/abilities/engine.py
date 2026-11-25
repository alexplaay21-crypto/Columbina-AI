"""Abilities Engine: какие способности и ограничения актуальны прямо сейчас.

Работает детерминированно (без запросов к API): по ключевым словам сообщения
и условиям контекста (ночь, полнолуние, погода, стадия, просьба о чуде...).
В промпт попадают 0–2 компактные заметки + реальные данные «лунного чувства».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AbilityContext:
    text: str
    conditions: set[str] = field(default_factory=set)
    stage: int = 1


@dataclass
class ActiveAbility:
    id: str
    name: str
    note: str
    score: float


class AbilitiesEngine:
    def __init__(self, abilities_cfg: dict[str, Any]):
        self.abilities: list[dict[str, Any]] = abilities_cfg.get("abilities", [])
        self.traits: dict[str, Any] = abilities_cfg.get("traits", {})
        self.by_id = {a["id"]: a for a in self.abilities}

    def _conditions_met(self, conds: list[str], ctx: AbilityContext) -> int:
        hits = 0
        for c in conds:
            if c.startswith("stage_gte_"):
                if ctx.stage >= int(c.rsplit("_", 1)[1]):
                    hits += 1
            elif c in ctx.conditions:
                hits += 1
        return hits

    def active(self, ctx: AbilityContext, limit: int = 2) -> list[ActiveAbility]:
        low = ctx.text.lower()
        out: list[ActiveAbility] = []
        for a in self.abilities:
            trig = a.get("triggers", {})
            kw_hits = sum(1 for k in trig.get("keywords", []) if k in low)
            cond_hits = self._conditions_met(trig.get("conditions", []), ctx)
            score = kw_hits * 1.0 + cond_hits * 0.8
            # Путь на луну без стадии 5 — заметка нужна только при прямом упоминании, чтобы уклониться.
            if a["id"] == "moon_path" and ctx.stage < 5 and kw_hits == 0:
                continue
            if a["id"] == "lunar_sense" and kw_hits == 0:
                score *= 0.5  # фоновые данные и так в контексте
            if score >= 1.0:
                out.append(ActiveAbility(a["id"], a["name"], " ".join(str(a["prompt_note"]).split()), score))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit]

    def spec(self, ability_id: str) -> dict[str, Any]:
        return self.by_id[ability_id]
