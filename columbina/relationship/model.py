"""Модель дружбы: доверие, стадии, вехи, откаты. Пользователь стадий не видит."""

from __future__ import annotations

from typing import Any

from ..memory.store import RelationshipRow


class RelationshipModel:
    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self.stages: dict[int, dict[str, Any]] = {int(k): v for k, v in config["stages"].items()}
        self.events: dict[str, float] = config.get("events", {})
        self.milestones: dict[str, list[str]] = config.get("milestones", {})

    def apply(self, rel: RelationshipRow, event: str) -> RelationshipRow:
        delta = self.events.get(event)
        if delta is None:
            return rel
        trust = rel.trust
        if delta > 0:
            # Медленный рост, который замедляется у потолка.
            trust += delta * (1.0 - trust * 0.5)
        else:
            trust += delta
        trust = max(0.0, min(1.0, trust))
        seen = list(rel.seen_events) + [event]
        milestones = list(rel.milestones)
        for m, needed in self.milestones.items():
            if m not in milestones and any(e in seen for e in needed):
                milestones.append(m)
        quality = rel.quality_sessions + (1 if event == "quality_conversation" else 0)
        out = RelationshipRow(trust=trust, stage=rel.stage, quality_sessions=quality,
                              milestones=milestones, seen_events=seen[-50:])
        out.stage = self.stage_for(out)
        return out

    def stage_for(self, rel: RelationshipRow) -> int:
        stage = 1
        for n in sorted(self.stages):
            s = self.stages[n]
            if rel.trust < float(s.get("min_trust", 0)):
                break
            if rel.quality_sessions < int(s.get("min_sessions", 0)):
                break
            req = s.get("requires_milestone")
            if req and req not in rel.milestones:
                break
            stage = n
        # Откат не больше чем на одну стадию за раз — дружба не рушится от одного эпизода.
        if stage < rel.stage - 1:
            stage = rel.stage - 1
        return stage

    def describe(self, rel: RelationshipRow) -> str:
        s = self.stages.get(rel.stage, self.stages[1])
        return f"{s['name']}. {s['behavior']} Готова делиться: {s['disclosure']}."
