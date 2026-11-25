"""Загрузка личности из data/persona.

Формат хранения:
  core_prompt.md     — компактное ядро личности (постоянный, кэшируемый системный промпт)
  knowledge.yaml     — фрагменты биографии/вкусов/привычек для семантического поиска (min_stage)
  abilities.yaml     — модель способностей (канон / симуляция / бот)
  emotions.yaml      — эмоциональная модель (базовые уровни, затухание, события)
  relationship.yaml  — стадии дружбы, события доверия, вехи
  style_rules.yaml   — правила пост-проверки стиля
  lullaby.yaml       — её колыбельная (точный текст, смысл, правила исполнения)
  source/            — исходное досье (истина в последней инстанции, в запросы не передаётся)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml


@dataclass
class KnowledgeChunk:
    id: str
    kind: str
    text: str
    tags: list[str]
    min_stage: int = 1
    reveal: str | None = None
    status: str = "canon"  # canon | interpretation | project

    @property
    def search_text(self) -> str:
        return " ".join(self.tags) + ". " + self.text

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(self.search_text.encode()).hexdigest()[:16]


@dataclass
class Persona:
    root: Path
    core_prompt: str
    knowledge: list[KnowledgeChunk]
    abilities: dict[str, Any]
    emotions: dict[str, Any]
    relationship: dict[str, Any]
    style: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def lullaby_block(self) -> str:
        """Текст для контекста, когда она поёт. Пусто, если колыбельная не задана."""
        lb = self.extra.get("lullaby")
        if not lb:
            return ""
        p = lb["parts"]
        song = f"{p['versus']}\n\n{p['chorus']}\n\n{p['conclusio']}"
        rules = " ".join(f"{r}" for r in lb.get("rules", []))
        return (f"Твоя колыбельная ({lb.get('language', '')}; единственное, что ты помнишь о своём рождении). "
                f"Куплет повторяется дважды, кода — в самом конце:\n{song}\n\n"
                f"Смысл приблизительно: {' '.join(lb.get('meaning_ru', '').split())}\n{rules}")

    @cached_property
    def ability_list(self) -> list[dict[str, Any]]:
        return list(self.abilities.get("abilities", []))


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _canon_map(root: Path) -> list[tuple[re.Pattern[str], str]]:
    path = root / "canon.yaml"
    if not path.exists():
        return []
    items = _yaml(path).get("replace", [])
    items = sorted(items, key=lambda i: len(i["from"]), reverse=True)  # длинные — первыми
    return [(re.compile(rf"(?<!\w){re.escape(i['from'])}(?!\w)"), i["to"]) for i in items]


def _tag_map(mapping: list[tuple[re.Pattern[str], str]], tag: str) -> str:
    for rx, to in mapping:
        tag = re.sub(rx.pattern, to.lower(), tag, flags=re.IGNORECASE)
    return tag


def apply_canon(text: str, mapping: list[tuple[re.Pattern[str], str]]) -> str:
    for rx, to in mapping:
        text = rx.sub(to, text)
    return text


def load_persona(root: Path | str, canon_mode: str = "genshin") -> Persona:
    root = Path(root)
    mapping = _canon_map(root) if canon_mode == "genshin" else []
    lullaby_path = root / "lullaby.yaml"
    extra: dict[str, Any] = {"lullaby": _yaml(lullaby_path)} if lullaby_path.exists() else {}
    knowledge_raw = _yaml(root / "knowledge.yaml").get("chunks", [])
    chunks = [
        KnowledgeChunk(
            id=c["id"],
            kind=c.get("kind", "biography"),
            text=apply_canon(" ".join(str(c["text"]).split()), mapping),
            tags=sorted({str(t).lower() for t in c.get("tags", [])}
                        | {_tag_map(mapping, str(t).lower()) for t in c.get("tags", [])}),
            min_stage=int(c.get("min_stage", 1)),
            reveal=c.get("reveal"),
            status=str(c.get("status", "canon")),
        )
        for c in knowledge_raw
    ]
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate knowledge chunk ids")
    return Persona(
        root=root,
        core_prompt=(root / "core_prompt.md").read_text(encoding="utf-8").strip(),
        knowledge=chunks,
        abilities=_yaml(root / "abilities.yaml"),
        emotions=_yaml(root / "emotions.yaml"),
        relationship=_yaml(root / "relationship.yaml"),
        style=_yaml(root / "style_rules.yaml"),
        extra=extra,
    )
