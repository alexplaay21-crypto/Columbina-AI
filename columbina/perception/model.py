"""Perception Engine: что именно дошло до Коломбины.

Каждый входящий элемент превращается в Percept. Главный принцип: она воспринимает ровно то,
что технически дошло до модели, и это явно сказано во внутреннем контексте.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Part, TextPart


@dataclass
class Percept:
    kind: str  # text | photo | voice | audio | video | video_note | document | link | forward | sticker | location | other
    delivered: bool
    # Как она это восприняла: seen | transcript | frames | read | text | metadata | nothing
    channel: str
    note: str  # строка для «Что дошло»
    text: str | None = None  # текст/расшифровка/извлечённое содержимое (идёт в сообщение пользователя)
    parts: list[Part] = field(default_factory=list)  # картинки/документы для модели
    history: str = ""  # как сохранить в истории (без бинарных данных)
    meta: dict[str, Any] = field(default_factory=dict)


CHANNEL_HINT = {
    "seen": "ты это видишь",
    "frames": "ты видишь только отдельные кадры",
    "described": "ты видишь это через подробное описание, а не своими глазами",
    "transcript": "голоса ты не слышишь — дошли только слова",
    "read": "ты это прочитала",
    "text": "",
    "metadata": "дошло только описание, без содержимого",
    "nothing": "до тебя это не дошло",
}


def render_perception(percepts: list[Percept]) -> str:
    """Секция «Что дошло» для внутреннего контекста. Пустая, если пришёл только обычный текст."""
    lines = []
    for p in percepts:
        if p.kind == "text" and p.channel == "text":
            continue
        hint = CHANNEL_HINT.get(p.channel, "")
        line = p.note + (f" ({hint})" if hint else "")
        lines.append("- " + line)
    return "\n".join(lines)


def build_user_parts(percepts: list[Percept]) -> tuple[list[Part], str]:
    """Части сообщения пользователя для модели и текстовая версия для истории."""
    parts: list[Part] = []
    texts: list[str] = []
    history: list[str] = []
    for p in percepts:
        parts.extend(p.parts)
        if p.text:
            texts.append(p.text)
        history.append(p.history or (p.text or ""))
    combined = "\n\n".join(t for t in texts if t.strip())
    if combined:
        parts.append(TextPart(combined))
    elif parts:
        parts.append(TextPart("(без подписи)"))
    else:
        parts.append(TextPart("(пустое сообщение)"))
    return parts, "\n".join(h for h in history if h.strip())
