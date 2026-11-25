"""Пост-обработка ответа: разметка реакций и частей, проверка стиля, анти-повторы."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ALLOWED_REACTIONS = {"🕊", "🌚", "🐳", "🍓", "😴", "🤔", "👀", "❤", "☃", "🥱", "🙈", "🤗", "🔥", "😁", "🤯", "😢"}
REACT_RE = re.compile(r"<react>\s*([^<]{1,8})\s*</react>")
# Модели без опыта с нашим форматом иногда пишут реакцию словами — превращаем её в настоящую реакцию.
TEXT_REACT_RE = re.compile(
    r"^\s*[\[(]?\s*(?:реакция|reaction|react)\s*[:：]?\s*([^\s\])]{1,4})\s*[\])]?[ \t]*\n?", re.IGNORECASE | re.MULTILINE)
SPLIT_RE = re.compile(r"\n\s*-{3,}\s*\n")


@dataclass
class ParsedReply:
    parts: list[str]
    reaction: str | None
    raw: str


@dataclass
class StyleVerdict:
    hard_violations: list[str] = field(default_factory=list)
    fixed_text: str = ""
    soft_fixes: list[str] = field(default_factory=list)


def parse_reply(text: str) -> ParsedReply:
    raw = text or ""
    reaction = None
    m = REACT_RE.search(raw) or TEXT_REACT_RE.search(raw)
    if m:
        emo = m.group(1).strip().replace("\ufe0f", "")
        if emo in ALLOWED_REACTIONS:
            reaction = emo
        raw = TEXT_REACT_RE.sub("", REACT_RE.sub("", raw)).strip()
    parts = [p.strip() for p in SPLIT_RE.split(raw) if p.strip()]
    if len(parts) == 1 and len(raw) > 320:
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        if 2 <= len(paras) <= 4 and all(len(p) < 700 for p in paras):
            parts = paras  # длинный ответ из абзацев — как несколько сообщений подряд
    if len(parts) > 3:
        parts = parts[:2] + ["\n\n".join(parts[2:])]
    return ParsedReply(parts=parts, reaction=reaction, raw=text)


class StyleGuard:
    def __init__(self, rules: dict[str, Any]):
        self.hard = {
            k: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in v["patterns"]]
            for k, v in rules.get("hard", {}).items()
        }
        soft = rules.get("soft", {})
        self.stage_dirs = [re.compile(p, re.MULTILINE) for p in soft.get("stage_directions", {}).get("patterns", [])]
        self.diminutives = [w.lower() for w in soft.get("diminutives", {}).get("words", [])]
        sig = rules.get("signature_markers", {})
        self.sig_window = int(sig.get("window", 6))
        self.sig_markers = {
            k: (re.compile(v["pattern"], re.IGNORECASE | re.MULTILINE), int(v["max"]), v["hint"])
            for k, v in sig.get("markers", {}).items()
        }
        self.max_excl = int(rules.get("max_exclamations", 1))

    def check(self, text: str) -> StyleVerdict:
        v = StyleVerdict(fixed_text=text)
        for name, pats in self.hard.items():
            if any(p.search(text) for p in pats):
                v.hard_violations.append(name)
        fixed = text
        for p in self.stage_dirs:
            if p.search(fixed):
                fixed = p.sub("", fixed)
                v.soft_fixes.append("stage_directions")
        low = fixed.lower()
        for w in self.diminutives:
            if re.search(rf"\b{re.escape(w)}\b", low):
                v.soft_fixes.append("diminutives")  # подсказка, не повод перегенерировать
                break
        # Не больше max_excl восклицательных знаков: остальные → точки.
        count = 0

        def _excl(m: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return m.group(0) if count <= self.max_excl else "."

        fixed2 = re.sub(r"!(?!\?)", _excl, fixed)
        if fixed2 != fixed:
            v.soft_fixes.append("exclamations")
        fixed = re.sub(r"[ \t]{2,}", " ", fixed2)
        fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip()
        v.fixed_text = fixed
        return v

    def overused(self, recent_replies: list[str], user_name: str | None = None) -> list[str]:
        """Подсказки против повторов: какие характерные приметы уже приелись."""
        window = recent_replies[-self.sig_window:]
        hints = []
        if user_name and len(user_name) > 1:
            n = sum(len(re.findall(rf"(?<!\w){re.escape(user_name)}(?!\w)", r, re.IGNORECASE)) for r in window)
            if n >= 2:
                hints.append("не обращайся по имени")
        if sum(1 for r in window if "\n- " in r or re.search(r"^\d\. ", r, re.M)) >= 2:
            hints.append("без списков — это разговор")
        for _, (rx, mx, hint) in self.sig_markers.items():
            n = sum(len(rx.findall(r)) for r in window)
            if n > mx:
                hints.append(hint)
        openings = [r.strip().split()[0].lower().strip(".,!?…") for r in window if r.strip()]
        if openings:
            top = max(set(openings), key=openings.count)
            if openings.count(top) >= 3:
                hints.append(f"не начинай ответ со слова «{top}»")
        return hints
