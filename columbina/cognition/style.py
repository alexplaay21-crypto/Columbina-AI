"""Пост-обработка ответа: разметка реакций и частей, проверка стиля, анти-повторы."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

ALLOWED_REACTIONS = {"🕊", "🌚", "🐳", "🍓", "😴", "🤔", "👀", "❤", "☃", "🥱", "🙈", "🤗", "🔥", "😁", "🤯", "😢"}
REACT_RE = re.compile(r"<react>\s*([^<]{1,8})\s*</react>")
EMPTY_TAG_RE = re.compile(r"<(react|sticker|gif|clip)>\s*</\1>", re.I)
LONE_TAG_RE = re.compile(r"</?(?:react|sticker|gif|clip)>", re.I)
# Модели без опыта с нашим форматом иногда пишут реакцию словами — превращаем её в настоящую реакцию.
TEXT_REACT_RE = re.compile(
    r"^\s*[\[(]?\s*(?:реакция|reaction|react)\s*[:：]?\s*([^\s\])]{1,4})\s*[\])]?[ \t]*\n?", re.IGNORECASE | re.MULTILINE)
SPLIT_RE = re.compile(r"\n\s*-{3,}\s*\n")
VOICE_RE = re.compile(r"<voice\s*/?>", re.I)
STICKER_RE = re.compile(r"<sticker>\s*([^<]{1,8})\s*</sticker>", re.I)
SILENCE_RE = re.compile(r"<silence\s*/?>", re.I)
GIF_RE = re.compile(r"<gif>\s*([^<]{2,60})\s*</gif>", re.I)
CLIP_RE = re.compile(r"<clip>\s*([^<]{1,80})\s*</clip>", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F900-\U0001F9FF]\ufe0f?")
ASSISTANTISH = re.compile(r"отличный вопрос|рада помочь|вот несколько|надеюсь,? это поможет|если хочешь,? (?:я )?могу|"
                          r"не стесняйся|в заключение|подытожим|во-первых.*во-вторых|я здесь, чтобы", re.I | re.S)


@dataclass
class ParsedReply:
    parts: list[str]
    reaction: str | None
    raw: str
    voice: bool = False  # она сама хочет ответить голосом (<voice>)
    sticker: str | None = None  # эмодзи стикера из её набора (<sticker>🌙</sticker>)
    silent: bool = False  # решила промолчать (<silence/>)
    gif: str | None = None  # запрос для GIF (<gif>снег падает</gif>)
    clip: str | None = None  # сохранённое голосовое по названию (<clip>колыбельная</clip>)


@dataclass
class StyleVerdict:
    hard_violations: list[str] = field(default_factory=list)
    fixed_text: str = ""
    soft_fixes: list[str] = field(default_factory=list)


HOLLOW_OPENING = re.compile(r"^(?:о,? )?(?:как )?(?:здорово|отлично|классно|круто|супер|замечательно|прекрасно|"
                            r"интересно|понимаю)[!.,]+\s+", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z«\"(])")
TO_USER = re.compile(r"(?<!\w)(?:ты|тебя|тебе|тобой|твой|твоя|твоё|твое|твои|вы|вас|вам)(?!\w)", re.I)


# Её «лунная» тема и красивости: в обычной болтовне — не чаще раза в несколько ответов.
LORE = re.compile(r"лун|куувах|куухен|хийси|hiisi|нод-кра|(?<!\w)дух|созвезд|звёзд|звезд|серебр|шеп(?:ч|от)|мерца|"
                  r"ветер[оа]к|сиян|фатуи|предвестн|царица|luonnotar|луоннотар|сёстр|сестр[аыуе]", re.I)
ABOUT_HER = re.compile(r"о себе|кто ты|где ты|откуда ты|твой (?:дом|зал)|как ты выглядишь|что (?:ты )?(?:делала|делаешь)|"
                       r"чем (?:ты )?занимал|как (?:у тебя )?(?:дела|день)|расскажи (?:про|о) (?:тебя|себя|свою)", re.I)


def parse_reply(text: str) -> ParsedReply:
    raw = text or ""
    reaction = None
    voice = bool(VOICE_RE.search(raw))
    sm = STICKER_RE.search(raw)
    sticker = sm.group(1).strip().replace("\ufe0f", "") if sm else None
    silent = bool(SILENCE_RE.search(raw))
    gm = GIF_RE.search(raw)
    gif = gm.group(1).strip() if gm else None
    cm = CLIP_RE.search(raw)
    clip = cm.group(1).strip() if cm else None
    raw = CLIP_RE.sub("", raw)
    raw = GIF_RE.sub("", SILENCE_RE.sub("", STICKER_RE.sub("", VOICE_RE.sub("", raw))))
    raw = EMPTY_TAG_RE.sub("", raw)  # пустые <react></react> и обломки тегов не уходят собеседнику текстом
    m = REACT_RE.search(raw) or TEXT_REACT_RE.search(raw)
    if m:
        emo = m.group(1).strip().replace("\ufe0f", "")
        if emo in ALLOWED_REACTIONS:
            reaction = emo
        raw = TEXT_REACT_RE.sub("", REACT_RE.sub("", raw)).strip()
    # Обломки эмодзи (склейка U+200D, селектор U+FE0F) остаются после вырезания тегов — убираем по краям.
    raw = LONE_TAG_RE.sub("", raw)
    parts = [q for p in SPLIT_RE.split(raw) if (q := p.strip().strip("\u200d\ufe0f").strip())]
    if len(parts) == 1 and len(raw) > 320:
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        if 2 <= len(paras) <= 4 and all(len(p) < 700 for p in paras):
            parts = paras  # длинный ответ из абзацев — как несколько сообщений подряд
    if len(parts) > 3:
        parts = parts[:2] + ["\n\n".join(parts[2:])]
    return ParsedReply(parts=parts, reaction=reaction, raw=text, voice=voice, sticker=sticker,
                       silent=silent and not parts and not gif and not clip, gif=gif, clip=clip)


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
        self.rng = random.Random()

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
        # Эмодзи — не больше одного на ответ.
        found = EMOJI.findall(fixed2)
        if len(found) > 1:
            first = found[0]
            fixed2 = re.sub(r"[ \t]+([.,!?…])", r"\1", EMOJI.sub("", fixed2))
            fixed2 = fixed2.rstrip() + " " + first if fixed2.rstrip().endswith(tuple(".!?…")) else fixed2
            v.soft_fixes.append("emoji")
        fixed = re.sub(r"[ \t]{2,}", " ", fixed2)
        fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip()
        v.fixed_text = fixed
        return v

    def de_interview(self, parts: list[str], prev_reply: str, user_text: str, rng: random.Random | None = None) -> list[str]:
        """Убирает «интервью»: пустую похвалу в начале, лишние вопросы и дежурный вопрос в конце.

        Модель любит схему «Здорово! Как прошло? Что делал?» — живой человек так не пишет.
        Оставляем максимум один вопрос; вопрос в конце снимаем, если прошлый ответ тоже кончался
        вопросом, или (иногда) если это дежурный вопрос к собеседнику. Содержательное не трогаем.
        """
        rng = rng or self.rng
        prev_q = prev_reply.rstrip().endswith("?")
        out: list[str] = []
        seen_q = False
        for part in parts:
            m = HOLLOW_OPENING.match(part)
            if m and len(part) - m.end() > 8:
                rest = part[m.end():]
                part = rest[0].upper() + rest[1:]
            sents = SENT_SPLIT.split(part)
            kept: list[str] = []
            for sent in sents:
                if sent.rstrip().endswith("?"):
                    if seen_q and kept and len(sent.split()) > 2:
                        continue  # второй вопрос подряд — уже допрос
                    seen_q = True
                kept.append(sent)
            out.append(" ".join(kept))
        last = SENT_SPLIT.split(out[-1]) if out else []
        if len(last) >= 2 and last[-1].rstrip().endswith("?") and "?" not in last[-2]:
            generic = bool(TO_USER.search(last[-1])) and len(last[-1]) < 90
            if prev_q or (generic and rng.random() < 0.6):
                out[-1] = " ".join(last[:-1])
        return [o.strip() for o in out if o.strip()] or parts

    def de_lore(self, parts: list[str], recent_replies: list[str], user_text: str) -> list[str]:
        """Убирает лишние «лунные» фразы и красивости, если собеседник не о том.

        Если спросили о ней или тема и так про луну/духов — оставляем до двух таких фраз.
        Иначе: одна фраза позволена, только если в трёх её прошлых ответах этого не было.
        """
        if LORE.search(user_text):
            return parts
        allowed = 2 if ABOUT_HER.search(user_text) else (0 if any(LORE.search(r) for r in recent_replies[-3:]) else 1)
        used = 0
        out: list[str] = []
        for part in parts:
            kept = []
            for sent in SENT_SPLIT.split(part):
                if LORE.search(sent):
                    used += 1
                    if used > allowed:
                        continue
                kept.append(sent)
            out.append(" ".join(kept))
        out = [o.strip() for o in out if o.strip()]
        return out or parts

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
        if sum(1 for r in window[-4:] if EMOJI.search(r.split("</react>")[-1])) >= 2:
            hints.append("без эмодзи в этот раз — ты ставила их слишком часто")
        # Защита характера: если начинает звучать как ассистент — мягко возвращаем к себе.
        drift = sum(1 for r in window if ASSISTANTISH.search(r))
        if drift >= 2:
            hints.append("ты начинаешь звучать как вежливый ассистент — говори как Коломбина: проще, со своим мнением, "
                         "без услужливых оборотов")
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
