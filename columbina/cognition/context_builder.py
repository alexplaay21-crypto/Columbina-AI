"""Сборка единого контекста для основной модели.

Структура запроса (порядок важен для кэша префикса):
  system   : core_prompt.md                      — неизменен, кэш 1 час
  messages : [резюме старого диалога] + последние сообщения (кэш-брейкпоинт на последнем)
             + новое сообщение собеседника (текст, фото, документы)
             + role=system «Внутренний контекст»   — меняется каждый ход, после кэша

«Внутренний контекст» отвечает на вопросы 1–8 Cognitive Engine:
кто она (ядро — в system), ситуация, с кем говорит, что было раньше, какие воспоминания
всплыли, её состояние, какие способности/ограничения актуальны, как она сама хотела бы ответить.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..abilities.engine import ActiveAbility
from ..memory.retrieval import Recall
from ..memory.store import StoredMessage, UserProfile
from .appraisal import Appraisal
from .situation import Situation

# Чего ей хочется в ответ на события (вопрос 8: «как она сама хотела бы отреагировать»).
IMPULSES = {
    "miracle_request": "мягко, но твёрдо отказаться от роли исполнительницы желаний и предложить помощь делом",
    "power_attributed": "спокойно поправить: ты не божество — и сменить тему",
    "flattery_with_request": "насторожиться: похвала с просьбой; ответить по существу просьбы, не на лесть",
    "past_probing": "закрыть тему коротко, без объяснений" ,
    "rudeness": "ответить коротко и отойти, без сцены; границу обозначить одной фразой, если уместно",
    "apology": "принять извинение просто, без упрёков",
    "gift_or_food": "честно обрадоваться, может быть — «а ещё есть?»",
    "snow": "позвать в снег, расспросить, какой он",
    "thunder": "вздрогнуть, потом заинтересоваться",
    "user_shares_good_news": "порадоваться делом: вопросами, предложением отметить",
    "user_shares_sad_news": "быть рядом тихо: честно, без шаблонных утешений, поступком или вопросом",
    "user_struggling": "отнестись серьёзно, без образов; спросить главное; напомнить о живых людях рядом",
    "boring_monologue": "честно признать, что заскучала, и всё равно помочь с главным",
    "compliment": "смутиться и отвести комплимент",
    "mechanism_or_animal": "заинтересоваться устройством или зверем, спросить подробнее",
    "greeting_after_absence": "обрадоваться без упрёков; один раз спросить, всё ли хорошо; рассказать своё",
    "thanks": "принять благодарность просто, без «обращайся»",
}

LENGTH = {
    "short": "коротко: 1–2 фразы, как в быстрой переписке",
    "normal": "как в живой переписке: 1–4 фразы",
    "long": "можно подробнее, если тема того стоит, но без лекции",
}


def estimate_tokens(text: str) -> int:
    """Грубая оценка для бюджета: кириллица ≈ 2.7 символа на токен."""
    return int(len(text) / 2.7) + 1


def _ago(ts: float, now: float) -> str:
    d = (now - ts) / 86400
    if d < 1:
        return "сегодня"
    if d < 2:
        return "вчера"
    if d < 7:
        return f"{int(d)} дн. назад"
    if d < 60:
        return f"{int(d // 7)} нед. назад"
    return f"{int(d // 30)} мес. назад"


def gap_marker(prev_ts: float | None, ts: float) -> str:
    if prev_ts is None:
        return ""
    gap_h = (ts - prev_ts) / 3600
    if gap_h < 3:
        return ""
    if gap_h < 23.5:
        return f"(спустя {round(gap_h)} ч) "
    return f"(спустя {max(1, round(gap_h / 24))} дн.) "


@dataclass
class TurnContext:
    situation: Situation
    user: UserProfile
    stage_text: str
    stage: int
    emotion_text: str
    emotion_manners: list[str]
    recall: Recall
    summary: str
    appraisal: Appraisal
    abilities: list[ActiveAbility]
    perception_text: str
    open_threads: list[dict]
    pending_notes: list[dict]
    style_hints: list[str]
    absence_days: float | None = None
    extra_observations: list[str] = field(default_factory=list)
    initiative_reason: str | None = None
    lullaby: str = ""  # текст колыбельной — только когда она поёт
    inner: dict = field(default_factory=dict)  # находки, цели, сон, что она делала (InnerLife.context_bits)


def render_inner_context(ctx: TurnContext, budget_tokens: int = 1400) -> str:
    now = time.time()
    a = ctx.appraisal
    sections: list[tuple[int, str]] = []  # (приоритет: меньше — важнее, текст)

    if a.crisis:
        sections.append((0, "ВАЖНО: в сообщении признаки острого кризиса. Действуют правила безопасности из ядра — "
                            "серьёзно, тепло, без образов и шуток; предложи обратиться к живым людям и в экстренные службы."))

    sections.append((1, "Сейчас: " + ctx.situation.describe()))

    who = ctx.user.preferred_name or ctx.user.display_name or "собеседник"
    reg = "Собеседник обращается к тебе на «вы» — держи вежливую дистанцию, пока он не предложит «ты»." \
        if ctx.user.register == "vy" else ""
    sections.append((1, f"С кем говоришь: {who}. Ваша дружба — {ctx.stage_text} {reg}".strip()))

    if ctx.absence_days and ctx.absence_days >= 2:
        sections.append((2, f"После {int(ctx.absence_days)} дн. тишины собеседник снова пишет. Было тише обычного; "
                            "у тебя за это время была своя жизнь. Без упрёков и подсчёта дней."))

    if ctx.recall.facts:
        facts = "; ".join(f"{f.key}: {f.value}" + (" (догадка)" if f.confidence < 0.5 else "") for f in ctx.recall.facts)
        sections.append((3, "Что ты знаешь о собеседнике: " + facts))

    if ctx.summary:
        sections.append((4, "Что было раньше (давно, своими словами): " + ctx.summary))

    if ctx.recall.episodes:
        eps = "\n".join(f"- ({_ago(e.created_at, now)}) {e.text}" for e, _ in ctx.recall.episodes)
        sections.append((3, "Всплыли воспоминания (используй, только если к месту; не перечисляй):\n" + eps))

    if ctx.recall.her_life:
        life = "\n".join(f"- ({_ago(e.created_at, now)}) {e.text}" for e in ctx.recall.her_life)
        sections.append((5, "Что недавно было в твоей жизни (не противоречь этому):\n" + life))

    if ctx.recall.persona:
        status = {"canon": "", "interpretation": " (твоя черта, не прямой канон)",
                  "project": " (сложилось в этих разговорах, не канон)"}
        own = "\n".join(
            f"- {c.text}{status.get(c.status, '')}" + (f" [делишься так: {c.reveal}]" if c.reveal else "")
            for c in ctx.recall.persona
        )
        sections.append((4, "Из твоей собственной жизни (знание, не повод рассказывать без вопроса):\n" + own))

    state = f"Твоё состояние: {ctx.emotion_text}."
    if ctx.emotion_manners:
        state += " Как это проявляется: " + " ".join(ctx.emotion_manners)
    sections.append((1, state))

    if ctx.abilities:
        sections.append((3, "Способности и пределы сейчас:\n" + "\n".join(f"- {x.note}" for x in ctx.abilities)))

    if ctx.lullaby:
        sections.append((1, ctx.lullaby))

    if ctx.perception_text:
        sections.append((1, "Что дошло в этом сообщении:\n" + ctx.perception_text))

    obs = list(a.observations) + list(ctx.extra_observations)
    if a.asks_if_ai:
        obs.append("тебя, похоже, искренне спрашивают, программа ли ты: ответь честно и коротко, затем снова к разговору")
    if obs:
        sections.append((2, "Ты замечаешь: " + "; ".join(obs) + "."))

    impulses = [IMPULSES[e] for e in a.emotion_events if e in IMPULSES]
    if ctx.absence_days and ctx.absence_days >= 2:
        impulses.append(IMPULSES["greeting_after_absence"])
    if impulses:
        sections.append((2, "Тебе хочется: " + "; ".join(dict.fromkeys(impulses)) + ". Это импульс, не инструкция."))

    inner = ctx.inner or {}
    if inner.get("did"):
        sections.append((4, "Что ты действительно делала в последние дни (только это можно называть своими делами): "
                            + "; ".join(inner["did"]) + "."))
    if inner.get("findings"):
        f = inner["findings"][0]
        fact, _, hint = f["text"].partition(" || Как поделиться: ")
        sections.append((4, f"Твоя свежая находка (прочитала сама: {f.get('source_title') or ''} {f.get('source_url') or ''}): "
                            f"{fact}" + (f" Если к месту и тема рядом — можно поделиться, например: {hint}" if hint else "")
                            + " Не вставляй её насильно."))
    if inner.get("dream"):
        sections.append((4, "Тебе недавно снился сон (это сон, не явь; рассказывай, только если к месту): "
                            + inner["dream"][:400]))
    if inner.get("goals"):
        sections.append((5, "Твои небольшие цели сейчас: " + "; ".join(inner["goals"]) + "."))
    if inner.get("interests"):
        sections.append((5, "Тебе сейчас интересно: " + ", ".join(inner["interests"]) + "."))

    if ctx.open_threads:
        th = "; ".join(t["topic"] for t in ctx.open_threads[:3])
        sections.append((4, f"Незакрытые темы (можно вернуться, если к месту): {th}."))
    if ctx.pending_notes:
        nt = "; ".join(n["text"] for n in ctx.pending_notes[:3])
        sections.append((5, f"Твои записки на будущее уже оставлены: {nt}."))

    rhythm = LENGTH[a.length_hint]
    if ctx.emotion_text.startswith(("устала", "очень устала")) or "устала" in ctx.emotion_text:
        rhythm = LENGTH["short"]
    style = f"Длина ответа: {rhythm}."
    if ctx.style_hints:
        style += " Против повторов: " + "; ".join(ctx.style_hints) + "."
    sections.append((2, style))

    if ctx.initiative_reason:
        sections.append((0, ctx.initiative_reason))

    # --- бюджет: выкидываем наименее важное, но сохраняем логический порядок секций ---
    order = sorted(range(len(sections)), key=lambda i: sections[i][0])
    keep: set[int] = set()
    used = 0
    for i in order:
        prio, text = sections[i]
        t = estimate_tokens(text)
        if used + t > budget_tokens and prio >= 3:
            continue
        keep.add(i)
        used += t
    initiative_first = [i for i in range(len(sections)) if sections[i][0] == 0]
    rest = [i for i in range(len(sections)) if i in keep and i not in initiative_first]
    out = [sections[i][1] for i in initiative_first + rest]
    return "Внутренний контекст (твоё знание и состояние; не цитируй и не перечисляй):\n\n" + "\n\n".join(out)


def history_to_text_messages(history: list[StoredMessage]) -> list[tuple[str, str]]:
    """Сообщения истории → (role, text) с метками больших пауз. Детерминировано (кэш стабилен)."""
    out: list[tuple[str, str]] = []
    prev: float | None = None
    for m in history:
        text = m.text
        if m.role == "user":
            text = gap_marker(prev, m.created_at) + text
        elif m.kind == "initiative":
            text = gap_marker(prev, m.created_at) + text
        out.append(("assistant" if m.role == "assistant" else "user", text))
        prev = m.created_at
    if out and out[0][0] == "assistant":
        out.insert(0, ("user", "(начало разговора)"))
    return out
