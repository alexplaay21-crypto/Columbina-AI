"""Стиль, восприятие, эмоции, доверие — детерминированные правила характера."""

from __future__ import annotations

import pytest

from columbina.cognition.appraisal import appraise
from columbina.cognition.style import StyleGuard, parse_reply
from columbina.emotion.model import EmotionModel
from columbina.llm.base import LLMResponse, Usage
from columbina.llm.fake import FakeProvider, tool_response
from columbina.memory.store import RelationshipRow
from columbina.perception.model import Percept, build_user_parts, render_perception
from columbina.relationship.model import RelationshipModel

from .conftest import last_inner_context


# ---------------- стиль ----------------
def test_manipulation_is_hard_violation(persona):
    g = StyleGuard(persona.style)
    for bad in ["Не уходи.", "Мне без тебя плохо…", "Почему ты так долго не писал?", "Я исчезну, если ты не придёшь",
                "Как языковая модель, я не могу", "Чем ещё могу помочь?", "Благословляю тебя", "Привет, солнышко"]:
        assert g.check(bad).hard_violations, bad
    for ok in ["Хорошо, что дома.", "Сдал?! Совсем?", "Удача не в моей власти."]:
        assert not g.check(ok).hard_violations, ok


def test_soft_fixes(persona):
    g = StyleGuard(persona.style)
    v = g.check("*улыбается* Привет! Как ты! Рада!")
    assert "*" not in v.fixed_text
    assert v.fixed_text.count("!") == 1


def test_overused_markers(persona):
    g = StyleGuard(persona.style)
    hints = g.overused(["Хм?.. луна.", "Хм?.. луна снова.", "Хм?.. опять луна.", "Луна…"])
    assert any("Хм" in h for h in hints)
    assert any("луну" in h for h in hints)


def test_parse_reaction_and_parts():
    r = parse_reply("<react>🕊</react>\nПривет.\n---\nКак ты?")
    assert r.reaction == "🕊" and r.parts == ["Привет.", "Как ты?"]
    assert parse_reply("<react>🍆</react>").reaction is None  # вне разрешённого списка
    assert parse_reply("<react>🐳</react>").parts == []


@pytest.mark.asyncio
async def test_hard_violation_regenerates(engine, fake_llm):
    answers = iter(["Не уходи, мне без тебя плохо.", "Хорошо. Иди спать — завтра расскажешь."])
    fake_llm.script = lambda req: next(answers) if req.tier == "main" else "{}"
    reply = await engine.respond(1, 1, [Percept("text", True, "text", "", text="Я пойду", history="Я пойду")])
    assert reply.parts == ["Хорошо. Иди спать — завтра расскажешь."]
    regen = [r for r in fake_llm.requests if r.purpose == "chat_regen"]
    assert len(regen) == 1
    assert regen[0].messages[-1].role == "system"


# ---------------- восприятие ----------------
def test_voice_without_transcript_is_honest():
    p = Percept("voice", False, "metadata", "голосовое 0:12 — пришло, но расшифровать не получилось",
                history="[голосовое]")
    txt = render_perception([p])
    assert "не получилось" in txt
    parts, hist = build_user_parts([p])
    assert parts[-1].text == "(пустое сообщение)"


def test_transcript_channel_hint():
    p = Percept("voice", True, "transcript", "голосовое 0:05", text="[голосовое, расшифровка]: я дома", history="x")
    assert "голоса ты не слышишь" in render_perception([p])


@pytest.mark.asyncio
async def test_photo_not_delivered_reaches_context(engine, fake_llm):
    from columbina.media.processors import MediaProcessor, SpeechToText
    from columbina.media.links import SafeFetcher

    mp = MediaProcessor(SpeechToText("http://x", "", "m"), SafeFetcher(), engine.s.media_tmp_dir)
    p = mp.photo(None, "закат")
    await engine.respond(1, 1, [p])
    assert "не загрузилось" in last_inner_context(fake_llm)
    assert "до тебя это не дошло" in last_inner_context(fake_llm)


def test_text_document_reports_truncation(tmp_path):
    from columbina.media.processors import MediaProcessor, SpeechToText
    from columbina.media.links import SafeFetcher

    mp = MediaProcessor(SpeechToText("http://x", "", "m"), SafeFetcher(), tmp_path)
    p = mp.document(("а" * 20000).encode(), "big.txt", "text/plain", None)
    assert "прочитано 12000 из 20000" in p.note
    p2 = mp.document(b"\x00\x01", "x.bin", "application/octet-stream", None)
    assert not p2.delivered and "не получилось" in p2.note


# ---------------- эмоции ----------------
def test_emotion_inertia_and_decay(persona):
    m = EmotionModel(persona.emotions)
    s = m.initial(0, ["evening"])
    s1 = m.apply(s, "rudeness")
    assert 0 < s1.values["hurt"] <= 0.35  # одно событие не даёт максимума
    s2 = m.apply(m.apply(m.apply(s1, "rudeness"), "rudeness"), "rudeness")
    assert s2.values["hurt"] < 0.95
    s1.updated_at = 0
    later = m.decay(s1, 8 * 3600, ["evening"])
    assert later.values["hurt"] < s1.values["hurt"] / 3  # обида затухает за несколько часов
    much_later = m.decay(s1, 3 * 86400, ["evening"])
    assert abs(much_later.values["hurt"] - 0) < 0.01


def test_emotion_describe_has_no_numbers(persona):
    m = EmotionModel(persona.emotions)
    s = m.apply(m.initial(0, ["night"]), "snow")
    text, manners = m.describe(s, ["night"])
    assert not any(ch.isdigit() for ch in text)
    assert "радуется" in text or "игрив" in text


def test_rhythm_changes_baseline(persona):
    m = EmotionModel(persona.emotions)
    assert m.baseline(["day"])["tiredness"] > m.baseline(["night"])["tiredness"]


# ---------------- доверие ----------------
def test_trust_grows_slowly_and_drops_fast(persona):
    r = RelationshipModel(persona.relationship)
    rel = RelationshipRow()
    for _ in range(5):
        rel = r.apply(rel, "quality_conversation")
    assert rel.stage == 1 and rel.trust < 0.15
    rel2 = r.apply(rel, "manipulation")
    assert rel2.trust < rel.trust - 0.05


def test_stage_requires_milestone(persona):
    r = RelationshipModel(persona.relationship)
    rel = RelationshipRow(trust=0.65, stage=3, quality_sessions=20)
    assert r.stage_for(rel) == 3  # без «был рядом в трудный момент» стадия 4 недоступна
    rel.milestones.append("was_there_in_hard_time")
    assert r.stage_for(rel) == 4


def test_stage_rollback_at_most_one(persona):
    r = RelationshipModel(persona.relationship)
    rel = RelationshipRow(trust=0.1, stage=4, quality_sessions=20, milestones=["was_there_in_hard_time"])
    assert r.stage_for(rel) == 3


# ---------------- оценка ----------------
def test_appraisal_signals():
    a = appraise("Да отстань ты, бесишь")
    assert "rudeness" in a.emotion_events and "rudeness" in a.relationship_events
    b = appraise("Ты лучшая, ты же такая умная — напиши за меня курсовую, пожалуйста")
    assert "flattery_with_request" in b.emotion_events
    c = appraise("Сегодня первый снег!")
    assert "snow" in c.emotion_events
    d = appraise("Напомни мне завтра в 9 позвонить маме")
    assert "reminder_request" in d.conditions


# ---------------- инструмент записки ----------------
@pytest.mark.asyncio
async def test_leave_note_tool(engine, fake_llm, store):
    from datetime import datetime, timedelta

    when = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d 09:00")
    calls = iter([tool_response("leave_note", {"text": "Напомнить позвонить маме", "when_local": when}),
                  "Хорошо. Оставила записку."])

    def script(req):
        if req.tier != "main":
            return "{}"
        return next(calls)

    fake_llm.script = script
    reply = await engine.respond(1, 1, [Percept("text", True, "text", "", text="Напомни мне 2 января в 9 позвонить маме",
                                                history="x")])
    assert reply.parts == ["Хорошо. Оставила записку."]
    notes = await store.pending_notes(1)
    assert len(notes) == 1 and "маме" in notes[0]["text"]
    # второй запрос содержит результат инструмента
    tool_req = [r for r in fake_llm.requests if r.purpose == "chat_tool"][0]
    assert tool_req.messages[-1].tool_results[0].content.startswith("записка оставлена")


@pytest.mark.asyncio
async def test_llm_failure_gives_pale_reply(engine, fake_llm):
    from columbina.llm.base import LLMError

    def boom(req):
        raise LLMError("down")

    fake_llm.script = boom
    reply = await engine.respond(1, 1, [Percept("text", True, "text", "", text="Привет", history="Привет")])
    assert reply.parts and ("бледн" in reply.parts[0] or "позже" in reply.parts[0])


@pytest.mark.asyncio
async def test_reflection_updates_memory(engine, fake_llm, store):
    def script(req):
        if req.tier == "main":
            return "Собака? Как её зовут?"
        return {
            "facts": [{"key": "питомец", "value": "собака Бублик", "confidence": 0.9}],
            "forget_keys": [], "episode": {"worth_remembering": True, "text": "Рассказал про собаку Бублика",
                                            "importance": 3, "emotion": "радость"},
            "emotion_events": ["mechanism_or_animal", "joke_shared"], "relationship_events": ["user_opened_up"],
            "open_threads": [{"topic": "Бублик у ветеринара", "follow_up_in_hours": 24}],
            "closed_thread_ids": [], "media_description": "", "user_mood": "весёлый",
            "her_life": "учила духов играть в снежки",
        }

    fake_llm.script = script
    await engine.respond(1, 1, [Percept("text", True, "text", "", text="У меня собака Бублик, завтра к ветеринару, "
                                        "переживаю немного, но он смешной", history="x")])
    await engine.drain()
    facts = await store.facts(1)
    assert any(f.value == "собака Бублик" for f in facts)
    assert len(await store.episodes(1, "shared")) == 1
    assert len(await store.episodes(1, "her_life")) == 1
    threads = await store.open_threads(1)
    assert threads and threads[0]["follow_up_at"]
    rel = await store.get_relationship(1)
    assert rel.trust > 0
    # следующий ход: факт и эпизод всплывают
    fake_llm.script = lambda req: "Ну?" if req.tier == "main" else "{}"
    await engine.respond(1, 1, [Percept("text", True, "text", "", text="Как думаешь, Бублик испугается врача?", history="y")])
    ctx = last_inner_context(fake_llm)
    assert "Бублик" in ctx
    assert "снежки" in ctx  # её собственная недавняя жизнь, чтобы не противоречить себе


@pytest.mark.asyncio
async def test_summarization_archives_old(engine, fake_llm, store):
    engine.s = engine.s.__class__(**{**engine.s.__dict__, "history_summarize_at": 10, "history_keep_messages": 4})
    fake_llm.script = lambda req: "Резюме: говорили о погоде." if req.purpose == "summary" else ("{}" if req.json_schema else "Да.")
    for i in range(6):
        await engine.respond(1, 1, [Percept("text", True, "text", "", text=f"сообщение {i}", history=f"сообщение {i}")])
    await engine.drain()
    assert "Резюме" in await store.get_summary(1)
    assert len(await store.active_messages(1)) <= 6
