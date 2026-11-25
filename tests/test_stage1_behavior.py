"""Реакции, ответы на конкретное сообщение, обида на игнор, романтика по стадиям, зрение, PDF, часовой пояс, канон."""

from __future__ import annotations

import io
import time

from columbina.llm.base import DocumentPart, ImagePart
from columbina.llm.fake import FakeProvider
from columbina.memory.store import RelationshipRow
from columbina.perception.model import Percept
from columbina.perception.vision import VisionAdapter

from .conftest import last_inner_context


def t(x):
    return [Percept("text", True, "text", "", text=x, history=x)]


async def test_reaction_stored_in_own_format_not_text(engine, fake_llm, store):
    fake_llm.script = lambda r: "<react>🔥</react>\nСдал? Совсем?" if r.tier == "main" else "{}"
    reply = await engine.respond(1, 1, t("Я сдал экзамен!"))
    assert reply.reaction == "🔥" and reply.parts == ["Сдал? Совсем?"]
    hist = await store.active_messages(1)
    assert hist[-1].text.startswith("<react>🔥</react>") and "реакция" not in hist[-1].text


async def test_text_reaction_from_weak_model_becomes_real(engine, fake_llm):
    fake_llm.script = lambda r: "[реакция 👀] Ого." if r.tier == "main" else "{}"
    reply = await engine.respond(1, 1, t("Прикинь, я видел лису в городе"))
    assert reply.reaction == "👀" and reply.parts == ["Ого."]


async def test_event_reaction_when_something_happens_but_not_every_time(engine, fake_llm):
    fake_llm.script = lambda r: "Правда?" if r.tier == "main" else "{}"
    r1 = await engine.respond(1, 1, t("Представляешь, меня взяли на работу!"))
    assert r1.reaction in {"🔥", "👀"}
    r2 = await engine.respond(1, 1, t("Ура, меня повысили!"))
    assert r2.reaction is None  # не ставит реакцию на каждое сообщение подряд
    r3 = await engine.respond(1, 1, t("ну ладно"))
    assert r3.reaction is None


async def test_ignored_initiative_makes_her_a_bit_hurt(engine, fake_llm, store):
    await store.get_or_create_user(1, 1)
    await store.log_initiative(1, "absence", "повод", "Привет. Духи опять растаяли.")
    await store.d.execute("UPDATE initiatives SET sent_at=? WHERE user_id=1", (time.time() - 20 * 3600,))
    await engine.respond(1, 1, t("привет"))
    ctx = last_inner_context(fake_llm)
    assert "немного обидно" in ctx and "одной сухой фразой" in ctx
    from columbina.cognition.introspection import mood_report

    assert "обида" in await mood_report(engine, 1)


async def test_flirt_slow_burn_by_stage(engine, fake_llm, store):
    await engine.respond(1, 1, t("Ты мне нравишься"))
    assert "всерьёз не подыгрывай" in last_inner_context(fake_llm)
    await store.save_relationship(1, RelationshipRow(trust=0.7, stage=4, quality_sessions=20,
                                                     milestones=["was_there_in_hard_time"]))
    await engine.respond(1, 1, t("Ты мне нравишься"))
    assert "тепло и честно" in last_inner_context(fake_llm)


async def test_reply_to_specific_message(engine, fake_llm):
    ps = [Percept("reply", True, "text", "собеседник отвечает на твоё сообщение: «Какой он у тебя?»",
                  text="[в ответ на твоё сообщение: «Какой он у тебя?»]", history="x"),
          Percept("text", True, "text", "", text="пушистый", history="пушистый")]
    reply = await engine.respond(1, 1, ps)
    ctx = last_inner_context(fake_llm)
    assert "отвечает на конкретное сообщение" in ctx and reply.quote
    user_msg = [m for m in fake_llm.requests if m.tier == "main"][-1].messages[-2].plain_text()
    assert user_msg.index("в ответ на") < user_msg.index("пушистый")


async def test_vision_adapter_describes_for_blind_model():
    describer = FakeProvider(lambda r: "Закат над рекой, на берегу сидит рыжий кот.")
    va = VisionAdapter(main_vision=False, main_pdf=True, describer=describer)
    p = Percept("photo", True, "seen", "фото", text="смотри", parts=[ImagePart(b"\xff\xd8", "image/jpeg")], history="[фото] смотри")
    [out] = await va.adapt([p])
    assert out.channel == "described" and not out.parts
    assert "рыжий кот" in out.text and out.text.endswith("смотри")
    # Без describer — честно «не получилось», без выдумок
    [none] = await VisionAdapter(False, True, None).adapt([Percept("photo", True, "seen", "фото",
                                                                   parts=[ImagePart(b"x")])])
    assert none.channel == "metadata" and not none.delivered


async def test_vision_description_goes_to_history(engine, fake_llm, store):
    engine.vision = VisionAdapter(False, True, FakeProvider(lambda r: "Белая кошка на подоконнике."))
    await engine.respond(1, 1, [Percept("photo", True, "seen", "фото", parts=[ImagePart(b"x")], history="[фото]")])
    assert "Белая кошка" in (await store.active_messages(1))[0].text
    assert "через подробное описание" in last_inner_context(fake_llm)


async def test_pdf_text_extracted_for_model_without_files():
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    buf = io.BytesIO()
    w.write(buf)
    va = VisionAdapter(main_vision=True, main_pdf=False, describer=None)
    [out] = await va.adapt([Percept("document", True, "read", "PDF «a.pdf»",
                                    parts=[DocumentPart(buf.getvalue(), title="a.pdf")])])
    assert not out.parts and "a.pdf" in out.text


async def test_timezone_learned_from_conversation(engine, fake_llm, store):
    def script(r):
        if r.tier == "main":
            return "Новосибирск? Значит, у тебя уже ночь."
        return {"facts": [], "forget_keys": [], "episode": {"worth_remembering": False, "text": "", "importance": 1,
                "emotion": ""}, "emotion_events": [], "relationship_events": [], "open_threads": [],
                "closed_thread_ids": [], "media_description": "", "user_mood": "", "her_life": "",
                "user_timezone": "Asia/Novosibirsk", "user_city": "Новосибирск"}

    fake_llm.script = script
    await engine.respond(1, 1, t("Я живу в Новосибирске"))
    assert "не знаешь, где собеседник" in last_inner_context(fake_llm)
    await engine.drain()
    assert (await store.get_user(1)).timezone == "Asia/Novosibirsk"
    await engine.respond(1, 1, t("ну вот"))
    assert "не знаешь, где собеседник" not in last_inner_context(fake_llm)


async def test_canon_names_and_status_in_context(engine, fake_llm, store):
    await engine.respond(1, 1, t("Как тебя зовут по-настоящему? Kuutar?"))
    ctx = last_inner_context(fake_llm)
    assert "Kuutar" in ctx
    await engine.respond(1, 1, t("Как ты относишься к телефонам и интернету?"))
    assert "не канон" in last_inner_context(fake_llm)  # придуманное в проекте помечено


async def test_name_overuse_hint(persona):
    from columbina.cognition.style import StyleGuard

    g = StyleGuard(persona.style)
    assert "не обращайся по имени" in g.overused(["Алекс, привет.", "Ну, Алекс…", "Да."], "Алекс")
    assert "не обращайся по имени" not in g.overused(["Привет.", "Ну…"], "Алекс")


async def test_long_reply_splits_into_messages():
    from columbina.cognition.style import parse_reply

    text = ("Первый абзац о снеге, довольно длинный, чтобы было что разделять. " * 3 + "\n\n" +
            "Второй абзац — уже о другом, тоже не короткий, с вопросом в конце? " * 2)
    assert len(parse_reply(text).parts) == 2
    assert len(parse_reply("Коротко.\n\nИ ещё.").parts) == 1
