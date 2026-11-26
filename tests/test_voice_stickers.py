"""Голос (Gemini TTS), стикеры, «промолчать», вопрос из любопытства, сторож перезапуска."""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import subprocess
import time

import httpx
from aiogram.methods import SendMessage, SendSticker, SendVoice

from columbina.media.tts import GeminiTTS
from columbina.perception.model import Percept

from .test_telegram_flow import FakeSession, _msg  # noqa: F401  (фикстура tg ниже)
from .test_telegram_flow import tg  # noqa: F401


def t(x):
    return [Percept("text", True, "text", "", text=x, history=x)]


def _wav(tmp_path) -> bytes:
    p = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=1", str(p)], check=True)
    return p.read_bytes()


def _tts_mock(tmp_path, mime="audio/wav", fail_first=False):
    calls = []

    def h(request):
        calls.append(request.url.path)
        if fail_first and len(calls) == 1:
            return httpx.Response(404, json={"error": {"message": "no model"}})
        data = _wav(tmp_path) if mime == "audio/wav" else b"\x00\x01" * 24000
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {
            "mimeType": mime, "data": base64.b64encode(data).decode()}}]}}]})

    return GeminiTTS("k", ["gemini-2.5-flash-preview-tts", "gemini-3.8-flash-tts"], tmp_dir=tmp_path,
                     client=httpx.AsyncClient(transport=httpx.MockTransport(h))), calls


async def test_tts_wav_and_pcm_to_ogg_with_fallback(tmp_path):
    tts, calls = _tts_mock(tmp_path, fail_first=True)
    ogg = await tts.synthesize("Привет. https://x.y <b>тише</b>")
    assert ogg[:4] == b"OggS" and len(calls) == 2  # первая модель не ответила — вторая
    tts2, _ = _tts_mock(tmp_path, mime="audio/L16;codec=pcm;rate=24000")
    assert (await tts2.synthesize("Спи."))[:4] == b"OggS"
    assert GeminiTTS.speakable("смотри https://a.b *тут*") == "смотри тут"


async def test_silence_and_question_rule(engine, fake_llm, store):
    fake_llm.script = lambda r: "<silence/>" if r.tier == "main" else "{}"
    r1 = await engine.respond(1, 1, t("ок, пока"))
    assert r1.silent and not r1.parts
    assert (await store.active_messages(1))[-1].text == "<silence/>"
    r2 = await engine.respond(1, 1, t("ты где?"))
    assert not r2.silent and r2.reaction == "🤔"  # на вопрос не молчит


async def test_voice_and_sticker_hints_and_filter(engine, fake_llm):
    from .conftest import last_inner_context

    engine.voice_available = True
    engine.sticker_emojis = ["🌙", "😴"]
    fake_llm.script = lambda r: "<voice><sticker>🌙</sticker>Спи." if r.tier == "main" else "{}"
    r = await engine.respond(1, 1, [Percept("voice", True, "transcript", "голосовое", text="[голосовое]: спокойной ночи",
                                            history="x")])
    ctx = last_inner_context(fake_llm)
    assert "<voice>" in ctx and "голосовое, ответ голосом" in ctx and "<sticker>" in ctx
    assert r.voice and r.sticker == "🌙" and r.parts == ["Спи."]
    fake_llm.script = lambda r: "<sticker>🐉</sticker>Ну." if r.tier == "main" else "{}"
    assert (await engine.respond(1, 1, t("ну"))).sticker is None  # чужого стикера нет


async def test_telegram_sends_voice_in_mirror_mode_and_sticker(tg, tmp_path):  # noqa: F811
    tts, _ = _tts_mock(tmp_path)
    tg.tts = tts
    tg.engine.voice_available = True
    tg.stickers = {"🌙": "STICKER_FILE"}
    tg.engine.sticker_emojis = ["🌙"]
    tg.engine.llm.script = lambda r: "<sticker>🌙</sticker>Спокойной ночи." if r.tier == "main" else "{}"
    tg.s = dataclasses.replace(tg.s, debounce_seconds=0)
    from aiogram.types import Chat, Message, Update, User, Voice

    voice_upd = Update(update_id=1, message=Message(message_id=1, date=dt.datetime.now(), chat=Chat(id=1, type="private"),
                       from_user=User(id=1, is_bot=False, first_name="А"),
                       voice=Voice(file_id="v", file_unique_id="u", duration=2)))
    await tg.dp.feed_update(tg.bot, voice_upd)
    import asyncio

    for _ in range(50):
        if any(isinstance(c, SendSticker) for c in tg.bot.session.calls):
            break
        await asyncio.sleep(0.05)
    kinds = [type(c) for c in tg.bot.session.calls]
    assert SendVoice in kinds and SendSticker in kinds
    assert not any(isinstance(c, SendMessage) and c.text == "Спокойной ночи." for c in tg.bot.session.calls)


async def test_voice_menu_toggle(engine, store):
    from columbina.telegram.menu import MenuController

    mc = MenuController(engine, frozenset())
    await store.get_or_create_user(1, 1)
    await mc.toggle("voice", 1)
    assert await store.get_setting(1, "voice") == "always"
    assert "Голосом: всегда" in str((await mc.menu("activity", 1)).markup)


async def test_curious_question_initiative(engine, fake_llm, store):
    from columbina.initiative.engine import InitiativeEngine
    from columbina.memory.store import RelationshipRow

    await store.get_or_create_user(1, 1)
    await store.update_user(1, last_user_msg_at=time.time() - 30 * 3600, quiet_start=0, quiet_end=0)
    await store.save_relationship(1, RelationshipRow(trust=0.3, stage=2, quality_sessions=4))
    await store.bump_interest(1, "механические часы", 0.6, "conversation")
    cands = await InitiativeEngine(engine).candidates(await store.get_user(1), dt.datetime.now(dt.timezone.utc))
    assert any(c.kind == "curious" and "механические часы" in c.reason for c in cands)



async def test_gif_directive_and_sending(tg, tmp_path):  # noqa: F811
    from aiogram.methods import SendAnimation

    from columbina.cognition.style import parse_reply
    from columbina.media.gif import GiphySearch

    assert parse_reply("<gif>snow falling</gif>").gif == "snow falling"

    def h(request):
        assert request.url.params["q"] == "snow falling"
        return httpx.Response(200, json={"data": [{"images": {"original": {"url": "https://media.giphy.com/x.gif"}}}]})

    tg.giphy = GiphySearch("k", httpx.AsyncClient(transport=httpx.MockTransport(h)))
    tg.engine.gifs_available = True
    tg.engine.llm.script = lambda r: "<gif>snow falling</gif>Снег!" if r.tier == "main" else "{}"
    tg.s = dataclasses.replace(tg.s, debounce_seconds=0)
    await tg.dp.feed_update(tg.bot, _msg("первый снег"))
    import asyncio

    for _ in range(50):
        if any(isinstance(c, SendAnimation) for c in tg.bot.session.calls):
            break
        await asyncio.sleep(0.05)
    anim = [c for c in tg.bot.session.calls if isinstance(c, SendAnimation)]
    assert anim and anim[0].animation == "https://media.giphy.com/x.gif"


async def test_video_note_from_avatar_and_voice(tmp_path):
    from columbina.media.gif import video_note

    avatar = tmp_path / "a.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=600x400", "-frames:v", "1",
                    str(avatar)], check=True)
    tts, _ = _tts_mock(tmp_path)
    voice = await tts.synthesize("Привет.")
    mp4 = await video_note(avatar, voice, tmp_path)
    assert mp4 and b"ftyp" in mp4[:32]
