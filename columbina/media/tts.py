"""Голос Коломбины: синтез речи через Gemini TTS (бесплатный ключ AI Studio) → OGG/Opus для Telegram.

Голос синтетический (готовый голос Gemini), не клон актрисы — как требует досье.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
from pathlib import Path

import httpx

from . import ffmpeg

log = logging.getLogger(__name__)

STYLE = "Скажи тихо, мягко и спокойно, как задумчивая девушка, немного по-ночному: "
SING = "Тихо напевай, как колыбельную, медленно и нежно: "


class GeminiTTS:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, models: list[str], voice: str = "Leda", tmp_dir: Path | None = None,
                 client: httpx.AsyncClient | None = None, max_chars: int = 900):
        self.api_key = api_key
        self.models = models
        self.voice = voice
        self.tmp_dir = tmp_dir or Path(tempfile.gettempdir())
        self.client = client or httpx.AsyncClient(timeout=90.0)
        self.max_chars = max_chars

    @property
    def available(self) -> bool:
        return bool(self.api_key) and ffmpeg.have_ffmpeg()

    @staticmethod
    def speakable(text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[*_`#<>\[\]]", "", text)
        return " ".join(text.split())

    async def synthesize(self, text: str, singing: bool = False) -> bytes | None:
        text = self.speakable(text)[: self.max_chars]
        if not text or not self.available:
            return None
        for model in self.models:
            # Старые TTS-модели понимают указание стиля; новые читают его вслух — им только текст.
            prompt = ((SING if singing else STYLE) + text) if "2.5" in model else text
            try:
                r = await self.client.post(self.URL.format(model=model), headers={"x-goog-api-key": self.api_key}, json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}}},
                })
                r.raise_for_status()
                part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
                return await self._to_ogg(base64.b64decode(part["data"]), part.get("mimeType", ""))
            except Exception as e:  # noqa: BLE001 — пробуем следующую модель
                log.info("TTS %s failed: %s", model, e)
        return None

    async def _to_ogg(self, audio: bytes, mime: str) -> bytes | None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=self.tmp_dir))
        try:
            src, dst = work / "in.raw", work / "out.ogg"
            src.write_bytes(audio)
            rate = re.search(r"rate=(\d+)", mime)
            args = ["ffmpeg", "-y", "-v", "error"]
            if "l16" in mime.lower() or "pcm" in mime.lower():
                args += ["-f", "s16le", "-ar", rate.group(1) if rate else "24000", "-ac", "1"]
            args += ["-i", str(src), "-c:a", "libopus", "-b:a", "32k", "-ac", "1", str(dst)]
            proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.DEVNULL,
                                                        stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(proc.wait(), 60)
            return dst.read_bytes() if dst.exists() and dst.stat().st_size > 0 else None
        finally:
            for f in work.iterdir():
                f.unlink(missing_ok=True)
            work.rmdir()
