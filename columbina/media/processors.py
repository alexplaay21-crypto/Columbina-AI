"""Обработчики мультимедиа. Каждый тип превращается в Percept (см. perception/model.py).

Все обработчики работают с байтами/файлами, а не с Telegram API: Telegram-слой только скачивает
файл и вызывает нужный обработчик. Это позволяет тестировать их отдельно.
"""

from __future__ import annotations

import io
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..llm.base import DocumentPart, ImagePart
from ..perception.model import Percept
from . import ffmpeg
from .links import SafeFetcher, UnsafeURL

log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1568
MAX_PDF_BYTES = 20 * 1024 * 1024
TEXT_EXT = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml", ".log", ".ini", ".srt"}


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def prepare_image(data: bytes) -> tuple[bytes, str]:
    """Уменьшает большие изображения (экономия токенов) и приводит к JPEG/PNG."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        if max(img.size) <= MAX_IMAGE_SIDE and fmt in ("JPEG", "PNG", "WEBP", "GIF") and len(data) < 4_500_000:
            mt = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}[fmt]
            return data, mt
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/jpeg"


# ---------------- Речь в текст ----------------
class SpeechToText:
    """Whisper-совместимый endpoint: POST {base}/audio/transcriptions (OpenAI, Groq, локальные серверы)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def transcribe(self, path: Path, language: str | None = "ru") -> str | None:
        if not self.available:
            return None
        files = {"file": (path.name, path.read_bytes(), "audio/ogg")}
        data = {"model": self.model}
        if language:
            data["language"] = language
        try:
            r = await self._client.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data, files=files,
            )
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
        except Exception as e:  # сеть/лимиты — честно сообщим, что голос не разобран
            log.warning("STT failed: %s", e)
            return None


@dataclass
class MediaProcessor:
    stt: SpeechToText
    fetcher: SafeFetcher
    tmp_dir: Path
    link_fetch_enabled: bool = True

    # ---------- фото ----------
    def photo(self, data: bytes | None, caption: str | None) -> Percept:
        if not data:
            return Percept("photo", False, "nothing", "фото — не загрузилось",
                           text=caption, history=f"[фото, не дошло]{' ' + caption if caption else ''}")
        img, mt = prepare_image(data)
        return Percept("photo", True, "seen", "фото", text=caption, parts=[ImagePart(img, mt)],
                       history=f"[фото]{' ' + caption if caption else ''}")

    # ---------- голос и аудио ----------
    async def audio(self, path: Path | None, kind: str, duration: float, caption: str | None = None,
                    title: str | None = None) -> Percept:
        label = {"voice": "голосовое", "audio": "аудиофайл"}.get(kind, "аудио")
        if title:
            label += f" «{title}»"
        dur = f" {fmt_duration(duration)}" if duration else ""
        if path is None:
            return Percept(kind, False, "nothing", f"{label}{dur} — файл не загрузился", text=caption,
                           history=f"[{label}, не дошло]")
        if kind == "audio" and duration > 600:
            return Percept(kind, False, "metadata", f"{label}{dur} — слишком длинный, не расшифрован",
                           text=caption, history=f"[{label}{dur}]")
        transcript = await self.stt.transcribe(path)
        prosody = ""
        try:
            if transcript and kind == "voice" and duration > 2:
                st = await ffmpeg.speech_stats(path)
                words = len(transcript.split())
                speaking = max(0.5, duration - st["pause_total"])
                wps = words / speaking
                pace = "быстро" if wps > 3.2 else "медленно" if wps < 1.8 else "в обычном темпе"
                pauses = "с долгими паузами" if st["pause_total"] > duration * 0.35 else ""
                prosody = f"; по звуку: говорит {pace}{', ' + pauses if pauses else ''}"
        except Exception:
            prosody = ""
        if transcript is None:
            note = f"{label}{dur} — пришло, но расшифровать не получилось"
            return Percept(kind, False, "metadata", note, text=caption, history=f"[{label}{dur}, не расшифровано]")
        if not transcript:
            note = f"{label}{dur} — в расшифровке нет слов (тишина, музыка или шум)"
            return Percept(kind, True, "transcript", note, text=caption, history=f"[{label}{dur}, без слов]")
        note = f"{label}{dur}{prosody}"
        text = f"[{label}, расшифровка]: {transcript}" + (f"\n{caption}" if caption else "")
        return Percept(kind, True, "transcript", note, text=text, history=text)

    # ---------- видео и кружки ----------
    async def video(self, path: Path | None, kind: str, duration: float, caption: str | None = None) -> Percept:
        label = "видеосообщение" if kind == "video_note" else "видео"
        dur = f" {fmt_duration(duration)}" if duration else ""
        if path is None:
            return Percept(kind, False, "nothing", f"{label}{dur} — файл не загрузился", text=caption,
                           history=f"[{label}, не дошло]")
        if not ffmpeg.have_ffmpeg():
            return Percept(kind, False, "metadata", f"{label}{dur} — открыть не получилось", text=caption,
                           history=f"[{label}{dur}]")
        real_dur, has_audio = await ffmpeg.duration_and_audio(path)
        duration = real_dur or duration
        # Кадры: 1 на ~5 секунд, от 2 до 6; без первых/последних 5% (часто чёрные).
        n = max(2, min(6, int(duration // 5) + 1)) if duration > 0 else 1
        span = duration * 0.9
        times = [duration * 0.05 + span * (i + 0.5) / n for i in range(n)] if duration > 0 else [0.0]
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=self.tmp_dir))
        try:
            frames = await ffmpeg.extract_frames(path, work / "frames", times)
            parts = [ImagePart(f.read_bytes(), "image/jpeg") for f in frames]
            transcript = None
            if has_audio:
                audio_path = work / "audio.ogg"
                if await ffmpeg.extract_audio(path, audio_path):
                    transcript = await self.stt.transcribe(audio_path)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        bits = [f"{label}{dur}: {len(parts)} кадр(ов) с интервалом ~{fmt_duration(duration / max(1, n))}"]
        if has_audio:
            bits.append("звук: " + ("расшифрована речь" if transcript else
                                     "речи не разобрать" if transcript == "" else "расшифровать не удалось"))
        else:
            bits.append("без звука")
        text_parts = []
        if transcript:
            text_parts.append(f"[{label}, расшифровка звука]: {transcript}")
        if caption:
            text_parts.append(caption)
        channel = "frames" if parts else "metadata"
        history = f"[{label}{dur}" + (f", речь: {transcript[:300]}" if transcript else "") + "]"
        return Percept(kind, bool(parts), channel, "; ".join(bits), text="\n".join(text_parts) or None,
                       parts=parts, history=history + (f" {caption}" if caption else ""),
                       meta={"frames": len(parts), "has_audio": has_audio})

    # ---------- документы ----------
    def document(self, data: bytes | None, filename: str, mime: str | None, caption: str | None) -> Percept:
        name = filename or "файл"
        ext = Path(name).suffix.lower()
        if data is None:
            return Percept("document", False, "nothing", f"документ «{name}» — не загрузился (возможно, слишком большой)",
                           text=caption, history=f"[документ «{name}», не дошёл]")
        if (mime or "").startswith("image/"):
            p = self.photo(data, caption)
            p.note = f"изображение файлом «{name}»"
            return p
        if ext == ".pdf" or mime == "application/pdf":
            if len(data) > MAX_PDF_BYTES:
                return Percept("document", False, "metadata", f"PDF «{name}» — слишком большой", text=caption,
                               history=f"[PDF «{name}»]")
            return Percept("document", True, "read", f"PDF «{name}»", text=caption,
                           parts=[DocumentPart(data, "application/pdf", title=name)],
                           history=f"[PDF «{name}»]{' ' + caption if caption else ''}")
        if ext == ".docx":
            try:
                import docx  # python-docx

                d = docx.Document(io.BytesIO(data))
                body = "\n".join(p.text for p in d.paragraphs)
            except Exception:
                body = None
            if body is not None:
                return self._text_doc(name, body, caption)
        if ext in TEXT_EXT or (mime or "").startswith("text/"):
            for enc in ("utf-8", "cp1251"):
                try:
                    return self._text_doc(name, data.decode(enc), caption)
                except UnicodeDecodeError:
                    continue
        return Percept("document", False, "metadata", f"файл «{name}» ({mime or 'неизвестный тип'}) — прочитать не получилось",
                       text=caption, history=f"[файл «{name}»]")

    def _text_doc(self, name: str, body: str, caption: str | None, limit: int = 12000) -> Percept:
        total = len(body)
        clipped = body[:limit]
        note = f"документ «{name}»" + (f" — прочитано {len(clipped)} из {total} символов" if total > limit else "")
        text = f"[документ «{name}»]:\n{clipped}" + (f"\n\n{caption}" if caption else "")
        return Percept("document", True, "read", note, text=text, history=f"[документ «{name}»]{' ' + caption if caption else ''}")

    # ---------- ссылки ----------
    async def link(self, url: str) -> Percept:
        if not self.link_fetch_enabled:
            return Percept("link", False, "metadata", f"ссылка {url} — открывать ссылки сейчас нельзя", history=f"[ссылка {url}]")
        try:
            page = await self.fetcher.fetch(url)
        except UnsafeURL as e:
            log.info("link blocked %s: %s", url, e)
            return Percept("link", False, "nothing", f"ссылка {url} — не открылась", history=f"[ссылка {url}, не открылась]")
        except Exception as e:
            log.info("link failed %s: %s", url, e)
            return Percept("link", False, "nothing", f"ссылка {url} — не открылась", history=f"[ссылка {url}, не открылась]")
        if not page.text.strip():
            return Percept("link", False, "metadata", f"ссылка {url} — страница без текста (может, видео или музыка)",
                           history=f"[ссылка {url}]")
        note = f"страница по ссылке{': «' + page.title + '»' if page.title else ''}" + (" — прочитано начало" if page.truncated else "")
        text = f"[содержимое ссылки {page.url}{' — ' + page.title if page.title else ''}]:\n{page.text}"
        return Percept("link", True, "read", note, text=text, history=f"[ссылка {url}{' — ' + page.title if page.title else ''}]")
