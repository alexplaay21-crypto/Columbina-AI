"""GIF через GIPHY (бесплатный ключ developers.giphy.com) и «кружок» из голоса + аватара."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class GiphySearch:
    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None, rating: str = "pg-13"):
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=15.0)
        self.rating = rating

    async def find(self, query: str) -> str | None:
        try:
            r = await self.client.get("https://api.giphy.com/v1/gifs/search", params={
                "api_key": self.api_key, "q": query[:60], "limit": 5, "rating": self.rating})
            r.raise_for_status()
            for it in r.json().get("data", []):
                img = it.get("images", {})
                url = (img.get("downsized_medium") or img.get("original") or {}).get("url")
                if url:
                    return url
        except Exception as e:  # noqa: BLE001
            log.info("giphy failed: %s", e)
        return None


async def video_note(avatar: Path, voice_ogg: bytes, tmp_dir: Path) -> bytes | None:
    """Кружок Telegram: квадратное видео 384×384 из картинки-аватара и голоса."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=tmp_dir))
    try:
        audio, out = work / "a.ogg", work / "v.mp4"
        audio.write_bytes(voice_ogg)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(avatar), "-i", str(audio),
            "-vf", "scale=384:384:force_original_aspect_ratio=increase,crop=384:384,format=yuv420p",
            "-c:v", "libx264", "-tune", "stillimage", "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k",
            "-shortest", "-movflags", "+faststart", str(out),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), 90)
        return out.read_bytes() if out.exists() and out.stat().st_size > 0 else None
    finally:
        for f in work.iterdir():
            f.unlink(missing_ok=True)
        work.rmdir()
