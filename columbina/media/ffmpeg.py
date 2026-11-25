"""Тонкие асинхронные обёртки над ffmpeg/ffprobe (без shell, аргументы списком)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path


class FFmpegMissing(RuntimeError):
    pass


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


async def _run(*args: str, timeout: float = 120.0) -> tuple[int, bytes, bytes]:
    if not have_ffmpeg():
        raise FFmpegMissing("ffmpeg/ffprobe не установлены")
    proc = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode or 0, out, err


async def probe(path: Path) -> dict:
    code, out, _ = await _run("ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path))
    if code != 0:
        return {}
    return json.loads(out or b"{}")


async def duration_and_audio(path: Path) -> tuple[float, bool]:
    info = await probe(path)
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return dur, has_audio


async def extract_audio(src: Path, dst: Path) -> bool:
    """В моно 16 кГц OGG/Opus — компактно для STT."""
    code, _, _ = await _run("ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "libopus", "-b:a", "32k", str(dst))
    return code == 0 and dst.exists() and dst.stat().st_size > 0


async def to_mp3(src: Path, dst: Path) -> bool:
    code, _, _ = await _run("ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "libmp3lame", "-b:a", "48k", str(dst))
    return code == 0 and dst.exists() and dst.stat().st_size > 0


async def extract_frames(src: Path, out_dir: Path, timestamps: list[float], max_side: int = 768) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for i, t in enumerate(timestamps):
        dst = out_dir / f"frame_{i:02d}.jpg"
        code, _, _ = await _run(
            "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
            "-vf", f"scale='min({max_side},iw)':-2", "-q:v", "4", str(dst),
        )
        if code == 0 and dst.exists():
            frames.append(dst)
    return frames


async def speech_stats(path: Path) -> dict:
    """Реальные акустические признаки: доля пауз и громкость (не «эмоции», а факты о звуке)."""
    code, _, err = await _run("ffmpeg", "-v", "info", "-i", str(path), "-af",
                              "silencedetect=noise=-35dB:d=0.6,volumedetect", "-f", "null", "-")
    text = err.decode(errors="ignore")
    silences = [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", text)]
    mean = re.search(r"mean_volume: (-?[\d.]+) dB", text)
    return {
        "pause_total": sum(silences),
        "pause_count": len(silences),
        "mean_volume_db": float(mean.group(1)) if mean else None,
    }
