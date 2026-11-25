"""Точка входа: сборка модулей и запуск бота.

    python -m columbina.main
"""

from __future__ import annotations

import asyncio
import logging

from .abilities.weather import WeatherService
from .cognition.engine import CognitiveEngine
from .config import load_settings, storage_target
from .initiative.engine import InitiativeEngine
from .initiative.scheduler import Scheduler
from .llm.anthropic_provider import AnthropicProvider
from .media.links import SafeFetcher
from .media.processors import MediaProcessor, SpeechToText
from .memory.embeddings import LocalHashEmbedder, VoyageEmbedder
from .memory.store import MemoryStore
from .persona.loader import load_persona
from .telegram.bot import TelegramInterface


async def build(settings=None):
    s = settings or load_settings()
    persona = load_persona(s.persona_dir)
    store = await MemoryStore(storage_target(s)).open()
    if s.embedding_provider == "voyage" and s.voyage_api_key:
        embedder = VoyageEmbedder(s.voyage_api_key, s.voyage_model)
    else:
        embedder = LocalHashEmbedder()
    llm = AnthropicProvider(s.main_model, s.background_model, s.main_effort, s.use_refusal_fallbacks)
    weather = WeatherService()
    engine = CognitiveEngine(s, persona, store, llm, embedder, weather)
    await engine.setup()
    s.media_tmp_dir.mkdir(parents=True, exist_ok=True)
    media = MediaProcessor(SpeechToText(s.stt_base_url, s.stt_api_key, s.stt_model), SafeFetcher(verify=s.link_ca_bundle or True),
                           s.media_tmp_dir, s.link_fetch_enabled)
    return s, engine, media


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан (см. .env.example)")
    s, engine, media = await build(settings)
    tg = TelegramInterface(s, engine, media)
    scheduler = Scheduler(InitiativeEngine(engine), tg.send_parts, s.initiative_tick_seconds)
    scheduler.start()
    try:
        await tg.run()
    finally:
        await scheduler.stop()
        await engine.drain()
        await engine.store.close()


if __name__ == "__main__":
    asyncio.run(main())
