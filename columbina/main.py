"""Точка входа: сборка модулей и запуск бота.

    python -m columbina.main
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .abilities.weather import WeatherService
from .cognition.engine import CognitiveEngine
from .config import load_settings, storage_target
from .initiative.engine import InitiativeEngine
from .initiative.scheduler import Scheduler
from .llm.factory import make_llm, make_vision, make_web_tools, resolved_models
from .media.links import SafeFetcher
from .media.processors import MediaProcessor, OpenRouterSpeechToText, SpeechToText
from .memory.embeddings import make_embedder
from .memory.store import MemoryStore
from .persona.loader import load_persona
from .telegram.bot import TelegramInterface


async def build(settings=None):
    s = settings or load_settings()
    persona = load_persona(s.persona_dir, s.canon_mode)
    store = await MemoryStore(storage_target(s)).open()
    embedder = make_embedder(s)
    llm = make_llm(s)
    weather = WeatherService()
    web = make_web_tools(s, store)
    engine = CognitiveEngine(s, persona, store, llm, embedder, weather, vision=make_vision(s), web_tools=web)
    try:
        await engine.setup()
    except BaseException as e:
        await store.close()  # иначе поток aiosqlite не даст процессу завершиться
        if not isinstance(e, httpx.HTTPStatusError):
            raise
        raise SystemExit(f"Эмбеддинги ({s.embedding_provider}) вернули HTTP {e.response.status_code}: "
                         "проверьте OPENROUTER_API_KEY / OPENROUTER_EMBEDDING_MODEL или поставьте EMBEDDING_PROVIDER=local")
    from .inner.life import InnerLife, Limits

    engine.inner = InnerLife(engine, s.persona_dir, Limits(research_per_day=s.research_per_day,
                                                           dreams_every_days=s.dreams_every_days), s.inner_life_enabled)
    s.media_tmp_dir.mkdir(parents=True, exist_ok=True)
    if s.stt_provider == "openrouter":
        stt = OpenRouterSpeechToText(s.openrouter_base_url, s.openrouter_api_key, s.openrouter_stt_model, s.media_tmp_dir)

        async def log_stt(cost: float, data: dict) -> None:
            from .llm.base import Usage

            u = data.get("usage") or {}
            await store.log_usage(None, "stt", s.openrouter_stt_model,
                                  Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)), cost)

        stt.on_usage = log_stt
    else:
        stt = SpeechToText(s.stt_base_url, s.stt_api_key, s.stt_model)
    link_reader = web.reader if web else SafeFetcher(verify=s.link_ca_bundle or True)
    media = MediaProcessor(stt, link_reader, s.media_tmp_dir, s.link_fetch_enabled)
    main_m, bg_m = resolved_models(s)
    logging.getLogger(__name__).info("LLM: %s (%s / %s), embeddings: %s, STT: %s, storage: %s",
                                     s.llm_provider, main_m, bg_m, s.embedding_provider, s.stt_provider,
                                     store.d.dialect)
    return s, engine, media


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан (см. .env.example)")
    s, engine, media = await build(settings)
    scheduler = None
    try:
        tg = TelegramInterface(s, engine, media)
        scheduler = Scheduler(InitiativeEngine(engine), tg.send_parts, s.initiative_tick_seconds, engine.inner,
                              heartbeat=s.media_tmp_dir.parent / "heartbeat")
        scheduler.start()
        await tg.run()
    finally:
        if scheduler:
            await scheduler.stop()
        await engine.drain()
        await engine.store.close()


if __name__ == "__main__":
    asyncio.run(main())
