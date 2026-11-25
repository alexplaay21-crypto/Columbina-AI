"""Простой асинхронный планировщик: проход Initiative Engine каждые N секунд."""

from __future__ import annotations

import asyncio
import logging

from .engine import InitiativeEngine

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, initiative: InitiativeEngine, send, interval_seconds: int = 600):
        self.initiative = initiative
        self.send = send
        self.interval = interval_seconds
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        await asyncio.sleep(30)
        while True:
            try:
                n = await self.initiative.tick(self.send)
                if n:
                    log.info("initiative: sent %d", n)
            except Exception:
                log.exception("scheduler tick failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
