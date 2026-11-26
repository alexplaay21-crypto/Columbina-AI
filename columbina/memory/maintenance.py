"""Гигиена памяти: дубликаты, противоречия, забывание неважного, очистка журналов. Раз в сутки, без LLM."""

from __future__ import annotations

import logging
import time

from .embeddings import cosine
from .store import MemoryStore

log = logging.getLogger(__name__)

DAY = 86400


class Maintenance:
    def __init__(self, store: MemoryStore, dup_threshold: float = 0.92):
        self.store = store
        self.dup = dup_threshold
        self._last_run: dict[int, float] = {}

    def due(self, user_id: int, now: float) -> bool:
        return now - self._last_run.get(user_id, 0) > DAY

    async def run(self, user_id: int, now: float | None = None) -> dict[str, int]:
        now = now or time.time()
        self._last_run[user_id] = now
        st = self.store
        report = {"merged": 0, "forgotten": 0, "pruned": 0}
        # 1. Дубликаты воспоминаний: сливаем почти одинаковые (оставляем более важное и старое).
        eps = [e for e in await st.episodes(user_id) if e.embedding is not None]
        removed: set[int] = set()
        for i, a in enumerate(eps):
            if a.id in removed:
                continue
            for b in eps[i + 1:]:
                if b.id in removed or a.kind != b.kind:
                    continue
                if cosine(a.embedding, b.embedding) >= self.dup:
                    await st.merge_episode(a.id, max(a.importance, b.importance),
                                           b.text if len(b.text) > len(a.text) else None)
                    removed.add(b.id)
        await st.delete_episodes(sorted(removed))
        report["merged"] = len(removed)
        # 2. Забывание: мелочи, к которым ни разу не возвращались за 90 дней; «своя жизнь» — через 30 дней.
        stale = [e.id for e in await st.episodes(user_id)
                 if (e.kind == "shared" and e.importance <= 2 and e.recall_count == 0 and now - e.created_at > 90 * DAY)
                 or (e.kind == "her_life" and now - e.created_at > 30 * DAY)]
        await st.delete_episodes(stale)
        report["forgotten"] = len(stale)
        # 3. Знания: неподтверждённые догадки старше 120 дней.
        report["pruned"] += await st.prune("knowledge", "created_at", now - 120 * DAY,
                                           "AND user_id=? AND kind='assumption'", (user_id,))
        # 4. Журналы: интернет — 30 дней, расходы — 90, события отношений — 180.
        report["pruned"] += await st.prune("web_log", "created_at", now - 30 * DAY, "AND user_id=?", (user_id,))
        report["pruned"] += await st.prune("usage", "created_at", now - 90 * DAY, "AND user_id=?", (user_id,))
        report["pruned"] += await st.prune("events", "created_at", now - 180 * DAY, "AND user_id=?", (user_id,))
        await st.decay_interests(user_id)
        if any(report.values()):
            log.info("memory maintenance %s: %s", user_id, report)
        return report
