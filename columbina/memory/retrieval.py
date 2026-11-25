"""Выбор релевантных воспоминаний: гибрид семантики, ключевых слов, давности и важности.

Не передаём всю базу: в контекст попадают top-k эпизодов, нужные факты и 0–3 фрагмента
её собственной биографии, доступные на текущей стадии дружбы.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..persona.loader import KnowledgeChunk
from .embeddings import Embedder, cosine, stems
from .store import Episode, Fact, MemoryStore


@dataclass
class Recall:
    episodes: list[tuple[Episode, float]]
    facts: list[Fact]
    persona: list[KnowledgeChunk]
    her_life: list[Episode]


def _keyword_overlap(query_stems: set[str], text: str) -> float:
    if not query_stems:
        return 0.0
    s = set(stems(text))
    return len(query_stems & s) / max(3, len(query_stems))


def _recency(created_at: float, now: float, half_life_days: float = 30.0) -> float:
    age_days = max(0.0, (now - created_at) / 86400)
    return math.pow(0.5, age_days / half_life_days)


class MemoryRetriever:
    def __init__(self, store: MemoryStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._persona_vecs: dict[str, np.ndarray] = {}

    async def index_persona(self, chunks: list[KnowledgeChunk]) -> None:
        missing: list[KnowledgeChunk] = []
        for c in chunks:
            cached = await self.store.get_persona_embedding(c.id, c.content_hash)
            if cached is not None and cached.shape[0] == self.embedder.dim:
                self._persona_vecs[c.id] = cached
            else:
                missing.append(c)
        if missing:
            vecs = await self.embedder.embed([c.search_text for c in missing])
            for c, v in zip(missing, vecs):
                self._persona_vecs[c.id] = v
                await self.store.set_persona_embedding(c.id, c.content_hash, v)

    async def _ensure_embeddings(self, episodes: list[Episode], facts: list[Fact]) -> None:
        """Досчитывает недостающие векторы (и пересчитывает при смене провайдера эмбеддингов)."""
        dim = self.embedder.dim
        ep_missing = [e for e in episodes if e.embedding is None or e.embedding.shape[0] != dim]
        f_missing = [f for f in facts if f.embedding is None or f.embedding.shape[0] != dim]
        if ep_missing:
            for e, v in zip(ep_missing, await self.embedder.embed([e.text for e in ep_missing])):
                e.embedding = v
                await self.store.set_embedding("episodes", e.id, v)
        if f_missing:
            for f, v in zip(f_missing, await self.embedder.embed([f"{f.key}: {f.value}" for f in f_missing])):
                f.embedding = v
                await self.store.set_embedding("facts", f.id, v)

    async def recall(
        self,
        user_id: int,
        query: str,
        stage: int,
        persona_chunks: list[KnowledgeChunk],
        top_k: int = 5,
        now: float | None = None,
    ) -> Recall:
        now = now or time.time()
        q_vec = (await self.embedder.embed([query], kind="query"))[0] if query.strip() else None
        q_stems = set(stems(query))

        episodes = await self.store.episodes(user_id)
        facts = await self.store.facts(user_id)
        await self._ensure_embeddings(episodes, facts)

        # --- эпизоды ---
        scored: list[tuple[Episode, float]] = []
        her_life: list[Episode] = []
        for ep in episodes:
            sim = cosine(q_vec, ep.embedding)
            kw = _keyword_overlap(q_stems, ep.text)
            score = 0.55 * sim + 0.25 * kw + 0.12 * _recency(ep.created_at, now) + 0.08 * (ep.importance / 5)
            if ep.kind == "her_life":
                # её недавняя «жизнь» нужна, чтобы не противоречить себе
                if now - ep.created_at < 3 * 86400:
                    her_life.append(ep)
                continue
            scored.append((ep, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        threshold = 0.22
        top = [(e, s) for e, s in scored[:top_k] if s >= threshold or e.importance >= 5]

        # --- факты ---
        pinned = [f for f in facts if f.pinned or f.key in {"имя", "name", "как зовут"}]
        pinned_ids = {f.id for f in pinned}
        other = [f for f in facts if f.id not in pinned_ids]
        other_scored = sorted(
            other,
            key=lambda f: 0.6 * cosine(q_vec, f.embedding) + 0.3 * _keyword_overlap(q_stems, f"{f.key} {f.value}")
            + 0.1 * _recency(f.updated_at, now, 60),
            reverse=True,
        )
        chosen_facts = pinned[:6] + other_scored[: max(0, 8 - len(pinned[:6]))]

        # --- её биография ---
        persona_scored: list[tuple[KnowledgeChunk, float]] = []
        for c in persona_chunks:
            if c.min_stage > stage:
                continue
            kw = _keyword_overlap(q_stems, " ".join(c.tags))
            tag_hit = any(t in query.lower() for t in c.tags if len(t) > 3)
            sim = cosine(q_vec, self._persona_vecs.get(c.id))
            score = 0.5 * sim + 0.3 * kw + (0.35 if tag_hit else 0.0)
            persona_scored.append((c, score))
        persona_scored.sort(key=lambda t: t[1], reverse=True)
        persona = [c for c, s in persona_scored[:3] if s >= 0.38][:2]

        await self.store.mark_recalled([e.id for e, _ in top])
        return Recall(episodes=top, facts=chosen_facts, persona=persona, her_life=her_life[-2:])
