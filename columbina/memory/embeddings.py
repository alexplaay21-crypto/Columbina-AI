"""Эмбеддинги для семантического поиска по памяти.

- VoyageEmbedder — Voyage AI (рекомендуемый Anthropic провайдер эмбеддингов, мультиязычный).
- LocalHashEmbedder — офлайн-вариант без зависимостей: хэширование n-грамм символов и основ слов.
  Хуже понимает синонимы, но бесплатен, детерминирован и годится для старта и тестов.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)


def stems(text: str, n: int = 5) -> list[str]:
    """Грубые основы слов для русского: первые n символов (достаточно для поиска «кофе/кофейный»)."""
    return [w[:n] for w in _WORD.findall(text.lower().replace("ё", "е")) if len(w) > 2]


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]: ...


class LocalHashEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = " " + text.lower().replace("ё", "е") + " "
        feats: list[tuple[str, float]] = []
        for st in stems(text):
            feats.append(("w:" + st, 2.0))
        for i in range(len(t) - 2):
            tri = t[i : i + 3]
            if tri.strip():
                feats.append(("c:" + tri, 0.5))
        for f, w in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[idx] += sign * w
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    async def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]:
        return [self._vec(t) for t in texts]


class VoyageEmbedder:
    URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str = "voyage-3.5-lite", dim: int = 1024):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]:
        if not texts:
            return []
        out: list[np.ndarray] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            r = await self._client.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"input": batch, "model": self.model, "input_type": "query" if kind == "query" else "document"},
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            for d in data:
                v = np.asarray(d["embedding"], dtype=np.float32)
                out.append(v / (np.linalg.norm(v) or 1.0))
        return out


class OpenAICompatibleEmbedder:
    """POST {base}/embeddings — OpenRouter (baai/bge-m3, qwen/qwen3-embedding-8b, …) или любой совместимый API."""

    def __init__(self, base_url: str, api_key: str, model: str, client: httpx.AsyncClient | None = None):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.dim: int | None = None  # узнаём по первому ответу
        self._client = client or httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]:
        if not texts:
            return []
        out: list[np.ndarray] = []
        for i in range(0, len(texts), 64):
            r = await self._client.post(
                self.url, headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts[i : i + 64], "encoding_format": "float"},
            )
            r.raise_for_status()
            for d in sorted(r.json()["data"], key=lambda d: d["index"]):
                v = np.asarray(d["embedding"], dtype=np.float32)
                out.append(v / (np.linalg.norm(v) or 1.0))
        self.dim = int(out[0].shape[0])
        return out


class FallbackEmbedder:
    """Пробует основной эмбеддер, при ошибке сети — локальный (с тем же размером недоступно,
    поэтому локальный используется только если основного вектора ещё нет в базе)."""

    def __init__(self, primary: Embedder, fallback: Embedder):
        self.primary, self.fallback = primary, fallback
        self.dim = primary.dim

    async def embed(self, texts: list[str], kind: str = "document") -> list[np.ndarray]:
        try:
            return await self.primary.embed(texts, kind)
        except Exception:
            return await self.fallback.embed(texts, kind)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def make_embedder(settings) -> Embedder:
    p = settings.embedding_provider
    if p == "voyage" and settings.voyage_api_key:
        return VoyageEmbedder(settings.voyage_api_key, settings.voyage_model)
    if p == "openrouter" and settings.openrouter_api_key:
        return OpenAICompatibleEmbedder(settings.openrouter_base_url, settings.openrouter_api_key,
                                        settings.openrouter_embedding_model)
    return LocalHashEmbedder()
