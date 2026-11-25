from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import pytest_asyncio

from columbina.cognition.engine import CognitiveEngine
from columbina.config import ROOT, Settings
from columbina.llm.fake import FakeProvider
from columbina.memory.embeddings import LocalHashEmbedder
from columbina.memory.store import MemoryStore
from columbina.persona.loader import load_persona


@pytest.fixture(scope="session")
def persona():
    return load_persona(ROOT / "data" / "persona")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return dataclasses.replace(Settings(), db_path=tmp_path / "t.sqlite3", media_tmp_dir=tmp_path / "tmp",
                               daily_budget_usd=0.0, debounce_seconds=0.0)


PG_URL = os.environ.get("TEST_DATABASE_URL", "")
BACKENDS = ["sqlite"] + (["postgres"] if PG_URL else [])


@pytest_asyncio.fixture(params=BACKENDS)
async def store(request, settings):
    """Каждый тест с памятью идёт на SQLite и (если задан TEST_DATABASE_URL) на PostgreSQL."""
    if request.param == "postgres":
        from columbina.memory.store import ALL_TABLES

        s = await MemoryStore(PG_URL).open()
        await s.d.execute("TRUNCATE " + ", ".join(ALL_TABLES) + " RESTART IDENTITY")
    else:
        s = await MemoryStore(settings.db_path).open()
    yield s
    await s.close()


@pytest.fixture
def fake_llm():
    return FakeProvider()


@pytest_asyncio.fixture
async def engine(settings, persona, store, fake_llm):
    e = CognitiveEngine(settings, persona, store, fake_llm, LocalHashEmbedder(), weather=None)
    await e.setup()
    yield e
    await e.drain()


def last_inner_context(fake: FakeProvider) -> str:
    """Текст «Внутреннего контекста» из последнего основного запроса."""
    for req in reversed(fake.requests):
        if req.tier == "main":
            sys_msgs = [m for m in req.messages if m.role == "system"]
            return sys_msgs[-1].plain_text() if sys_msgs else ""
    return ""
