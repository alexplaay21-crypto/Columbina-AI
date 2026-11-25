"""Долговременная память: SQLite (разработка, тесты) или PostgreSQL (прод) — один и тот же код.

Слои памяти:
  messages        — рабочий контекст (сырые последние сообщения) + архив
  summaries       — суммаризация старого диалога (скользящее резюме)
  episodes        — эпизодические воспоминания (общие события, события из её «жизни»)
  facts           — факты о собеседнике (ключ → значение, с уверенностью)
  relationship    — история и состояние отношений
  emotion_state   — её эмоциональное состояние (сохраняется между диалогами)
  events          — журнал эмоциональных/отношенческих событий (изменения отношения)
  threads, notes  — незакрытые темы и записки (для инициативы)

Векторы эмбеддингов хранятся как float32-байты (BLOB/BYTEA). Поиск идёт в пределах одного
собеседника (сотни–тысячи записей), поэтому оценка в Python быстрая при любом числе пользователей.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .db import Database, open_database

TABLES = """
CREATE TABLE IF NOT EXISTS users (
    user_id {BIGINT} PRIMARY KEY,
    chat_id {BIGINT} NOT NULL,
    display_name TEXT,
    preferred_name TEXT,
    timezone TEXT,
    city TEXT,
    lat {REAL}, lon {REAL},
    initiative_mode TEXT NOT NULL DEFAULT 'normal',
    quiet_start INTEGER NOT NULL DEFAULT 23,
    quiet_end INTEGER NOT NULL DEFAULT 9,
    register TEXT NOT NULL DEFAULT 'ty',
    created_at {REAL} NOT NULL,
    last_user_msg_at {REAL},
    last_bot_msg_at {REAL},
    sessions INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id {PK},
    user_id {BIGINT} NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at {REAL} NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'chat'
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, archived, id);
CREATE TABLE IF NOT EXISTS summaries (
    user_id {BIGINT} PRIMARY KEY,
    text TEXT NOT NULL,
    covered_until_id {BIGINT} NOT NULL,
    updated_at {REAL} NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id {PK},
    user_id {BIGINT} NOT NULL,
    kind TEXT NOT NULL DEFAULT 'shared',
    text TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 2,
    emotion TEXT,
    created_at {REAL} NOT NULL,
    last_recalled_at {REAL},
    recall_count INTEGER NOT NULL DEFAULT 0,
    embedding {BLOB}
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
CREATE TABLE IF NOT EXISTS facts (
    id {PK},
    user_id {BIGINT} NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence {REAL} NOT NULL DEFAULT 0.7,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at {REAL} NOT NULL,
    updated_at {REAL} NOT NULL,
    embedding {BLOB},
    UNIQUE(user_id, key)
);
CREATE TABLE IF NOT EXISTS relationship (
    user_id {BIGINT} PRIMARY KEY,
    trust {REAL} NOT NULL DEFAULT 0.0,
    stage INTEGER NOT NULL DEFAULT 1,
    quality_sessions INTEGER NOT NULL DEFAULT 0,
    milestones TEXT NOT NULL DEFAULT '[]',
    seen_events TEXT NOT NULL DEFAULT '[]',
    updated_at {REAL} NOT NULL
);
CREATE TABLE IF NOT EXISTS emotion_state (
    user_id {BIGINT} PRIMARY KEY,
    vals TEXT NOT NULL,
    mood {REAL} NOT NULL DEFAULT 0.0,
    last_cause TEXT,
    updated_at {REAL} NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id {PK},
    user_id {BIGINT} NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    detail TEXT,
    created_at {REAL} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, id);
CREATE TABLE IF NOT EXISTS threads (
    id {PK},
    user_id {BIGINT} NOT NULL,
    topic TEXT NOT NULL,
    follow_up_at {REAL},
    status TEXT NOT NULL DEFAULT 'open',
    created_at {REAL} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id, status);
CREATE TABLE IF NOT EXISTS notes (
    id {PK},
    user_id {BIGINT} NOT NULL,
    text TEXT NOT NULL,
    due_at {REAL} NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at {REAL} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_due ON notes(status, due_at);
CREATE TABLE IF NOT EXISTS initiatives (
    id {PK},
    user_id {BIGINT} NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT,
    text TEXT,
    sent_at {REAL} NOT NULL,
    answered INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_initiatives_user ON initiatives(user_id, id);
CREATE TABLE IF NOT EXISTS usage (
    id {PK},
    user_id {BIGINT},
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER,
    cache_read INTEGER, cache_write INTEGER,
    cost_usd {REAL} NOT NULL,
    created_at {REAL} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage(user_id, created_at);
CREATE TABLE IF NOT EXISTS web_log (
    id {PK},
    user_id {BIGINT},
    tool TEXT NOT NULL,
    arg TEXT,
    ok INTEGER NOT NULL DEFAULT 1,
    created_at {REAL} NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_web_log_user ON web_log(user_id, created_at);
CREATE TABLE IF NOT EXISTS persona_embeddings (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    embedding {BLOB} NOT NULL
);
"""

TYPES = {
    "sqlite": {"BIGINT": "INTEGER", "REAL": "REAL", "BLOB": "BLOB", "PK": "INTEGER PRIMARY KEY AUTOINCREMENT"},
    "postgres": {"BIGINT": "BIGINT", "REAL": "DOUBLE PRECISION", "BLOB": "BYTEA", "PK": "BIGSERIAL PRIMARY KEY"},
}

ALL_TABLES = ["users", "messages", "summaries", "episodes", "facts", "relationship", "emotion_state", "events",
              "threads", "notes", "initiatives", "usage", "web_log", "persona_embeddings"]


def schema_for(dialect: str) -> str:
    s = TABLES.format(**TYPES[dialect])
    return ("PRAGMA journal_mode=WAL;\n" + s) if dialect == "sqlite" else s


def to_blob(vec: np.ndarray | None) -> bytes | None:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


@dataclass
class UserProfile:
    user_id: int
    chat_id: int
    display_name: str | None = None
    preferred_name: str | None = None
    timezone: str | None = None
    city: str | None = None
    lat: float | None = None
    lon: float | None = None
    initiative_mode: str = "normal"
    quiet_start: int = 23
    quiet_end: int = 9
    register: str = "ty"
    created_at: float = 0.0
    last_user_msg_at: float | None = None
    last_bot_msg_at: float | None = None
    sessions: int = 0


@dataclass
class StoredMessage:
    id: int
    role: str
    text: str
    created_at: float
    kind: str = "chat"


@dataclass
class Episode:
    id: int
    kind: str
    text: str
    importance: int
    emotion: str | None
    created_at: float
    recall_count: int = 0
    embedding: np.ndarray | None = None


@dataclass
class Fact:
    id: int
    key: str
    value: str
    confidence: float
    pinned: bool
    updated_at: float
    embedding: np.ndarray | None = None


@dataclass
class RelationshipRow:
    trust: float = 0.0
    stage: int = 1
    quality_sessions: int = 0
    milestones: list[str] = field(default_factory=list)
    seen_events: list[str] = field(default_factory=list)



class MemoryStore:
    """`path` — путь к файлу SQLite или URL `postgresql://user:pass@host/db`."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self.db: Database | None = None

    async def open(self) -> "MemoryStore":
        if not self.path.startswith(("postgres://", "postgresql://")) and self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await open_database(self.path)
        await self.db.executescript(schema_for(self.db.dialect))
        return self

    async def close(self) -> None:
        if self.db:
            await self.db.close()

    @property
    def d(self) -> Database:
        assert self.db is not None, "MemoryStore.open() не вызван"
        return self.db

    # ---------- users ----------
    async def get_or_create_user(self, user_id: int, chat_id: int, display_name: str | None = None) -> UserProfile:
        row = await self.d.fetchone("SELECT * FROM users WHERE user_id=?", (user_id,))
        if row is None:
            now = time.time()
            await self.d.execute(
                "INSERT INTO users(user_id, chat_id, display_name, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(user_id) DO NOTHING",
                (user_id, chat_id, display_name, now),
            )
            return UserProfile(user_id=user_id, chat_id=chat_id, display_name=display_name, created_at=now)
        return UserProfile(**row)

    async def get_user(self, user_id: int) -> UserProfile | None:
        row = await self.d.fetchone("SELECT * FROM users WHERE user_id=?", (user_id,))
        return UserProfile(**row) if row else None

    async def list_users(self) -> list[UserProfile]:
        return [UserProfile(**r) for r in await self.d.fetchall("SELECT * FROM users ORDER BY user_id")]

    USER_COLUMNS = {"chat_id", "display_name", "preferred_name", "timezone", "city", "lat", "lon", "initiative_mode",
                    "quiet_start", "quiet_end", "register", "last_user_msg_at", "last_bot_msg_at", "sessions"}

    async def update_user(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        bad = set(fields) - self.USER_COLUMNS
        if bad:
            raise ValueError(f"unknown user columns: {bad}")
        cols = ", ".join(f"{k}=?" for k in fields)
        await self.d.execute(f"UPDATE users SET {cols} WHERE user_id=?", (*fields.values(), user_id))

    # ---------- messages ----------
    async def add_message(self, user_id: int, role: str, text: str, kind: str = "chat", at: float | None = None) -> int:
        return await self.d.insert(
            "INSERT INTO messages(user_id, role, text, created_at, kind) VALUES (?,?,?,?,?)",
            (user_id, role, text, float(at or time.time()), kind),
        )

    @staticmethod
    def _msg(r: dict[str, Any]) -> StoredMessage:
        return StoredMessage(r["id"], r["role"], r["text"], r["created_at"], r["kind"])

    async def active_messages(self, user_id: int) -> list[StoredMessage]:
        rows = await self.d.fetchall(
            "SELECT id, role, text, created_at, kind FROM messages WHERE user_id=? AND archived=0 ORDER BY id",
            (user_id,),
        )
        return [self._msg(r) for r in rows]

    async def last_messages(self, user_id: int, n: int, role: str | None = None) -> list[StoredMessage]:
        q = "SELECT id, role, text, created_at, kind FROM messages WHERE user_id=?"
        args: list[Any] = [user_id]
        if role:
            q += " AND role=?"
            args.append(role)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(n)
        return [self._msg(r) for r in reversed(await self.d.fetchall(q, args))]

    async def update_message_text(self, message_id: int, text: str) -> None:
        await self.d.execute("UPDATE messages SET text=? WHERE id=?", (text, message_id))

    async def archive_messages_upto(self, user_id: int, last_id: int) -> None:
        await self.d.execute("UPDATE messages SET archived=1 WHERE user_id=? AND id<=?", (user_id, last_id))

    # ---------- summaries ----------
    async def get_summary(self, user_id: int) -> str:
        row = await self.d.fetchone("SELECT text FROM summaries WHERE user_id=?", (user_id,))
        return row["text"] if row else ""

    async def set_summary(self, user_id: int, text: str, covered_until_id: int) -> None:
        await self.d.execute(
            "INSERT INTO summaries(user_id, text, covered_until_id, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET text=excluded.text, covered_until_id=excluded.covered_until_id, "
            "updated_at=excluded.updated_at",
            (user_id, text, covered_until_id, time.time()),
        )

    # ---------- episodes ----------
    async def add_episode(
        self,
        user_id: int,
        text: str,
        importance: int = 2,
        emotion: str | None = None,
        kind: str = "shared",
        embedding: np.ndarray | None = None,
        at: float | None = None,
    ) -> int:
        return await self.d.insert(
            "INSERT INTO episodes(user_id, kind, text, importance, emotion, created_at, embedding) VALUES (?,?,?,?,?,?,?)",
            (user_id, kind, text, int(importance), emotion, float(at or time.time()), to_blob(embedding)),
        )

    async def episodes(self, user_id: int, kind: str | None = None) -> list[Episode]:
        q = "SELECT * FROM episodes WHERE user_id=?"
        args: list[Any] = [user_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        rows = await self.d.fetchall(q + " ORDER BY id", args)
        return [
            Episode(r["id"], r["kind"], r["text"], r["importance"], r["emotion"], r["created_at"],
                    r["recall_count"], from_blob(r["embedding"]))
            for r in rows
        ]

    async def set_embedding(self, table: str, row_id: int, vec: np.ndarray) -> None:
        assert table in ("episodes", "facts")
        await self.d.execute(f"UPDATE {table} SET embedding=? WHERE id=?", (to_blob(vec), row_id))

    async def mark_recalled(self, episode_ids: list[int]) -> None:
        if not episode_ids:
            return
        now = time.time()
        await self.d.executemany(
            "UPDATE episodes SET recall_count=recall_count+1, last_recalled_at=? WHERE id=?",
            [(now, i) for i in episode_ids],
        )

    # ---------- facts ----------
    async def upsert_fact(
        self, user_id: int, key: str, value: str, confidence: float = 0.7,
        embedding: np.ndarray | None = None, pinned: bool = False,
    ) -> None:
        now = time.time()
        await self.d.execute(
            "INSERT INTO facts(user_id, key, value, confidence, pinned, created_at, updated_at, embedding) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, "
            "confidence=CASE WHEN facts.confidence > excluded.confidence THEN facts.confidence ELSE excluded.confidence END, "
            "updated_at=excluded.updated_at, "
            "embedding=COALESCE(excluded.embedding, facts.embedding), "
            "pinned=CASE WHEN facts.pinned > excluded.pinned THEN facts.pinned ELSE excluded.pinned END",
            (user_id, key.strip().lower()[:80], value.strip()[:400], float(confidence), int(pinned), now, now,
             to_blob(embedding)),
        )

    async def delete_fact(self, user_id: int, key: str) -> None:
        await self.d.execute("DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key.strip().lower()))

    async def facts(self, user_id: int) -> list[Fact]:
        rows = await self.d.fetchall("SELECT * FROM facts WHERE user_id=? ORDER BY updated_at DESC", (user_id,))
        return [
            Fact(r["id"], r["key"], r["value"], r["confidence"], bool(r["pinned"]), r["updated_at"], from_blob(r["embedding"]))
            for r in rows
        ]

    # ---------- relationship ----------
    async def get_relationship(self, user_id: int) -> RelationshipRow:
        r = await self.d.fetchone("SELECT * FROM relationship WHERE user_id=?", (user_id,))
        if not r:
            return RelationshipRow()
        return RelationshipRow(r["trust"], r["stage"], r["quality_sessions"], json.loads(r["milestones"]),
                               json.loads(r["seen_events"]))

    async def save_relationship(self, user_id: int, rel: RelationshipRow) -> None:
        await self.d.execute(
            "INSERT INTO relationship(user_id, trust, stage, quality_sessions, milestones, seen_events, updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET trust=excluded.trust, stage=excluded.stage, "
            "quality_sessions=excluded.quality_sessions, milestones=excluded.milestones, "
            "seen_events=excluded.seen_events, updated_at=excluded.updated_at",
            (user_id, float(rel.trust), int(rel.stage), int(rel.quality_sessions), json.dumps(rel.milestones),
             json.dumps(rel.seen_events[-50:]), time.time()),
        )

    # ---------- emotion ----------
    async def get_emotion(self, user_id: int) -> tuple[dict[str, float], float, str | None, float] | None:
        r = await self.d.fetchone("SELECT * FROM emotion_state WHERE user_id=?", (user_id,))
        if not r:
            return None
        return json.loads(r["vals"]), r["mood"], r["last_cause"], r["updated_at"]

    async def save_emotion(self, user_id: int, vals: dict[str, float], mood: float, last_cause: str | None, at: float) -> None:
        await self.d.execute(
            "INSERT INTO emotion_state(user_id, vals, mood, last_cause, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET vals=excluded.vals, mood=excluded.mood, "
            "last_cause=excluded.last_cause, updated_at=excluded.updated_at",
            (user_id, json.dumps({k: round(v, 4) for k, v in vals.items()}), float(mood), last_cause, float(at)),
        )

    # ---------- events log ----------
    async def log_event(self, user_id: int, kind: str, source: str, detail: str | None = None) -> None:
        await self.d.execute(
            "INSERT INTO events(user_id, kind, source, detail, created_at) VALUES (?,?,?,?,?)",
            (user_id, kind, source, (detail or "")[:300], time.time()),
        )

    async def recent_events(self, user_id: int, n: int = 20) -> list[tuple[str, str, str, float]]:
        rows = await self.d.fetchall(
            "SELECT kind, source, detail, created_at FROM events WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, n)
        )
        return [(r["kind"], r["source"], r["detail"], r["created_at"]) for r in rows]

    # ---------- threads & notes ----------
    async def add_thread(self, user_id: int, topic: str, follow_up_at: float | None) -> None:
        await self.d.execute(
            "INSERT INTO threads(user_id, topic, follow_up_at, created_at) VALUES (?,?,?,?)",
            (user_id, topic[:300], float(follow_up_at) if follow_up_at is not None else None, time.time()),
        )

    async def open_threads(self, user_id: int) -> list[dict[str, Any]]:
        return await self.d.fetchall(
            "SELECT * FROM threads WHERE user_id=? AND status='open' ORDER BY id DESC LIMIT 10", (user_id,)
        )

    async def close_thread(self, thread_id: int, status: str = "done") -> None:
        await self.d.execute("UPDATE threads SET status=? WHERE id=?", (status, thread_id))

    async def add_note(self, user_id: int, text: str, due_at: float) -> int:
        return await self.d.insert(
            "INSERT INTO notes(user_id, text, due_at, created_at) VALUES (?,?,?,?)",
            (user_id, text[:500], float(due_at), time.time()),
        )

    async def due_notes(self, now: float) -> list[dict[str, Any]]:
        return await self.d.fetchall(
            "SELECT * FROM notes WHERE status='pending' AND due_at<=? ORDER BY due_at", (float(now),)
        )

    async def pending_notes(self, user_id: int) -> list[dict[str, Any]]:
        return await self.d.fetchall(
            "SELECT * FROM notes WHERE user_id=? AND status='pending' ORDER BY due_at", (user_id,)
        )

    async def set_note_status(self, note_id: int, status: str) -> None:
        await self.d.execute("UPDATE notes SET status=? WHERE id=?", (status, note_id))

    # ---------- initiatives ----------
    async def log_initiative(self, user_id: int, kind: str, reason: str, text: str) -> None:
        await self.d.execute(
            "INSERT INTO initiatives(user_id, kind, reason, text, sent_at) VALUES (?,?,?,?,?)",
            (user_id, kind, reason, text, time.time()),
        )

    async def last_initiative(self, user_id: int) -> dict[str, Any] | None:
        return await self.d.fetchone("SELECT * FROM initiatives WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,))

    async def initiatives_since(self, user_id: int, since: float) -> list[dict[str, Any]]:
        return await self.d.fetchall(
            "SELECT * FROM initiatives WHERE user_id=? AND sent_at>=?", (user_id, float(since))
        )

    async def mark_initiatives_answered(self, user_id: int) -> None:
        await self.d.execute("UPDATE initiatives SET answered=1 WHERE user_id=? AND answered=0", (user_id,))

    # ---------- usage ----------
    async def log_usage(self, user_id: int | None, purpose: str, model: str, usage: Any, cost: float) -> None:
        await self.d.execute(
            "INSERT INTO usage(user_id, purpose, model, input_tokens, output_tokens, cache_read, cache_write, cost_usd, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, purpose, model, int(usage.input_tokens), int(usage.output_tokens), int(usage.cache_read_tokens),
             int(usage.cache_write_tokens), float(cost), time.time()),
        )

    async def cost_since(self, user_id: int | None, since: float) -> float:
        if user_id is None:
            r = await self.d.fetchone("SELECT COALESCE(SUM(cost_usd),0) AS c FROM usage WHERE created_at>=?", (float(since),))
        else:
            r = await self.d.fetchone(
                "SELECT COALESCE(SUM(cost_usd),0) AS c FROM usage WHERE user_id=? AND created_at>=?", (user_id, float(since))
            )
        return float(r["c"])

    async def usage_report(self, since: float) -> list[dict[str, Any]]:
        return await self.d.fetchall(
            "SELECT purpose, model, COUNT(*) AS n, SUM(input_tokens) AS inp, SUM(output_tokens) AS out, "
            "SUM(cache_read) AS cr, SUM(cache_write) AS cw, SUM(cost_usd) AS cost FROM usage WHERE created_at>=? "
            "GROUP BY purpose, model ORDER BY cost DESC",
            (float(since),),
        )

    # ---------- web ----------
    async def log_web(self, user_id: int | None, tool: str, arg: str, ok: bool) -> None:
        await self.d.execute("INSERT INTO web_log(user_id, tool, arg, ok, created_at) VALUES (?,?,?,?,?)",
                             (user_id, tool, arg, int(ok), time.time()))

    async def web_count_since(self, user_id: int | None, since: float) -> int:
        r = await self.d.fetchone("SELECT COUNT(*) AS n FROM web_log WHERE user_id=? AND created_at>=?",
                                  (user_id, float(since)))
        return int(r["n"])

    async def web_recent(self, user_id: int | None, n: int = 10) -> list[dict[str, Any]]:
        return await self.d.fetchall("SELECT * FROM web_log WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, n))

    # ---------- persona embeddings cache ----------
    async def get_persona_embedding(self, chunk_id: str, content_hash: str) -> np.ndarray | None:
        r = await self.d.fetchone(
            "SELECT embedding FROM persona_embeddings WHERE chunk_id=? AND content_hash=?", (chunk_id, content_hash)
        )
        return from_blob(r["embedding"]) if r else None

    async def set_persona_embedding(self, chunk_id: str, content_hash: str, vec: np.ndarray) -> None:
        await self.d.execute(
            "INSERT INTO persona_embeddings(chunk_id, content_hash, embedding) VALUES (?,?,?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET content_hash=excluded.content_hash, embedding=excluded.embedding",
            (chunk_id, content_hash, to_blob(vec)),
        )

    # ---------- privacy ----------
    async def forget_user(self, user_id: int) -> None:
        for table in ("messages", "summaries", "episodes", "facts", "relationship", "emotion_state",
                      "events", "threads", "notes", "initiatives", "web_log"):
            await self.d.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
