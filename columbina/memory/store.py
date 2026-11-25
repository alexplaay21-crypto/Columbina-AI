"""Долговременная память: SQLite (aiosqlite).

Слои памяти:
  messages        — рабочий контекст (сырые последние сообщения) + архив
  summaries       — суммаризация старого диалога (скользящее резюме)
  episodes        — эпизодические воспоминания (общие события, события из её «жизни»)
  facts           — факты о собеседнике (ключ → значение, с уверенностью)
  relationship    — история и состояние отношений
  emotion_state   — её эмоциональное состояние (сохраняется между диалогами)
  events          — журнал эмоциональных/отношенческих событий (изменения отношения)
  threads, notes  — незакрытые темы и записки (для инициативы)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    display_name TEXT,
    preferred_name TEXT,
    timezone TEXT,
    city TEXT,
    lat REAL, lon REAL,
    initiative_mode TEXT NOT NULL DEFAULT 'normal',
    quiet_start INTEGER NOT NULL DEFAULT 23,
    quiet_end INTEGER NOT NULL DEFAULT 9,
    register TEXT NOT NULL DEFAULT 'ty',
    created_at REAL NOT NULL,
    last_user_msg_at REAL,
    last_bot_msg_at REAL,
    sessions INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'chat'
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, archived, id);
CREATE TABLE IF NOT EXISTS summaries (
    user_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    covered_until_id INTEGER NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL DEFAULT 'shared',
    text TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 2,
    emotion TEXT,
    created_at REAL NOT NULL,
    last_recalled_at REAL,
    recall_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.7,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    embedding BLOB,
    UNIQUE(user_id, key)
);
CREATE TABLE IF NOT EXISTS relationship (
    user_id INTEGER PRIMARY KEY,
    trust REAL NOT NULL DEFAULT 0.0,
    stage INTEGER NOT NULL DEFAULT 1,
    quality_sessions INTEGER NOT NULL DEFAULT 0,
    milestones TEXT NOT NULL DEFAULT '[]',
    seen_events TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS emotion_state (
    user_id INTEGER PRIMARY KEY,
    vals TEXT NOT NULL,
    mood REAL NOT NULL DEFAULT 0.0,
    last_cause TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    detail TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id, id);
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    follow_up_at REAL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    due_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS initiatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT,
    text TEXT,
    sent_at REAL NOT NULL,
    answered INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER,
    cache_read INTEGER, cache_write INTEGER,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage(user_id, created_at);
CREATE TABLE IF NOT EXISTS persona_embeddings (
    chunk_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


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
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.db: aiosqlite.Connection | None = None

    async def open(self) -> "MemoryStore":
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript(SCHEMA)
        await self.db.commit()
        return self

    async def close(self) -> None:
        if self.db:
            await self.db.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self.db is not None, "MemoryStore.open() не вызван"
        return self.db

    # ---------- users ----------
    async def get_or_create_user(self, user_id: int, chat_id: int, display_name: str | None = None) -> UserProfile:
        row = await (await self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,))).fetchone()
        if row is None:
            now = time.time()
            await self.conn.execute(
                "INSERT INTO users(user_id, chat_id, display_name, created_at) VALUES (?,?,?,?)",
                (user_id, chat_id, display_name, now),
            )
            await self.conn.commit()
            return UserProfile(user_id=user_id, chat_id=chat_id, display_name=display_name, created_at=now)
        return UserProfile(**{k: row[k] for k in row.keys()})

    async def get_user(self, user_id: int) -> UserProfile | None:
        row = await (await self.conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,))).fetchone()
        return UserProfile(**{k: row[k] for k in row.keys()}) if row else None

    async def list_users(self) -> list[UserProfile]:
        rows = await (await self.conn.execute("SELECT * FROM users")).fetchall()
        return [UserProfile(**{k: r[k] for k in r.keys()}) for r in rows]

    async def update_user(self, user_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        await self.conn.execute(f"UPDATE users SET {cols} WHERE user_id=?", (*fields.values(), user_id))
        await self.conn.commit()

    # ---------- messages ----------
    async def add_message(self, user_id: int, role: str, text: str, kind: str = "chat", at: float | None = None) -> int:
        cur = await self.conn.execute(
            "INSERT INTO messages(user_id, role, text, created_at, kind) VALUES (?,?,?,?,?)",
            (user_id, role, text, at or time.time(), kind),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def active_messages(self, user_id: int) -> list[StoredMessage]:
        rows = await (
            await self.conn.execute(
                "SELECT id, role, text, created_at, kind FROM messages WHERE user_id=? AND archived=0 ORDER BY id",
                (user_id,),
            )
        ).fetchall()
        return [StoredMessage(r["id"], r["role"], r["text"], r["created_at"], r["kind"]) for r in rows]

    async def last_messages(self, user_id: int, n: int, role: str | None = None) -> list[StoredMessage]:
        q = "SELECT id, role, text, created_at, kind FROM messages WHERE user_id=?"
        args: list[Any] = [user_id]
        if role:
            q += " AND role=?"
            args.append(role)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(n)
        rows = await (await self.conn.execute(q, args)).fetchall()
        return [StoredMessage(r["id"], r["role"], r["text"], r["created_at"], r["kind"]) for r in reversed(rows)]

    async def update_message_text(self, message_id: int, text: str) -> None:
        await self.conn.execute("UPDATE messages SET text=? WHERE id=?", (text, message_id))
        await self.conn.commit()

    async def archive_messages_upto(self, user_id: int, last_id: int) -> None:
        await self.conn.execute("UPDATE messages SET archived=1 WHERE user_id=? AND id<=?", (user_id, last_id))
        await self.conn.commit()

    # ---------- summaries ----------
    async def get_summary(self, user_id: int) -> str:
        row = await (await self.conn.execute("SELECT text FROM summaries WHERE user_id=?", (user_id,))).fetchone()
        return row["text"] if row else ""

    async def set_summary(self, user_id: int, text: str, covered_until_id: int) -> None:
        await self.conn.execute(
            "INSERT INTO summaries(user_id, text, covered_until_id, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET text=excluded.text, covered_until_id=excluded.covered_until_id, "
            "updated_at=excluded.updated_at",
            (user_id, text, covered_until_id, time.time()),
        )
        await self.conn.commit()

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
        cur = await self.conn.execute(
            "INSERT INTO episodes(user_id, kind, text, importance, emotion, created_at, embedding) VALUES (?,?,?,?,?,?,?)",
            (user_id, kind, text, int(importance), emotion, at or time.time(), to_blob(embedding)),
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def episodes(self, user_id: int, kind: str | None = None) -> list[Episode]:
        q = "SELECT * FROM episodes WHERE user_id=?"
        args: list[Any] = [user_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        rows = await (await self.conn.execute(q + " ORDER BY id", args)).fetchall()
        return [
            Episode(r["id"], r["kind"], r["text"], r["importance"], r["emotion"], r["created_at"],
                    r["recall_count"], from_blob(r["embedding"]))
            for r in rows
        ]

    async def set_embedding(self, table: str, row_id: int, vec: np.ndarray) -> None:
        assert table in ("episodes", "facts")
        await self.conn.execute(f"UPDATE {table} SET embedding=? WHERE id=?", (to_blob(vec), row_id))
        await self.conn.commit()

    async def mark_recalled(self, episode_ids: list[int]) -> None:
        if not episode_ids:
            return
        now = time.time()
        await self.conn.executemany(
            "UPDATE episodes SET recall_count=recall_count+1, last_recalled_at=? WHERE id=?",
            [(now, i) for i in episode_ids],
        )
        await self.conn.commit()

    # ---------- facts ----------
    async def upsert_fact(
        self, user_id: int, key: str, value: str, confidence: float = 0.7,
        embedding: np.ndarray | None = None, pinned: bool = False,
    ) -> None:
        now = time.time()
        await self.conn.execute(
            "INSERT INTO facts(user_id, key, value, confidence, pinned, created_at, updated_at, embedding) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, "
            "confidence=MAX(facts.confidence, excluded.confidence), updated_at=excluded.updated_at, "
            "embedding=COALESCE(excluded.embedding, facts.embedding), pinned=MAX(facts.pinned, excluded.pinned)",
            (user_id, key.strip().lower()[:80], value.strip()[:400], confidence, int(pinned), now, now, to_blob(embedding)),
        )
        await self.conn.commit()

    async def delete_fact(self, user_id: int, key: str) -> None:
        await self.conn.execute("DELETE FROM facts WHERE user_id=? AND key=?", (user_id, key.strip().lower()))
        await self.conn.commit()

    async def facts(self, user_id: int) -> list[Fact]:
        rows = await (await self.conn.execute("SELECT * FROM facts WHERE user_id=? ORDER BY updated_at DESC", (user_id,))).fetchall()
        return [
            Fact(r["id"], r["key"], r["value"], r["confidence"], bool(r["pinned"]), r["updated_at"], from_blob(r["embedding"]))
            for r in rows
        ]

    # ---------- relationship ----------
    async def get_relationship(self, user_id: int) -> RelationshipRow:
        r = await (await self.conn.execute("SELECT * FROM relationship WHERE user_id=?", (user_id,))).fetchone()
        if not r:
            return RelationshipRow()
        return RelationshipRow(r["trust"], r["stage"], r["quality_sessions"], json.loads(r["milestones"]), json.loads(r["seen_events"]))

    async def save_relationship(self, user_id: int, rel: RelationshipRow) -> None:
        await self.conn.execute(
            "INSERT INTO relationship(user_id, trust, stage, quality_sessions, milestones, seen_events, updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET trust=excluded.trust, stage=excluded.stage, "
            "quality_sessions=excluded.quality_sessions, milestones=excluded.milestones, "
            "seen_events=excluded.seen_events, updated_at=excluded.updated_at",
            (user_id, rel.trust, rel.stage, rel.quality_sessions, json.dumps(rel.milestones),
             json.dumps(rel.seen_events[-50:]), time.time()),
        )
        await self.conn.commit()

    # ---------- emotion ----------
    async def get_emotion(self, user_id: int) -> tuple[dict[str, float], float, str | None, float] | None:
        r = await (await self.conn.execute("SELECT * FROM emotion_state WHERE user_id=?", (user_id,))).fetchone()
        if not r:
            return None
        return json.loads(r["vals"]), r["mood"], r["last_cause"], r["updated_at"]

    async def save_emotion(self, user_id: int, vals: dict[str, float], mood: float, last_cause: str | None, at: float) -> None:
        await self.conn.execute(
            "INSERT INTO emotion_state(user_id, vals, mood, last_cause, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET vals=excluded.vals, mood=excluded.mood, "
            "last_cause=excluded.last_cause, updated_at=excluded.updated_at",
            (user_id, json.dumps({k: round(v, 4) for k, v in vals.items()}), mood, last_cause, at),
        )
        await self.conn.commit()

    # ---------- events log ----------
    async def log_event(self, user_id: int, kind: str, source: str, detail: str | None = None) -> None:
        await self.conn.execute(
            "INSERT INTO events(user_id, kind, source, detail, created_at) VALUES (?,?,?,?,?)",
            (user_id, kind, source, (detail or "")[:300], time.time()),
        )
        await self.conn.commit()

    async def recent_events(self, user_id: int, n: int = 20) -> list[tuple[str, str, str, float]]:
        rows = await (await self.conn.execute(
            "SELECT kind, source, detail, created_at FROM events WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, n)
        )).fetchall()
        return [(r["kind"], r["source"], r["detail"], r["created_at"]) for r in rows]

    # ---------- threads & notes ----------
    async def add_thread(self, user_id: int, topic: str, follow_up_at: float | None) -> None:
        await self.conn.execute(
            "INSERT INTO threads(user_id, topic, follow_up_at, created_at) VALUES (?,?,?,?)",
            (user_id, topic[:300], follow_up_at, time.time()),
        )
        await self.conn.commit()

    async def open_threads(self, user_id: int) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT * FROM threads WHERE user_id=? AND status='open' ORDER BY id DESC LIMIT 10", (user_id,)
        )).fetchall()
        return [dict(r) for r in rows]

    async def close_thread(self, thread_id: int, status: str = "done") -> None:
        await self.conn.execute("UPDATE threads SET status=? WHERE id=?", (status, thread_id))
        await self.conn.commit()

    async def add_note(self, user_id: int, text: str, due_at: float) -> int:
        cur = await self.conn.execute(
            "INSERT INTO notes(user_id, text, due_at, created_at) VALUES (?,?,?,?)", (user_id, text[:500], due_at, time.time())
        )
        await self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    async def due_notes(self, now: float) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT * FROM notes WHERE status='pending' AND due_at<=? ORDER BY due_at", (now,)
        )).fetchall()
        return [dict(r) for r in rows]

    async def pending_notes(self, user_id: int) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT * FROM notes WHERE user_id=? AND status='pending' ORDER BY due_at", (user_id,)
        )).fetchall()
        return [dict(r) for r in rows]

    async def set_note_status(self, note_id: int, status: str) -> None:
        await self.conn.execute("UPDATE notes SET status=? WHERE id=?", (status, note_id))
        await self.conn.commit()

    # ---------- initiatives ----------
    async def log_initiative(self, user_id: int, kind: str, reason: str, text: str) -> None:
        await self.conn.execute(
            "INSERT INTO initiatives(user_id, kind, reason, text, sent_at) VALUES (?,?,?,?,?)",
            (user_id, kind, reason, text, time.time()),
        )
        await self.conn.commit()

    async def last_initiative(self, user_id: int) -> dict[str, Any] | None:
        r = await (await self.conn.execute(
            "SELECT * FROM initiatives WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
        )).fetchone()
        return dict(r) if r else None

    async def initiatives_since(self, user_id: int, since: float) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT * FROM initiatives WHERE user_id=? AND sent_at>=?", (user_id, since)
        )).fetchall()
        return [dict(r) for r in rows]

    async def mark_initiatives_answered(self, user_id: int) -> None:
        await self.conn.execute("UPDATE initiatives SET answered=1 WHERE user_id=? AND answered=0", (user_id,))
        await self.conn.commit()

    # ---------- usage ----------
    async def log_usage(self, user_id: int | None, purpose: str, model: str, usage: Any, cost: float) -> None:
        await self.conn.execute(
            "INSERT INTO usage(user_id, purpose, model, input_tokens, output_tokens, cache_read, cache_write, cost_usd, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, purpose, model, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
             usage.cache_write_tokens, cost, time.time()),
        )
        await self.conn.commit()

    async def cost_since(self, user_id: int | None, since: float) -> float:
        if user_id is None:
            r = await (await self.conn.execute("SELECT COALESCE(SUM(cost_usd),0) c FROM usage WHERE created_at>=?", (since,))).fetchone()
        else:
            r = await (await self.conn.execute(
                "SELECT COALESCE(SUM(cost_usd),0) c FROM usage WHERE user_id=? AND created_at>=?", (user_id, since)
            )).fetchone()
        return float(r["c"])

    async def usage_report(self, since: float) -> list[dict[str, Any]]:
        rows = await (await self.conn.execute(
            "SELECT purpose, model, COUNT(*) n, SUM(input_tokens) inp, SUM(output_tokens) out, SUM(cache_read) cr, "
            "SUM(cache_write) cw, SUM(cost_usd) cost FROM usage WHERE created_at>=? GROUP BY purpose, model ORDER BY cost DESC",
            (since,),
        )).fetchall()
        return [dict(r) for r in rows]

    # ---------- persona embeddings cache ----------
    async def get_persona_embedding(self, chunk_id: str, content_hash: str) -> np.ndarray | None:
        r = await (await self.conn.execute(
            "SELECT embedding FROM persona_embeddings WHERE chunk_id=? AND content_hash=?", (chunk_id, content_hash)
        )).fetchone()
        return from_blob(r["embedding"]) if r else None

    async def set_persona_embedding(self, chunk_id: str, content_hash: str, vec: np.ndarray) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO persona_embeddings(chunk_id, content_hash, embedding) VALUES (?,?,?)",
            (chunk_id, content_hash, to_blob(vec)),
        )
        await self.conn.commit()

    # ---------- privacy ----------
    async def forget_user(self, user_id: int) -> None:
        for table in ("messages", "summaries", "episodes", "facts", "relationship", "emotion_state",
                      "events", "threads", "notes", "initiatives"):
            await self.conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        await self.conn.commit()
