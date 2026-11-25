"""Тонкий слой над базой данных: один и тот же SQL для SQLite (разработка, тесты) и PostgreSQL (прод).

SQL в MemoryStore пишется переносимо: плейсхолдеры `?` (для Postgres переводятся в `$1, $2…`),
`ON CONFLICT … DO UPDATE`, `CASE` вместо `MAX(a, b)`. Различается только схема (см. store.py).
"""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence


class Database(Protocol):
    dialect: str

    async def execute(self, sql: str, args: Sequence[Any] = ()) -> None: ...
    async def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None: ...
    async def insert(self, sql: str, args: Sequence[Any] = ()) -> int: ...
    async def fetchone(self, sql: str, args: Sequence[Any] = ()) -> dict[str, Any] | None: ...
    async def fetchall(self, sql: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]: ...
    async def executescript(self, script: str) -> None: ...
    async def close(self) -> None: ...


class SQLiteDatabase:
    dialect = "sqlite"

    def __init__(self, path: str):
        self.path = path
        self.conn = None

    async def open(self) -> "SQLiteDatabase":
        import aiosqlite

        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        return self

    async def execute(self, sql, args=()):
        await self.conn.execute(sql, tuple(args))
        await self.conn.commit()

    async def executemany(self, sql, rows):
        await self.conn.executemany(sql, [tuple(r) for r in rows])
        await self.conn.commit()

    async def insert(self, sql, args=()):
        cur = await self.conn.execute(sql, tuple(args))
        await self.conn.commit()
        return int(cur.lastrowid)

    async def fetchone(self, sql, args=()):
        row = await (await self.conn.execute(sql, tuple(args))).fetchone()
        return dict(row) if row is not None else None

    async def fetchall(self, sql, args=()):
        rows = await (await self.conn.execute(sql, tuple(args))).fetchall()
        return [dict(r) for r in rows]

    async def executescript(self, script):
        await self.conn.executescript(script)
        await self.conn.commit()

    async def close(self):
        if self.conn is not None:
            await self.conn.close()


_PLACEHOLDER = re.compile(r"\?")


def to_pg(sql: str) -> str:
    """`?` → `$1, $2, …` (в нашем SQL нет `?` внутри строковых литералов)."""
    n = 0

    def repl(_: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"${n}"

    return _PLACEHOLDER.sub(repl, sql)


class PostgresDatabase:
    dialect = "postgres"

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 10):
        self.dsn = dsn
        self.min_size, self.max_size = min_size, max_size
        self.pool = None

    async def open(self) -> "PostgresDatabase":
        import asyncpg

        self.pool = await asyncpg.create_pool(self.dsn, min_size=self.min_size, max_size=self.max_size,
                                              command_timeout=30)
        return self

    async def execute(self, sql, args=()):
        await self.pool.execute(to_pg(sql), *args)

    async def executemany(self, sql, rows):
        await self.pool.executemany(to_pg(sql), [tuple(r) for r in rows])

    async def insert(self, sql, args=()):
        return int(await self.pool.fetchval(to_pg(sql) + " RETURNING id", *args))

    async def fetchone(self, sql, args=()):
        row = await self.pool.fetchrow(to_pg(sql), *args)
        return dict(row) if row is not None else None

    async def fetchall(self, sql, args=()):
        return [dict(r) for r in await self.pool.fetch(to_pg(sql), *args)]

    async def executescript(self, script):
        async with self.pool.acquire() as conn:
            await conn.execute(script)

    async def close(self):
        if self.pool is not None:
            await self.pool.close()


async def open_database(url_or_path: str) -> Database:
    if url_or_path.startswith(("postgres://", "postgresql://")):
        return await PostgresDatabase(url_or_path).open()
    return await SQLiteDatabase(url_or_path).open()
