"""Перенос памяти из SQLite в PostgreSQL.

    python scripts/migrate_sqlite_to_pg.py var/columbina.sqlite3 postgresql://columbina:pass@localhost:5432/columbina

Целевая база должна быть пустой (схема создаётся автоматически). С --force данные дописываются поверх.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from columbina.memory.store import ALL_TABLES, MemoryStore  # noqa: E402

SERIAL_TABLES = {"messages", "episodes", "facts", "events", "threads", "notes", "initiatives", "usage"}


async def main(src: str, dst: str, force: bool) -> None:
    if not Path(src).exists():
        raise SystemExit(f"нет файла {src}")
    s = await MemoryStore(src).open()
    d = await MemoryStore(dst).open()
    if d.d.dialect != "postgres":
        raise SystemExit("второй аргумент должен быть postgresql://…")
    try:
        if not force:
            for t in ALL_TABLES:
                r = await d.d.fetchone(f"SELECT COUNT(*) AS n FROM {t}")
                if r["n"]:
                    raise SystemExit(f"в целевой таблице {t} уже есть данные; используйте --force")
        for t in ALL_TABLES:
            rows = await s.d.fetchall(f"SELECT * FROM {t}")
            if not rows:
                print(f"{t:<20} 0")
                continue
            cols = list(rows[0].keys())
            sql = f"INSERT INTO {t}({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) ON CONFLICT DO NOTHING"
            await d.d.executemany(sql, [[r[c] for c in cols] for r in rows])
            if t in SERIAL_TABLES:
                await d.d.execute(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), "
                                  f"(SELECT COALESCE(MAX(id), 1) FROM {t}))")
            print(f"{t:<20} {len(rows)}")
    finally:
        await s.close()
        await d.close()
    print("готово")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path")
    ap.add_argument("postgres_url")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.sqlite_path, a.postgres_url, a.force))
