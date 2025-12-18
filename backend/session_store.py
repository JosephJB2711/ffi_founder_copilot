# backend/session_store.py
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path(__file__).resolve().parent / "sessions.sqlite3"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA foreign_keys=ON;")
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);")


def touch_session(session_id: str) -> None:
    now = int(time.time())
    with _conn() as c:
        c.execute("""
        INSERT INTO sessions(session_id, created_at, updated_at)
        VALUES(?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at;
        """, (session_id, now, now))


def get_summary(session_id: str) -> str:
    with _conn() as c:
        row = c.execute("SELECT summary FROM sessions WHERE session_id=?", (session_id,)).fetchone()
    return row[0] if row else ""


def set_summary(session_id: str, summary: str) -> None:
    now = int(time.time())
    with _conn() as c:
        c.execute("""
        INSERT INTO sessions(session_id, summary, created_at, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at;
        """, (session_id, summary, now, now))


def append_message(session_id: str, role: str, content: str) -> None:
    now = int(time.time())
    with _conn() as c:
        c.execute("INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                  (session_id, role, content, now))
        c.execute("UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id))


def get_last_messages(session_id: str, limit: int = 12) -> List[Dict[str, str]]:
    with _conn() as c:
        rows = c.execute("""
            SELECT role, content
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]


def count_messages(session_id: str) -> int:
    with _conn() as c:
        row = c.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()
    return int(row[0]) if row else 0


def delete_oldest_messages(session_id: str, keep_last: int) -> None:
    # Löscht alle bis auf die letzten keep_last messages.
    with _conn() as c:
        c.execute("""
            DELETE FROM messages
            WHERE session_id=?
              AND id NOT IN (
                SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?
              )
        """, (session_id, session_id, keep_last))
