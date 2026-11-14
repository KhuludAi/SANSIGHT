import sqlite3
from typing import List, Optional

from shared.config import DB_PATH, VITALS
from shared.schema import Reading


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                vital TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                source TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_astro_vital_time
            ON readings (astronaut_id, vital, timestamp)
        """)


def insert_reading(r: Reading) -> int:
    unit = r.unit or VITALS[r.vital]["unit"]
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO readings (astronaut_id, timestamp, vital, value, unit, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r.astronaut_id, r.timestamp.isoformat(), r.vital, r.value, unit, r.source),
        )
        return cur.lastrowid


def get_readings(astronaut_id: str, vital: Optional[str] = None, limit: int = 100) -> List[dict]:
    query = "SELECT * FROM readings WHERE astronaut_id = ?"
    params = [astronaut_id]
    if vital:
        query += " AND vital = ?"
        params.append(vital)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_latest(astronaut_id: str) -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT r.* FROM readings r
            JOIN (
                SELECT vital, MAX(timestamp) AS ts
                FROM readings WHERE astronaut_id = ? GROUP BY vital
            ) m ON r.vital = m.vital AND r.timestamp = m.ts
            WHERE r.astronaut_id = ?
        """, (astronaut_id, astronaut_id)).fetchall()
        return [dict(row) for row in rows]