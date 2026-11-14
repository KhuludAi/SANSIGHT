
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "sansight.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oct_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                astronaut_id TEXT NOT NULL,
                eye TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rnfl_um REAL,
                quality REAL NOT NULL,
                device TEXT NOT NULL,
                UNIQUE(astronaut_id, eye, timestamp)
            )
        """)


def save_measurement(astronaut_id, measurement):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO oct_measurements
            (astronaut_id, eye, timestamp, rnfl_um, quality, device)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            astronaut_id,
            measurement.eye,
            measurement.timestamp,
            measurement.rnfl_um,
            measurement.quality,
            measurement.device
        ))


def load_measurements(astronaut_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT eye, timestamp, rnfl_um, quality, device
            FROM oct_measurements
            WHERE astronaut_id = ?
            ORDER BY timestamp ASC, id ASC
        """, (astronaut_id,)).fetchall()

    return [dict(row) for row in rows]
