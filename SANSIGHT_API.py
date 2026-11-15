
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import urllib.error
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from SANS_MONITOR import SANSMonitor, OCTMeasurement
from ADAPTIVE_SCHEDULER import AdaptiveScheduler


# ==========================================
# SANSIGHT API
# ==========================================

app = FastAPI(
    title="SANSIGHT API",
    version="0.2.0",
    description="SANS monitoring with persistent SQLite storage"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# DATABASE
# ==========================================

DB_PATH = Path(__file__).resolve().parent / "sansight.db"

# Protect the prototype's in-memory state and database writes.
state_lock = threading.RLock()

monitors = {}


def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
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


def load_measurements(astronaut_id):
    with connect_db() as conn:
        rows = conn.execute("""
            SELECT eye, timestamp, rnfl_um, quality, device
            FROM oct_measurements
            WHERE astronaut_id = ?
        """, (astronaut_id,)).fetchall()

    # Sort by actual datetime, not timestamp text.
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: datetime.fromisoformat(
            row["timestamp"]
        ).astimezone(timezone.utc)
    )


def save_measurement(astronaut_id, measurement):
    with connect_db() as conn:
        conn.execute("""
            INSERT INTO oct_measurements
            (
                astronaut_id,
                eye,
                timestamp,
                rnfl_um,
                quality,
                device
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            astronaut_id,
            measurement.eye,
            measurement.timestamp,
            measurement.rnfl_um,
            measurement.quality,
            measurement.device
        ))


init_db()


# ==========================================
# INPUT SCHEMA
# ==========================================

class MeasurementInput(BaseModel):
    astronaut_id: str = Field(min_length=1)
    eye: Literal["R", "L"]
    timestamp: datetime
    rnfl_um: float | None = None
    quality: float = Field(ge=0, le=1)
    device: str = "OCT"


# ==========================================
# MONITOR STATE
# ==========================================

def get_monitor(astronaut_id: str):
    with state_lock:
        if astronaut_id not in monitors:
            monitor = SANSMonitor()

            for row in load_measurements(astronaut_id):
                measurement = OCTMeasurement(
                    eye=row["eye"],
                    timestamp=row["timestamp"],
                    rnfl_um=row["rnfl_um"],
                    quality=row["quality"],
                    device=row["device"]
                )

                monitor.process(measurement)

            monitors[astronaut_id] = monitor

        return monitors[astronaut_id]


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SANSIGHT",
        "version": "0.2.0",
        "database": "SQLite"
    }


# ==========================================
# ADD OCT MEASUREMENT
# ==========================================

@app.post("/sans/measurement")
def add_measurement(data: MeasurementInput):
    if data.timestamp.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="Timestamp must include a timezone"
        )

    measurement = OCTMeasurement(
        eye=data.eye,
        timestamp=data.timestamp.isoformat(),
        rnfl_um=data.rnfl_um,
        quality=data.quality,
        device=data.device
    )

    with state_lock:
        monitor = get_monitor(data.astronaut_id)
        eye_state = monitor.eyes[data.eye]

        # Reject older or repeated timestamps before changing state.
        if (
            eye_state.last_time is not None
            and data.timestamp <= eye_state.last_time
        ):
            raise HTTPException(
                status_code=409,
                detail="Out-of-order or duplicate measurement"
            )

        # Persist first, then update the in-memory monitor.
        # This prevents a database failure from silently losing data.
        try:
            save_measurement(data.astronaut_id, measurement)
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=409,
                detail="Measurement already exists"
            )
        except sqlite3.Error:
            raise HTTPException(
                status_code=500,
                detail="Could not save measurement"
            )

        result = monitor.process(measurement)

    return {
        "astronaut_id": data.astronaut_id,
        "measurement_result": result
    }


# ==========================================
# SANS STATUS
# ==========================================

@app.get("/sans/status/{astronaut_id}")
def get_status(astronaut_id: str):
    with state_lock:
        monitor = get_monitor(astronaut_id)

        eyes = {}

        for eye, state in monitor.eyes.items():
            eyes[eye] = {
                "baseline_rnfl_um": state.baseline,
                "estimated_rnfl_um": state.estimate,
                "uncertainty_um": round(
                    state.variance ** 0.5, 2
                ),
                "last_valid_scan": (
                    state.last_valid_time.isoformat()
                    if state.last_valid_time else None
                ),
                "history": list(state.history)
            }

        return {
            "astronaut_id": astronaut_id,
            "eyes": eyes
        }


# ==========================================
# ADAPTIVE SCAN SCHEDULING
# ==========================================

@app.get("/sans/schedule/{astronaut_id}")
def get_schedule(astronaut_id: str):
    with state_lock:
        monitor = get_monitor(astronaut_id)
        scheduler = AdaptiveScheduler(monitor)

        now = datetime.now(timezone.utc)

        return {
            "astronaut_id": astronaut_id,
            "generated_at": now.isoformat(),
            "recommendations": scheduler.recommend(now)
        }



# ==========================================
# ASTRONAUT OVERVIEW
# ==========================================

@app.get("/astronaut/{astronaut_id}/overview")
def astronaut_overview(astronaut_id: str):
    return {
        "astronaut_id": astronaut_id,
        "sans": get_status(astronaut_id),
        "vitals": fetch_vitals(astronaut_id)
    }



# ==========================================
# VITAL SIGNS API INTEGRATION
# ==========================================

VITALS_API_URL = "http://127.0.0.1:8000"


def fetch_vitals(astronaut_id: str):
    """
    Fetch latest vital signs, trend analysis,
    data quality, and alerts from Vital Signs API.
    """

    def get_json(path):
        url = f"{VITALS_API_URL}{path}"

        with urllib.request.urlopen(
            url, timeout=5
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    # Get latest vital-sign readings
    try:
        latest = get_json(
            f"/latest/{astronaut_id}"
        )

    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError
    ):
        return {
            "integration_status": "unavailable",
            "data": None,
            "analyses": {},
            "message": "Vital Signs API is unavailable."
        }

    # Get trend analysis, data quality, and alerts
    analyses = {}

    for vital in ("hr", "spo2", "glucose", "hrv"):
        try:
            analyses[vital] = get_json(
                f"/trends/{astronaut_id}/{vital}"
            )

        except (
            urllib.error.URLError,
            ValueError,
            TimeoutError
        ):
            analyses[vital] = {
                "status": "unavailable",
                "message": "Analysis could not be retrieved."
            }

    return {
        "integration_status": "connected",
        "data": latest,
        "analyses": analyses
    }
