from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "vitals.db"

# Vitals we track, with unit and normal range (low, high)
VITALS = {
    "hr":      {"name": "Heart Rate",              "unit": "bpm",   "normal": (60, 100)},
    "spo2":    {"name": "Blood Oxygen Saturation", "unit": "%",     "normal": (95, 100)},
    "glucose": {"name": "Blood Glucose",           "unit": "mg/dL", "normal": (70, 140)},
    "hrv":     {"name": "Heart Rate Variability",  "unit": "ms",    "normal": (20, 100)},
}