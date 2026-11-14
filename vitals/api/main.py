from functools import lru_cache
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from shared.config import BASE_DIR, VITALS
from shared.schema import Reading
from vitals.db.database import get_latest, get_readings, init_db, insert_reading
from vitals.processing.heart_rate import classify_hr
from vitals.processing.spo2 import classify_spo2

app = FastAPI(title="SANSIGHT Vitals API")
init_db()

MODELS = BASE_DIR / "models_saved"
HR_TEST = BASE_DIR / "data" / "raw" / "heart_rate" / "Test Data"


@lru_cache(maxsize=None)
def load_model(filename: str):
    path = MODELS / filename
    if not path.exists():
        raise HTTPException(503, f"Model file {filename} not found. Train it first.")
    return joblib.load(path)


def status_for(vital: str, value: float) -> dict:
    if vital == "hr":
        return classify_hr(value)
    if vital == "spo2":
        return classify_spo2(value)
    low, high = VITALS[vital]["normal"]
    state = "low" if value < low else "high" if value > high else "normal"
    return {"state": state}


@app.get("/health")
def health():
    return {"status": "ok", "models_found": sorted(p.name for p in MODELS.glob("*.joblib"))}


# ---------- Readings (stored in SQLite) ----------
@app.post("/readings")
def add_reading(reading: Reading):
    new_id = insert_reading(reading)
    return {"id": new_id, "status": status_for(reading.vital, reading.value)}


@app.get("/readings/{astronaut_id}")
def list_readings(astronaut_id: str, vital: Optional[str] = None, limit: int = 100):
    return get_readings(astronaut_id, vital, limit)


@app.get("/latest/{astronaut_id}")
def latest(astronaut_id: str):
    rows = get_latest(astronaut_id)
    for r in rows:
        r["status"] = status_for(r["vital"], r["value"])
    return rows


# ---------- Models ----------
@app.post("/predict/glucose")
def predict_glucose(data: Dict[str, object]):
    bundle = load_model("glucose_model.joblib")
    row = dict(data)
    if "GENDER" in row:
        row["GENDER"] = {"M": 0, "F": 1}.get(str(row["GENDER"]).upper())
    if "DIABETIC" in row:
        row["DIABETIC"] = {"Y": 1, "N": 0}.get(str(row["DIABETIC"]).upper())
    X = pd.DataFrame([row]).reindex(columns=bundle["features"])
    X = X.apply(pd.to_numeric, errors="coerce")
    value = float(bundle["model"].predict(X)[0])
    return {
        "estimated_glucose_mg_dl": round(value, 1),
        "status": status_for("glucose", value),
        "note": "Demo estimate, average error is about 39 mg/dL.",
    }


@app.post("/predict/stress")
def predict_stress(data: Dict[str, float]):
    bundle = load_model("heart_rate_stress_model.joblib")
    missing = [f for f in bundle["features"] if f not in data]
    if missing:
        raise HTTPException(422, {"missing_features": missing})
    X = pd.DataFrame([data])[bundle["features"]]
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    return {
        "prediction": str(model.predict(X)[0]),
        "probabilities": {c: round(float(p), 3) for c, p in zip(bundle["classes"], proba)},
    }


@lru_cache(maxsize=1)
def _stress_samples() -> pd.DataFrame:
    time = pd.read_csv(next(HR_TEST.rglob("time_domain_features_test.csv")), nrows=5000)
    freq = pd.read_csv(next(HR_TEST.rglob("frequency_domain_features_test.csv")))
    nonlin = pd.read_csv(next(HR_TEST.rglob("heart_rate_non_linear_features_test.csv")))
    return time.merge(freq, on="uuid").merge(nonlin, on="uuid").dropna()


@app.get("/demo/stress-sample")
def stress_sample():
    """A random real row from the test set, with its true label, for demos."""
    bundle = load_model("heart_rate_stress_model.joblib")
    row = _stress_samples().sample(1).iloc[0]
    return {
        "true_condition": str(row["condition"]),
        "features": {f: float(row[f]) for f in bundle["features"]},
    }


@app.post("/predict/diabetes-risk")
def predict_diabetes(data: Dict[str, object]):
    bundle = load_model("diabetes_risk_model.joblib")
    pipe = bundle["model"]
    cat_cols = pipe.named_steps["encode"].transformers_[0][2]
    row = {}
    for f in bundle["features"]:
        if f in data:
            row[f] = data[f]
        else:
            row[f] = "Unknown" if f in cat_cols else 0
    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[0]
    return {
        "prediction": str(pipe.predict(X)[0]),
        "probabilities": {str(c): round(float(p), 3) for c, p in zip(pipe.classes_, proba)},
        "note": "Readmission risk from hospital records, about 58% accurate. Not a diagnosis.",
    }