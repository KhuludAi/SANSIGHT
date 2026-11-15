
from functools import lru_cache
from typing import Dict, Optional
from vitals.processing.data_quality import check_data_quality, validate_alert_quality
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from shared.config import BASE_DIR, VITALS
from shared.schema import Reading
from vitals.db.database import (
    get_latest,
    get_readings,
    init_db,
    insert_reading,
)
from vitals.processing.heart_rate import classify_hr
from vitals.processing.spo2 import classify_spo2
from vitals.processing.trend_detection import (
    analyze_trend,
    evaluate_alert,
)

app = FastAPI(title="SANSIGHT Vitals API")
init_db()

MODELS = BASE_DIR / "models_saved"
HR_TEST = BASE_DIR / "data" / "raw" / "heart_rate" / "Test Data"


# ---------- Model Loading ----------

@lru_cache(maxsize=None)
def load_model(filename: str):
    path = MODELS / filename

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file {filename} not found."
        )

    return joblib.load(path)


# ---------- Vital Status ----------

def status_for(vital: str, value: float) -> dict:
    if vital == "hr":
        return classify_hr(value)

    if vital == "spo2":
        return classify_spo2(value)

    low, high = VITALS[vital]["normal"]

    state = (
        "low" if value < low
        else "high" if value > high
        else "normal"
    )

    return {"state": state}


# ---------- Health ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_found": sorted(
            p.name for p in MODELS.glob("*.joblib")
        ),
    }


# ---------- Readings ----------

@app.post("/readings")
def add_reading(reading: Reading):
    new_id = insert_reading(reading)

    return {
        "id": new_id,
        "status": status_for(
            reading.vital,
            reading.value
        ),
    }


@app.get("/readings/{astronaut_id}")
def list_readings(
    astronaut_id: str,
    vital: Optional[str] = None,
    limit: int = 100
):
    return get_readings(
        astronaut_id,
        vital,
        limit
    )


@app.get("/latest/{astronaut_id}")
def latest(astronaut_id: str):
    rows = get_latest(astronaut_id)

    for row in rows:
        row["status"] = status_for(
            row["vital"],
            row["value"]
        )

    return rows



# ---------- Trend Detection & Alerts ----------

@app.get("/trends/{astronaut_id}/{vital}")
def get_vital_trend(
    astronaut_id: str,
    vital: str
):
    if vital not in VITALS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vital: {vital}"
        )

    readings = get_readings(
        astronaut_id=astronaut_id,
        vital=vital,
        limit=30
    )

    # Analyze trend
    result = analyze_trend(readings)

    # Check data quality
    quality = check_data_quality(readings)

    # Evaluate alert
    alert = evaluate_alert(vital, result)

    # Validate alert against data quality
    alert = validate_alert_quality(alert, quality)

    result["alert"] = alert["severity"] == "watch"

    return {
        "astronaut_id": astronaut_id,
        "vital": vital,
        "analysis": result,
        "data_quality": quality,
        "alert_details": alert
    }


# ---------- Glucose Model (Legacy Demo) ----------

@app.post("/predict/glucose")
def predict_glucose(data: Dict[str, object]):
    bundle = load_model("glucose_model.joblib")

    row = dict(data)

    if "GENDER" in row:
        row["GENDER"] = {
            "M": 0,
            "F": 1
        }.get(str(row["GENDER"]).upper())

    if "DIABETIC" in row:
        row["DIABETIC"] = {
            "Y": 1,
            "N": 0
        }.get(str(row["DIABETIC"]).upper())

    X = pd.DataFrame([row]).reindex(
        columns=bundle["features"]
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    value = float(
        bundle["model"].predict(X)[0]
    )

    return {
        "estimated_glucose_mg_dl": round(value, 1),
        "status": status_for("glucose", value),
        "note": (
            "Demo estimate only. "
            "Average error is about 39 mg/dL."
        ),
    }


# ---------- Stress Prediction ----------

@app.post("/predict/stress")
def predict_stress(data: Dict[str, float]):
    bundle = load_model(
        "heart_rate_stress_model.joblib"
    )

    missing = [
        feature
        for feature in bundle["features"]
        if feature not in data
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing}
        )

    X = pd.DataFrame([data])[
        bundle["features"]
    ]

    model = bundle["model"]
    proba = model.predict_proba(X)[0]

    return {
        "prediction": str(model.predict(X)[0]),
        "probabilities": {
            str(c): round(float(p), 3)
            for c, p in zip(
                bundle["classes"],
                proba
            )
        },
    }


# ---------- Stress Demo Samples ----------

@lru_cache(maxsize=1)
def _stress_samples() -> pd.DataFrame:
    time = pd.read_csv(
        next(
            HR_TEST.rglob(
                "time_domain_features_test.csv"
            )
        ),
        nrows=5000
    )

    freq = pd.read_csv(
        next(
            HR_TEST.rglob(
                "frequency_domain_features_test.csv"
            )
        )
    )

    nonlin = pd.read_csv(
        next(
            HR_TEST.rglob(
                "heart_rate_non_linear_features_test.csv"
            )
        )
    )

    return (
        time.merge(freq, on="uuid")
        .merge(nonlin, on="uuid")
        .dropna()
    )


@app.get("/demo/stress-sample")
def stress_sample():
    bundle = load_model(
        "heart_rate_stress_model.joblib"
    )

    if not HR_TEST.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "HRV test dataset is missing. "
                "The trained model is available, "
                "but demo samples require the original CSV files."
            )
        )

    required_files = [
        "time_domain_features_test.csv",
        "frequency_domain_features_test.csv",
        "heart_rate_non_linear_features_test.csv",
    ]

    for filename in required_files:
        if not any(HR_TEST.rglob(filename)):
            raise HTTPException(
                status_code=503,
                detail=f"Missing HRV test file: {filename}"
            )

    samples = _stress_samples()

    if samples.empty:
        raise HTTPException(
            status_code=503,
            detail="No complete HRV test samples are available."
        )

    row = samples.sample(1).iloc[0]

    return {
        "true_condition": str(row["condition"]),
        "features": {
            feature: float(row[feature])
            for feature in bundle["features"]
        },
    }


# ---------- Diabetes Risk (Legacy Demo) ----------

@app.post("/predict/diabetes-risk")
def predict_diabetes(data: Dict[str, object]):
    bundle = load_model(
        "diabetes_risk_model.joblib"
    )

    pipe = bundle["model"]

    cat_cols = (
        pipe.named_steps["encode"]
        .transformers_[0][2]
    )

    row = {}

    for feature in bundle["features"]:
        if feature in data:
            row[feature] = data[feature]
        else:
            row[feature] = (
                "Unknown"
                if feature in cat_cols
                else 0
            )

    X = pd.DataFrame([row])

    proba = pipe.predict_proba(X)[0]

    return {
        "prediction": str(pipe.predict(X)[0]),
        "probabilities": {
            str(c): round(float(p), 3)
            for c, p in zip(
                pipe.classes_,
                proba
            )
        },
        "note": (
            "Hospital readmission risk demo. "
            "Not an astronaut health diagnosis."
        ),
    }

# ---------- Data Quality ----------

@app.get("/quality/{astronaut_id}/{vital}")
def get_data_quality(astronaut_id: str, vital: str):
    if vital not in VITALS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown vital: {vital}"
        )

    readings = get_readings(
        astronaut_id=astronaut_id,
        vital=vital,
        limit=30
    )

    quality = check_data_quality(readings)

    return {
        "astronaut_id": astronaut_id,
        "vital": vital,
        "data_quality": quality
    }
