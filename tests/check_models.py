import joblib
import pandas as pd
from shared.config import BASE_DIR

M = BASE_DIR / "models_saved"
RAW = BASE_DIR / "data" / "raw"

# 1. Heart rate stress
b = joblib.load(M / "heart_rate_stress_model.joblib")
d = RAW / "heart_rate" / "Test Data"
t = pd.read_csv(next(d.rglob("time_domain_features_test.csv")), nrows=3000)
f = pd.read_csv(next(d.rglob("frequency_domain_features_test.csv")))
n = pd.read_csv(next(d.rglob("heart_rate_non_linear_features_test.csv")))
df = t.merge(f, on="uuid").merge(n, on="uuid").dropna()
pred = b["model"].predict(df[b["features"]])
print("Stress model: correct", round((pred == df["condition"]).mean() * 100, 1), "%")

# 2. Glucose
b = joblib.load(M / "glucose_model.joblib")
g = pd.read_csv(RAW / "glucose_level_estimation.csv").dropna(subset=["GLUCOSE_LEVEL"])
g["GENDER"] = g["GENDER"].map({"M": 0, "F": 1})
g["DIABETIC"] = g["DIABETIC"].map({"Y": 1, "N": 0})
pred = b["model"].predict(g[b["features"]])
print("Glucose model: average mistake", round(abs(pred - g["GLUCOSE_LEVEL"]).mean(), 1), "mg/dL")

# 3. Diabetes risk
b = joblib.load(M / "diabetes_risk_model.joblib")
x = pd.read_csv(RAW / "blood_pressure.csv", nrows=3000).replace("?", pd.NA)
y = x["label"]
x = x.drop(columns=["id", "patient_no", "weight", "label"])
cat = x.select_dtypes(include=["object", "string"]).columns
x[cat] = x[cat].fillna("Unknown").astype(str)
pred = b["model"].predict(x[b["features"]])
print("Diabetes risk model: correct", round((pred == y).mean() * 100, 1), "%")