import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from shared.config import BASE_DIR

HR_DIR = BASE_DIR / "data" / "raw" / "heart_rate"
OUT = BASE_DIR / "models_saved" / "heart_rate_stress_model.joblib"


def load(folder, split):
    """Join the 3 feature files of one split on uuid."""
    d = HR_DIR / folder
    time = pd.read_csv(next(d.rglob(f"time_domain_features_{split}.csv")))
    freq = pd.read_csv(next(d.rglob(f"frequency_domain_features_{split}.csv")))
    nonlin = pd.read_csv(next(d.rglob(f"heart_rate_non_linear_features_{split}.csv")))
    return time.merge(freq, on="uuid").merge(nonlin, on="uuid").dropna()


train = load("Train Data", "train")
test = load("Test Data", "test")
print("Train rows:", len(train), "| Test rows:", len(test))
print("uuids shared between train and test:", len(set(train["uuid"]) & set(test["uuid"])))
print("Train classes:\n", train["condition"].value_counts())

# HR is dropped because the test files don't have it
features = [c for c in train.columns
            if c not in ("uuid", "datasetId", "condition", "HR") and c in test.columns]
print("Features used:", len(features))

# Use 40% of the training rows so training stays fast
sample = train.sample(frac=0.4, random_state=42)

model = RandomForestClassifier(
    n_estimators=120, min_samples_leaf=3, class_weight="balanced",
    n_jobs=-1, random_state=42,
)
model.fit(sample[features], sample["condition"])

# Check on the official test set
pred = model.predict(test[features])
print(classification_report(test["condition"], pred))

# Retrain on all training rows and save
model.fit(train[features], train["condition"])
OUT.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": features, "classes": list(model.classes_)}, OUT)
print("Saved:", OUT)