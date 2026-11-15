import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from shared.config import BASE_DIR

DATA = BASE_DIR / "data" / "raw" / "blood_pressure.csv"
OUT = BASE_DIR / "models_saved" / "diabetes_risk_model.joblib"

# 1. Load and clean
df = pd.read_csv(DATA)
df = df.replace("?", pd.NA)
groups = df["patient_no"]
y = df["label"]
X = df.drop(columns=["id", "patient_no", "weight", "label"])   # weight is almost all missing

cat_cols = X.select_dtypes(include="object").columns.tolist()
X[cat_cols] = X[cat_cols].fillna("Unknown").astype(str)
print("Rows:", len(X), "| patients:", groups.nunique())
print(y.value_counts())

# 2. Pipeline: text -> numbers, then a random forest
pipe = Pipeline([
    ("encode", ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)],
        remainder="passthrough")),
    ("model", RandomForestClassifier(
        n_estimators=80, min_samples_leaf=20, max_depth=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
])

# 3. Split by patient, so nobody is in both train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = pipe.predict(X.iloc[test_idx])
print(classification_report(y.iloc[test_idx], pred))

# 4. Train on everything and save
pipe.fit(X, y)
OUT.parent.mkdir(exist_ok=True)
joblib.dump({"model": pipe, "features": list(X.columns)}, OUT, compress=3)
print("Saved:", OUT)