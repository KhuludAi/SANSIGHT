import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from shared.config import BASE_DIR

DATA = BASE_DIR / "data" / "raw" / "glucose_level_estimation.csv"
OUT = BASE_DIR / "models_saved" / "glucose_model.joblib"

# 1. Load and clean
df = pd.read_csv(DATA)
df = df.dropna(subset=["GLUCOSE_LEVEL"])
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["DIABETIC"] = df["DIABETIC"].map({"Y": 1, "N": 0})
print("Rows used:", len(df))

features = [c for c in df.columns if c != "GLUCOSE_LEVEL"]
X = df[features]
y = df["GLUCOSE_LEVEL"]

# 2. Two candidate models (missing values filled with the median)
models = {
    "LinearRegression": make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
    "RandomForest": make_pipeline(SimpleImputer(strategy="median"),
                                  RandomForestRegressor(n_estimators=200, random_state=42)),
}

# 3. Compare them with 5-fold cross-validation
scores = {}
for name, model in models.items():
    mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    scores[name] = mae
    print(f"{name}: average error = {mae:.2f} mg/dL")

best_name = min(scores, key=scores.get)
print("Best model:", best_name)

# 4. Check on data the model has never seen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best = models[best_name]
best.fit(X_train, y_train)
pred = best.predict(X_test)
print(f"Test MAE: {mean_absolute_error(y_test, pred):.2f} mg/dL")
print(f"Test R2:  {r2_score(y_test, pred):.2f}")

# 5. Train on everything and save
best.fit(X, y)
OUT.parent.mkdir(exist_ok=True)
joblib.dump({"model": best, "features": features}, OUT)
print("Saved:", OUT)