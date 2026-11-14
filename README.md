# Vital Signs Module

Models and rules for vital signs monitoring.

## What is inside

- Heart rate and stress model (no stress / interruption / time pressure)
- SpO2 rule (oxygen level)
- Heart rate rule (low / normal / high)
- Glucose estimate model
- Diabetes readmission risk model (uses blood_pressure.csv, which is a diabetes hospital dataset)
- FastAPI backend

## How to use

1. pip install -r requirements.txt
2. Get the datasets from Shafeeq and put them in data/raw/ (they are not on GitHub)
3. Train the models:
   python -m vitals.models.train_heart_rate
   python -m vitals.models.train_glucose
   python -m vitals.models.train_diabetes_risk
4. Run the API:
   uvicorn vitals.api.main:app --reload
5. Open http://127.0.0.1:8000/docs

## Honest results

- Stress model: about 100% on the dataset's own test split. Not tested on new people.
- Glucose model: average mistake about 39 mg/dL. Demo estimate only.
- Diabetes risk model: about 54% correct overall, about 30% for the 30-day return group.
