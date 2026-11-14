def classify_hr(bpm: float) -> dict:
    """Turn a heart rate in bpm into a status."""
    if bpm < 60:
        return {"state": "low", "status": "bradycardia range"}
    if bpm <= 100:
        return {"state": "normal", "status": "normal resting heart rate"}
    return {"state": "high", "status": "tachycardia range"}