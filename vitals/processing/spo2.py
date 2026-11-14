def classify_spo2(value: float) -> dict:
    """Turn an SpO2 percentage into a level."""
    if value < 85:
        level, text = 1, "severe low oxygen"
    elif value < 91:
        level, text = 2, "moderate low oxygen"
    elif value < 95:
        level, text = 3, "mild low oxygen"
    else:
        level, text = 4, "normal"
    return {"level": level, "status": text}