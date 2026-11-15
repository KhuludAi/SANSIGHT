
from datetime import datetime
from statistics import mean


# ---------- Trend Detection ----------

def analyze_trend(readings):
    """Analyze vital trends, duration, and rate of change."""

    if len(readings) < 3:
        return {
            "trend": "insufficient_data",
            "alert": False,
            "message": "At least 3 readings are required."
        }

    # Sort readings chronologically
    readings = sorted(
        readings,
        key=lambda r: datetime.fromisoformat(
            r["timestamp"].replace("Z", "+00:00")
        )
    )

    values = [float(r["value"]) for r in readings]

    # Analyze the latest three readings
    recent = values[-3:]
    recent_readings = readings[-3:]

    if recent[0] < recent[1] < recent[2]:
        trend = "increasing"

    elif recent[0] > recent[1] > recent[2]:
        trend = "decreasing"

    else:
        trend = "stable_or_fluctuating"

    # Change in measured value
    change = recent[-1] - recent[0]

    # Time interval between first and last recent reading
    start_time = datetime.fromisoformat(
        recent_readings[0]["timestamp"].replace("Z", "+00:00")
    )

    end_time = datetime.fromisoformat(
        recent_readings[-1]["timestamp"].replace("Z", "+00:00")
    )

    duration_minutes = (
        end_time - start_time
    ).total_seconds() / 60

    # Rate of change per minute
    if duration_minutes > 0:
        rate_per_minute = round(
            change / duration_minutes, 4
        )
    else:
        rate_per_minute = None

    return {
        "trend": trend,
        "alert": False,
        "latest_value": recent[-1],
        "change": round(change, 2),
        "average": round(mean(values), 2),
        "readings_analyzed": len(values),
        "duration_minutes": round(duration_minutes, 2),
        "rate_per_minute": rate_per_minute,
        "message": (
            "A directional trend was detected. Review the readings."
            if trend in ("increasing", "decreasing")
            else "No consistent directional trend detected."
        )
    }


# ---------- Alert Evaluation ----------

def evaluate_alert(vital, trend_result):
    """Experimental trend alerts, not medical diagnoses."""

    trend = trend_result.get("trend")

    if trend == "insufficient_data":
        return {
            "severity": "unknown",
            "message": "At least 3 readings are required."
        }

    if trend not in ("increasing", "decreasing"):
        return {
            "severity": "normal",
            "message": "No consistent directional trend detected."
        }

    change = abs(trend_result.get("change", 0))

    # Demo thresholds only — not clinical thresholds
    thresholds = {
        "hr": 10,
        "spo2": 3,
        "glucose": 20,
        "hrv": 10
    }

    threshold = thresholds.get(vital)

    if threshold is None:
        return {
            "severity": "unknown",
            "message": "Unsupported vital sign."
        }

    if change >= threshold:
        return {
            "severity": "watch",
            "message": (
                "A notable directional change was detected. "
                "Review recent readings and measurement context."
            )
        }

    return {
        "severity": "normal",
        "message": "Directional change is below the demo alert threshold."
    }
