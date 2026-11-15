
from datetime import datetime
from statistics import mean


def analyze_trend(readings):
    """Analyze a vital sign's recent readings in chronological order."""

    if len(readings) < 3:
        return {
            "trend": "insufficient_data",
            "alert": False,
            "message": "At least 3 readings are required."
        }

    readings = sorted(
        readings,
        key=lambda r: datetime.fromisoformat(
            r["timestamp"].replace("Z", "+00:00")
        )
    )

    values = [float(r["value"]) for r in readings]

    # Compare the latest 3 consecutive readings.
    recent = values[-3:]

    if recent[0] < recent[1] < recent[2]:
        trend = "increasing"
    elif recent[0] > recent[1] > recent[2]:
        trend = "decreasing"
    else:
        trend = "stable_or_fluctuating"

    change = recent[-1] - recent[0]

    return {
        "trend": trend,
        "alert": trend in ("increasing", "decreasing"),
        "latest_value": recent[-1],
        "change": round(change, 2),
        "average": round(mean(values), 2),
        "readings_analyzed": len(values),
        "message": (
            "A directional trend was detected. Review the readings."
            if trend in ("increasing", "decreasing")
            else "No consistent directional trend detected."
        )
    }

def evaluate_alert(vital, trend_result):
    """Demo-only alert classification based on trend direction."""

    trend = trend_result.get("trend")

    if trend == "insufficient_data":
        return {
            "severity": "unknown",
            "message": "More readings are needed."
        }

    if trend in ("increasing", "decreasing"):
        return {
            "severity": "watch",
            "message": (
                "A consistent trend was detected. "
                "Check activity, measurement quality, "
                "and recent readings."
            )
        }

    return {
        "severity": "normal",
        "message": "No consistent directional trend detected."
    }
