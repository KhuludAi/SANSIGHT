
from datetime import datetime, timezone


def check_data_quality(readings, max_gap_minutes=60):
    """Check timestamp integrity of vital-sign readings."""

    if not readings:
        return {
            "quality": "insufficient_data",
            "reliable": False,
            "issues": ["No readings available."]
        }

    issues = []

    try:
        timestamps = [
            datetime.fromisoformat(
                r["timestamp"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            for r in readings
        ]
    except (ValueError, TypeError, KeyError):
        return {
            "quality": "invalid",
            "reliable": False,
            "issues": ["Invalid or missing timestamp."]
        }

    # Duplicate timestamps
    if len(timestamps) != len(set(timestamps)):
        issues.append("Duplicate timestamps detected.")

    # The database returns newest readings first.
    ordered = sorted(timestamps)

    # Check gaps between consecutive readings
    gaps = [
        (ordered[i] - ordered[i - 1]).total_seconds() / 60
        for i in range(1, len(ordered))
    ]

    large_gaps = [
        gap for gap in gaps
        if gap > max_gap_minutes
    ]

    if large_gaps:
        issues.append(
            f"{len(large_gaps)} time gap(s) exceed "
            f"{max_gap_minutes} minutes."
        )

    if len(readings) < 3:
        issues.append("Fewer than 3 readings available.")

    return {
        "quality": "warning" if issues else "good",
        "reliable": len(issues) == 0,
        "issues": issues,
        "readings_checked": len(readings),
        "max_gap_minutes": max_gap_minutes
    }

# ---------- Alert Integrity ----------

def validate_alert_quality(alert, quality):
    """
    Prevent trend-based alerts from being treated as reliable
    when timestamp quality checks fail.
    """

    if not quality.get("reliable", False):
        return {
            "severity": "data_warning",
            "message": (
                "Data quality issues detected. "
                "Review missing or inconsistent readings "
                "before interpreting the trend."
            ),
            "original_severity": alert.get("severity"),
            "quality_issues": quality.get("issues", [])
        }

    return alert
