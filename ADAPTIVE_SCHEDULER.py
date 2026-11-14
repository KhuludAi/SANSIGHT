
from datetime import datetime, timedelta
import json

from SANS_MONITOR import SANSMonitor, OCTMeasurement


class AdaptiveScheduler:
    def __init__(self, monitor):
        self.monitor = monitor

    def recommend(self, current_time):
        recommendations = []

        for eye, state in self.monitor.eyes.items():

            if state.estimate is None:
                recommendations.append({
                    "eye": eye,
                    "priority": "HIGH",
                    "next_scan_days": 0,
                    "recommended_date": (
                        current_time.date().isoformat()
                    ),
                    "change_from_baseline_um": None,
                    "uncertainty_um": round(
                        state.variance ** 0.5, 2
                    ),
                    "days_since_valid_scan": None,
                    "reason": "No valid baseline measurement"
                })
                continue

            change = abs(
                state.estimate - state.baseline
            )

            uncertainty = state.variance ** 0.5

            if state.last_valid_time:
                days_since_valid = max(
                    0,
                    (
                        current_time - state.last_valid_time
                    ).total_seconds() / 86400
                )
            else:
                days_since_valid = float("inf")

            recent_flags = (
                state.history[-1]["flags"]
                if state.history else []
            )

            if "LOW_SCAN_QUALITY" in recent_flags:
                priority = "HIGH"
                days = 0
                reason = "Repeat low-quality OCT scan"

            elif days_since_valid >= 10:
                priority = "HIGH"
                days = 0
                reason = "Long gap since last valid OCT scan"

            elif change >= 8:
                priority = "HIGH"
                days = 1
                reason = "Increasing change from baseline"

            elif uncertainty >= 4:
                priority = "HIGH"
                days = 1
                reason = "High measurement uncertainty"

            elif change >= 4 or uncertainty >= 3:
                priority = "MEDIUM"
                days = 3
                reason = "Moderate change or uncertainty"

            else:
                priority = "LOW"
                days = 7
                reason = "Stable estimated measurements"

            recommendations.append({
                "eye": eye,
                "priority": priority,
                "next_scan_days": days,
                "recommended_date": (
                    current_time + timedelta(days=days)
                ).date().isoformat(),
                "change_from_baseline_um": round(
                    change, 2
                ),
                "uncertainty_um": round(
                    uncertainty, 2
                ),
                "days_since_valid_scan": round(
                    days_since_valid, 2
                ),
                "reason": reason
            })

        return sorted(
            recommendations,
            key=lambda item: (
                {
                    "HIGH": 0,
                    "MEDIUM": 1,
                    "LOW": 2
                }[item["priority"]],
                -item["uncertainty_um"],
                -abs(
                    item["change_from_baseline_um"] or 0
                ),
                -(
                    item["days_since_valid_scan"]
                    if item["days_since_valid_scan"] is not None
                    else 999
                )
            )
        )


if __name__ == "__main__":
    monitor = SANSMonitor()
    start = datetime(2026, 11, 1, 10, 0)

    demo = [
        ("R", 0, 98, 0.95),
        ("L", 0, 96, 0.94),
        ("R", 4, 101, 0.92),
        ("L", 4, 97, 0.91),
        ("R", 8, 106, 0.90),
        ("L", 8, None, 0.90),
        ("R", 12, 110, 0.45),
        ("L", 12, 99, 0.93),
        ("R", 16, 113, 0.94)
    ]

    for eye, day, rnfl, quality in demo:
        monitor.process(OCTMeasurement(
            eye=eye,
            timestamp=(
                start + timedelta(days=day)
            ).isoformat(),
            rnfl_um=rnfl,
            quality=quality
        ))

    scheduler = AdaptiveScheduler(monitor)

    recommendations = scheduler.recommend(
        start + timedelta(days=16)
    )

    print("\n=== SANSIGHT ADAPTIVE SCHEDULING ===")

    for recommendation in recommendations:
        print(json.dumps(recommendation, indent=2))
