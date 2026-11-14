
import random
import statistics
from datetime import datetime, timedelta

from SANS_MONITOR import SANSMonitor, OCTMeasurement
from ADAPTIVE_SCHEDULER import AdaptiveScheduler

MISSIONS = 100
DAYS = 32
SCAN_BUDGET = 8
CHANGE_DAY = 10
DETECTION_THRESHOLD = 5.0
SEED = 42

rng = random.Random(SEED)
START = datetime(2026, 11, 1, 10, 0)

SCAN_DAYS = [0, 0, 4, 8, 12, 16, 20, 24]


def simulate_mission():
    affected_eye = rng.choice(["R", "L"])
    progression = rng.uniform(0.8, 1.5)

    baseline = {"R": 98.0, "L": 96.0}

    observations = {}

    for day in range(DAYS):
        for eye in ["R", "L"]:
            true_value = baseline[eye]

            if eye == affected_eye and day >= CHANGE_DAY:
                true_value += progression * (
                    day - CHANGE_DAY
                )

            quality = rng.uniform(0.5, 1.0)
            missing = rng.random() < 0.08

            observations[(eye, day)] = {
                "value": (
                    None if missing else
                    true_value + rng.gauss(0, 1.0)
                ),
                "quality": quality
            }

    def evaluate(policy):
        monitor = SANSMonitor()
        scheduler = AdaptiveScheduler(monitor)

        detection_day = None
        schedule = []

        for index, day in enumerate(SCAN_DAYS):

            if index == 0:
                eye = "R"

            elif index == 1:
                eye = "L"

            elif policy == "fixed":
                # Alternate eyes after baseline scans.
                eye = (
                    "R" if (index - 2) % 2 == 0
                    else "L"
                )

            else:
                recommendations = scheduler.recommend(
                    START + timedelta(days=day)
                )
                eye = recommendations[0]["eye"]

            obs = observations[(eye, day)]

            result = monitor.process(OCTMeasurement(
                eye=eye,
                timestamp=(
                    START + timedelta(days=day)
                ).isoformat(),
                rnfl_um=obs["value"],
                quality=obs["quality"]
            ))

            schedule.append((eye, day))

            if (
                detection_day is None
                and eye == affected_eye
                and result["accepted"]
                and result["change_from_baseline_um"] is not None
                and result["change_from_baseline_um"]
                >= DETECTION_THRESHOLD
            ):
                detection_day = day

        return {
            "detection_day": detection_day,
            "scans": len(schedule),
            "schedule": schedule
        }

    return evaluate("fixed"), evaluate("adaptive")


results = [
    simulate_mission()
    for _ in range(MISSIONS)
]

fixed_days = [
    fixed["detection_day"]
    for fixed, adaptive in results
    if fixed["detection_day"] is not None
]

adaptive_days = [
    adaptive["detection_day"]
    for fixed, adaptive in results
    if adaptive["detection_day"] is not None
]

both = [
    (fixed["detection_day"], adaptive["detection_day"])
    for fixed, adaptive in results
    if fixed["detection_day"] is not None
    and adaptive["detection_day"] is not None
]

print("\n=== SANSIGHT REAL ENGINE COMPARISON ===")
print("Simulated missions:", MISSIONS)
print("Scan budget:", SCAN_BUDGET)
print("Missing data probability: 8%")
print("Scan quality range: 0.5 to 1.0")

print("\nFIXED SCHEDULING")
print(
    "Detection rate:",
    round(len(fixed_days) / MISSIONS * 100, 2),
    "%"
)

print("\nADAPTIVE SCHEDULING")
print(
    "Detection rate:",
    round(len(adaptive_days) / MISSIONS * 100, 2),
    "%"
)

print("\nMISSIONS DETECTED BY BOTH:", len(both))

if both:
    advantages = [
        fixed_day - adaptive_day
        for fixed_day, adaptive_day in both
    ]

    print(
        "Mean advantage (days):",
        round(statistics.mean(advantages), 2)
    )

    print(
        "Adaptive earlier:",
        sum(value > 0 for value in advantages)
    )

    print(
        "Fixed earlier:",
        sum(value < 0 for value in advantages)
    )

    print(
        "Same detection day:",
        sum(value == 0 for value in advantages)
    )

print(
    "\nEqual scan budgets confirmed:",
    all(
        fixed["scans"] == adaptive["scans"] == SCAN_BUDGET
        for fixed, adaptive in results
    )
)
