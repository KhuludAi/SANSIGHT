
import random
import statistics

SEED = 42
MISSIONS = 100
DAYS = 30
SCAN_BUDGET = 8
CHANGE_DAY = 10
DETECTION_THRESHOLD = 5.0

rng = random.Random(SEED)


def simulate_mission():
    # One eye develops a gradual change.
    affected_eye = rng.choice(["R", "L"])

    baseline = {"R": 98.0, "L": 96.0}
    progression = rng.uniform(0.8, 1.5)

    def true_value(eye, day):
        value = baseline[eye]

        if eye == affected_eye and day >= CHANGE_DAY:
            value += progression * (day - CHANGE_DAY)

        return value

    # Same underlying mission for both policies.
    noise = {
        (eye, day): rng.gauss(0, 1.0)
        for eye in ["R", "L"]
        for day in range(DAYS)
    }

    def observe(eye, day):
        return true_value(eye, day) + noise[(eye, day)]

    def run_fixed():
        # Eight total scans: alternate eyes every 4 days.
        schedule = [
            ("R" if i % 2 == 0 else "L", i * 4)
            for i in range(SCAN_BUDGET)
        ]
        return evaluate(schedule)

    def run_adaptive():
        # Demonstration policy: prioritize the eye
        # with the largest observed change.
        schedule = []
        last_change = {"R": 0.0, "L": 0.0}
        last_scanned = {"R": -100, "L": -100}

        # Two baseline scans.
        schedule.extend([("R", 0), ("L", 0)])
        last_scanned["R"] = 0
        last_scanned["L"] = 0

        for i in range(2, SCAN_BUDGET):
            day = i * 4

            scores = {
                eye: (
                    abs(last_change[eye])
                    + 0.3 * (day - last_scanned[eye])
                )
                for eye in ["R", "L"]
            }

            eye = max(scores, key=scores.get)
            schedule.append((eye, day))

            measurement = observe(eye, day)
            last_change[eye] = measurement - baseline[eye]
            last_scanned[eye] = day

        return evaluate(schedule)

    def evaluate(schedule):
        detection_day = None

        for eye, day in schedule:
            measurement = observe(eye, day)

            if (
                eye == affected_eye
                and day >= CHANGE_DAY
                and measurement - baseline[eye]
                >= DETECTION_THRESHOLD
            ):
                detection_day = day
                break

        return detection_day, schedule

    fixed_day, fixed_schedule = run_fixed()
    adaptive_day, adaptive_schedule = run_adaptive()

    return fixed_day, adaptive_day


results = [simulate_mission() for _ in range(MISSIONS)]

fixed_detected = [f for f, a in results if f is not None]
adaptive_detected = [a for f, a in results if a is not None]

print("\n=== SANSIGHT SCHEDULING COMPARISON ===")
print("Simulated missions:", MISSIONS)
print("Scan budget per mission:", SCAN_BUDGET)

print("\nFIXED SCHEDULING")
print("Detection rate:", round(
    100 * len(fixed_detected) / MISSIONS, 2
), "%")

print("\nADAPTIVE SCHEDULING")
print("Detection rate:", round(
    100 * len(adaptive_detected) / MISSIONS, 2
), "%")

both = [
    (f, a)
    for f, a in results
    if f is not None and a is not None
]

if both:
    delays = [f - a for f, a in both]
    print("\nMISSIONS DETECTED BY BOTH:", len(both))
    print(
        "Mean detection-time advantage (days):",
        round(statistics.mean(delays), 2)
    )

print("\nPositive advantage means adaptive detected earlier.")
print("Negative advantage means fixed detected earlier.")
