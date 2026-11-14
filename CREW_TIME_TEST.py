
from datetime import datetime, timedelta
from SANS_MONITOR import SANSMonitor, OCTMeasurement
from ADAPTIVE_SCHEDULER import AdaptiveScheduler

SCAN_MINUTES = 15
AVAILABLE_MINUTES = 15

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
        timestamp=(start + timedelta(days=day)).isoformat(),
        rnfl_um=rnfl,
        quality=quality
    ))

scheduler = AdaptiveScheduler(monitor)
recommendations = scheduler.recommend(
    start + timedelta(days=16)
)

capacity = AVAILABLE_MINUTES // SCAN_MINUTES
selected = recommendations[:capacity]
deferred = recommendations[capacity:]

print("\n=== SANSIGHT CREW-TIME OPTIMIZATION ===")
print("Available crew time:", AVAILABLE_MINUTES, "minutes")
print("Time per OCT scan:", SCAN_MINUTES, "minutes")

print("\nSELECTED SCANS:")
for item in selected:
    print(
        item["eye"],
        "| Priority:", item["priority"],
        "| Reason:", item["reason"]
    )

print("\nDEFERRED SCANS:")
for item in deferred:
    print(
        item["eye"],
        "| Priority:", item["priority"],
        "| Recommended date:", item["recommended_date"]
    )

print("\nCrew time used:", len(selected) * SCAN_MINUTES, "minutes")
