
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
import json


@dataclass
class OCTMeasurement:
    eye: str
    timestamp: str
    rnfl_um: float | None
    quality: float
    device: str = "OCT"


class EyeMonitor:
    def __init__(self, eye):
        self.eye = eye
        self.baseline = None
        self.estimate = None
        self.variance = 25.0
        self.last_time = None
        self.last_valid_time = None
        self.history = []

    def process(self, measurement):
        if measurement.eye != self.eye:
            raise ValueError("Wrong eye")

        timestamp = datetime.fromisoformat(
            measurement.timestamp
        )

        flags = []

        if self.last_time and timestamp <= self.last_time:
            return {
                "eye": self.eye,
                "timestamp": measurement.timestamp,
                "accepted": False,
                "flags": ["OUT_OF_ORDER"]
            }

        if measurement.rnfl_um is None:
            flags.append("MISSING_MEASUREMENT")

        if not 0 <= measurement.quality <= 1:
            flags.append("INVALID_SCAN_QUALITY")
        elif measurement.quality < 0.6:
            flags.append("LOW_SCAN_QUALITY")

        if measurement.rnfl_um is not None:
            if not 40 <= measurement.rnfl_um <= 250:
                flags.append("OUT_OF_RANGE")

        # Predict uncertainty growth with elapsed time.
        if self.last_time is not None:
            elapsed_days = (
                timestamp - self.last_time
            ).total_seconds() / 86400

            self.variance += max(elapsed_days, 0) * 0.5

        if self.last_valid_time is not None:
            gap_days = (
                timestamp - self.last_valid_time
            ).total_seconds() / 86400

            if gap_days > 14:
                flags.append("LONG_VALID_SCAN_GAP")

        invalid_flags = {
            "MISSING_MEASUREMENT",
            "INVALID_SCAN_QUALITY",
            "LOW_SCAN_QUALITY",
            "OUT_OF_RANGE"
        }

        accepted = not any(
            flag in invalid_flags for flag in flags
        )

        self.last_time = timestamp

        if accepted:
            value = float(measurement.rnfl_um)

            if self.estimate is None:
                self.estimate = value
                self.baseline = value

            else:
                measurement_variance = (
                    9.0 / max(measurement.quality, 0.1)
                )

                kalman_gain = (
                    self.variance /
                    (self.variance + measurement_variance)
                )

                self.estimate += kalman_gain * (
                    value - self.estimate
                )

                self.variance *= 1 - kalman_gain

            self.last_valid_time = timestamp

        result = {
            "eye": self.eye,
            "timestamp": measurement.timestamp,
            "accepted": accepted,
            "flags": flags,
            "raw_rnfl_um": measurement.rnfl_um,
            "estimated_rnfl_um": (
                round(self.estimate, 2)
                if self.estimate is not None else None
            ),
            "change_from_baseline_um": (
                round(self.estimate - self.baseline, 2)
                if self.baseline is not None else None
            ),
            "uncertainty_um": round(
                sqrt(self.variance), 2
            )
        }

        self.history.append(result)
        return result


class SANSMonitor:
    def __init__(self):
        self.eyes = {
            "R": EyeMonitor("R"),
            "L": EyeMonitor("L")
        }

    def process(self, measurement):
        if measurement.eye not in self.eyes:
            raise ValueError("Eye must be R or L")

        return self.eyes[measurement.eye].process(
            measurement
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
        measurement = OCTMeasurement(
            eye=eye,
            timestamp=(
                start + timedelta(days=day)
            ).isoformat(),
            rnfl_um=rnfl,
            quality=quality
        )

        result = monitor.process(measurement)
        print(json.dumps(result, indent=2))
