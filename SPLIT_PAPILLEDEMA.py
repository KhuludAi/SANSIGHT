
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "papilledema_inventory.csv"
OUTPUT = ROOT / "papilledema_split.csv"

random.seed(42)

with INPUT.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Remove exact duplicates
unique = {}
conflicts = []

for row in rows:
    digest = row["sha256"]

    if digest in unique:
        if row["class"] != unique[digest]["class"]:
            conflicts.append((row["path"], unique[digest]["path"]))
        continue

    unique[digest] = row

if conflicts:
    print("Cross-class conflicts:", len(conflicts))
    raise SystemExit("Resolve label conflicts before splitting.")

# Group images by patient
groups = defaultdict(list)

for row in unique.values():
    groups[row["patient_group"]].append(row)

output = []

for cls in ["Normal", "Papilledema", "Pseudo papilledema"]:
    group_keys = [
        key for key in groups
        if key.startswith(cls + ":")
    ]

    random.shuffle(group_keys)

    n = len(group_keys)
    n_train = round(n * 0.70)
    n_val = max(1, round(n * 0.15))

    for i, key in enumerate(group_keys):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for row in groups[key]:
            output.append({
                "path": row["path"],
                "class": row["class"],
                "patient_group": row["patient_group"],
                "split": split
            })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "class", "patient_group", "split"]
    )
    writer.writeheader()
    writer.writerows(output)

print("\n=== PATIENT-LEVEL SPLIT ===")

for split in ["train", "val", "test"]:
    subset = [r for r in output if r["split"] == split]

    print(
        split,
        "images:", len(subset),
        "classes:", dict(Counter(r["class"] for r in subset)),
        "patient groups:", len(set(r["patient_group"] for r in subset))
    )

print("\nSaved:", OUTPUT)
