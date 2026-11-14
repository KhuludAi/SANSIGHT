
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib
import csv

ROOT = (
    Path(__file__).resolve().parent
    / "OCT_papilledema"
    / "Original Data"
    / "OCT Data with Patient ID"
)

CLASSES = ["Normal", "Papilledema", "Pseudo papilledema"]
records = []
issues = []
seen_hashes = {}

for cls in CLASSES:
    for path in (ROOT / cls).rglob("*.jpg"):
        patient_id = path.stem.split("_")[0]
        group_id = f"{cls}:{patient_id}"

        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if digest in seen_hashes:
                issues.append([
                    "exact_duplicate",
                    str(path),
                    seen_hashes[digest]
                ])
            else:
                seen_hashes[digest] = str(path)

            if width < 100 or height < 100:
                issues.append([
                    "small_image", str(path), f"{width}x{height}"
                ])

            records.append([
                str(path), cls, group_id,
                width, height, digest
            ])

        except Exception as e:
            issues.append(["unreadable", str(path), str(e)])

OUT = Path(__file__).resolve().parent

with (OUT / "papilledema_inventory.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "path", "class", "patient_group",
        "width", "height", "sha256"
    ])
    writer.writerows(records)

with (OUT / "papilledema_issues.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["issue", "path", "details"])
    writer.writerows(issues)

print("\n=== SANSIGHT DATA INSPECTION ===")
print("Valid images:", len(records))
print("Images by class:", dict(Counter(r[1] for r in records)))
print("Image dimensions:", Counter(
    (r[3], r[4]) for r in records
).most_common(10))
print("Issues:", dict(Counter(r[0] for r in issues)))
print("Reports saved successfully.")
