from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import csv

ROOT = Path(__file__).resolve().parent / "OCT"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

hashes = defaultdict(list)
corrupted = []
total = 0

print("Starting OCT inspection...\n", flush=True)

for split in ["train", "test"]:
    for label in CLASSES:
        folder = ROOT / split / label

        if not folder.is_dir():
            print(f"MISSING FOLDER: {folder}", flush=True)
            continue

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        print(f"{split}/{label}: {len(files)} images", flush=True)

        for path in files:
            try:
                with Image.open(path) as img:
                    img.verify()

                sha = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(
                        lambda: f.read(1024 * 1024), b""
                    ):
                        sha.update(chunk)

                relative = str(path.relative_to(ROOT))
                hashes[sha.hexdigest()].append(
                    (relative, split, label)
                )
                total += 1

            except Exception as e:
                corrupted.append((str(path), str(e)))

duplicates = []
cross_split = []

for group in hashes.values():
    if len(group) > 1:
        original = group[0]
        for duplicate in group[1:]:
            duplicates.append(
                (duplicate[0], original[0])
            )

        splits = {item[1] for item in group}
        if len(splits) > 1:
            cross_split.append(group)

with open("oct_duplicates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["duplicate", "original"])
    writer.writerows(duplicates)

with open("oct_corrupted.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "error"])
    writer.writerows(corrupted)

print("\n========== RESULTS ==========")
print("Valid images:", total)
print("Exact duplicate copies:", len(duplicates))
print("Corrupted images:", len(corrupted))
print("Cross-split duplicate groups:", len(cross_split))
print("Reports saved. No files deleted.")