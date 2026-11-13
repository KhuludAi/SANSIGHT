import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "OCT"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

with open(ROOT / "oct_duplicates.csv",
          newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        a = row["duplicate"].replace("\\", "/")
        b = row["original"].replace("\\", "/")
        union(a, b)

groups = defaultdict(list)

for path in DATA.glob("train/*/*"):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    rel = path.relative_to(DATA).as_posix()
    groups[find(rel)].append(rel)

selected = []
conflicts = []
excluded = 0

for files in groups.values():
    labels = {Path(p).parent.name for p in files}

    if len(labels) > 1:
        conflicts.extend(files)
        excluded += len(files)
    else:
        selected.append(sorted(files)[0])
        excluded += len(files) - 1

with open(ROOT / "oct_train_clean.csv", "w",
          newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    for p in sorted(selected):
        writer.writerow([p, Path(p).parent.name])

with open(ROOT / "oct_conflicts.csv", "w",
          newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path"])
    for p in sorted(conflicts):
        writer.writerow([p])

print("Training images:", len(selected))
print("Excluded images:", excluded)
print("Conflicting images:", len(conflicts))
print("Saved: oct_train_clean.csv")
print("Saved: oct_conflicts.csv")