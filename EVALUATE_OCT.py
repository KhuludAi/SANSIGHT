
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "OCT"
MODEL_PATH = ROOT / "sansight_oct_best.pth"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
LABELS = {name: i for i, name in enumerate(CLASSES)}

BATCH_SIZE = 8

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Test transformations: same normalization as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class OCTTestDataset(Dataset):
    def __init__(self):
        self.samples = []

        for label in CLASSES:
            folder = DATA / "test" / label

            if not folder.exists():
                raise FileNotFoundError(folder)

            files = [
                p for p in folder.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

            for path in files:
                self.samples.append((path, LABELS[label]))

        if not self.samples:
            raise RuntimeError("No test images found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            image = img.convert("RGB")
            image = transform(image)

        return image, label

# Load untouched test set
test_ds = OCTTestDataset()

print("Test images:", len(test_ds))

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# Rebuild model
model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])

# Evaluate
predictions = []
targets = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)

        preds = outputs.argmax(dim=1)

        predictions.extend(preds.cpu().tolist())
        targets.extend(labels.tolist())

# Metrics
accuracy = accuracy_score(targets, predictions)
macro_f1 = f1_score(
    targets, predictions, average="macro"
)

print("\n========== TEST RESULTS ==========")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")

report = classification_report(
    targets,
    predictions,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Confusion matrix
cm = confusion_matrix(
    targets,
    predictions,
    labels=list(range(len(CLASSES)))
)

print("CONFUSION MATRIX")
print(cm)

# Save results
(ROOT / "results").mkdir(exist_ok=True)

with open(
    ROOT / "results" / "test_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
).to_csv(
    ROOT / "results" / "confusion_matrix.csv"
)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES, rotation=30)
ax.set_yticklabels(CLASSES)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("SANSIGHT OCT - Test Confusion Matrix")

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(
    ROOT / "results" / "confusion_matrix.png",
    dpi=200
)
plt.close(fig)

print("\nEvaluation completed.")
print("Results saved in:", ROOT / "results")
