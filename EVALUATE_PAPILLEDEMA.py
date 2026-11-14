
from pathlib import Path
import csv

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent
CLASSES = ["Normal", "Papilledema", "Pseudo papilledema"]

class TestDataset(Dataset):
    def __init__(self):
        with (ROOT / "papilledema_split.csv").open(
            encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

        self.rows = [r for r in rows if r["split"] == "test"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        with Image.open(row["path"]) as img:
            image = img.convert("RGB")

        return (
            self.transform(image),
            CLASSES.index(row["class"])
        )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    ROOT / "sansight_papilledema_best.pth",
    map_location=device,
    weights_only=True
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

dataset = TestDataset()
loader = DataLoader(dataset, batch_size=16, shuffle=False)

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

print("\n=== SANSIGHT FINAL TEST ===")
print("Test images:", len(dataset))
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
print("Macro F1:", round(
    f1_score(y_true, y_pred, average="macro", zero_division=0), 4
))

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_true, y_pred,
    labels=list(range(len(CLASSES)))
))
