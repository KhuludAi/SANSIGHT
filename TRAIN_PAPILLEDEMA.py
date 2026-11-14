
from pathlib import Path
import csv
import random
import copy

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "papilledema_split.csv"

CLASSES = ["Normal", "Papilledema", "Pseudo papilledema"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

SEED = 42
EPOCHS = 15
BATCH_SIZE = 16
LR = 0.0001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class OCTDataset(Dataset):
    def __init__(self, split, transform):
        with CSV_PATH.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.rows = [r for r in rows if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        with Image.open(row["path"]) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        label = CLASS_TO_IDX[row["class"]]
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_ds = OCTDataset("train", train_transform)
val_ds = OCTDataset("val", eval_transform)

train_labels = [CLASS_TO_IDX[r["class"]] for r in train_ds.rows]
counts = np.bincount(train_labels, minlength=len(CLASSES))

weights = [1.0 / counts[label] for label in train_labels]

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    sampler=sampler, num_workers=0
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_f1 = -1
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    macro_f1 = f1_score(
        y_true, y_pred,
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_ds):.4f} | "
        f"Val Macro F1: {macro_f1:.4f}",
        flush=True
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": CLASSES,
            "best_val_macro_f1": best_f1,
            "epoch": epoch + 1,
        }, ROOT / "sansight_papilledema_best.pth")

print("\nBest validation Macro F1:", round(best_f1, 4))
print("Model saved: sansight_papilledema_best.pth")
