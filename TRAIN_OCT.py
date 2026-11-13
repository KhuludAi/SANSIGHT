
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "OCT"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
LABELS = {name: i for i, name in enumerate(CLASSES)}

BATCH_SIZE = 8
EPOCHS = 3
LR = 0.0001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Load prepared split
df = pd.read_csv(ROOT / "oct_train_val_split.csv")

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class OCTDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = DATA / row["path"]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        label = LABELS[row["label"]]

        return image, label

train_ds = OCTDataset(train_df, train_transform)
val_ds = OCTDataset(val_df, val_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

# Class weights
counts = Counter(train_df["label"])

weights = torch.tensor([
    len(train_df) / (len(CLASSES) * counts[c])
    for c in CLASSES
], dtype=torch.float32).to(device)

print("Class weights:", weights)

# Pretrained ResNet18
model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0001
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)

best_f1 = -1.0

# Training
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Step {step+1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    # Validation
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)

            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    accuracy = accuracy_score(targets, predictions)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro"
    )

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {total_loss/len(train_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if macro_f1 > best_f1:
        best_f1 = macro_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "best_macro_f1": best_f1,
            "epoch": epoch + 1
        }, ROOT / "sansight_oct_best.pth")

        print("Best model saved!")

print("\nTraining finished.")
print("Best Macro F1:", best_f1)
