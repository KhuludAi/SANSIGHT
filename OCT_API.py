
from pathlib import Path
from io import BytesIO

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sansight_papilledema_best.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

CLASSES = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

app = FastAPI(title="SANSIGHT OCT API")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "module": "oct_classification",
        "device": str(device)
    }

@app.post("/oct/analyze")
async def analyze_oct(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg", "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB"
        )

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(torch.argmax(probabilities).item())

    return {
        "module": "oct_classification",
        "prediction": CLASSES[index],
        "confidence": round(
            float(probabilities[index].item()), 4
        ),
        "probabilities": {
            name: round(float(probabilities[i].item()), 4)
            for i, name in enumerate(CLASSES)
        },
        "model": "sansight_papilledema_best",
        "disclaimer": (
            "Research prototype. Not a SANS diagnosis "
            "or a clinically validated assessment."
        )
    }
