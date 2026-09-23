import gc
import io
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI(title="Fake Currency Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "model"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).resolve().parents[1] / "frontend" / "dist"))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

stage1_model = None
stage2_model = None


def build_model(path: Path) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    state = torch.load(path, map_location=device, mmap=True, weights_only=True)
    model.load_state_dict(state)
    del state
    gc.collect()
    return model.to(device).eval()


@app.on_event("startup")
def load_models():
    global stage1_model, stage2_model
    stage1_model = build_model(MODEL_DIR / "best_model.pth")
    stage2_model = build_model(MODEL_DIR / "best_real_fake_model.pth")


def predict(model: nn.Module, image: Image.Image) -> float:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(tensor).item()


def grade(is_real: bool, conf: float) -> str:
    if is_real:
        return "A" if conf >= 0.95 else "B" if conf >= 0.80 else "C" if conf >= 0.60 else "D"
    return "D" if conf >= 0.95 else "C" if conf >= 0.80 else "B" if conf >= 0.60 else "A"


def risk(is_real: bool, conf: float) -> str:
    if is_real:
        return "LOW" if conf >= 0.90 else "MEDIUM" if conf >= 0.70 else "HIGH"
    return "HIGH" if conf >= 0.90 else "MEDIUM" if conf >= 0.70 else "LOW"


def strength(p: float) -> str:
    distance = abs(p - 0.5) * 100
    return "BORDERLINE" if distance < 10 else "MODERATE" if distance <= 30 else "STRONG"


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    p1 = predict(stage1_model, image)
    is_currency = p1 > 0.5
    stage1 = {
        "is_currency": is_currency,
        "confidence": round((p1 if is_currency else 1 - p1) * 100, 2),
    }
    if not is_currency:
        return {"timestamp": datetime.now().isoformat(), "stage1": stage1, "stage2": None}

    p2 = predict(stage2_model, image)
    is_real = p2 > 0.5
    conf = p2 if is_real else 1 - p2
    return {
        "timestamp": datetime.now().isoformat(),
        "stage1": stage1,
        "stage2": {
            "classification": "REAL" if is_real else "FAKE",
            "confidence": round(conf * 100, 2),
            "percent_real": round(p2 * 100, 2),
            "percent_fake": round((1 - p2) * 100, 2),
            "grade": grade(is_real, conf),
            "risk": risk(is_real, conf),
            "stars": max(1, min(5, round(p2 * 5))),
            "decision_strength": strength(p2),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": stage1_model is not None and stage2_model is not None}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
