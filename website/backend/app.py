import csv
import gc
import io
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import chat
import feedback
from quality import assess_quality
from ratelimit import SlidingWindow

app = FastAPI(title="Fake Currency Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ["MODEL_DIR"]) if "MODEL_DIR" in os.environ else HERE.parent.parent / "model"
FRONTEND_DIR = Path(os.environ["FRONTEND_DIR"]) if "FRONTEND_DIR" in os.environ else HERE.parent / "frontend" / "dist"

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


def analyze(image: Image.Image) -> dict:
    quality = assess_quality(image)
    p1 = predict(stage1_model, image)
    is_currency = p1 > 0.5
    stage1 = {
        "is_currency": is_currency,
        "confidence": round((p1 if is_currency else 1 - p1) * 100, 2),
    }
    if not is_currency:
        return {"timestamp": datetime.now().isoformat(), "quality": quality, "stage1": stage1, "stage2": None}

    p2 = predict(stage2_model, image)
    is_real = p2 > 0.5
    conf = p2 if is_real else 1 - p2
    return {
        "timestamp": datetime.now().isoformat(),
        "quality": quality,
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


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return analyze(image)


MAX_BATCH = 20


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH} images per batch")

    results = []
    summary = {"total": len(files), "real": 0, "fake": 0, "not_currency": 0, "errors": 0}
    for f in files:
        try:
            image = Image.open(io.BytesIO(await f.read())).convert("RGB")
        except Exception:
            summary["errors"] += 1
            results.append({"filename": f.filename, "error": "Invalid image file"})
            continue

        analysis = await run_in_threadpool(analyze, image)
        if analysis["stage2"] is None:
            summary["not_currency"] += 1
        elif analysis["stage2"]["classification"] == "REAL":
            summary["real"] += 1
        else:
            summary["fake"] += 1
        results.append({"filename": f.filename, **analysis})

    return {"summary": summary, "results": results}


class FeedbackIn(BaseModel):
    predicted: Literal["REAL", "FAKE", "NOT_NOTE"]
    actual: Literal["REAL", "FAKE", "NOT_NOTE"]
    confidence: float | None = Field(default=None, ge=0, le=100)
    quality: Literal["GOOD", "FAIR", "POOR"] | None = None
    comment: str = Field(default="", max_length=300)


@app.post("/api/feedback")
def submit_feedback(body: FeedbackIn):
    try:
        feedback_id = feedback.add_feedback(**body.model_dump())
    except feedback.FeedbackFull:
        raise HTTPException(status_code=503, detail="Feedback storage is full")
    return {"id": feedback_id}


@app.get("/api/feedback/export")
def export_feedback(format: Literal["json", "csv"] = "json", x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")

    rows = feedback.list_feedback()
    if format == "json":
        return {"summary": feedback.summary(), "rows": rows}

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["id", "created_at", "predicted", "actual", "confidence", "quality", "comment"])
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=feedback.csv"},
    )


chat_per_visitor = SlidingWindow(int(os.environ.get("CHAT_PER_MINUTE", "6")), 60)
chat_site_wide = SlidingWindow(int(os.environ.get("CHAT_PER_HOUR", "300")), 3600)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=chat.MAX_MESSAGE_CHARS)


class ScanContext(BaseModel):
    verdict: Literal["REAL", "FAKE", "NOT_NOTE"]
    confidence: float = Field(ge=0, le=100)
    grade: Literal["A", "B", "C", "D"] | None = None
    risk: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    decision_strength: Literal["BORDERLINE", "MODERATE", "STRONG"] | None = None
    quality: Literal["GOOD", "FAIR", "POOR"] | None = None


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    scan: ScanContext | None = None


def visitor_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.get("/api/chat/status")
def chat_status():
    return {"enabled": chat.enabled()}


@app.post("/api/chat")
def chat_endpoint(body: ChatIn, request: Request):
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=422, detail="Last message must be from the user")
    if not chat.enabled():
        raise HTTPException(status_code=503, detail="Chat is not available right now")
    if not chat_per_visitor.allow(visitor_key(request)) or not chat_site_wide.allow("site"):
        raise HTTPException(status_code=429, detail="Too many messages. Please wait a minute.", headers={"Retry-After": "60"})

    try:
        reply = chat.ask(
            [m.model_dump() for m in body.messages],
            body.scan.model_dump() if body.scan else None,
        )
    except chat.ChatRateLimited:
        raise HTTPException(status_code=429, detail="The assistant is busy. Please try again shortly.", headers={"Retry-After": "30"})
    except chat.ChatDisabled:
        raise HTTPException(status_code=503, detail="Chat is not available right now")
    except chat.ChatUnavailable:
        raise HTTPException(status_code=502, detail="The assistant could not answer. Please try again.")
    return {"reply": reply}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": stage1_model is not None and stage2_model is not None}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
