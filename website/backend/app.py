import csv
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
import memory
from explain import _backbone_features, classify_with_cam, edge_share
from imaging import MAX_UPLOAD_BYTES, ImageRejected, load_image
from quality import assess_quality
from ratelimit import SlidingWindow

app = FastAPI(title="Indian Currency Authenticator API")
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
    # Build on the meta device and attach the memory-mapped weights directly, so the
    # weights live in reclaimable file-backed pages instead of a second private copy.
    with torch.device("meta"):
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, assign=True)
    return model.to(device).eval()


@app.on_event("startup")
def load_models():
    global stage1_model, stage2_model
    stage1_model = build_model(MODEL_DIR / "best_model.pth")
    stage2_model = build_model(MODEL_DIR / "best_real_fake_model.pth")


RECENT_SCANS: dict[str, dict] = {}


def predict_with_embedding(model: nn.Module, image: Image.Image) -> tuple[float, np.ndarray]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = _backbone_features(model, tensor)
        pooled = torch.flatten(model.avgpool(features), 1)
        prob = model.fc(pooled).item()
        norm = torch.norm(pooled, dim=1, keepdim=True)
        emb = (pooled / (norm + 1e-7))[0].cpu().numpy()
        return prob, emb


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


def analyze(image: Image.Image, original_size: tuple[int, int] | None = None, explain: bool = False) -> dict:
    note_token = secrets.token_hex(8)
    quality = assess_quality(image, original_size)
    p1, emb1 = predict_with_embedding(stage1_model, image)

    # Check Stage 1 memory for prior corrections
    mem1 = memory.lookup_memory(emb1, stage=1)
    if mem1:
        if mem1["label"] == "CURRENCY":
            p1 = 0.99 if mem1["action"] == "override" else (p1 * 0.4 + 0.6)
        elif mem1["label"] == "NOT_NOTE":
            p1 = 0.01 if mem1["action"] == "override" else (p1 * 0.4)

    is_currency = p1 > 0.5
    stage1 = {
        "is_currency": is_currency,
        "confidence": round((p1 if is_currency else 1 - p1) * 100, 2),
    }
    if mem1:
        stage1["memory_match"] = mem1

    if not is_currency:
        RECENT_SCANS[note_token] = {"emb1": emb1, "emb2": None}
        if len(RECENT_SCANS) > 100:
            RECENT_SCANS.pop(next(iter(RECENT_SCANS)))
        return {
            "note_token": note_token,
            "timestamp": datetime.now().isoformat(),
            "quality": quality,
            "stage1": stage1,
            "stage2": None,
        }

    p2, cam, emb2 = classify_with_cam(stage2_model, transform(image).unsqueeze(0).to(device))

    # Check Stage 2 memory for prior corrections
    mem2 = memory.lookup_memory(emb2, stage=2)
    if mem2:
        if mem2["label"] == "REAL":
            p2 = 0.98 if mem2["action"] == "override" else (p2 * 0.4 + 0.58)
        elif mem2["label"] == "FAKE":
            p2 = 0.02 if mem2["action"] == "override" else (p2 * 0.4 + 0.01)

    is_real = p2 > 0.5
    conf = p2 if is_real else 1 - p2
    stage2_dict = {
        "classification": "REAL" if is_real else "FAKE",
        "confidence": round(conf * 100, 2),
        "percent_real": round(p2 * 100, 2),
        "percent_fake": round((1 - p2) * 100, 2),
        "grade": grade(is_real, conf),
        "risk": risk(is_real, conf),
        "stars": max(1, min(5, round(p2 * 5))),
        "decision_strength": strength(p2),
    }
    if mem2:
        stage2_dict["memory_match"] = mem2

    result = {
        "note_token": note_token,
        "timestamp": datetime.now().isoformat(),
        "quality": quality,
        "stage1": stage1,
        "stage2": stage2_dict,
    }
    if explain:
        result["explanation"] = {
            "target": "REAL" if is_real else "FAKE",
            "grid": np.round(cam, 3).tolist(),
            "edge_share": round(edge_share(cam), 3),
        }

    RECENT_SCANS[note_token] = {"emb1": emb1, "emb2": emb2}
    if len(RECENT_SCANS) > 100:
        RECENT_SCANS.pop(next(iter(RECENT_SCANS)))
    return result


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    try:
        image, original_size = load_image(await file.read(MAX_UPLOAD_BYTES + 1))
    except ImageRejected as e:
        raise HTTPException(status_code=400, detail=str(e))
    return analyze(image, original_size, explain=True)


MAX_BATCH = 20


@app.post("/api/batch")
async def batch(files: list[UploadFile] = File(...)):
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH} images per batch")

    results = []
    summary = {"total": len(files), "real": 0, "fake": 0, "not_currency": 0, "errors": 0}
    for f in files:
        try:
            image, original_size = load_image(await f.read(MAX_UPLOAD_BYTES + 1))
        except ImageRejected as e:
            summary["errors"] += 1
            results.append({"filename": f.filename, "error": str(e)})
            continue

        analysis = await run_in_threadpool(analyze, image, original_size)
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
    note_token: str | None = None


@app.post("/api/feedback")
def submit_feedback(body: FeedbackIn):
    try:
        feedback_id = feedback.add_feedback(
            predicted=body.predicted,
            actual=body.actual,
            confidence=body.confidence,
            quality=body.quality,
            comment=body.comment,
        )
    except feedback.FeedbackFull:
        raise HTTPException(status_code=503, detail="Feedback storage is full")

    # If this feedback is a correction, store the image embedding in memory!
    memory_registered = False
    if body.note_token and body.note_token in RECENT_SCANS and body.predicted != body.actual:
        scan_data = RECENT_SCANS[body.note_token]
        comment_text = body.comment or f"User correction ({body.predicted} -> {body.actual})"
        if body.actual == "NOT_NOTE" and scan_data.get("emb1") is not None:
            memory.add_memory(scan_data["emb1"], stage=1, label="NOT_NOTE", comment=comment_text)
            memory_registered = True
        elif body.predicted == "NOT_NOTE" and body.actual in ("REAL", "FAKE") and scan_data.get("emb1") is not None:
            memory.add_memory(scan_data["emb1"], stage=1, label="CURRENCY", comment=comment_text)
            if scan_data.get("emb2") is not None:
                memory.add_memory(scan_data["emb2"], stage=2, label=body.actual, comment=comment_text)
            memory_registered = True
        elif body.actual in ("REAL", "FAKE") and scan_data.get("emb2") is not None:
            memory.add_memory(scan_data["emb2"], stage=2, label=body.actual, comment=comment_text)
            memory_registered = True

    return {"id": feedback_id, "memory_registered": memory_registered}


@app.get("/api/memory/list")
def list_memories(x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return {"memories": memory.list_memories()}


class MemoryAction(BaseModel):
    id: int
    approved: bool = True


@app.post("/api/memory/approve")
def approve_memory(body: MemoryAction, x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    ok = memory.review_memory(body.id, body.approved)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "ok", "id": body.id, "approved": body.approved}


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
    focus: Literal[
        "upper left", "upper centre", "upper right",
        "middle left", "centre", "middle right",
        "lower left", "lower centre", "lower right",
        "spread across the photo",
    ] | None = None


class ChatIn(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    scan: ScanContext | None = None


def visitor_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
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
