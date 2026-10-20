import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any
from PIL import Image

logger = logging.getLogger("clip_gate")

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or ""
CLIP_API_URL = os.environ.get(
    "CLIP_API_URL",
    "https://router.huggingface.co/hf-inference/models/openai/clip-vit-base-patch32",
)
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
STAGE1_MODE = os.environ.get("STAGE1_MODE", "auto").lower()  # "auto", "clip", "resnet"

POSITIVE_LABELS = [
    "an Indian Rupee currency banknote",
    "an Indian paper currency note",
    "a genuine Indian banknote of paper money",
]

NEGATIVE_LABELS = [
    "a photograph of everyday objects, furniture, or animals",
    "a white document, invoice, receipt, or plain paper",
    "a page of a book, newspaper, or identity card",
    "a toy, electronic gadget, or blank background",
]

CANDIDATE_LABELS = POSITIVE_LABELS + NEGATIVE_LABELS


def is_clip_configured() -> bool:
    """Return True if an API token or custom endpoint is configured for CLIP."""
    if STAGE1_MODE == "resnet":
        return False
    return bool(HF_TOKEN) or ("localhost" in CLIP_API_URL or "127.0.0.1" in CLIP_API_URL)


def image_to_base64(image: Image.Image, max_dimension: int = 448) -> str:
    """Resize image if needed and encode as JPEG base64 string."""
    img = image.convert("RGB")
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_clip(image: Image.Image, timeout: float = 4.0) -> dict[str, Any] | None:
    """
    Query OpenAI CLIP zero-shot image classification via Hugging Face Serverless API.
    Returns structured result dict or None on network error/unauthenticated state.
    """
    if not is_clip_configured():
        return None

    try:
        b64_image = image_to_base64(image)
        payload = {
            "image": b64_image,
            "parameters": {
                "candidate_labels": CANDIDATE_LABELS,
            },
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "IndianCurrencyAuthenticator/1.0",
        }
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        req = urllib.request.Request(CLIP_API_URL, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning(f"CLIP endpoint returned HTTP {resp.status}")
                return None
            body = resp.read().decode("utf-8")
            results = json.loads(body)

        if not isinstance(results, list) or len(results) == 0:
            logger.warning(f"Unexpected response format from CLIP: {body[:150]}")
            return None

        # Aggregate positive vs negative scores
        pos_score = sum(item.get("score", 0.0) for item in results if item.get("label") in POSITIVE_LABELS)
        neg_score = sum(item.get("score", 0.0) for item in results if item.get("label") in NEGATIVE_LABELS)
        total = pos_score + neg_score

        if total > 0:
            norm_pos = pos_score / total
        else:
            norm_pos = 0.5

        top_label = results[0].get("label", "Unknown")
        top_score = results[0].get("score", 0.0)

        is_currency = norm_pos >= 0.5
        conf = round((norm_pos if is_currency else 1.0 - norm_pos) * 100, 2)

        return {
            "is_currency": is_currency,
            "confidence": conf,
            "model": CLIP_MODEL_NAME,
            "gate_type": "clip_zero_shot",
            "p_raw": norm_pos,
            "top_label": top_label,
            "top_score": round(top_score, 4),
        }

    except urllib.error.HTTPError as e:
        logger.warning(f"CLIP API HTTP error: {e.code} - {e.reason}")
        return None
    except urllib.error.URLError as e:
        logger.warning(f"CLIP API connection error: {e.reason}")
        return None
    except Exception as e:
        logger.warning(f"CLIP inference exception: {e}")
        return None
