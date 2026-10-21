# Implementation Specification: Indian Currency Authenticator

Comprehensive technical architecture, machine learning pipeline, backend/frontend implementation details, and deployment specification for the **Indian Currency Authenticator** system.

---

## 1. Executive Summary & System Overview

The **Indian Currency Authenticator** is an end-to-end, multi-stage machine learning system designed to inspect, detect, and authenticate Indian paper currency (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 denominations). 

The platform pairs an advanced multi-stage computer vision inference pipeline with an interactive ATM/CRT-themed web application, an automated image quality screener, real-time explainable AI (Grad-CAM heatmaps), a zero-retraining **Feature Embedding Memory Layer**, an **OpenAI CLIP Foundation Gate**, and a low-latency LLM Help Desk powered by Google Gemini.

### Key Capabilities
- **Hybrid Stage 1 Currency Gate:** Rejects non-banknote images before Stage 2 using either an open-vocabulary **OpenAI CLIP (`ViT-B/32`)** zero-shot vision-language foundation gate or a supervised **ResNet-50** binary classifier with automatic seamless failover.
- **Stage 2 Authenticity Classification:** Distinguishes genuine from counterfeit currency across all modern Mahatma Gandhi series denominations using a fine-tuned ResNet-50 deep CNN.
- **Explainable AI (XAI):** Generates zero-overhead $7 \times 7$ Grad-CAM saliency heatmaps highlighting regions pushing the model toward genuine or counterfeit verdicts, complete with edge-share warning heuristics.
- **Feature Embedding Memory Layer (Zero-Retraining Learning):** Extracts 2048-dimensional normalized visual embeddings from the neural backbone. When users report misclassifications, the note's visual signature is stored in an in-memory vector cache; subsequent scans of that note apply instant memory overrides without requiring expensive model retraining in Colab.
- **Deterministic Heuristic Layer (Stage 3):** Synthesizes raw network probabilities into human-readable metrics: Authenticity Grade (A–D), Risk Level (Low/Med/High), Star Rating (1–5), and Decision Strength.
- **Image Quality Pre-screener:** Evaluates blur (Laplacian variance), glare, underexposure/overexposure, contrast, and resolution before user misinterpretation.
- **512 MB RAM Optimization:** Operates reliably within Render's free tier (512 MB RAM ceiling) using memory-mapped weights (`mmap=True`), libjpeg draft decoding, glibc arena limits, and serverless CLIP inference (0 MB added server RAM).
- **Sub-Second LLM Help Desk:** Integrated conversational assistant using `gemini-2.5-flash` with zero-thinking latency optimization (`thinking_budget=0`) and dynamic failover to `gemini-2.5-flash-lite`, passing scan diagnostics with anti-hallucination guardrails.

---

## 2. End-to-End System Architecture

```
                    ┌─────────────────────────┐
                    │   Input Image Upload    │
                    │ (Single / Batch / Cam)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Image Ingestion & Qual. │
                    │ - Bounded JPEG Draft    │
                    │ - EXIF Transpose        │
                    │ - Laplacian Blur / Glare│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Feature Extraction     │
                    │  2048-D Embedding (emb1)│
                    └────────────┬────────────┘
                                 │
                     Prior Correction in Memory?
                   ┌─────────────┴─────────────┐
                   │                           │
             YES (sim ≥ 0.92)             NO (No match)
                   │                           │
                   ▼                           ▼
        ┌────────────────────┐      ┌─────────────────────────┐
        │  Memory Override   │      │         STAGE 1         │
        │ Instant Correction │      │ OpenAI CLIP (ViT-B/32)  │
        └──────────┬─────────┘      │ Fallback: ResNet-50     │
                   │                └────────────┬────────────┘
                   │                             │
                   │               Is Banknote?  │ p1 > 0.5
                   │             ┌───────────────┴───────────────┐
                   │             │                               │
                   │       NO (p1 ≤ 0.5)                   YES (p1 > 0.5)
                   │             │                               │
                   │             ▼                               ▼
                   │  ┌────────────────────┐          ┌────────────────────┐
                   │  │   NOT A CURRENCY   │          │      STAGE 2       │
                   │  │ Confidence: 1 - p1 │          │ Real vs Fake Note  │
                   │  │ Quality Diagnostics│          │ (ResNet-50 Model)  │
                   │  └──────────┬─────────┘          └──────────┬─────────┘
                   │             │                               │
                   └─────────────┼───────────────────────────────┤
                                 │                               │
                                 │                      Memory Check (emb2)?
                                 │                    ┌──────────┴──────────┐
                                 │                    │                     │
                                 │              YES (sim ≥ 0.92)       NO (No match)
                                 │                    │                     │
                                 │                    ▼                     ▼
                                 │         ┌────────────────────┐ ┌────────────────────┐
                                 │         │  Memory Override   │ │ ResNet-50 + GradCAM│
                                 │         │ Instant Real/Fake  │ │ 7x7 Attention Grid │
                                 │         └──────────┬─────────┘ └──────────┬─────────┘
                                 │                    │                      │
                                 │                    └───────────┬──────────┘
                                 │                                │
                                 │                                ▼
                                 │                    ┌───────────────────────┐
                                 │                    │        STAGE 3        │
                                 │                    │    Result Synthesis   │
                                 │                    │ - Grade (A, B, C, D)  │
                                 │                    │ - Risk (Low/Med/High) │
                                 │                    │ - Rating (1 - 5 ★)    │
                                 │                    │ - Decision Strength   │
                                 │                    └───────────┬───────────┘
                                 │                                │
            ┌────────────────────┴────────────────────────────────┴────────────────────┐
            ▼                                                                          ▼
┌───────────────────────┐                                                  ┌───────────────────────┐
│   Frontend ATM View   │                                                  │    Gemini Help Desk   │
│ - CRT/ATM UI Displays │ ◄── [User Correction Feedback] ──┐               │ Contextual Guidance   │
│ - Heatmap Canvas      │                                  │               │ Security Feature Info │
│ - Gate Badge (CLIP)   │                                  ▼               │ Sub-second Flash 2.5  │
│ - Memory Glow Badge   │                     ┌─────────────────────────┐  └───────────────────────┘
└───────────────────────┘                     │ Feature Embedding Store │
                                              │ (SQLite Vector Memory)  │
                                              │ Overrides future scans  │
                                              └─────────────────────────┘
```

---

## 3. Deep Learning Pipeline & Model Specifications

### 3.1 ResNet-50 Backbone Models (Stage 1 & Stage 2)

Both Stage 1 and Stage 2 utilize a **ResNet-50** deep convolutional backbone (pretrained on ImageNet-1K) with identical custom binary classification heads:

- **Backbone:** ResNet-50 (`torchvision.models.resnet50(weights=None)`)
- **Backbone Feature Output:** 2,048 feature maps ($7 \times 7$ spatial resolution after `layer4` conv blocks and before average pooling).
- **Custom Head Architecture:**
  ```python
  nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=2048, out_features=256),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(in_features=256, out_features=1),
      nn.Sigmoid()
  )
  ```
- **Total Parameters:** Approximately 24.03 Million parameters per model (~92 MB stored weight files).
- **Input Dimensions:** $3 \times 224 \times 224$ pixels.
- **Normalization:** ImageNet distribution ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 3.2 Label Conventions & Standard Logic

> [!IMPORTANT]
> **Label Orientation:** For both stages, probability $> 0.5$ designates the target positive class:
> - **Stage 1 (Currency Detection):** $\text{prob} > 0.5 \implies$ **CURRENCY**; $\le 0.5 \implies$ **NON-CURRENCY**.
> - **Stage 2 (Real/Fake Classification):** $\text{prob} > 0.5 \implies$ **REAL NOTE** ($\text{Label} = 1$); $\le 0.5 \implies$ **FAKE NOTE** ($\text{Label} = 0$).

### 3.3 Training Configuration & Hyperparameters (Google Colab)

Models were trained in Google Colab (GPU accelerated):
- **Optimizer:** Adam ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 1 \times 10^{-4}$)
- **Batch Size:** 32
- **Loss Function:** `nn.BCEWithLogitsLoss()`
- **Learning Rate Scheduler:** `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)`
- **Early Stopping:** Patience = 5 epochs (Stage 1 best checkpoint saved at Epoch 2).
- **Dataset Partitioning:** 70% Train / 15% Validation / 15% Test (Stratified sampling, seed = 42).
- **Data Augmentations:** `RandomHorizontalFlip(p=0.5)`, `RandomVerticalFlip(p=0.5)`, `RandomRotation(degrees=15)`, `ColorJitter(brightness=0.2, contrast=0.2)`.

### 3.4 Verified Empirical Performance

All metrics reflect genuine test evaluations on held-out test sets:

#### Stage 1: Currency Gate ($N_{\text{test}} = 1,318$)
- **Overall Accuracy:** 91.96%
- **AUC-ROC:** 0.9877
- **Currency Class ($n=1,116$):** Precision = 1.00, Recall = 0.91
- **Non-Currency Class ($n=202$):** Precision = 0.66, Recall = 0.98
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN (Non-Currency)} & \text{FP} \\ \text{FN} & \text{TP (Currency)} \end{pmatrix} = \begin{pmatrix} 198 & 4 \\ 102 & 1014 \end{pmatrix}$$
- **Dataset Size:** 7,442 currency images + 1,344 non-currency images = 8,786 total images.

#### Stage 2: Real vs. Fake Classifier ($N_{\text{test}} = 1,117$)
- **Overall Accuracy:** ~99.46%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN (Fake)} & \text{FP} \\ \text{FN} & \text{TP (Real)} \end{pmatrix} = \begin{pmatrix} 375 & 1 \\ 5 & 736 \end{pmatrix}$$
- **Dataset Source:** Kaggle Indian Currency Real vs Fake Notes across ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 denominations.

---

### 3.5 OpenAI CLIP (`ViT-B/32`) Zero-Shot Gating (`clip_gate.py`)

To eliminate false-positive non-currency leaks without retraining ResNet-50 in Colab, the system includes an **OpenAI CLIP ViT-B/32** zero-shot multimodal foundation gate.

- **Model:** `openai/clip-vit-base-patch32` (Contrastive Language-Image Pretraining, 400M pre-trained pairs).
- **Mechanism:** Text-image cosine similarity in a shared 512-dimensional latent embedding space.
- **Candidate Label Sets:**
  - *Positive Labels:*
    - `"an Indian Rupee currency banknote"`
    - `"an Indian paper currency note"`
    - `"a genuine Indian banknote of paper money"`
  - *Negative Labels:*
    - `"a photograph of everyday objects, furniture, or animals"`
    - `"a white document, invoice, receipt, or plain paper"`
    - `"a page of a book, newspaper, or identity card"`
    - `"a toy, electronic gadget, or blank background"`
- **Score Aggregation:**
  $$\text{pos\_score} = \sum_{l \in \text{POS}} S(I, l), \quad \text{neg\_score} = \sum_{l \in \text{NEG}} S(I, l)$$
  $$\text{norm\_pos} = \frac{\text{pos\_score}}{\text{pos\_score} + \text{neg\_score}}$$
  Image passes Stage 1 if $\text{norm\_pos} \ge 0.50$.
- **0 MB Server RAM Architecture:** Executes via Hugging Face Serverless Inference API router (`router.huggingface.co/hf-inference`), adding zero weight memory to Render's 512 MB container.
- **Fail-Safe Fallback:** If `HF_TOKEN` is unset or an API timeout occurs, the system automatically falls back to the local ResNet-50 binary classifier with zero downtime.

---

### 3.6 Feature Embedding Memory Layer (`memory.py`)

The system implements a **zero-retraining continuous learning layer** that learns from mistake reports without needing model retraining:

- **Embedding Extraction:** During every forward pass through ResNet-50, the 2048-dimensional feature vector immediately preceding the classification head is extracted:
  $$\mathbf{e} = \text{Flatten}(\text{AdaptiveAvgPool2d}(\mathbf{F}_{layer4})) \in \mathbb{R}^{2048}$$
  $$\hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2 + 10^{-7}}$$
- **Similarity Metric:** Fast in-memory cosine dot product against all active approved memories:
  $$\text{sim}(\hat{\mathbf{e}}, \mathbf{m}_i) = \hat{\mathbf{e}} \cdot \mathbf{m}_i \in [-1, 1]$$
- **Decision Thresholds:**
  - **Override ($\text{sim} \ge 0.92$):** Direct visual match to a previously corrected note. The model verdict is overridden instantly (e.g. $p_1 = 0.99$ or $0.01$).
  - **Blend ($0.85 \le \text{sim} < 0.92$):** High visual affinity. The model's raw probability is blended with the prior human correction:
    $$p_{\text{blended}} = 0.4 \times p_{\text{raw}} + 0.6 \times p_{\text{target}}$$
  - **Ignore ($\text{sim} < 0.85$):** Standard forward pass verdict is preserved.
- **Persistence & Administration:** Stored in SQLite (`memory.db`). Includes moderation endpoints (`/api/memory/list`, `/api/memory/approve`) and automatic registration on user feedback corrections (`POST /api/feedback`).

---

## 4. Backend Implementation (`website/backend`)

The backend is built with **FastAPI** (`app.py`), structured for low-latency inference, bounded memory, and zero memory leaks.

### 4.1 Memory Optimization for 512 MB RAM (Render Free Plan)

Because the service runs on a constrained 512 MB container, several memory-saving techniques are deployed:

1. **Meta-Device Skeleton with Memory-Mapped Weights:**
   ```python
   def build_model(path: Path) -> nn.Module:
       with torch.device("meta"):
           model = models.resnet50(weights=None)
           model.fc = nn.Sequential(...)
       state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
       model.load_state_dict(state, assign=True)
       return model.to(device).eval()
   ```
   *Benefit:* Avoids duplicating the ~92 MB model tensor memory into private heap memory; weights reside in reclaimable OS page cache.
2. **Bounded Image Decoding (`imaging.py`):**
   - Limits file uploads to $15 \text{ MB}$.
   - Sets `Image.MAX_IMAGE_PIXELS = 50_000_000` to defend against decompression bombs.
   - Leverages `image.draft("RGB", (1024, 1024))` so libjpeg decodes lower-frequency DCT coefficients, avoiding multi-megapixel decompression in RAM.
   - Transposes EXIF orientations and applies `.thumbnail((1024, 1024))`.
3. **OS Arena & Threading Constraints:**
   - `MALLOC_ARENA_MAX=2`: Prevents glibc malloc from allocating separate memory arenas per core.
   - `OMP_NUM_THREADS=1`: Prevents PyTorch CPU worker thread inflation.

### 4.2 Stage 3 Post-Processing & Result Synthesis

Raw probabilities ($p_2 \in [0, 1]$) are converted into user-friendly metrics:

- **Stars (Authenticity Rating):** $\text{Stars} = \max(1, \min(5, \text{round}(5 \times p_2)))$
- **Authenticity Grade (Verdict-Aware):**
  - If **REAL**:
    - $p_2 \ge 0.95 \implies \mathbf{A}$
    - $p_2 \ge 0.80 \implies \mathbf{B}$
    - $p_2 \ge 0.60 \implies \mathbf{C}$
    - $p_2 < 0.60 \implies \mathbf{D}$
  - If **FAKE**:
    - $(1-p_2) \ge 0.95 \implies \mathbf{D}$
    - $(1-p_2) \ge 0.80 \implies \mathbf{C}$
    - $(1-p_2) \ge 0.60 \implies \mathbf{B}$
    - $(1-p_2) < 0.60 \implies \mathbf{A}$
- **Risk Level:**
  - If **REAL**: Low ($\ge 0.90$), Medium ($\ge 0.70$), High ($< 0.70$)
  - If **FAKE**: High ($\ge 0.90$), Medium ($\ge 0.70$), Low ($< 0.70$)
- **Decision Strength:**
  - $\text{Distance} = |p_2 - 0.5| \times 100$
  - $< 10\% \implies \text{BORDERLINE}$
  - $10\% - 30\% \implies \text{MODERATE}$
  - $> 30\% \implies \text{STRONG}$

### 4.3 Explainability Engine: Head-Only Grad-CAM (`explain.py`)

1. Backbone forward pass runs within `torch.no_grad()` to extract the $2048 \times 7 \times 7$ feature tensor from `layer4`.
2. The feature tensor is detached and set to `requires_grad_(True)`.
3. Only the lightweight classifier head forward & backward pass runs with `torch.enable_grad()` against the pre-sigmoid logit.
4. Channel weights:
   $$w_k = \frac{1}{49} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial y_{\text{logit}}}{\partial A_{i, j}^k}$$
5. Activation heatmap: $L_{\text{CAM}} = \text{ReLU}\left(\sum_k w_k A^k\right)$.
6. Computation cost: Only ~9 ms extra per scan (154 ms vs 145 ms).
7. **Edge Share Warning:** Computes the proportion of heat in perimeter cells:
   $$\text{edge\_share} = 1.0 - \frac{\sum_{i=2}^6 \sum_{j=2}^6 L_{\text{CAM}}(i, j)}{\sum_{\text{all}} L_{\text{CAM}}}$$
   If $\text{edge\_share} \ge 0.65$, a warning indicates background influence.

### 4.4 Automated Image Quality Assessment (`quality.py`)

Evaluates 5 computer-vision checks on a downscaled $512 \times 512$ thumbnail:
1. **Sharpness:** Variance of discrete Laplacian: $\sigma^2_{\text{Lap}} < 60.0$ (Fail), $< 150.0$ (Warn), $\ge 150.0$ (OK).
2. **Brightness / Exposure:** Mean pixel intensity: Dark fail $< 40$, warn $< 70$; Overexposed fail $> 225$, warn $> 200$.
3. **Contrast:** Standard deviation of intensities: Fail $< 14.0$, Warn $< 28.0$.
4. **Glare:** Ratio of saturated pixels ($\ge 250$): Fail $> 20\%$, Warn $> 8\%$.
5. **Resolution:** Minimum side of original image: Fail $< 200\text{ px}$, Warn $< 400\text{ px}$.
- Rating: `POOR` (if any fail), `FAIR` (if any warn), or `GOOD`.

### 4.5 LLM Help Desk & Fast Inference (`chat.py`)

- **Primary Model:** `gemini-2.5-flash` via Google GenAI SDK (`google-genai`).
- **Sub-Second Latency:** Configured with `thinking_budget=0` to disable dynamic multi-second reasoning delays, cutting response latency from 15 seconds to under 1 second.
- **Dynamic Fallback:** Automatically switches to `gemini-2.5-flash-lite` if the primary model is busy or rate-limited.
- **Multi-Turn Sanitization:** Normalizes conversation turns to guarantee strictly alternating `user` / `model` sequences, preventing API schema rejections on consecutive user queries.
- **Context Injection:** Injects latest scan verdict, confidence, grade, risk, quality, and attention focus with strict anti-hallucination boundaries.
- **Rate Limiting:** In-memory sliding window limiter (`ratelimit.py`) enforcing 6 queries/min per visitor IP and 300 queries/hour site-wide.

### 4.6 API Endpoint Specification

| Method | Route | Description | Request Body | Response |
|--------|-------|-------------|--------------|----------|
| `POST` | `/api/detect` | Single image scan + Grad-CAM + Memory check | `multipart/form-data` (`file`) | JSON: Stage 1, Stage 2, Quality, Explanation, Memory Match |
| `POST` | `/api/batch` | Batch scan (max 20 images) | `multipart/form-data` (`files`) | JSON: Summary counts + individual results |
| `POST` | `/api/feedback` | Log user correction & auto-register memory | JSON: `{predicted, actual, confidence, quality, comment, note_token}` | `{"id": <int>, "memory_registered": bool}` |
| `GET` | `/api/feedback/export` | Export feedback (CSV/JSON) | Header: `X-Admin-Token` | CSV file attachment or JSON list |
| `GET` | `/api/memory/list` | View active feature embeddings | Header: `X-Admin-Token` | JSON: List of vector memories with stage & labels |
| `POST` | `/api/memory/approve` | Approve or deactivate memory | JSON: `{id: <int>, approved: bool}`, Header: `X-Admin-Token` | `{"status": "ok", "id": <int>, "approved": bool}` |
| `GET` | `/api/chat/status` | Help desk availability check | None | `{"enabled": true/false}` |
| `POST` | `/api/chat` | Query AI help desk | JSON: `{messages: [...], scan: {...}}` | `{"reply": "<string>"}` |
| `GET` | `/health` | Container health probe | None | `{"status": "ok", "models_loaded": bool, "stage1_gate": {...}}` |

---

## 5. Frontend Implementation (`website/frontend`)

The frontend is built using **React 18** and bundled with **Vite**, designed as a retro ATM hardware kiosk interface.

### 5.1 Component Structure

```
App.jsx (Root & ATM Frame)
 ├── Topbar (Brand title + LED status indicator)
 ├── Navigation Modes (SINGLE NOTE | BATCH | HELP)
 ├── Display Section
 │    ├── IdleScreen (Dropzone, upload button, camera capture)
 │    ├── ScanScreen (Animated progress steps + CRT scanline)
 │    ├── ResultScreen
 │    │    ├── Heatmap.jsx (HTML5 Canvas overlaying 7x7 Grad-CAM grid)
 │    │    ├── QualityNote.jsx (Displaying quality status & retake warning)
 │    │    ├── FeedbackPrompt.jsx (User feedback correction modal)
 │    │    ├── Gate Architecture Badge (OpenAI CLIP ViT-B/32 vs ResNet-50)
 │    │    ├── Memory Match Badge (Glowing CRT indicator on memory overrides)
 │    │    └── Meter & Receipt (Authenticity metrics visualization)
 │    ├── BatchScreen.jsx (Multi-file upload, row status, batch totals)
 │    └── ChatScreen.jsx (Conversational help desk with suggestion pills)
 └── Tray Footer (ATM Note Return Slot)
```

### 5.2 Key Frontend Mechanics
- **Canvas Heatmap Renderer (`Heatmap.jsx`):** Renders the $7 \times 7$ normalized Grad-CAM matrix onto an offscreen canvas, applies bicubic upscaling to $112 \times 112$, and maps green tones for genuine features and red tones for counterfeit signals with alpha transparency over the banknote.
- **Memory Override Indicators:** Renders an animated glowing badge `[MEMORY VERIFIED: ...]` whenever a scan verdict was steered by prior user feedback.
- **Stage 1 Architecture Pill:** Exposes whether Stage 1 was evaluated by `OpenAI CLIP ViT-B/32` or `ResNet-50`.
- **Cold-Start Resilience (`ChatScreen.jsx`):** Employs an exponential retry strategy (3 attempts, 2.5s delay) to query `/api/chat/status`, gracefully handling Render free container spin-up.

---

## 6. DevOps & Deployment Pipeline

### 6.1 Multi-Stage Docker Build (`Dockerfile`)

```dockerfile
# Stage 1: Build React Frontend
FROM node:20-slim AS frontend
WORKDIR /web
COPY website/frontend/package.json website/frontend/package-lock.json ./
RUN npm ci
COPY website/frontend/ ./
RUN npm run build

# Stage 2: Python Runtime Environment
FROM python:3.11-slim
WORKDIR /app

# Install CPU-only PyTorch wheels to minimize image size
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY website/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY website/backend/*.py ./
COPY model/ ./model/
COPY --from=frontend /web/dist ./frontend-dist

# Verification: Prevent deploying Git LFS pointer text stubs
RUN for f in model/best_model.pth model/best_real_fake_model.pth; do \
      test "$(stat -c%s $f)" -gt 1000000 || { echo "$f is a Git LFS pointer, not the model"; exit 1; }; \
    done

ENV MODEL_DIR=/app/model \
    FRONTEND_DIR=/app/frontend-dist \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### 6.2 Render Environment Variables

| Variable | Required | Description |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | Yes | API key for Gemini 2.5 Flash chatbot assistance |
| `HF_TOKEN` | Optional | Free Hugging Face token to activate OpenAI CLIP ViT-B/32 Stage 1 |
| `STAGE1_MODE` | Optional | `"auto"` (default), `"clip"`, or `"resnet"` |
| `ADMIN_TOKEN` | Optional | Secret key for feedback and memory admin endpoints |
| `CHAT_MODEL` | Optional | Defaults to `gemini-2.5-flash` |
| `CHAT_FALLBACK_MODEL` | Optional | Defaults to `gemini-2.5-flash-lite` |

---

## 7. Local Development & Operational Guide

### 7.1 Running the Backend Locally

```powershell
cd website\backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Run FastAPI development server
uvicorn app:app --reload --port 8000
```

### 7.2 Running the Frontend Locally

```powershell
cd website\frontend
npm install
npm run dev
# App will launch on http://localhost:5173, proxying /api requests to localhost:8000
```

---

## 8. Governance, Strict Rules & Known Limitations

### 8.1 Critical Project Rules (Zero Exception)
1. **Untracked `Detection/` Folder:** Under no circumstances should `Detection/` (3,960 untracked files) be touched, scanned, evaluated, or committed to git.
2. **Local Research Paper (`report/`):** Excluded via `.git/info/exclude`. Must never be tracked or pushed to remote repositories.
3. **Commit Authorship Rotation:** Rotate authorship strictly across the three collaborators:
   $$\text{Ayush Trivedi} \longrightarrow \text{Akshat Vidyarthi} \longrightarrow \text{Harsh Thakkar}$$
   *(Next scheduled author: **Harsh Thakkar <harsht1103@gmail.com>**)*. Commits must never include automated AI co-author trailers.
4. **Data Integrity:** Never fabricate or extrapolate benchmark metrics. All reported numbers must strictly originate from actual Colab runs.

### 8.2 Known Technical Caveats & Considerations
- **Double-Sigmoid Calibration:** Models were trained with `BCEWithLogitsLoss` over Sigmoid outputs. While classification thresholds ($0.5$) are fully functional, intermediate probabilities are slightly uncalibrated. Authenticity grades and decision strengths serve as qualitative heuristics rather than rigorous Bayesian probabilities.
- **Render Cold Starts:** The free hosting plan spins down upon idle; the frontend includes status polling and connection indicators to guide users during cold wakeups.
