# Implementation Specification: Indian Currency Authenticator

Comprehensive technical architecture, machine learning pipeline, backend/frontend implementation details, and deployment specification for the **Indian Currency Authenticator** system.

---

## 1. Executive Summary & System Overview

The **Indian Currency Authenticator** (formerly *Currency ATM*) is an end-to-end, multi-stage machine learning system designed to inspect, detect, and authenticate Indian paper currency (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 denominations). 

The platform pairs a 3-stage computer vision inference pipeline with an interactive ATM-themed web application, an automated image quality assessment screener, real-time explainable AI (Grad-CAM heatmaps), and an LLM-powered Help Desk powered by Google Gemini.

### Key Capabilities
- **Two-Stage Deep Learning Gating:** Rejects non-banknote images before subjecting banknotes to real/fake classification.
- **Explainable AI (XAI):** Generates zero-overhead $7 \times 7$ Grad-CAM saliency heatmaps highlighting regions influencing classification.
- **Deterministic Heuristic Layer:** Synthesizes raw network probabilities into human-readable metrics: Authenticity Grade (A–D), Risk Level (Low/Med/High), Star Rating (1–5), and Decision Strength.
- **Image Quality Pre-screener:** Detects blur (Laplacian variance), glare, underexposure/overexposure, low contrast, and insufficient resolution before user misinterpretation.
- **Constrained Memory Footprint:** Engineered to run reliably within a 512 MB RAM ceiling (Render Free Tier) using memory-mapped weights (`mmap=True`), libjpeg draft decoding, and glibc arena restrictions.
- **LLM Knowledge Assistant:** Integrated help desk using `gemini-3.8-flash` with dynamic failover to `gemini-3.5-flash-lite`, passing scan context with strict anti-hallucination prompt boundaries.

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
                    │         STAGE 1         │
                    │ Binary Currency Gate    │
                    │ (ResNet-50 Classifier)  │
                    └────────────┬────────────┘
                                 │
                    Is Banknote? │ p1 > 0.5
                  ┌──────────────┴──────────────┐
                  │                             │
             NO (p1 ≤ 0.5)                 YES (p1 > 0.5)
                  │                             │
                  ▼                             ▼
       ┌────────────────────┐        ┌────────────────────┐
       │   NOT A CURRENCY   │        │      STAGE 2       │
       │ Confidence: 1 - p1 │        │ Real vs Fake Note  │
       │ Quality Diagnostics│        │ (ResNet-50 Model)  │
       └────────────────────┘        └──────────┬─────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                         ┌────────────────────┐  ┌────────────────────┐
                         │   Grad-CAM (XAI)   │  │   Classification   │
                         │ Head-only Gradients│  │ Probability p2     │
                         │ 7x7 Attention Grid │  │ p2 > 0.5 = REAL    │
                         └──────────┬─────────┘  └──────────┬─────────┘
                                    │                       │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │        STAGE 3        │
                                    │    Result Synthesis   │
                                    │ - Grade (A, B, C, D)  │
                                    │ - Risk (Low/Med/High) │
                                    │ - Rating (1 - 5 ★)    │
                                    │ - Decision Strength   │
                                    └───────────┬───────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
             ┌───────────────────────┐                     ┌───────────────────────┐
             │   Frontend ATM View   │                     │    Gemini Help Desk   │
             │ - CRT/ATM UI Displays │                     │ Contextual Guidance   │
             │ - Heatmap Canvas      │                     │ Security Feature Info │
             │ - User Feedback Loop  │                     │ Failover Orchestration│
             └───────────────────────┘                     └───────────────────────┘
```

---

## 3. Deep Learning Pipeline & Model Specifications

Both Stage 1 and Stage 2 utilize a **ResNet-50** convolutional backbone (pretrained on ImageNet-1K) with identical custom binary classification heads.

### 3.1 Model Architecture

- **Backbone:** ResNet-50 (`torchvision.models.resnet50(weights=None)`)
- **Backbone Feature Output:** 2,048 feature maps ($7 \times 7$ spatial resolution after `layer4` conv blocks and before average pooling).
- **Custom Head:**
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
> **Label Orientation:** For both stages, outputs are mapped so that probability $> 0.5$ designates the target positive class:
> - **Stage 1 (Currency Detection):** $\text{prob} > 0.5 \implies$ **CURRENCY**; $\le 0.5 \implies$ **NON-CURRENCY**.
> - **Stage 2 (Real/Fake Classification):** $\text{prob} > 0.5 \implies$ **REAL NOTE** ($\text{Label} = 1$); $\le 0.5 \implies$ **FAKE NOTE** ($\text{Label} = 0$).

### 3.3 Training Configuration & Hyperparameters (Google Colab)

Models were trained across Phase 1 to Phase 3 in Google Colab (GPU accelerated):
- **Optimizer:** Adam ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 1 \times 10^{-4}$)
- **Batch Size:** 32
- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=...)` *(Note: Applied to Sigmoid output activations; saved checkpoints operate reliably with standard decision thresholds)*.
- **Learning Rate Scheduler:** `ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)`
- **Early Stopping:** Patience = 5 epochs (Stage 1 stopped at Epoch 7; best checkpoint saved at Epoch 2).
- **Dataset Partitioning:** 70% Train / 15% Validation / 15% Test (Stratified sampling, seed = 42).
- **Data Augmentations:**
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomVerticalFlip(p=0.5)`
  - `RandomRotation(degrees=15)`
  - `ColorJitter(brightness=0.2, contrast=0.2)`

### 3.4 Verified Empirical Performance

All metrics reflect genuine test evaluations on held-out test sets:

#### Stage 1: Currency Gate ($N_{\text{test}} = 1,318$)
- **Overall Accuracy:** 91.96%
- **AUC-ROC:** 0.9877
- **Currency Class ($n=1,116$):** Precision = 1.00, Recall = 0.91
- **Non-Currency Class ($n=202$):** Precision = 0.66, Recall = 0.98
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN (Non-Currency)} & \text{FP} \\ \text{FN} & \text{TP (Currency)} \end{pmatrix} = \begin{pmatrix} 198 & 4 \\ 102 & 1014 \end{pmatrix}$$
- **Dataset Size:** 7,442 currency images + 1,344 downloaded non-currency images = 8,786 total images (Split: 6,150 train / 1,318 val / 1,318 test).

#### Stage 2: Real vs. Fake Classifier ($N_{\text{test}} = 1,117$)
- **Overall Accuracy:** ~99.46%
- **Confusion Matrix:**
  $$\begin{pmatrix} \text{TN (Fake)} & \text{FP} \\ \text{FN} & \text{TP (Real)} \end{pmatrix} = \begin{pmatrix} 375 & 1 \\ 5 & 736 \end{pmatrix}$$
- **Dataset Source:** Kaggle Indian Currency Real vs Fake Notes across ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000 denominations.

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

- **Stars (Authenticity Rating):**
  $$\text{Stars} = \max(1, \min(5, \text{round}(5 \times p_2)))$$
  *Note: Stars denote probability of authenticity, not model confidence.*
- **Authenticity Grade (Verdict-Aware):**
  - If **REAL**:
    - $p_2 \ge 0.95 \implies \mathbf{A}$
    - $p_2 \ge 0.80 \implies \mathbf{B}$
    - $p_2 \ge 0.60 \implies \mathbf{C}$
    - $p_2 < 0.60 \implies \mathbf{D}$
  - If **FAKE**:
    - $(1-p_2) \ge 0.95 \implies \mathbf{D}$ (confident counterfeit)
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

To deliver visual explanations without doubling runtime or memory:
1. Backbone forward pass runs within `torch.no_grad()` to extract the $2048 \times 7 \times 7$ feature tensor from `layer4`.
2. The feature tensor is detached and set to `requires_grad_(True)`.
3. Only the lightweight classifier head forward & backward pass runs with `torch.enable_grad()` against the pre-sigmoid logit.
4. Channel weights are calculated via global average pooling of gradients:
   $$w_k = \frac{1}{7 \times 7} \sum_{i=1}^7 \sum_{j=1}^7 \frac{\partial y_{\text{logit}}}{\partial A_{i, j}^k}$$
5. Activation heatmap:
   $$L_{\text{CAM}} = \text{ReLU}\left(\sum_k w_k A^k\right)$$
6. Computation cost: Only ~9 ms extra per scan (154 ms vs 145 ms).
7. **Edge Share Warning:** Computes the proportion of heat residing in the perimeter cells:
   $$\text{edge\_share} = 1.0 - \frac{\sum_{i=2}^6 \sum_{j=2}^6 L_{\text{CAM}}(i, j)}{\sum_{\text{all}} L_{\text{CAM}}}$$
   If $\text{edge\_share} \ge 0.65$, the system triggers a warning that background edges may be influencing the verdict.

### 4.4 Automated Image Quality Assessment (`quality.py`)

Before returning verdicts, the image is evaluated across 5 computer-vision checks on a downscaled $512 \times 512$ thumbnail:
1. **Sharpness:** Variance of the discrete Laplacian operator:
   $$\Delta I(x, y) = -4 I(x, y) + I(x-1, y) + I(x+1, y) + I(x, y-1) + I(x, y+1)$$
   - Fail: $\sigma^2_{\text{Lap}} < 60.0$ | Warn: $< 150.0$ | OK: $\ge 150.0$.
2. **Brightness / Exposure:** Mean pixel intensity (Dark fail $< 40$, warn $< 70$; Overexposed fail $> 225$, warn $> 200$).
3. **Contrast:** Standard deviation of pixel intensities (Fail $< 14.0$, Warn $< 28.0$).
4. **Glare:** Ratio of pixels saturated at $\ge 250$ (Fail $> 20\%$, Warn $> 8\%$).
5. **Resolution:** Minimum side of original image (Fail $< 200\text{ px}$, Warn $< 400\text{ px}$).
- **Overall Rating:** `POOR` (if any fail), `FAIR` (if any warn), or `GOOD`. `POOR` flags `retake: true`.

### 4.5 LLM Help Desk & Fallback Orchestration (`chat.py`)

- **Primary Model:** `gemini-3.8-flash` via Google GenAI SDK (`google-genai`).
- **Fallback Model:** `gemini-3.5-flash-lite` (automatically engaged on 429 rate limits or 503 upstream overloads).
- **Scan Context Injection:** Passes the user's latest scan results (verdict, confidence, grade, risk, quality, and coarse attention sector) into the system prompt.
- **Strict Grounding Guardrails:** System prompt explicitly instructs the LLM that Grad-CAM attention does *not* indicate specific security feature failures, preventing fabricated security claims.
- **Rate Limiting:** In-memory sliding window limiter (`ratelimit.py`) enforcing 6 queries/min per visitor IP and 300 queries/hour site-wide.

### 4.6 API Endpoint Specification

| Method | Route | Description | Request Body | Response |
|--------|-------|-------------|--------------|----------|
| `POST` | `/api/detect` | Single image scan + Grad-CAM | `multipart/form-data` (`file`) | JSON: Stage 1, Stage 2, Quality, Explanation |
| `POST` | `/api/batch` | Batch scan (max 20 images) | `multipart/form-data` (`files`) | JSON: Summary counts + individual results |
| `POST` | `/api/feedback` | Log user correction | JSON: `{predicted, actual, confidence, quality, comment}` | `{"id": <int>}` |
| `GET` | `/api/feedback/export` | Export feedback (CSV/JSON) | Header: `X-Admin-Token` | CSV file attachment or JSON list |
| `GET` | `/api/chat/status` | Help desk availability check | None | `{"enabled": true/false}` |
| `POST` | `/api/chat` | Query AI help desk | JSON: `{messages: [...], scan: {...}}` | `{"reply": "<string>"}` |
| `GET` | `/health` | Container health probe | None | `{"status": "ok", "models_loaded": true}` |

---

## 5. Frontend Implementation (`website/frontend`)

The frontend is built using **React 18** and bundled with **Vite**, designed as a retro ATM hardware kiosk interface.

### 5.1 Component Structure

```
App.jsx (Root & ATM Frame)
 ├── Topbar (Brand title + LED indicator)
 ├── Navigation Modes (SINGLE NOTE | BATCH | HELP)
 ├── Display Section
 │    ├── IdleScreen (Dropzone, upload button, camera capture)
 │    ├── ScanScreen (Animated progress steps + CRT scanline)
 │    ├── ResultScreen
 │    │    ├── Heatmap.jsx (HTML5 Canvas overlaying 7x7 Grad-CAM grid)
 │    │    ├── QualityNote.jsx (Displaying quality status & retake warning)
 │    │    ├── FeedbackPrompt.jsx (User feedback thumbs-up / correction)
 │    │    └── Meter & Receipt (Authenticity metrics visualization)
 │    ├── BatchScreen.jsx (Multi-file upload, row status, batch totals)
 │    └── ChatScreen.jsx (Conversational help desk with suggestion pills)
 └── Tray Footer (ATM Note Return Slot)
```

### 5.2 Key Frontend Mechanics
- **Canvas Heatmap Renderer (`Heatmap.jsx`):** Renders the $7 \times 7$ normalized Grad-CAM matrix onto an offscreen canvas, applies bicubic upscaling to $112 \times 112$, and maps green tones for genuine features and red tones for counterfeit signals with alpha transparency over the original banknote image.
- **Cold-Start Resilience (`ChatScreen.jsx`):** Employs an exponential retry strategy (3 attempts, 2.5s delay) to query `/api/chat/status`, gracefully handling Render free container spin-up rather than falsely displaying "offline".
- **Responsive Mobile Camera Hook:** Supports native device camera triggers using `capture="environment"` attribute toggles.

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

### 6.2 Render Configuration (`render.yaml`)

```yaml
services:
  - type: web
    name: currency-atm
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: free
    healthCheckPath: /health
    envVars:
      - key: ADMIN_TOKEN
        sync: false
      - key: GEMINI_API_KEY
        sync: false
      - key: CHAT_PROVIDER
        value: gemini
      - key: CHAT_MODEL
        value: gemini-3.8-flash
      - key: CHAT_FALLBACK_MODEL
        value: gemini-3.5-flash-lite
```

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

### 7.3 Testing with Docker Locally

```bash
docker build -t indian-currency-authenticator .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key-here" indian-currency-authenticator
```

---

## 8. Governance, Strict Rules & Known Limitations

### 8.1 Critical Project Rules (Zero Exception)
1. **Untracked `Detection/` Folder:** Under no circumstances should `Detection/` (3,960 untracked files) be touched, scanned, evaluated, or committed to git.
2. **Local Research Paper (`report/`):** Excluded via `.git/info/exclude`. Must never be tracked or pushed to remote repositories.
3. **Commit Authorship Rotation:** Rotate authorship strictly across the three collaborators:
   $$\text{Ayush Trivedi} \longrightarrow \text{Akshat Vidyarthi} \longrightarrow \text{Harsh Thakkar}$$
   *(Next scheduled author: **Akshat Vidyarthi**)*. Commits must never include automated AI co-author trailers.
4. **Data Integrity:** Never fabricate or extrapolate benchmark metrics. All reported numbers must strictly originate from actual Colab runs.

### 8.2 Known Technical Caveats & Considerations
- **Double-Sigmoid Calibration:** Models were trained with `BCEWithLogitsLoss` over Sigmoid outputs. While classification thresholds ($0.5$) are fully functional, intermediate probabilities are slightly uncalibrated. Authenticity grades and decision strengths serve as qualitative heuristics rather than rigorous Bayesian probabilities.
- **Stage 1 Non-Currency Precision (0.66):** Due to the limited non-currency dataset ($N \approx 1,344$), approximately 102 non-currency test images leaked past Stage 1 as currency. Expanding diverse negative non-currency images remains an active recommendation.
- **Render Cold Starts:** The free hosting plan spins down upon idle; the frontend includes status polling and connection indicators to guide users during cold wakeups.
