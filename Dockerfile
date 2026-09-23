FROM node:20-slim AS frontend
WORKDIR /web
COPY website/frontend/package.json website/frontend/package-lock.json ./
RUN npm ci
COPY website/frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY website/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY website/backend/app.py ./
COPY model/ ./model/
COPY --from=frontend /web/dist ./frontend-dist

# Fail the build if Git LFS pointers were copied instead of the real weights.
RUN for f in model/best_model.pth model/best_real_fake_model.pth; do \
      test "$(stat -c%s $f)" -gt 1000000 || { echo "$f is a Git LFS pointer, not the model"; exit 1; }; \
    done

ENV MODEL_DIR=/app/model \
    FRONTEND_DIR=/app/frontend-dist \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
