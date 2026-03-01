# ═══════════════════════════════════════════════════════════════════════
#  CCPA Violation Detector — Optimised Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════════════
#  • python:3.11-slim base (no full OS — ~150 MB)
#  • Multi-stage: build deps stay out of the runtime image
#  • CPU-only PyTorch (~200 MB vs ~700 MB with CUDA)
#  • Models baked in at build time for instant startup
#  • Layer ordering maximises Docker cache hits
# ═══════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-only system dependencies (never reach runtime image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ cmake make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install CPU-only PyTorch first (smaller than full torch)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime deps only
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY download_model.py .
COPY ccpa_statute.pdf .
COPY requirements.txt .

# Pre-download models during build (baked into image)
RUN python download_model.py

# Expose the API port
EXPOSE 8000

# Health-check for Docker / Codespaces
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn — 1 worker to save memory on Codespaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
