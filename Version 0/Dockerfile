# ============================================================
# CCPA Compliance Analyzer — Dockerfile
# Multi-stage build with model pre-download
# ============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ── System dependencies ──────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ─────────────────────────────
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────
COPY app/ ./app/
COPY download_model.py .
COPY startup.sh .

# ── Copy CCPA statute PDF ────────────────────────────────────
RUN mkdir -p /app/data
COPY ccpa_statute.pdf /app/data/ccpa_statute.pdf

# ── Pre-download models (large layer — cached by Docker) ─────
# Build with:  docker build --build-arg HF_TOKEN=xxx .
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 download_model.py

# ── Expose port ──────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start server ─────────────────────────────────────────────
RUN chmod +x startup.sh
CMD ["bash", "startup.sh"]