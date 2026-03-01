"""
FastAPI application — exposes /health and /analyze endpoints.

The pipeline is initialised once at startup so subsequent requests
are fast (no re-loading models on every call).
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.models import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.reasoning_rag import ReasoningRAGPipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CCPA Violation Detector",
    description="Analyses business practices against the CCPA statute using Reasoning RAG with Mixture-of-Experts.",
    version="1.0.0",
)

# Global pipeline instance (loaded at startup)
pipeline: ReasoningRAGPipeline | None = None


@app.on_event("startup")
async def startup_event():
    """Initialise the full RAG pipeline at container start."""
    global pipeline
    t0 = time.time()
    logger.info("🚀 Starting pipeline initialisation …")
    pipeline = ReasoningRAGPipeline()
    pipeline.initialise()
    logger.info("✅ Pipeline ready in %.1f s", time.time() - t0)


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint expected by the test harness."""
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyse a natural-language prompt describing a business practice
    and return whether it violates the CCPA.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    logger.info("Analysing: %.80s…", request.prompt)
    t0 = time.time()

    try:
        result = pipeline.analyze(request.prompt)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        # Return a safe fallback rather than a 500
        result = AnalyzeResponse(harmful=False, articles=[])

    logger.info(
        "Result: harmful=%s articles=%s (%.2f s)",
        result.harmful, result.articles, time.time() - t0,
    )
    return result
