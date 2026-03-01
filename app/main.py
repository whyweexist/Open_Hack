"""
FastAPI server for CCPA Compliance Analyzer.

Endpoints:
  GET  /health  → returns 200 when system is ready
  POST /analyze → analyzes a business practice for CCPA violations
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from app.config import PDF_PATH
from app.analyzer import CCPAAnalyzer
from app.rag_engine import RAGEngine
from app.pdf_processor import extract_text_from_pdf, chunk_pdf_by_pages

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global instances ─────────────────────────────────────────
analyzer = CCPAAnalyzer()
startup_complete = False


# ── Request / Response models ────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str


class AnalyzeResponse(BaseModel):
    harmful: bool
    articles: List[str]


# ── Lifespan (startup / shutdown) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components at startup."""
    global startup_complete

    logger.info("=" * 60)
    logger.info("CCPA Compliance Analyzer — Starting up")
    logger.info("=" * 60)

    try:
        # 1. Process PDF with page indexing
        logger.info(f"Processing PDF: {PDF_PATH}")
        page_index = None
        pdf_chunks = None

        if os.path.exists(PDF_PATH):
            page_index = extract_text_from_pdf(PDF_PATH)
            pdf_chunks = chunk_pdf_by_pages(page_index)
            logger.info(
                f"PDF page index built: {page_index.total_pages} pages, "
                f"{len(page_index.section_pages)} sections mapped"
            )
            for sec_id, (start, end) in sorted(
                page_index.section_pages.items()
            ):
                logger.info(f"  {sec_id}: pages {start}-{end}")
        else:
            logger.warning(
                f"PDF not found at {PDF_PATH}; "
                "using built-in knowledge base only"
            )

        # 2. Initialize RAG engine
        rag_engine = RAGEngine()
        rag_engine.initialize(
            pdf_chunks=pdf_chunks,
            page_index=page_index,
        )

        # 3. Initialize LLM analyzer
        analyzer.initialize(rag_engine=rag_engine)

        startup_complete = True
        logger.info("=" * 60)
        logger.info("CCPA Compliance Analyzer — Ready to serve")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        startup_complete = False

    yield

    # Cleanup
    logger.info("Shutting down CCPA Compliance Analyzer")


# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="CCPA Compliance Analyzer",
    description="Analyzes business practices for CCPA violations",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 200 when the system is fully initialized and ready.
    """
    if startup_complete and analyzer.is_ready:
        return {"status": "healthy"}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_practice(request: AnalyzeRequest):
    """
    Analyze a business practice for CCPA violations.

    Accepts a JSON body with a 'prompt' field describing the practice.
    Returns a JSON object with 'harmful' (bool) and 'articles' (list).
    """
    if not startup_complete or not analyzer.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready",
        )

    if not request.prompt or not request.prompt.strip():
        return AnalyzeResponse(harmful=False, articles=[])

    logger.info(f"Analyzing: {request.prompt[:100]}...")

    # Run analysis in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, analyzer.analyze, request.prompt
    )

    return AnalyzeResponse(
        harmful=result["harmful"],
        articles=result["articles"],
    )


# ── Error handlers ───────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content={"harmful": False, "articles": []},
    )