"""
Centralized configuration for the CCPA Violation Detection System.
All tuneable knobs live here so nothing is scattered across modules.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CCPA_PDF_PATH = BASE_DIR / "ccpa_statute.pdf"

# ── Embedding model ─────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384

# ── Quantized LLM ───────────────────────────────────────────────────────
LLM_MODEL_REPO = os.getenv(
    "LLM_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
)
LLM_MODEL_FILE = os.getenv(
    "LLM_MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
LLM_CONTEXT_LENGTH = int(os.getenv("LLM_CONTEXT_LENGTH", "2048"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))

# ── RAG / chunking ──────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
TOP_K_EXPERTS = int(os.getenv("TOP_K_EXPERTS", "2"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

# ── Server ───────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
