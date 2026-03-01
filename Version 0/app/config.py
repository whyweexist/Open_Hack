"""
Configuration constants for the CCPA Compliance Analyzer.
All settings can be overridden via environment variables.
"""

import os

# ── Model Configuration ──────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── HuggingFace Token (for gated models) ─────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ── Server Configuration ─────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── RAG Configuration ────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── LLM Generation Parameters ────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))

# ── Paths ─────────────────────────────────────────────────────
PDF_PATH = os.getenv("PDF_PATH", "/app/data/ccpa_statute.pdf")
INDEX_PATH = os.getenv("INDEX_PATH", "/app/data/faiss_index")

# ── Valid CCPA Sections (for output validation) ───────────────
VALID_SECTIONS = [
    "Section 1798.100",
    "Section 1798.105",
    "Section 1798.106",
    "Section 1798.110",
    "Section 1798.115",
    "Section 1798.120",
    "Section 1798.121",
    "Section 1798.125",
    "Section 1798.130",
    "Section 1798.135",
    "Section 1798.140",
    "Section 1798.145",
    "Section 1798.150",
    "Section 1798.155",
]