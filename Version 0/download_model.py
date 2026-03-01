"""
Pre-download model weights during Docker build.
This ensures fast container startup without network dependency.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_models():
    hf_token = os.getenv("HF_TOKEN", None)

    # ── Download LLM ─────────────────────────────────────────
    model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    logger.info(f"Downloading LLM: {model_name}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            model_name,
            token=hf_token,
            ignore_patterns=["*.gguf", "*.ggml"],  # skip GGUF files
        )
        logger.info(f"LLM downloaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to download LLM: {e}")
        sys.exit(1)

    # ── Download Embedding Model ─────────────────────────────
    embedding_model = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    logger.info(f"Downloading embedding model: {embedding_model}")

    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(embedding_model)
        logger.info(f"Embedding model downloaded: {embedding_model}")
    except Exception as e:
        logger.error(f"Failed to download embedding model: {e}")
        sys.exit(1)

    logger.info("All models downloaded successfully")


if __name__ == "__main__":
    download_models()