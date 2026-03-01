"""
Model Download Script — run during Docker build to pre-download
the quantized GGUF LLM and the sentence-transformer embedding model
so the container starts instantly.
"""

import os
import sys
from pathlib import Path


def download_models():
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download GGUF quantized LLM ───────────────────────────────
    repo_id = os.getenv("LLM_MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    filename = os.getenv("LLM_MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

    target = model_dir / filename
    if not target.exists():
        print(f"⬇  Downloading GGUF model: {repo_id}/{filename}")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            print("✅ GGUF model downloaded")
        except Exception as exc:
            print(f"⚠  GGUF download failed (system will use heuristic fallback): {exc}")
    else:
        print(f"✅ GGUF model already present: {target}")

    # ── 2. Pre-download sentence-transformer ─────────────────────────
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"⬇  Pre-downloading embedding model: {model_name}")
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name)
        print("✅ Embedding model cached")
    except Exception as exc:
        print(f"⚠  Embedding model download failed: {exc}")
        sys.exit(1)

    print("\n🎉 All models ready!")


if __name__ == "__main__":
    download_models()
