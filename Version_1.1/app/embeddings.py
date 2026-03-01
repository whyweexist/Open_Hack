"""
Embedding Engine — wraps sentence-transformers with optional ONNX
quantisation for fast CPU inference.

The engine lazily loads the model on first use and caches it.
"""

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Thin wrapper around SentenceTransformer with quantisation hooks."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        use_quantized: bool = True,
    ):
        self.model_name = model_name
        self.use_quantized = use_quantized
        self._model = None
        self.dim = EMBEDDING_DIM

    # ── Lazy model loading ─────────────────────────────────────────────
    def _load_model(self):
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s (quantized=%s)",
                     self.model_name, self.use_quantized)

        backend = "onnx" if self.use_quantized else None

        try:
            if self.use_quantized:
                # Use ONNX backend for quantized CPU inference
                self._model = SentenceTransformer(
                    self.model_name,
                    backend=backend,
                )
            else:
                self._model = SentenceTransformer(self.model_name)
        except Exception:
            # Fallback: load without quantization
            logger.warning("Quantized load failed — falling back to default")
            self._model = SentenceTransformer(self.model_name)

        logger.info("Embedding model loaded (dim=%d)", self.dim)

    # ── Public API ─────────────────────────────────────────────────────
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode one or more texts into dense vectors."""
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query — convenience wrapper."""
        return self.encode(text)[0]
