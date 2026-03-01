"""
FAISS Vector Store — stores chunk embeddings with metadata and
supports fast approximate nearest-neighbour search.

Metadata (section_id, page, domain, …) is kept in a parallel list
so results carry full provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search hit."""
    text: str
    score: float
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """In-memory FAISS index with metadata tracking."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        import faiss

        self.dim = dim
        # Inner-product index (embeddings are already L2-normalised ⇒ cosine)
        self.index = faiss.IndexFlatIP(dim)
        self._metadata: List[Dict[str, Any]] = []
        self._texts: List[str] = []

    # ── Ingest ─────────────────────────────────────────────────────────
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add vectors + metadata to the store."""
        assert len(embeddings) == len(texts) == len(metadatas)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self._texts.extend(texts)
        self._metadata.extend(metadatas)
        logger.info("Added %d vectors (total: %d)", len(texts), self.index.ntotal)

    # ── Search ─────────────────────────────────────────────────────────
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Return top-k nearest chunks, optionally filtered by domain."""
        if self.index.ntotal == 0:
            return []

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        # Fetch more candidates when filtering by domain
        fetch_k = top_k * 3 if domain_filter else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results: List[SearchResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if domain_filter and meta.get("domain") != domain_filter:
                continue
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=float(score),
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results

    @property
    def size(self) -> int:
        return self.index.ntotal
