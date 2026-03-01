"""
Mixture-of-Experts (MoE) Router — implements a sparse gating network
that routes incoming queries to the most relevant domain experts.

Architecture
────────────
1. Each expert registers a *description embedding* that captures its
   domain focus (data-collection, deletion, opt-out, discrimination …).
2. The gating network computes cosine similarity between the query
   embedding and every expert description.
3. Top-K experts (with scores above a threshold) are activated.
4. Activated experts run their specialised retrieval + analysis.
5. Results are aggregated with expert-weight blending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from app.config import TOP_K_EXPERTS, SIMILARITY_THRESHOLD

if TYPE_CHECKING:
    from app.embeddings import EmbeddingEngine
    from app.experts import Expert

logger = logging.getLogger(__name__)


@dataclass
class ExpertActivation:
    """An expert that the gating network has selected."""
    expert: "Expert"
    gate_score: float


class MoERouter:
    """Sparse gating network over domain experts."""

    def __init__(
        self,
        embedding_engine: "EmbeddingEngine",
        top_k: int = TOP_K_EXPERTS,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.threshold = threshold

        self._experts: List["Expert"] = []
        self._expert_embeddings: Optional[np.ndarray] = None  # (N, dim)

    # ── Registration ───────────────────────────────────────────────────
    def register_experts(self, experts: List["Expert"]) -> None:
        """Register domain experts and pre-compute their embeddings."""
        self._experts = experts

        descriptions = [e.description for e in experts]
        self._expert_embeddings = self.embedding_engine.encode(
            descriptions, normalize=True
        )
        logger.info("MoE router registered %d experts", len(experts))

    # ── Gating ─────────────────────────────────────────────────────────
    def route(self, query: str) -> List[ExpertActivation]:
        """
        Compute gate scores and return activated experts (top-K, above
        threshold), sorted by relevance descending.
        """
        if not self._experts or self._expert_embeddings is None:
            return []

        query_emb = self.embedding_engine.encode_query(query)  # (dim,)
        scores = self._compute_gate_scores(query_emb)

        # Sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        activations: List[ExpertActivation] = []
        for idx, score in ranked[: self.top_k]:
            if score < self.threshold:
                continue
            activations.append(
                ExpertActivation(expert=self._experts[idx], gate_score=score)
            )

        # If no expert exceeds threshold, activate the best one anyway
        # so we always produce an answer.
        if not activations and ranked:
            best_idx, best_score = ranked[0]
            activations.append(
                ExpertActivation(
                    expert=self._experts[best_idx], gate_score=best_score
                )
            )

        logger.debug(
            "Routed query to %d expert(s): %s",
            len(activations),
            [(a.expert.name, f"{a.gate_score:.3f}") for a in activations],
        )
        return activations

    # ── Internal ───────────────────────────────────────────────────────
    def _compute_gate_scores(self, query_emb: np.ndarray) -> np.ndarray:
        """Cosine similarity between query and every expert description."""
        # query_emb: (dim,)   expert_embs: (N, dim)  → (N,)
        return self._expert_embeddings @ query_emb
