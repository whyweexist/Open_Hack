"""
Domain Experts — specialised retrieval + analysis modules for each
area of the CCPA statute.

Each expert owns:
  • a human-readable description (used by the MoE gating network)
  • a dedicated FAISS sub-index containing only its domain chunks
  • domain-specific keyword patterns for fast pre-filtering
  • section IDs it covers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.vector_store import FAISSVectorStore, SearchResult
from app.config import TOP_K_RETRIEVAL, EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class ExpertOpinion:
    """Structured output from one expert."""
    expert_name: str
    domain: str
    relevant_sections: List[str]
    retrieved_chunks: List[SearchResult]
    confidence: float                       # 0-1
    reasoning_hint: str                     # short sentence for LLM


class Expert:
    """Base domain expert."""

    def __init__(
        self,
        name: str,
        domain: str,
        description: str,
        section_ids: List[str],
        keywords: List[str],
    ):
        self.name = name
        self.domain = domain
        self.description = description
        self.section_ids = section_ids
        self.keywords = [kw.lower() for kw in keywords]
        self.vector_store = FAISSVectorStore(dim=EMBEDDING_DIM)

    # ── Populate the expert's private vector store ─────────────────────
    def add_chunks(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.vector_store.add(embeddings, texts, metadatas)

    # ── Retrieval ──────────────────────────────────────────────────────
    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = TOP_K_RETRIEVAL
    ) -> List[SearchResult]:
        """Retrieve top-k chunks using normalized query embedding."""
        # Ensure query is normalized for cosine similarity search
        normalized_query = self._normalize_embedding(query_embedding)
        return self.vector_store.search(normalized_query, top_k=top_k)
    
    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    # ── Keyword pre-check ──────────────────────────────────────────────
    def keyword_relevance(self, text: str) -> float:
        """Quick 0-1 score based on keyword match count."""
        text_lower = text.lower()
        hits = sum(1 for kw in self.keywords if kw in text_lower)
        return min(hits / max(len(self.keywords), 1), 1.0)

    # ── Analyse ────────────────────────────────────────────────────────
    def analyse(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> ExpertOpinion:
        """Run retrieval and produce an expert opinion."""
        # Validate inputs
        if query_embedding is None or len(query_embedding) == 0:
            logger.warning("%s: Invalid query embedding", self.name)
            return self._empty_opinion(query)
        
        results = self.retrieve(query_embedding, top_k)
        kw_score = self.keyword_relevance(query)

        # Calculate confidence from retrieval results
        if results:
            # Use max score from top results (most relevant chunk)
            max_ret_score = max(r.score for r in results)
            # Average of top-3 results
            avg_ret_score = np.mean([r.score for r in results[:3]])
            # Weighted combination favoring maximum relevance
            ret_score = 0.4 * max_ret_score + 0.6 * avg_ret_score
        else:
            ret_score = 0.0

        # Combine retrieval confidence with keyword relevance
        confidence = 0.7 * ret_score + 0.3 * kw_score
        confidence = float(np.clip(confidence, 0.0, 1.0))

        sections_found = list(
            {r.metadata.get("section_id", "") for r in results if r.metadata}
        )

        return ExpertOpinion(
            expert_name=self.name,
            domain=self.domain,
            relevant_sections=sections_found,
            retrieved_chunks=results,
            confidence=confidence,
            reasoning_hint=self._build_hint(query, results),
        )
    
    def _empty_opinion(self, query: str) -> ExpertOpinion:
        """Return a zero-confidence opinion when no retrieval happens."""
        return ExpertOpinion(
            expert_name=self.name,
            domain=self.domain,
            relevant_sections=[],
            retrieved_chunks=[],
            confidence=0.0,
            reasoning_hint=f"Expert '{self.name}' could not analyse query.",
        )

    def _build_hint(self, query: str, results: List[SearchResult]) -> str:
        """Create a human-readable reasoning hint."""
        if not results:
            return f"No relevant {self.domain} chunks found."
        
        # Get unique sections and compute average score
        secs = list({r.metadata.get("section_id", "?") for r in results})
        avg_score = np.mean([r.score for r in results]) if results else 0.0
        
        return (
            f"Expert '{self.name}' found {len(results)} relevant chunk(s) "
            f"from section(s) {', '.join(secs)} (avg relevance: {avg_score:.2f})."
        )


# ═══════════════════════════════════════════════════════════════════════
# Concrete expert instances
# ═══════════════════════════════════════════════════════════════════════

def create_data_collection_expert() -> Expert:
    return Expert(
        name="Data Collection & Disclosure",
        domain="data_collection",
        description=(
            "Analyses whether a business properly informs consumers about "
            "data collection practices, provides required notices, discloses "
            "categories of personal information collected, and responds to "
            "consumer requests within 45 days. Covers CCPA sections "
            "1798.100, 1798.110, and 1798.130."
        ),
        section_ids=["1798.100", "1798.110", "1798.130"],
        keywords=[
            "collect", "collection", "disclose", "disclosure", "privacy policy",
            "notice", "inform", "categories", "personal information",
            "browsing history", "geolocation", "biometric", "purpose",
            "undisclosed", "hidden", "secret", "not mention", "doesn't mention",
        ],
    )


def create_deletion_rights_expert() -> Expert:
    return Expert(
        name="Deletion Rights",
        domain="deletion_rights",
        description=(
            "Analyses whether a business honours consumer requests to "
            "delete personal information, responds within deadlines, and "
            "directs service providers to delete data. Covers CCPA "
            "section 1798.105."
        ),
        section_ids=["1798.105"],
        keywords=[
            "delete", "deletion", "erase", "remove", "right to delete",
            "request to delete", "ignore", "refuse", "keep records",
            "keeping", "ignoring", "not deleting",
        ],
    )


def create_opt_out_sale_expert() -> Expert:
    return Expert(
        name="Opt-Out & Sale of Data",
        domain="opt_out_sale",
        description=(
            "Analyses whether a business sells personal information without "
            "consumer consent, provides opt-out mechanisms, displays 'Do Not "
            "Sell My Personal Information' links, and obtains required consent "
            "for minors under 16. Covers CCPA sections 1798.115, 1798.120, "
            "and 1798.135."
        ),
        section_ids=["1798.115", "1798.120", "1798.135"],
        keywords=[
            "sell", "sale", "selling", "sold", "opt-out", "opt out",
            "third party", "broker", "data broker", "minor", "child",
            "children", "under 16", "under 13", "parental consent",
            "14-year", "teenager", "without consent", "without informing",
            "do not sell",
        ],
    )


def create_non_discrimination_expert() -> Expert:
    return Expert(
        name="Non-Discrimination",
        domain="non_discrimination",
        description=(
            "Analyses whether a business discriminates against consumers "
            "who exercise their CCPA rights by denying services, charging "
            "higher prices, or providing lower quality. Covers CCPA "
            "section 1798.125."
        ),
        section_ids=["1798.125"],
        keywords=[
            "discriminat", "price", "pricing", "penalty", "deny",
            "higher price", "different rate", "different quality",
            "punish", "retaliat", "opted out", "exercise rights",
            "charge more", "lower quality",
        ],
    )


def create_all_experts() -> list[Expert]:
    """Factory: return all four domain experts."""
    return [
        create_data_collection_expert(),
        create_deletion_rights_expert(),
        create_opt_out_sale_expert(),
        create_non_discrimination_expert(),
    ]
