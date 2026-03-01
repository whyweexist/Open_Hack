"""
Reasoning RAG Pipeline — the top-level orchestrator.

Flow
────
1. Embed the incoming query.
2. **PageIndex tree search** — traverse the hierarchical CCPA tree
   to find the most relevant sections via reasoning-based retrieval.
3. Route through the MoE gating network → select top-K experts.
4. Each activated expert retrieves domain-specific chunks.
5. **Merge** PageIndex results with expert-retrieved chunks.
6. Build a structured reasoning prompt with the combined context.
7. Send to quantized LLM for chain-of-thought analysis.
8. Parse and validate the JSON output.
9. If LLM is unavailable / fails → fall back to an embedding-based
   heuristic classifier (ensures the system ALWAYS returns a result).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ccpa_knowledge import get_all_section_texts, get_sections_by_domain
from app.chunker import Chunk, SectionAwareChunker
from app.config import (
    CCPA_PDF_PATH, SIMILARITY_THRESHOLD, TOP_K_RETRIEVAL,
    PAGEINDEX_ENABLED, PAGEINDEX_TOP_K,
)
from app.embeddings import EmbeddingEngine
from app.experts import Expert, ExpertOpinion, create_all_experts
from app.llm_engine import QuantizedLLMEngine
from app.models import AnalyzeResponse
from app.moe_router import MoERouter
from app.page_index import PageIndexRetriever, build_page_index, TreeSearchResult
from app.pdf_processor import PDFProcessor
from app.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ── Violation patterns for heuristic fallback ────────────────────────
_VIOLATION_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern": "selling personal information without opt-out or consent or informing",
        "sections": ["1798.120"],
        "harmful": True,
    },
    {
        "pattern": "sell customer data to third party broker without notice opt-out",
        "sections": ["1798.120", "1798.115"],
        "harmful": True,
    },
    {
        "pattern": "collect browsing history geolocation biometric data without disclosing privacy policy",
        "sections": ["1798.100"],
        "harmful": True,
    },
    {
        "pattern": "undisclosed data collection hidden tracking without informing",
        "sections": ["1798.100", "1798.110"],
        "harmful": True,
    },
    {
        "pattern": "ignore refuse deny delete deletion request keeping records not deleting",
        "sections": ["1798.105"],
        "harmful": True,
    },
    {
        "pattern": "charge higher price discriminate deny service consumer opted out exercise rights",
        "sections": ["1798.125"],
        "harmful": True,
    },
    {
        "pattern": "sell data minor child children under 16 under 13 without parental consent",
        "sections": ["1798.120"],
        "harmful": True,
    },
    {
        "pattern": "data breach security unauthorized access unencrypted nonredacted",
        "sections": ["1798.150"],
        "harmful": True,
    },
    # ── Compliant patterns ──
    {
        "pattern": "clear privacy policy opt out at any time compliant",
        "sections": [],
        "harmful": False,
    },
    {
        "pattern": "deleted personal data within 45 days verified consumer request compliant",
        "sections": [],
        "harmful": False,
    },
    {
        "pattern": "do not sell my personal information link homepage",
        "sections": [],
        "harmful": False,
    },
    {
        "pattern": "equal service pricing regardless privacy rights non-discriminatory",
        "sections": [],
        "harmful": False,
    },
    {
        "pattern": "schedule meeting discuss project unrelated",
        "sections": [],
        "harmful": False,
    },
]


class ReasoningRAGPipeline:
    """End-to-end pipeline: query → AnalyzeResponse."""

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.llm_engine = QuantizedLLMEngine()
        self.moe_router = MoERouter(self.embedding_engine)
        self.global_store = FAISSVectorStore()
        self.experts: List[Expert] = []
        self._pattern_embeddings: Optional[np.ndarray] = None
        self._page_index_retriever: Optional[PageIndexRetriever] = None
        self._ready = False

    # ── Initialisation ─────────────────────────────────────────────────
    def initialise(self) -> None:
        """Load models, parse CCPA data, build indices."""
        logger.info("Initialising Reasoning RAG Pipeline …")

        # 1. Load embedding model
        self.embedding_engine.encode("warmup")

        # 2. Load LLM (may fail gracefully)
        self.llm_engine.load()

        # 3. Prepare CCPA sections (PDF + fallback knowledge base)
        sections = self._load_ccpa_sections()

        # 4. Chunk sections
        chunker = SectionAwareChunker()
        chunks = chunker.chunk_sections(sections)

        # 5. Embed all chunks
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.embedding_engine.encode(chunk_texts)
        chunk_metadatas = [
            {
                "section_id": c.section_id,
                "title": c.title,
                "page": c.page,
                "domain": c.domain,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        # 6. Populate global store
        self.global_store.add(chunk_embeddings, chunk_texts, chunk_metadatas)

        # 7. Create experts and populate their sub-indices
        self.experts = create_all_experts()
        for expert in self.experts:
            domain_mask = [
                i for i, c in enumerate(chunks) if c.domain == expert.domain
            ]
            if domain_mask:
                e_embs = chunk_embeddings[domain_mask]
                e_texts = [chunk_texts[i] for i in domain_mask]
                e_metas = [chunk_metadatas[i] for i in domain_mask]
                expert.add_chunks(e_embs, e_texts, e_metas)

        # 8. Register experts with MoE router
        self.moe_router.register_experts(self.experts)

        # 9. Pre-compute violation pattern embeddings (for fallback)
        patterns = [p["pattern"] for p in _VIOLATION_PATTERNS]
        self._pattern_embeddings = self.embedding_engine.encode(patterns)

        # 10. Build PageIndex tree (reasoning-based hierarchical retrieval)
        if PAGEINDEX_ENABLED:
            try:
                _root, self._page_index_retriever = build_page_index(
                    sections, self.embedding_engine, top_k_per_level=2
                )
                logger.info("PageIndex tree built with %d leaves",
                            len(_root.all_leaves()))
            except Exception as exc:
                logger.warning("PageIndex build failed: %s", exc)

        self._ready = True
        logger.info(
            "Pipeline ready — %d chunks, %d experts, LLM=%s, PageIndex=%s",
            len(chunks),
            len(self.experts),
            "available" if self.llm_engine.is_available else "fallback-only",
            "active" if self._page_index_retriever else "disabled",
        )

    # ── Main analysis ──────────────────────────────────────────────────
    def analyze(self, prompt: str) -> AnalyzeResponse:
        """Public entry-point: analyse a business practice prompt."""
        if not self._ready:
            self.initialise()

        query_embedding = self.embedding_engine.encode_query(prompt)

        # Step 1 — PageIndex tree search (reasoning-based retrieval)
        page_index_results: List[TreeSearchResult] = []
        if self._page_index_retriever:
            page_index_results = self._page_index_retriever.search(
                prompt, top_k=PAGEINDEX_TOP_K
            )
            logger.debug(
                "PageIndex found %d results: %s",
                len(page_index_results),
                [(r.node.section_id, f"{r.score:.3f}") for r in page_index_results],
            )

        # Step 2 — MoE routing → expert opinions
        activations = self.moe_router.route(prompt)
        expert_opinions: List[ExpertOpinion] = []
        for act in activations:
            opinion = act.expert.analyse(prompt, query_embedding)
            opinion.confidence *= act.gate_score  # weight by gate
            expert_opinions.append(opinion)

        # Step 3 — Build context (merge PageIndex + expert chunks)
        context = self._build_context(expert_opinions, page_index_results)

        # Step 4 — LLM reasoning (with fallback)
        result = None
        if self.llm_engine.is_available and context:
            result = self.llm_engine.classify_violation(context, prompt)

        # Step 5 — Validate / fallback
        if result and self._validate_result(result):
            return AnalyzeResponse(
                harmful=result["harmful"],
                articles=self._normalise_articles(result.get("articles", [])),
            )

        # Fallback: heuristic classifier (also uses PageIndex sections)
        logger.info("Using heuristic fallback for classification")
        return self._heuristic_classify(
            prompt, query_embedding, expert_opinions, page_index_results
        )

    # ── Context building ───────────────────────────────────────────────
    def _build_context(
        self,
        opinions: List[ExpertOpinion],
        page_index_results: Optional[List[TreeSearchResult]] = None,
    ) -> str:
        """Merge deduplicated PageIndex + expert chunks into a prompt context."""
        seen_texts: set[str] = set()
        context_parts: list[str] = []

        # Priority 1: PageIndex tree-search results (hierarchically retrieved)
        if page_index_results:
            for pir in page_index_results:
                key = pir.node.text[:100] if pir.node.text else pir.node.summary[:100]
                if key and key not in seen_texts:
                    seen_texts.add(key)
                    path_str = " → ".join(pir.path)
                    content = pir.node.text or pir.node.summary
                    context_parts.append(
                        f"[PageIndex | Section {pir.node.section_id} "
                        f"| Page {pir.node.start_page} | Path: {path_str}]\n"
                        f"{content}"
                    )

        # Priority 2: MoE expert chunks
        for op in sorted(opinions, key=lambda o: o.confidence, reverse=True):
            for chunk in op.retrieved_chunks:
                key = chunk.text[:100]
                if key not in seen_texts:
                    seen_texts.add(key)
                    sec = chunk.metadata.get("section_id", "?")
                    pg = chunk.metadata.get("page", "?")
                    context_parts.append(
                        f"[MoE Expert | Section {sec} | Page {pg}]\n{chunk.text}"
                    )

        return "\n\n---\n\n".join(context_parts[:12])  # cap at 12 chunks

    # ── Heuristic fallback classifier ──────────────────────────────────
    def _heuristic_classify(
        self,
        prompt: str,
        query_embedding: np.ndarray,
        expert_opinions: List[ExpertOpinion],
        page_index_results: Optional[List[TreeSearchResult]] = None,
    ) -> AnalyzeResponse:
        """
        Embedding-similarity classifier that matches the query against
        known violation / compliance patterns, boosted by PageIndex results.
        """
        # Pattern matching
        scores = self._pattern_embeddings @ query_embedding  # (N,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_pattern = _VIOLATION_PATTERNS[best_idx]

        # Aggregate expert confidence for harmful sections
        expert_harmful_score = 0.0
        expert_sections: set[str] = set()
        for op in expert_opinions:
            if op.confidence > 0.3 and op.relevant_sections:
                expert_harmful_score = max(expert_harmful_score, op.confidence)
                expert_sections.update(op.relevant_sections)

        # Boost with PageIndex tree-search results
        pi_sections: set[str] = set()
        pi_max_score = 0.0
        if page_index_results:
            for pir in page_index_results:
                if pir.node.section_id:
                    pi_sections.add(pir.node.section_id)
                    pi_max_score = max(pi_max_score, pir.score)

        # Decision logic
        if best_pattern["harmful"] and best_score > 0.45:
            # Merge pattern sections + expert sections + PageIndex sections
            sections = set(best_pattern["sections"]) | expert_sections | pi_sections
            return AnalyzeResponse(
                harmful=True,
                articles=[f"Section {s}" for s in sorted(sections) if s],
            )

        if not best_pattern["harmful"] and best_score > 0.5:
            return AnalyzeResponse(harmful=False, articles=[])

        # Edge case: expert or PageIndex signals strong violation
        combined_sections = expert_sections | pi_sections
        combined_score = max(expert_harmful_score, pi_max_score)
        if combined_score > 0.5 and combined_sections:
            return AnalyzeResponse(
                harmful=True,
                articles=[f"Section {s}" for s in sorted(combined_sections)],
            )

        # Default: not harmful
        return AnalyzeResponse(harmful=False, articles=[])

    # ── Helpers ────────────────────────────────────────────────────────
    def _load_ccpa_sections(self) -> List[Dict[str, Any]]:
        """Try PDF extraction first, fall back to built-in knowledge."""
        sections: List[Dict[str, Any]] = []

        # Try PDF
        try:
            processor = PDFProcessor(CCPA_PDF_PATH)
            pdf_sections = processor.extract_sections()
            if pdf_sections:
                sections = [
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "text": s.text,
                        "page": s.page_start,
                        "domain": s.domain,
                    }
                    for s in pdf_sections
                ]
                logger.info("Loaded %d sections from PDF", len(sections))
        except Exception as exc:
            logger.warning("PDF extraction failed: %s", exc)

        # Always merge with built-in knowledge (fills gaps)
        builtin = get_all_section_texts()
        existing_ids = {s["section_id"] for s in sections}
        for bi in builtin:
            if bi["section_id"] not in existing_ids:
                sections.append(bi)

        logger.info("Total CCPA sections available: %d", len(sections))
        return sections

    @staticmethod
    def _validate_result(result: Dict[str, Any]) -> bool:
        """Check LLM output has the expected shape."""
        if not isinstance(result, dict):
            return False
        if "harmful" not in result or not isinstance(result["harmful"], bool):
            return False
        if "articles" not in result or not isinstance(result["articles"], list):
            return False
        # If harmful, articles must be non-empty
        if result["harmful"] and len(result["articles"]) == 0:
            return False
        # If not harmful, articles must be empty
        if not result["harmful"] and len(result["articles"]) > 0:
            return False
        return True

    @staticmethod
    def _normalise_articles(articles: list) -> List[str]:
        """Ensure consistent 'Section 1798.XXX' format."""
        normalised: List[str] = []
        for art in articles:
            art = str(art).strip()
            # Already has "Section" prefix
            if art.lower().startswith("section"):
                normalised.append(art)
            # Raw number like "1798.120"
            elif art.replace(".", "").replace(" ", "").isdigit():
                normalised.append(f"Section {art}")
            else:
                normalised.append(art)
        return normalised
