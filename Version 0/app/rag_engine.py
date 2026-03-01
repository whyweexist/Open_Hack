"""
RAG (Retrieval-Augmented Generation) Engine.

Builds a FAISS vector index over CCPA sections and PDF chunks,
then retrieves the top-K most relevant sections for a given query.
Supports hybrid retrieval: semantic similarity + keyword boosting.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from app.ccpa_knowledge import CCPA_SECTIONS, get_section_text
from app.pdf_processor import PageChunk, PageIndex
from app.config import TOP_K_RETRIEVAL, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval engine combining:
      1. Dense semantic search (sentence-transformers + FAISS)
      2. Keyword-based boosting from CCPA knowledge base
      3. Page-index awareness from PDF
    """

    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents: List[Dict] = []
        self.page_index: Optional[PageIndex] = None
        self._ready = False

    def initialize(
        self,
        pdf_chunks: Optional[List[PageChunk]] = None,
        page_index: Optional[PageIndex] = None,
    ):
        """
        Build the FAISS index from CCPA knowledge base + PDF chunks.
        """
        logger.info("Initializing RAG engine...")
        self.page_index = page_index

        # 1. Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._ready = False
            return

        # 2. Prepare documents from knowledge base
        self.documents = []
        texts_to_embed = []

        for sec_id, sec_data in CCPA_SECTIONS.items():
            # Main section text
            doc_text = (
                f"{sec_id} — {sec_data['title']}\n"
                f"{sec_data['text']}\n"
                f"Violation criteria: {'; '.join(sec_data['violation_criteria'])}"
            )
            self.documents.append({
                "text": doc_text,
                "section_id": sec_id,
                "source": "knowledge_base",
                "page": (
                    page_index.section_pages.get(sec_id, (None, None))[0]
                    if page_index else None
                ),
                "keywords": sec_data.get("keywords", []),
            })
            texts_to_embed.append(doc_text)

            # Also add violation criteria as separate entries for better recall
            for criterion in sec_data["violation_criteria"]:
                vc_text = f"{sec_id}: {criterion}"
                self.documents.append({
                    "text": vc_text,
                    "section_id": sec_id,
                    "source": "violation_criterion",
                    "page": None,
                    "keywords": sec_data.get("keywords", []),
                })
                texts_to_embed.append(vc_text)

        # 3. Add PDF chunks if available
        if pdf_chunks:
            for chunk in pdf_chunks:
                self.documents.append({
                    "text": chunk.text,
                    "section_id": chunk.section_id,
                    "source": "pdf",
                    "page": chunk.page_number,
                    "keywords": [],
                })
                texts_to_embed.append(chunk.text)
            logger.info(f"Added {len(pdf_chunks)} PDF chunks to index")

        # 4. Compute embeddings
        logger.info(f"Computing embeddings for {len(texts_to_embed)} documents...")
        embeddings = self.embedding_model.encode(
            texts_to_embed,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 5. Build FAISS index
        import faiss
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim)
        self.index.add(embeddings.astype(np.float32))

        self._ready = True
        logger.info(
            f"RAG engine ready: {len(self.documents)} docs, "
            f"dimension={dimension}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> List[Dict]:
        """
        Retrieve the top-K most relevant CCPA sections for a query.

        Uses hybrid approach:
          - Dense semantic similarity (FAISS)
          - Keyword boosting (from knowledge base)
          - De-duplication by section ID

        Returns:
            List of dicts with keys: section_id, text, score, page
        """
        if not self._ready or self.embedding_model is None:
            logger.warning("RAG engine not ready, returning all sections")
            return self._fallback_retrieve()

        # 1. Dense retrieval
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Retrieve more than top_k to allow for de-duplication
        k = min(top_k * 4, len(self.documents))
        scores, indices = self.index.search(query_embedding, k)

        # 2. Keyword boosting
        query_lower = query.lower()
        keyword_scores: Dict[str, float] = {}
        for sec_id, sec_data in CCPA_SECTIONS.items():
            boost = 0.0
            for kw in sec_data.get("keywords", []):
                if kw.lower() in query_lower:
                    boost += 0.15
            keyword_scores[sec_id] = min(boost, 0.5)  # Cap boost

        # 3. Combine scores and de-duplicate by section
        section_scores: Dict[str, float] = {}
        section_texts: Dict[str, str] = {}
        section_pages: Dict[str, Optional[int]] = {}

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            sec_id = doc.get("section_id")
            if not sec_id:
                continue

            combined_score = float(score) + keyword_scores.get(sec_id, 0.0)

            if sec_id not in section_scores or combined_score > section_scores[sec_id]:
                section_scores[sec_id] = combined_score
                # Use knowledge base text if available (more complete)
                if doc["source"] == "knowledge_base":
                    section_texts[sec_id] = doc["text"]
                elif sec_id not in section_texts:
                    section_texts[sec_id] = doc["text"]
                if doc.get("page"):
                    section_pages[sec_id] = doc["page"]

        # 4. Sort by score and return top_k
        sorted_sections = sorted(
            section_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for sec_id, score in sorted_sections:
            # Use knowledge base text for completeness
            text = section_texts.get(sec_id, get_section_text(sec_id))
            results.append({
                "section_id": sec_id,
                "text": text,
                "score": score,
                "page": section_pages.get(sec_id),
            })

        logger.info(
            f"Retrieved {len(results)} sections: "
            f"{[r['section_id'] for r in results]}"
        )
        return results

    def _fallback_retrieve(self) -> List[Dict]:
        """Fallback: return all sections when RAG is not available."""
        results = []
        for sec_id, sec_data in CCPA_SECTIONS.items():
            results.append({
                "section_id": sec_id,
                "text": get_section_text(sec_id),
                "score": 1.0,
                "page": None,
            })
        return results

    @property
    def is_ready(self) -> bool:
        return self._ready