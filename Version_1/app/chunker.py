"""
Section-aware Chunker — splits CCPA text into overlapping, fixed-size
chunks while preserving section and page metadata.

This feeds the FAISS vector store so the retriever knows *where* each
chunk came from (section, page, position).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""
    text: str
    section_id: str
    title: str
    page: int
    domain: str
    chunk_index: int          # position within the section
    total_chunks: int         # total chunks in the section


class SectionAwareChunker:
    """
    Splits section-level texts into fixed-size overlapping character
    windows, keeping each chunk tagged with its origin.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ─────────────────────────────────────────────────────
    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Accept a list of dicts with keys:
            section_id, title, text, page, domain
        and return chunked output.
        """
        all_chunks: List[Chunk] = []
        for sec in sections:
            sec_chunks = self._split_text(sec["text"])
            total = len(sec_chunks)
            for idx, text in enumerate(sec_chunks):
                all_chunks.append(
                    Chunk(
                        text=text,
                        section_id=sec["section_id"],
                        title=sec.get("title", ""),
                        page=sec.get("page", 0),
                        domain=sec.get("domain", "general_compliance"),
                        chunk_index=idx,
                        total_chunks=total,
                    )
                )
        logger.info(
            "Chunked %d sections → %d chunks (size=%d, overlap=%d)",
            len(sections), len(all_chunks), self.chunk_size, self.chunk_overlap,
        )
        return all_chunks

    # ── Internal ───────────────────────────────────────────────────────
    def _split_text(self, text: str) -> List[str]:
        """Fixed-size character-window split with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at a sentence boundary (. or \n) for readability
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_at = max(last_period, last_newline)
                if break_at > self.chunk_size // 2:
                    chunk = text[start : start + break_at + 1]
                    end = start + break_at + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]
