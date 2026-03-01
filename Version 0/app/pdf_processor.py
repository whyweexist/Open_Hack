"""
PDF Processor with Page-Level Indexing.

Extracts text from each page of the CCPA statute PDF, identifies
section boundaries, and builds a page index mapping each CCPA
section to its page range.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PageChunk:
    """A chunk of text with page-level metadata."""
    text: str
    page_number: int
    section_id: Optional[str] = None
    chunk_index: int = 0


@dataclass
class PageIndex:
    """
    Maps CCPA sections to their page ranges in the PDF.
    Provides lookup by section number or by page.
    """
    section_pages: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    page_texts: Dict[int, str] = field(default_factory=dict)
    total_pages: int = 0

    def get_pages_for_section(self, section_id: str) -> Optional[Tuple[int, int]]:
        return self.section_pages.get(section_id)

    def get_text_for_page(self, page_num: int) -> str:
        return self.page_texts.get(page_num, "")

    def get_sections_on_page(self, page_num: int) -> List[str]:
        result = []
        for sid, (start, end) in self.section_pages.items():
            if start <= page_num <= end:
                result.append(sid)
        return result


# ── Section number regex pattern ──────────────────────────────
SECTION_PATTERN = re.compile(
    r'(?:Section\s+|§\s*|SEC\.\s+)'
    r'(1798\.\d{3}(?:\.\d+)?)',
    re.IGNORECASE
)

SECTION_PATTERN_BARE = re.compile(
    r'\b(1798\.\d{3}(?:\.\d+)?)\b'
)


def extract_text_from_pdf(pdf_path: str) -> PageIndex:
    """
    Extract text from each page of the PDF and build a page index.

    Returns:
        PageIndex with page texts and section-to-page mappings.
    """
    page_index = PageIndex()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF processing")
        return page_index

    try:
        doc = fitz.open(pdf_path)
        page_index.total_pages = len(doc)

        section_first_page: Dict[str, int] = {}
        section_last_page: Dict[str, int] = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            page_index.page_texts[page_num + 1] = text  # 1-based

            # Find section references on this page
            for match in SECTION_PATTERN.finditer(text):
                sec_num = match.group(1)
                sec_id = f"Section {sec_num}"
                if sec_id not in section_first_page:
                    section_first_page[sec_id] = page_num + 1
                section_last_page[sec_id] = page_num + 1

            # Also try bare number pattern
            for match in SECTION_PATTERN_BARE.finditer(text):
                sec_num = match.group(1)
                sec_id = f"Section {sec_num}"
                if sec_id not in section_first_page:
                    section_first_page[sec_id] = page_num + 1
                section_last_page[sec_id] = page_num + 1

        doc.close()

        # Build page ranges
        for sec_id in section_first_page:
            start = section_first_page[sec_id]
            end = section_last_page.get(sec_id, start)
            page_index.section_pages[sec_id] = (start, end)

        logger.info(
            f"PDF processed: {page_index.total_pages} pages, "
            f"{len(page_index.section_pages)} sections found"
        )

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")

    return page_index


def chunk_pdf_by_pages(
    page_index: PageIndex,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[PageChunk]:
    """
    Split extracted PDF text into overlapping chunks with page references.

    Args:
        page_index: The PageIndex from extract_text_from_pdf
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of PageChunk objects
    """
    chunks: List[PageChunk] = []
    chunk_idx = 0

    for page_num in sorted(page_index.page_texts.keys()):
        text = page_index.page_texts[page_num].strip()
        if not text:
            continue

        # Determine which sections are on this page
        sections_on_page = page_index.get_sections_on_page(page_num)
        primary_section = sections_on_page[0] if sections_on_page else None

        # Split long pages into chunks
        if len(text) <= chunk_size:
            chunks.append(PageChunk(
                text=text,
                page_number=page_num,
                section_id=primary_section,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1
        else:
            # Sliding window chunking
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end]

                # Try to break at sentence boundary
                if end < len(text):
                    last_period = chunk_text.rfind('.')
                    if last_period > chunk_size // 2:
                        end = start + last_period + 1
                        chunk_text = text[start:end]

                chunks.append(PageChunk(
                    text=chunk_text,
                    page_number=page_num,
                    section_id=primary_section,
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1
                start = end - chunk_overlap
                if start >= len(text):
                    break

    logger.info(f"Created {len(chunks)} chunks from PDF")
    return chunks


def extract_sections_from_text(full_text: str) -> Dict[str, str]:
    """
    Parse raw PDF text and extract individual section texts.

    Returns:
        Dict mapping section ID → section text
    """
    sections: Dict[str, str] = {}

    # Split by section headers
    parts = re.split(
        r'(?=(?:Section\s+|§\s*|SEC\.\s+)1798\.\d{3})',
        full_text,
        flags=re.IGNORECASE
    )

    for part in parts:
        match = SECTION_PATTERN.search(part[:200])
        if match:
            sec_num = match.group(1)
            sec_id = f"Section {sec_num}"
            sections[sec_id] = part.strip()

    return sections