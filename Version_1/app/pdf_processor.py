"""
PDF Processor — Extracts text from the CCPA statute PDF with **page indexing**.

Uses PyMuPDF (fitz) for fast, accurate extraction.  Each returned block
carries the originating page number so downstream components can cite sources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Raw text from a single PDF page."""
    page_number: int
    text: str


@dataclass
class SectionContent:
    """A logical CCPA section extracted from the PDF."""
    section_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    domain: str = "general_compliance"


# Regex that matches CCPA section headers like "1798.100." or "§ 1798.100"
_SECTION_PATTERN = re.compile(
    r"(?:§\s*|Section\s+|SEC\.\s+)?(1798\.\d{3})\b\.?"
)

# Map section ranges to expert domains
_DOMAIN_MAP = {
    "1798.100": "data_collection",
    "1798.105": "deletion_rights",
    "1798.110": "data_collection",
    "1798.115": "opt_out_sale",
    "1798.120": "opt_out_sale",
    "1798.125": "non_discrimination",
    "1798.130": "data_collection",
    "1798.135": "opt_out_sale",
    "1798.140": "general_compliance",
    "1798.145": "general_compliance",
    "1798.150": "general_compliance",
    "1798.155": "general_compliance",
    "1798.185": "general_compliance",
    "1798.190": "general_compliance",
    "1798.192": "general_compliance",
    "1798.198": "general_compliance",
    "1798.199": "general_compliance",
}


class PDFProcessor:
    """Extracts and structures text from the CCPA statute PDF."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

    # ── Page-level extraction ──────────────────────────────────────────
    def extract_pages(self) -> List[PageContent]:
        """Return one ``PageContent`` per PDF page (1-indexed)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not installed — skipping PDF extraction")
            return []

        if not self.pdf_path.exists():
            logger.warning("PDF not found at %s", self.pdf_path)
            return []

        pages: List[PageContent] = []
        try:
            doc = fitz.open(str(self.pdf_path))
            for idx, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append(PageContent(page_number=idx + 1, text=text))
            doc.close()
            logger.info("Extracted %d pages from %s", len(pages), self.pdf_path.name)
        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc)

        return pages

    # ── Section-level extraction ───────────────────────────────────────
    def extract_sections(
        self, pages: Optional[List[PageContent]] = None
    ) -> List[SectionContent]:
        """
        Parse page-level text into logical CCPA sections.

        The algorithm scans for section-header patterns and groups
        consecutive text under each header until the next header appears.
        """
        if pages is None:
            pages = self.extract_pages()

        if not pages:
            return []

        # Merge all page text while tracking page boundaries
        merged_lines: list[tuple[int, str]] = []  # (page_num, line)
        for pc in pages:
            for line in pc.text.splitlines():
                merged_lines.append((pc.page_number, line))

        sections: List[SectionContent] = []
        current_id: Optional[str] = None
        current_title = ""
        current_lines: list[str] = []
        current_page_start = 1
        current_page_end = 1

        for page_num, line in merged_lines:
            match = _SECTION_PATTERN.search(line)
            if match:
                # Flush previous section
                if current_id and current_lines:
                    sections.append(
                        SectionContent(
                            section_id=current_id,
                            title=current_title,
                            text="\n".join(current_lines),
                            page_start=current_page_start,
                            page_end=current_page_end,
                            domain=_DOMAIN_MAP.get(
                                current_id, "general_compliance"
                            ),
                        )
                    )
                current_id = match.group(1)
                current_title = line.strip()
                current_lines = [line.strip()]
                current_page_start = page_num
                current_page_end = page_num
            else:
                if current_id:
                    current_lines.append(line)
                    current_page_end = page_num

        # Flush last section
        if current_id and current_lines:
            sections.append(
                SectionContent(
                    section_id=current_id,
                    title=current_title,
                    text="\n".join(current_lines),
                    page_start=current_page_start,
                    page_end=current_page_end,
                    domain=_DOMAIN_MAP.get(current_id, "general_compliance"),
                )
            )

        logger.info("Parsed %d CCPA sections from PDF", len(sections))
        return sections
