"""
PageIndex — Local hierarchical tree-index for reasoning-based RAG.

Inspired by VectifyAI/PageIndex (https://github.com/VectifyAI/PageIndex),
this module builds a **hierarchical tree structure** from the CCPA statute
PDF and supports **tree-search retrieval** using embedding similarity at
each level — no OpenAI API required.

Architecture
────────────
                    ┌──────────────────────┐
                    │  CCPA Statute (root)  │
                    └──────┬───────────────┘
               ┌───────────┼───────────────┐
               ▼           ▼               ▼
        ┌──────────┐ ┌──────────┐   ┌──────────┐
        │ Title 1  │ │ Title 2  │   │ Title N  │
        │(General) │ │(Rights)  │   │(Enforce) │
        └────┬─────┘ └────┬─────┘   └──────────┘
             ▼            ▼
        ┌─────────┐  ┌─────────┐
        │§1798.100│  │§1798.120│  …
        └─────────┘  └─────────┘

Tree search: at each level, compute similarity between query and
node summaries → descend into best-matching children → collect
leaf text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PageIndexNode:
    """A single node in the hierarchical page-index tree."""
    title: str
    node_id: str
    start_page: int
    end_page: int
    summary: str
    text: str = ""
    section_id: str = ""
    domain: str = "general_compliance"
    children: List["PageIndexNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "node_id": self.node_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "section_id": self.section_id,
            "domain": self.domain,
            "children": [c.to_dict() for c in self.children],
        }

    def all_leaves(self) -> List["PageIndexNode"]:
        """Recursively collect all leaf nodes."""
        if self.is_leaf():
            return [self]
        leaves: list[PageIndexNode] = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves


@dataclass
class TreeSearchResult:
    """Result from a tree-search retrieval."""
    node: PageIndexNode
    score: float
    path: List[str]  # node_ids from root to this node


# ═══════════════════════════════════════════════════════════════════════
# Tree builder
# ═══════════════════════════════════════════════════════════════════════

class PageIndexBuilder:
    """
    Builds a PageIndex tree from CCPA sections.

    Unlike the upstream VectifyAI version (which uses GPT-4o for ToC
    detection and tree construction), this builder works entirely offline
    by leveraging the known structure of the CCPA statute.
    """

    # ── Domain grouping for mid-level tree nodes ───────────────────────
    _DOMAIN_GROUPS = {
        "Consumer Rights — Data Collection & Disclosure": {
            "domain": "data_collection",
            "sections": ["1798.100", "1798.110", "1798.130"],
            "summary": (
                "Consumer rights related to knowing what personal information "
                "is collected, disclosure requirements, and response timelines."
            ),
        },
        "Consumer Rights — Right to Deletion": {
            "domain": "deletion_rights",
            "sections": ["1798.105"],
            "summary": (
                "Consumer right to request deletion of personal information "
                "and business obligations to comply."
            ),
        },
        "Consumer Rights — Opt-Out of Sale": {
            "domain": "opt_out_sale",
            "sections": ["1798.115", "1798.120", "1798.135"],
            "summary": (
                "Consumer right to opt out of the sale of personal information, "
                "minor protections, and 'Do Not Sell' link requirements."
            ),
        },
        "Consumer Rights — Non-Discrimination": {
            "domain": "non_discrimination",
            "sections": ["1798.125"],
            "summary": (
                "Prohibition on discriminating against consumers who exercise "
                "CCPA rights, including pricing and service quality."
            ),
        },
        "General Provisions": {
            "domain": "general_compliance",
            "sections": ["1798.140", "1798.145", "1798.150", "1798.155"],
            "summary": (
                "Definitions, exemptions, data-breach right of action, and "
                "administrative enforcement penalties."
            ),
        },
    }

    def build_from_sections(
        self, sections: List[Dict[str, Any]]
    ) -> PageIndexNode:
        """
        Build a 3-level tree:
            Root → Domain-group → Individual section (leaf)

        Parameters
        ----------
        sections : list of dicts with keys
            section_id, title, text, page, domain
        """
        section_map = {s["section_id"]: s for s in sections}
        node_counter = 0

        group_nodes: List[PageIndexNode] = []
        for group_title, group_info in self._DOMAIN_GROUPS.items():
            node_counter += 1
            child_nodes: List[PageIndexNode] = []

            for sid in group_info["sections"]:
                sec = section_map.get(sid)
                if not sec:
                    continue
                node_counter += 1
                child_nodes.append(
                    PageIndexNode(
                        title=sec.get("title", f"Section {sid}"),
                        node_id=f"N{node_counter:04d}",
                        start_page=sec.get("page", 0),
                        end_page=sec.get("page", 0),
                        summary=sec["text"][:300],
                        text=sec["text"],
                        section_id=sid,
                        domain=sec.get("domain", group_info["domain"]),
                    )
                )

            if child_nodes:
                group_node = PageIndexNode(
                    title=group_title,
                    node_id=f"N{node_counter:04d}",
                    start_page=child_nodes[0].start_page,
                    end_page=child_nodes[-1].end_page,
                    summary=group_info["summary"],
                    domain=group_info["domain"],
                    children=child_nodes,
                )
                group_nodes.append(group_node)

        root = PageIndexNode(
            title="California Consumer Privacy Act (CCPA)",
            node_id="N0000",
            start_page=1,
            end_page=max((g.end_page for g in group_nodes), default=1),
            summary=(
                "The California Consumer Privacy Act of 2018 gives California "
                "residents the right to know what personal information is "
                "collected, to delete it, to opt-out of its sale, and to "
                "non-discrimination for exercising these rights."
            ),
            children=group_nodes,
        )

        total_leaves = len(root.all_leaves())
        logger.info(
            "PageIndex tree built: %d group nodes, %d leaf sections",
            len(group_nodes), total_leaves,
        )
        return root


# ═══════════════════════════════════════════════════════════════════════
# Tree-search retriever
# ═══════════════════════════════════════════════════════════════════════

class PageIndexRetriever:
    """
    Reasoning-based tree-search retrieval.

    At each level of the tree the retriever:
    1. Embeds every child node's summary.
    2. Computes cosine similarity with the query embedding.
    3. Selects the top-K most relevant children.
    4. Recurses into those children.
    5. Returns the collected leaf-node texts as context.
    """

    def __init__(self, embedding_engine: Any, top_k_per_level: int = 2):
        self.embedding_engine = embedding_engine
        self.top_k_per_level = top_k_per_level
        self._root: Optional[PageIndexNode] = None
        self._node_embeddings: Dict[str, np.ndarray] = {}  # node_id → emb

    # ── Index the tree ─────────────────────────────────────────────────
    def index_tree(self, root: PageIndexNode) -> None:
        """Pre-compute embeddings for every node in the tree."""
        self._root = root
        self._precompute_embeddings(root)
        logger.info(
            "PageIndex retriever indexed %d nodes", len(self._node_embeddings)
        )

    def _precompute_embeddings(self, node: PageIndexNode) -> None:
        text = node.summary or node.title
        emb = self.embedding_engine.encode_query(text)
        self._node_embeddings[node.node_id] = emb
        for child in node.children:
            self._precompute_embeddings(child)

    # ── Tree search ────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[TreeSearchResult]:
        """
        Traverse the tree top-down, selecting the most relevant
        branches at each level and collecting leaf results.
        """
        if self._root is None:
            return []

        query_emb = self.embedding_engine.encode_query(query)
        results: List[TreeSearchResult] = []
        self._tree_search(
            node=self._root,
            query_emb=query_emb,
            path=[],
            results=results,
            remaining=top_k,
        )
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _tree_search(
        self,
        node: PageIndexNode,
        query_emb: np.ndarray,
        path: List[str],
        results: List[TreeSearchResult],
        remaining: int,
    ) -> None:
        current_path = path + [node.node_id]

        if node.is_leaf():
            score = float(query_emb @ self._node_embeddings[node.node_id])
            results.append(
                TreeSearchResult(node=node, score=score, path=current_path)
            )
            return

        # Score each child
        child_scores: List[Tuple[float, PageIndexNode]] = []
        for child in node.children:
            emb = self._node_embeddings.get(child.node_id)
            if emb is not None:
                score = float(query_emb @ emb)
                child_scores.append((score, child))

        # Select top-K children to explore (sparse routing)
        child_scores.sort(key=lambda x: x[0], reverse=True)
        selected = child_scores[: self.top_k_per_level]

        for _score, child in selected:
            if remaining <= 0:
                break
            before = len(results)
            self._tree_search(child, query_emb, current_path, results, remaining)
            remaining -= len(results) - before

    # ── Pretty-print the tree (for debugging) ──────────────────────────
    def print_tree(self, node: Optional[PageIndexNode] = None, indent: int = 0):
        if node is None:
            node = self._root
        if node is None:
            return
        prefix = "  " * indent
        leaf_mark = "🍃" if node.is_leaf() else "📁"
        pages = f"pp.{node.start_page}-{node.end_page}"
        print(f"{prefix}{leaf_mark} [{node.node_id}] {node.title} ({pages})")
        for child in node.children:
            self.print_tree(child, indent + 1)


# ═══════════════════════════════════════════════════════════════════════
# Convenience: build + index in one call
# ═══════════════════════════════════════════════════════════════════════

def build_page_index(
    sections: List[Dict[str, Any]],
    embedding_engine: Any,
    top_k_per_level: int = 2,
) -> Tuple[PageIndexNode, PageIndexRetriever]:
    """
    One-shot helper: build the tree and index it for retrieval.

    Returns (root_node, retriever).
    """
    builder = PageIndexBuilder()
    root = builder.build_from_sections(sections)

    retriever = PageIndexRetriever(
        embedding_engine=embedding_engine,
        top_k_per_level=top_k_per_level,
    )
    retriever.index_tree(root)

    return root, retriever
