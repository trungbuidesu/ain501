"""Types for RAG knowledge entries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """One row from knowledge JSON (indexed by English `query` text)."""

    query: str
    caption_vi: str
    priority: int  # 0=hazard, 1=navigation, 2=info
    source_file: str


@dataclass(frozen=True, slots=True)
class ScoredEntry:
    """Retrieval hit with cosine similarity (inner product on L2-normalized vectors)."""

    entry: KnowledgeEntry
    score: float
