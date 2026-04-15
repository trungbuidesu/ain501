"""RAG Lite: MiniLM + FAISS knowledge retrieval."""

from __future__ import annotations

from src.rag.knowledge_base import KnowledgeBase, load_json_entries
from src.rag.minilm_embedder import MiniLMEmbedder
from src.rag.types import KnowledgeEntry, ScoredEntry

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "MiniLMEmbedder",
    "ScoredEntry",
    "load_json_entries",
]
