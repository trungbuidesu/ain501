"""FAISS-backed knowledge base over MiniLM embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, cast

import faiss  # type: ignore[import-untyped]
import numpy as np

from src.rag.minilm_embedder import MiniLMEmbedder
from src.rag.types import KnowledgeEntry, ScoredEntry


class _EmbedderLike(Protocol):
    """Anything that can produce normalized vectors (MiniLM or test double)."""

    dim: int

    def encode(self, text: str) -> np.ndarray: ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json_entries(path: Path) -> list[KnowledgeEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array")
    out: list[KnowledgeEntry] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        q = str(row.get("query", "")).strip()
        cap = str(row.get("caption_vi", "")).strip()
        if not q or not cap:
            continue
        pr = row.get("priority", 2)
        try:
            priority = int(pr)
        except (TypeError, ValueError):
            priority = 2
        priority = max(0, min(2, priority))
        out.append(
            KnowledgeEntry(
                query=q,
                caption_vi=cap,
                priority=priority,
                source_file=path.name,
            )
        )
    return out


class KnowledgeBase:
    """FAISS IndexFlatIP over L2-normalized MiniLM vectors.

    Loads JSON knowledge from ``knowledge_dir``.
    """

    def __init__(
        self,
        model_path: Path | str,
        knowledge_dir: Path | str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.3,
        embedder: _EmbedderLike | None = None,
    ) -> None:
        self._root = _repo_root()
        self.model_path = Path(model_path)
        self.knowledge_dir = (
            Path(knowledge_dir)
            if Path(knowledge_dir).is_absolute()
            else self._root / knowledge_dir
        )
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self._embedder = cast(
            _EmbedderLike,
            embedder if embedder is not None else MiniLMEmbedder(self.model_path),
        )
        self._entries: list[KnowledgeEntry] = []
        self._index: Any = None
        self._dim: int = 0
        self._build()

    def _build(self) -> None:
        paths = sorted(self.knowledge_dir.glob("*.json"))
        for p in paths:
            self._entries.extend(load_json_entries(p))
        if not self._entries:
            self._index = None
            return
        vectors: list[np.ndarray] = []
        for e in self._entries:
            vectors.append(self._embedder.encode(e.query))
        mat = np.stack(vectors, axis=0).astype(np.float32)
        self._dim = int(mat.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(mat)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def search(self, text: str) -> list[ScoredEntry] | None:
        """Return top-k scored entries, or None if best score is below threshold."""

        if not text.strip() or self._index is None or not self._entries:
            return None
        q = self._embedder.encode(text).reshape(1, -1).astype(np.float32)
        scores, idxs = self._index.search(q, min(self.top_k, len(self._entries)))
        scores_flat = scores[0]
        idxs_flat = idxs[0]
        best = float(scores_flat[0])
        if best < self.min_similarity:
            return None
        hits: list[ScoredEntry] = []
        for i in range(len(scores_flat)):
            if idxs_flat[i] < 0:
                continue
            sc = float(scores_flat[i])
            if sc < self.min_similarity:
                break
            hits.append(ScoredEntry(entry=self._entries[int(idxs_flat[i])], score=sc))
        return hits if hits else None
