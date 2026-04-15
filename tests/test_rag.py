"""Tests for RAG knowledge base and caption orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.caption.orchestrator import CaptionOrchestrator, build_rag_query_string
from src.caption.template_engine import SpatialTemplateEngine
from src.rag.knowledge_base import KnowledgeBase, load_json_entries
from src.rag.minilm_embedder import MiniLMEmbedder


class _DeterministicEmbedder:
    """384-dim L2-normalized vectors from text hash (no ONNX)."""

    dim = 384

    def encode(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v


def test_build_rag_query_string_includes_scene() -> None:
    s = build_rag_query_string(
        [],
        scene_type="urban_street",
        frame_width=100,
        frame_height=100,
    )
    assert "urban street" in s


def test_load_json_entry_count_matches_repo_knowledge() -> None:
    root = Path(__file__).resolve().parents[1]
    kd = root / "data" / "knowledge"
    paths = sorted(kd.glob("*.json"))
    total = 0
    for p in paths:
        total += len(load_json_entries(p))
    assert total == 80


def test_knowledge_base_search_deterministic_embedder(tmp_path: Path) -> None:
    kb_dir = tmp_path / "k"
    kb_dir.mkdir()
    data = [
        {
            "query": "car front close",
            "caption_vi": "Cảnh báo: xe ô tô phía trước, rất gần",
            "priority": 0,
        },
        {"query": "xyzabc nonsense", "caption_vi": "Không dùng", "priority": 2},
    ]
    (kb_dir / "a.json").write_text(json.dumps(data), encoding="utf-8")
    emb = _DeterministicEmbedder()
    kb = KnowledgeBase(
        tmp_path / "m.onnx",
        kb_dir,
        top_k=3,
        min_similarity=0.3,
        embedder=emb,
    )
    assert kb.entry_count == 2
    hits = kb.search("car front close")
    assert hits is not None
    cap0 = hits[0].entry.caption_vi
    assert "xe ô tô" in cap0 or "Cảnh báo" in cap0


def test_knowledge_base_person_dog(tmp_path: Path) -> None:
    kb_dir = tmp_path / "k"
    kb_dir.mkdir()
    data = [
        {
            "query": "person dog leash",
            "caption_vi": "Một người đang dắt chó",
            "priority": 2,
        },
    ]
    (kb_dir / "rel.json").write_text(json.dumps(data), encoding="utf-8")
    kb = KnowledgeBase(
        tmp_path / "m.onnx",
        kb_dir,
        embedder=_DeterministicEmbedder(),
    )
    hits = kb.search("person dog leash")
    assert hits is not None
    assert "dắt chó" in hits[0].entry.caption_vi


def test_orchestrator_rag_disabled_uses_template() -> None:
    eng = SpatialTemplateEngine(empty_policy="silent")
    orch = CaptionOrchestrator(
        template_engine=eng,
        knowledge_base=None,
        rag_enabled=False,
    )
    dets = [
        {
            "label": "person",
            "confidence": 0.9,
            "bbox": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            "count": 1,
        }
    ]
    events = orch.generate(
        dets,
        frame_width=100,
        frame_height=100,
        merge_tier1=False,
    )
    assert len(events) >= 1
    assert orch.describe_sources(events) == ["template"]


@pytest.mark.skipif(
    not Path("models/minilm_l6.onnx").is_file(),
    reason="MiniLM ONNX not present",
)
def test_minilm_embedder_smoke() -> None:
    emb = MiniLMEmbedder("models/minilm_l6.onnx")
    v = emb.encode("test query")
    assert v.shape[0] == 384
    assert abs(float(np.linalg.norm(v)) - 1.0) < 0.01
