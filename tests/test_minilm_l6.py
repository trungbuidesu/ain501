# -*- coding: utf-8 -*-
"""Tests for MiniLM-L6 text embedding utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.training.scripts import minilm_l6


class _FakeSentenceModel:
    """Small fake model that returns deterministic normalized vectors."""

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        vectors = np.stack([self._vector_for_text(text) for text in texts], axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms
        if convert_to_numpy:
            return vectors.astype(np.float32)
        raise ValueError("fake model supports convert_to_numpy=True only")

    def _vector_for_text(self, text: str) -> np.ndarray:
        key = text.casefold()
        if "clear" in key or "walkway" in key:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "stairs" in key or "steps" in key:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if "crosswalk" in key or "crossing" in key:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.2, 0.2, 0.2], dtype=np.float32)


def test_generate_template_pairs_and_split() -> None:
    """Template builder creates rows and split assignment keeps all splits."""

    rows = minilm_l6.generate_template_pairs(seed=501)
    assert len(rows) > 0
    assert all(row.anchor and row.positive and row.negative for row in rows)
    split_rows = minilm_l6.assign_splits(rows, train_ratio=0.8, val_ratio=0.1, seed=501)
    splits = {row.split for row in split_rows}
    assert splits == {"train", "val", "test"}


def test_write_read_pairs_csv_roundtrip(tmp_path: Path) -> None:
    """Pairs CSV writer/reader preserves row content."""

    rows = minilm_l6.assign_splits(
        minilm_l6.generate_template_pairs(seed=123),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=123,
    )
    path = tmp_path / "pairs.csv"
    minilm_l6.write_pairs_csv(path, rows)
    loaded = minilm_l6.read_pairs_csv(path)
    assert len(loaded) == len(rows)
    assert loaded[0].anchor == rows[0].anchor
    assert loaded[0].split == rows[0].split


def test_evaluate_retrieval_metrics_with_fake_model() -> None:
    """Retrieval reports strong metrics for coherent synthetic examples."""

    rows = [
        minilm_l6.PairRow(
            anchor="The path is clear ahead.",
            positive="The walkway in front is clear.",
            negative="Stairs are right in front.",
            source="navigation",
            split="val",
        ),
        minilm_l6.PairRow(
            anchor="Stairs are visible.",
            positive="A staircase is visible ahead.",
            negative="The crosswalk lines are bright.",
            source="stairs",
            split="val",
        ),
    ]
    report = minilm_l6.evaluate_retrieval(_FakeSentenceModel(), rows, split="val", k=1)  # type: ignore[arg-type]
    assert report["queries"] == 2
    assert report["recall_at_k"] >= 0.5
    assert report["mrr"] > 0.0


def test_build_pairs_dataset_command_writes_file(tmp_path: Path) -> None:
    """build-pairs command helper writes output CSV and reports split counts."""

    output_csv = tmp_path / "minilm_pairs.csv"
    args = argparse.Namespace(
        output_csv=output_csv,
        seed=501,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    report = minilm_l6.build_pairs_dataset(args)
    assert output_csv.exists()
    assert report["rows"] > 0
    assert set(report["split_counts"]) == {"train", "val", "test"}
