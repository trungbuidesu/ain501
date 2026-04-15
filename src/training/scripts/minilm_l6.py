"""CLI for all-MiniLM-L6-v2 embedding, fine-tuning, ONNX, and benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml  # type: ignore[import-untyped]
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
)
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
    TripletLoss,
)
from torch.utils.data import DataLoader

from src.utils import TrainingLogger

DEFAULT_CONFIG = Path("configs/models/minilm_l6.yaml")
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PAIRS_CSV = Path("data/processed/text_embedding/minilm_pairs.csv")
DEFAULT_ONNX_PATH = Path("models/minilm_l6.onnx")


@dataclass(frozen=True)
class PairRow:
    """Training/evaluation row with one positive and one negative pair."""

    anchor: str
    positive: str
    negative: str
    source: str
    split: str


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config mapping."""

    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def _config_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section, {})
    return value if isinstance(value, dict) else {}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build parser for MiniLM CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe")
    describe.add_argument("--model-path", default=DEFAULT_MODEL_ID)

    build_pairs = subparsers.add_parser("build-pairs")
    build_pairs.add_argument("--output-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    build_pairs.add_argument("--seed", type=int, default=501)
    build_pairs.add_argument("--train-ratio", type=float, default=0.8)
    build_pairs.add_argument("--val-ratio", type=float, default=0.1)

    similarity = subparsers.add_parser("evaluate-similarity")
    similarity.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    similarity.add_argument("--output-json", type=Path)

    retrieval = subparsers.add_parser("evaluate-retrieval")
    retrieval.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    retrieval.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    retrieval.add_argument("--split", choices=["train", "val", "test"], default="val")
    retrieval.add_argument("--k", type=int, default=5)
    retrieval.add_argument("--output-json", type=Path)

    fine_tune = subparsers.add_parser("fine-tune")
    fine_tune.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    fine_tune.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    fine_tune.add_argument("--objective", choices=["infonce", "triplet"])
    fine_tune.add_argument("--epochs", type=int)
    fine_tune.add_argument("--batch-size", type=int)
    fine_tune.add_argument("--lr", type=float)
    fine_tune.add_argument("--warmup-ratio", type=float)
    fine_tune.add_argument("--weight-decay", type=float)
    fine_tune.add_argument("--margin", type=float)
    fine_tune.add_argument("--seed", type=int)
    fine_tune.add_argument("--run-name")
    fine_tune.add_argument("--checkpoint-dir", type=Path)
    fine_tune.add_argument("--log-dir", type=Path)
    fine_tune.add_argument("--report-dir", type=Path)
    fine_tune.add_argument("--k", type=int, default=5)

    export = subparsers.add_parser("export-onnx")
    export.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    export.add_argument("--output", type=Path, default=DEFAULT_ONNX_PATH)
    export.add_argument("--max-seq-length", type=int)
    export.add_argument("--normalize-embeddings", action="store_true")
    export.add_argument("--opset", type=int, default=17)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--runtime", choices=["pytorch", "onnxruntime"])
    benchmark.add_argument("--model-path", default=DEFAULT_MODEL_ID)
    benchmark.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--warmup", type=int)
    benchmark.add_argument("--iterations", type=int)
    benchmark.add_argument("--max-seq-length", type=int)
    benchmark.add_argument("--profile-name")
    benchmark.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "describe":
        print(json.dumps(describe_model(args.model_path), indent=2))
        return 0
    if args.command == "build-pairs":
        report = build_pairs_dataset(args)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "evaluate-similarity":
        report = evaluate_similarity(args.model_path)
        _write_json_if_requested(report, args.output_json)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "evaluate-retrieval":
        rows = read_pairs_csv(args.pairs_csv)
        model = SentenceTransformer(args.model_path)
        report = evaluate_retrieval(model, rows, split=args.split, k=args.k)
        _write_json_if_requested(report, args.output_json)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "fine-tune":
        report = fine_tune_model(args, config)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "export-onnx":
        report = export_onnx_model(args, config)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "benchmark":
        report = benchmark_model(args, config)
        _write_json_if_requested(report, args.output_json)
        print(json.dumps(report, indent=2))
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


def _write_json_if_requested(payload: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def describe_model(model_path: str) -> dict[str, Any]:
    """Describe all-MiniLM-L6 architecture and pooling strategy."""

    model = SentenceTransformer(model_path)
    first_module = model._first_module()
    auto_model = getattr(first_module, "auto_model", None)
    tokenizer = getattr(first_module, "tokenizer", None)
    pooling_module = model._last_module()
    hidden_size = (
        int(getattr(auto_model.config, "hidden_size", 384))
        if auto_model
        else 384
    )
    num_layers = (
        int(getattr(auto_model.config, "num_hidden_layers", 6))
        if auto_model
        else 6
    )
    max_seq_length = int(getattr(model, "max_seq_length", 128))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) if tokenizer else 0
    return {
        "model_path": model_path,
        "backbone_family": "MiniLM (distilled BERT-family encoder)",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "max_seq_length": max_seq_length,
        "pooling": {
            "strategy": "mean_pooling",
            "normalize_embeddings_default": True,
            "module": type(pooling_module).__name__,
        },
        "vocab_size": vocab_size,
    }


def build_pairs_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Create positive/negative pairs from caption templates."""

    if (
        args.train_ratio <= 0.0
        or args.val_ratio < 0.0
        or args.train_ratio + args.val_ratio >= 1.0
    ):
        raise ValueError("ratios must satisfy train > 0, val >= 0, train + val < 1")
    rows = generate_template_pairs(args.seed)
    split_rows = assign_splits(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_pairs_csv(args.output_csv, split_rows)
    split_counts: dict[str, int] = {}
    for row in split_rows:
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
    return {
        "output_csv": str(args.output_csv),
        "rows": len(split_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "sources": sorted({row.source for row in split_rows}),
    }


def generate_template_pairs(seed: int) -> list[PairRow]:
    """Generate deterministic caption-template positives and negatives."""

    rng = random.Random(seed)
    templates = {
        "navigation": [
            "The sidewalk ahead is clear and wide.",
            "The walkway in front is unobstructed.",
            "There is a clear pedestrian path ahead.",
            "The path is free of obstacles and easy to walk.",
        ],
        "stairs": [
            "A staircase is directly in front of the user.",
            "Steps are visible ahead and go upward.",
            "The user is approaching a flight of stairs.",
            "There are stairs right in front of the camera.",
        ],
        "crosswalk": [
            "A crosswalk is visible at the intersection.",
            "Pedestrian crossing lines appear ahead.",
            "The scene shows a marked crossing area.",
            "Crosswalk stripes are clearly visible.",
        ],
        "indoor_text": [
            "A sign with text is visible on the wall.",
            "There is readable text in this indoor scene.",
            "The camera captures a board with printed words.",
            "An indoor sign contains visible writing.",
        ],
    }
    sources = list(templates.keys())
    rows: list[PairRow] = []
    for source_index, source in enumerate(sources):
        sentences = templates[source]
        negative_source = sources[(source_index + 1) % len(sources)]
        negatives = templates[negative_source]
        for index, anchor in enumerate(sentences):
            positive = sentences[(index + 1) % len(sentences)]
            negative = negatives[index % len(negatives)]
            rows.append(
                PairRow(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    source=source,
                    split="train",
                )
            )
    rng.shuffle(rows)
    return rows


def assign_splits(
    rows: list[PairRow],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[PairRow]:
    """Assign train/val/test splits in a source-balanced manner."""

    rng = random.Random(seed)
    grouped: dict[str, list[PairRow]] = {}
    for row in rows:
        grouped.setdefault(row.source, []).append(row)
    output: list[PairRow] = []
    for source in sorted(grouped):
        bucket = list(grouped[source])
        rng.shuffle(bucket)
        split_names = _split_names(
            len(bucket),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        output.extend(
            PairRow(
                anchor=row.anchor,
                positive=row.positive,
                negative=row.negative,
                source=row.source,
                split=split_name,
            )
            for row, split_name in zip(bucket, split_names, strict=True)
        )
    return output


def _split_names(total: int, *, train_ratio: float, val_ratio: float) -> list[str]:
    """Return deterministic split names for a bucket."""

    if total <= 0:
        return []
    test_ratio = 1.0 - train_ratio - val_ratio
    targets = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [math.floor(target) for target in targets]
    remainder = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (targets[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    if total >= 3:
        for index in range(3):
            if counts[index] == 0 and targets[index] > 0:
                donor = max(range(3), key=lambda donor_index: counts[donor_index])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return ["train"] * counts[0] + ["val"] * counts[1] + ["test"] * counts[2]


def write_pairs_csv(path: Path, rows: list[PairRow]) -> None:
    """Write pair rows to CSV."""

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["anchor", "positive", "negative", "source", "split"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_pairs_csv(path: Path) -> list[PairRow]:
    """Read pair CSV and validate required fields."""

    if not path.exists():
        raise ValueError(f"pairs csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        raw_rows = list(csv.DictReader(file))
    required = {"anchor", "positive", "negative", "source", "split"}
    rows: list[PairRow] = []
    for index, row in enumerate(raw_rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {index} missing columns: {sorted(missing)}")
        rows.append(
            PairRow(
                anchor=str(row["anchor"]),
                positive=str(row["positive"]),
                negative=str(row["negative"]),
                source=str(row["source"]),
                split=str(row["split"]),
            )
        )
    return rows


def evaluate_similarity(model_path: str) -> dict[str, Any]:
    """Evaluate cosine similarity on curated similar/dissimilar sentence pairs."""

    model = SentenceTransformer(model_path)
    probes = [
        ("The path ahead is clear.", "The walkway in front is unobstructed.", 1),
        ("A staircase is ahead.", "Steps are visible in front.", 1),
        ("Crosswalk lines are visible.", "A pedestrian crossing can be seen.", 1),
        ("The path ahead is clear.", "A staircase is directly ahead.", 0),
        ("A sign with text is visible.", "The room is empty with no sign.", 0),
        ("Crosswalk lines are visible.", "The elevator doors are closed.", 0),
    ]
    similarities: list[dict[str, Any]] = []
    positives: list[float] = []
    negatives: list[float] = []
    for anchor, other, label in probes:
        embeddings = model.encode(
            [anchor, other],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        score = float(np.dot(embeddings[0], embeddings[1]))
        similarities.append(
            {
                "anchor": anchor,
                "other": other,
                "label": label,
                "cosine_similarity": score,
            }
        )
        if label == 1:
            positives.append(score)
        else:
            negatives.append(score)
    avg_pos = float(np.mean(positives)) if positives else float("nan")
    avg_neg = float(np.mean(negatives)) if negatives else float("nan")
    return {
        "model_path": model_path,
        "samples": similarities,
        "avg_positive_similarity": avg_pos,
        "avg_negative_similarity": avg_neg,
        "similarity_gap": avg_pos - avg_neg,
        "quality_check_passed": avg_pos > avg_neg,
    }


def fine_tune_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune MiniLM with InfoNCE or Triplet objective."""

    training_cfg = _config_section(config, "training")
    output_cfg = _config_section(config, "output")
    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 501))
    set_random_seed(seed)
    rows = read_pairs_csv(args.pairs_csv)
    train_rows = [row for row in rows if row.split == "train"]
    val_rows = [row for row in rows if row.split == "val"]
    if not train_rows:
        raise ValueError("pairs csv must contain train split rows")
    if not val_rows:
        raise ValueError("pairs csv must contain val split rows")

    objective = str(args.objective or training_cfg.get("objective", "infonce")).lower()
    epochs = int(
        args.epochs
        if args.epochs is not None
        else training_cfg.get("epochs", 8)
    )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else training_cfg.get("batch_size", 32)
    )
    lr = float(args.lr if args.lr is not None else training_cfg.get("lr", 2e-5))
    warmup_ratio = float(
        args.warmup_ratio
        if args.warmup_ratio is not None
        else training_cfg.get("warmup_ratio", 0.1)
    )
    weight_decay = float(
        args.weight_decay
        if args.weight_decay is not None
        else training_cfg.get("weight_decay", 0.01)
    )
    margin = float(
        args.margin
        if args.margin is not None
        else training_cfg.get("margin", 0.2)
    )
    run_name = args.run_name or time.strftime("minilm_l6_%Y%m%d_%H%M%S")
    checkpoint_dir = Path(
        str(
            args.checkpoint_dir
            or output_cfg.get("checkpoint_dir", "models/text/minilm_l6")
        )
    )
    log_dir = Path(str(args.log_dir or output_cfg.get("log_dir", "runs/minilm_l6")))
    report_dir = Path(
        str(args.report_dir or output_cfg.get("report_dir", "reports/minilm_l6"))
    )
    output_path = checkpoint_dir / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model_path)
    baseline_retrieval = evaluate_retrieval(model, rows, split="val", k=args.k)

    train_examples: list[InputExample]
    if objective == "triplet":
        train_examples = [
            InputExample(texts=[row.anchor, row.positive, row.negative])
            for row in train_rows
        ]
    else:
        train_examples = [
            InputExample(texts=[row.anchor, row.positive])
            for row in train_rows
        ]
        objective = "infonce"

    train_loader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
    if objective == "triplet":
        train_loss = TripletLoss(model=model, triplet_margin=margin)
    else:
        train_loss = MultipleNegativesRankingLoss(model=model)

    evaluator = build_similarity_evaluator(val_rows)
    warmup_steps = int(len(train_loader) * epochs * warmup_ratio)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_path),
        optimizer_params={"lr": lr},
        weight_decay=weight_decay,
        show_progress_bar=True,
    )

    trained_model = SentenceTransformer(str(output_path))
    final_retrieval = evaluate_retrieval(trained_model, rows, split="val", k=args.k)
    summary = {
        "run_name": run_name,
        "model_path": args.model_path,
        "output_path": str(output_path),
        "objective": objective,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "margin": margin,
        "baseline_retrieval": baseline_retrieval,
        "final_retrieval": final_retrieval,
        "retrieval_delta_recall_at_k": (
            final_retrieval["recall_at_k"] - baseline_retrieval["recall_at_k"]
        ),
        "retrieval_delta_mrr": final_retrieval["mrr"] - baseline_retrieval["mrr"],
    }
    summary_path = report_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with TrainingLogger(
        project="ain501",
        run_name=run_name,
        backends=["tensorboard"],
        log_dir=str(log_dir),
        config={
            "objective": objective,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup_steps": warmup_steps,
        },
    ) as logger:
        logger.log_scalar(
            "retrieval/recall_before",
            float(baseline_retrieval["recall_at_k"]),
            step=0,
        )
        logger.log_scalar(
            "retrieval/mrr_before",
            float(baseline_retrieval["mrr"]),
            step=0,
        )
        logger.log_scalar(
            "retrieval/recall_after",
            float(final_retrieval["recall_at_k"]),
            step=epochs,
        )
        logger.log_scalar(
            "retrieval/mrr_after",
            float(final_retrieval["mrr"]),
            step=epochs,
        )
    summary["summary_json"] = str(summary_path)
    return summary


def build_similarity_evaluator(rows: list[PairRow]) -> EmbeddingSimilarityEvaluator:
    """Create validation evaluator for cosine similarity agreement."""

    anchors = [row.anchor for row in rows]
    positives = [row.positive for row in rows]
    negatives = [row.negative for row in rows]
    sentence1 = anchors + anchors
    sentence2 = positives + negatives
    scores = [1.0] * len(anchors) + [0.0] * len(anchors)
    return EmbeddingSimilarityEvaluator(sentence1, sentence2, scores)


def evaluate_retrieval(
    model: SentenceTransformer,
    rows: list[PairRow],
    *,
    split: str,
    k: int,
) -> dict[str, Any]:
    """Evaluate retrieval metrics Recall@k and MRR for one split."""

    selected_rows = [row for row in rows if row.split == split]
    if not selected_rows:
        raise ValueError(f"split '{split}' has no rows in pairs dataset")
    candidates = sorted(
        {row.positive for row in selected_rows}
        | {row.negative for row in selected_rows}
    )
    candidate_embeddings = model.encode(
        candidates,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    candidate_matrix = np.asarray(candidate_embeddings, dtype=np.float32)
    candidate_index = {text: idx for idx, text in enumerate(candidates)}
    hits = 0
    reciprocal_ranks: list[float] = []
    for row in selected_rows:
        query_embedding = model.encode(
            [row.anchor],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_vector = np.asarray(query_embedding[0], dtype=np.float32)
        similarities = candidate_matrix @ query_vector
        ranked_indices = np.argsort(-similarities)
        positive_idx = candidate_index[row.positive]
        if positive_idx in ranked_indices[:k]:
            hits += 1
        rank_pos = int(np.where(ranked_indices == positive_idx)[0][0]) + 1
        reciprocal_ranks.append(1.0 / rank_pos)
    recall_at_k = hits / len(selected_rows)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return {
        "split": split,
        "k": k,
        "queries": len(selected_rows),
        "candidate_count": len(candidates),
        "recall_at_k": recall_at_k,
        "mrr": mrr,
    }


class _OnnxEmbeddingWrapper(nn.Module):
    """ONNX export wrapper that applies mean pooling + optional normalization."""

    def __init__(self, auto_model: nn.Module, normalize_embeddings: bool) -> None:
        super().__init__()
        self.auto_model = auto_model
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = outputs.last_hidden_state
        expanded_mask = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        )
        summed = torch.sum(token_embeddings * expanded_mask, dim=1)
        counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        if self.normalize_embeddings:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled


def export_onnx_model(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Export MiniLM encoder to ONNX with mean pooling."""

    model_cfg = _config_section(config, "model")
    output_cfg = _config_section(config, "output")
    normalize = bool(
        args.normalize_embeddings or model_cfg.get("normalize_embeddings", True)
    )
    model = SentenceTransformer(args.model_path)
    first_module = model._first_module()
    auto_model = getattr(first_module, "auto_model", None)
    tokenizer = getattr(first_module, "tokenizer", None)
    if auto_model is None or tokenizer is None:
        raise ValueError(
            "SentenceTransformer model does not expose transformer module/tokenizer"
        )
    # Force eager attention path for stable ONNX tracing on recent transformers.
    if hasattr(auto_model.config, "_attn_implementation"):
        auto_model.config._attn_implementation = "eager"
    max_seq_length = int(
        args.max_seq_length or model_cfg.get("max_seq_length", model.max_seq_length)
    )
    output_path = Path(
        str(args.output or output_cfg.get("onnx_path", "models/minilm_l6.onnx"))
    )
    wrapper = _OnnxEmbeddingWrapper(
        auto_model.eval(),
        normalize_embeddings=normalize,
    ).cpu()
    dummy = tokenizer(
        ["The path ahead is clear."],
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(torch.long)
    attention_mask = dummy["attention_mask"].to(torch.long)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,
    )
    return {
        "model_path": args.model_path,
        "onnx_path": str(output_path),
        "max_seq_length": max_seq_length,
        "normalize_embeddings": normalize,
        "opset": args.opset,
    }


def benchmark_model(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Benchmark PyTorch/SentenceTransformer or ONNXRuntime latency."""

    benchmark_cfg = _config_section(config, "benchmark")
    model_cfg = _config_section(config, "model")
    eval_cfg = _config_section(config, "evaluation")
    runtime = str(args.runtime or benchmark_cfg.get("runtime", "onnxruntime"))
    batch_size = int(args.batch_size or benchmark_cfg.get("batch_size", 1))
    warmup = int(args.warmup or benchmark_cfg.get("warmup", 20))
    iterations = int(args.iterations or benchmark_cfg.get("iterations", 100))
    max_seq_length = int(args.max_seq_length or model_cfg.get("max_seq_length", 128))
    profile_name = str(
        args.profile_name or benchmark_cfg.get("profile_name", "current-cpu")
    )
    target_latency_ms = float(eval_cfg.get("target_latency_ms", 5.0))

    if runtime == "pytorch":
        timings = _benchmark_pytorch(
            model_path=args.model_path,
            batch_size=batch_size,
            warmup=warmup,
            iterations=iterations,
            max_seq_length=max_seq_length,
        )
    else:
        timings = _benchmark_onnxruntime(
            onnx_path=args.onnx_path,
            model_path=args.model_path,
            batch_size=batch_size,
            warmup=warmup,
            iterations=iterations,
            max_seq_length=max_seq_length,
        )

    latency_p50 = statistics.median(timings)
    latency_p95 = percentile(timings, 0.95)
    return {
        "profile_name": profile_name,
        "runtime": runtime,
        "cpu": platform.processor() or platform.machine(),
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_p50": latency_p50,
        "latency_ms_p95": latency_p95,
        "target_latency_ms": target_latency_ms,
        "target_under_5ms": latency_p50 < target_latency_ms,
    }


def _benchmark_pytorch(
    *,
    model_path: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    max_seq_length: int,
) -> list[float]:
    model = SentenceTransformer(model_path)
    texts = ["The path ahead is clear."] * batch_size
    _ = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    timings: list[float] = []
    for _ in range(warmup):
        _ = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    for _ in range(iterations):
        start = time.perf_counter()
        _ = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        batch_ms = (time.perf_counter() - start) * 1000.0
        timings.append(batch_ms / batch_size)
    model.max_seq_length = max_seq_length
    return timings


def _benchmark_onnxruntime(
    *,
    onnx_path: Path,
    model_path: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    max_seq_length: int,
) -> list[float]:
    import onnxruntime as ort

    if not onnx_path.exists():
        raise ValueError(f"onnx model not found: {onnx_path}")
    model = SentenceTransformer(model_path)
    tokenizer = getattr(model._first_module(), "tokenizer", None)
    if tokenizer is None:
        raise ValueError("tokenizer not found on SentenceTransformer first module")
    texts = ["The path ahead is clear."] * batch_size
    encoded = tokenizer(
        texts,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {
        "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
        "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
    }
    for _ in range(warmup):
        _ = session.run(None, inputs)
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, inputs)
        batch_ms = (time.perf_counter() - start) * 1000.0
        timings.append(batch_ms / batch_size)
    return timings


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""

    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def set_random_seed(seed: int) -> None:
    """Set deterministic random seeds for Python, NumPy and torch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    sys.exit(main())
