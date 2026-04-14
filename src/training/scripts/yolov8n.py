# -*- coding: utf-8 -*-
"""Code-first YOLOv8n CLI for dataset YAMLs, dry-runs, and guarded execution."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

DEFAULT_CONFIG = Path("configs/models/yolov8n.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config mapping."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a config section as a mapping."""

    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _path_from(config: dict[str, Any], section: str, key: str, fallback: str) -> Path:
    """Read a path-like config setting."""

    return Path(str(_section(config, section).get(key, fallback)))


def architecture_summary() -> dict[str, Any]:
    """Return a YOLOv8n architecture summary for the project checklist."""

    return {
        "model": "yolov8n",
        "source": "Ultralytics YOLOv8 nano checkpoint",
        "backbone": {
            "family": "CSPDarknet-style",
            "blocks": ["Conv", "C2f", "SPPF"],
            "notes": "nano depth/width scaling with C2f feature extraction",
        },
        "neck": {
            "family": "PAN-FPN",
            "notes": "top-down and bottom-up multi-scale feature fusion",
        },
        "head": {
            "type": "decoupled anchor-free detection head",
            "outputs": ["box regression", "class logits", "distribution focal bins"],
        },
        "detection": {
            "anchor_free": True,
            "loss_components": ["box", "cls", "dfl"],
            "postprocess": "confidence filtering followed by NMS",
        },
    }


def read_class_names(classes_path: Path) -> list[str]:
    """Read class names from a newline-delimited `classes.txt` file."""

    if not classes_path.exists():
        raise ValueError(f"missing classes file: {classes_path}")
    names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"classes file is empty: {classes_path}")
    return names


def build_data_yaml(dataset_root: Path, class_names: Sequence[str]) -> dict[str, Any]:
    """Build an Ultralytics data.yaml payload for a canonical YOLO dataset root."""

    payload: dict[str, Any] = {
        "path": str(dataset_root),
        "train": split_images_value(dataset_root, "train", fallback="val"),
        "val": split_images_value(dataset_root, "val", fallback="train"),
        "nc": len(class_names),
        "names": list(class_names),
    }
    if (dataset_root / "images" / "test").exists():
        payload["test"] = "images/test"
    return payload


def split_images_value(dataset_root: Path, split: str, fallback: str) -> str:
    """Return an images/<split> path, falling back when the split is absent."""

    requested = dataset_root / "images" / split
    if requested.exists():
        return f"images/{split}"
    fallback_path = dataset_root / "images" / fallback
    if fallback_path.exists():
        return f"images/{fallback}"
    return f"images/{split}"


def write_data_yaml(
    dataset_root: Path,
    classes_path: Path,
    output: Path,
    execute: bool,
) -> dict[str, Any]:
    """Create or dry-run an Ultralytics data.yaml payload."""

    class_names = read_class_names(classes_path)
    payload = build_data_yaml(dataset_root, class_names)
    report = {
        "dry_run": not execute,
        "output": str(output),
        "dataset_root": str(dataset_root),
        "data": payload,
    }
    if execute:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return report


def training_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return normalized YOLOv8n training defaults."""

    training = _section(config, "training")
    return {
        "imgsz": int(training.get("imgsz", 640)),
        "epochs": int(training.get("epochs", 100)),
        "batch": int(training.get("batch", 16)),
        "lr0": float(training.get("lr0", 0.01)),
        "mosaic": float(training.get("mosaic", 1.0)),
        "mixup": float(training.get("mixup", 0.1)),
        "multi_scale": bool(training.get("multi_scale", True)),
    }


def train_plan(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Build the Ultralytics training invocation as JSON-serializable data."""

    dataset = _section(config, "dataset")
    model = _section(config, "model")
    outputs = _section(config, "output")
    defaults = training_defaults(config)
    plan = {
        "weights": args.weights or str(model.get("weights", "yolov8n.pt")),
        "data": str(args.data_yaml or dataset.get("data_yaml", "")),
        "epochs": args.epochs or defaults["epochs"],
        "batch": args.batch or defaults["batch"],
        "imgsz": args.imgsz or defaults["imgsz"],
        "lr0": args.lr0 if args.lr0 is not None else defaults["lr0"],
        "mosaic": args.mosaic if args.mosaic is not None else defaults["mosaic"],
        "mixup": args.mixup if args.mixup is not None else defaults["mixup"],
        "multi_scale": (
            args.multi_scale
            if args.multi_scale is not None
            else defaults["multi_scale"]
        ),
        "device": args.device,
        "project": str(args.project or outputs.get("run_dir", "runs/yolov8n")),
        "name": args.name,
    }
    if not plan["data"]:
        raise ValueError("missing data yaml path")
    return plan


def evaluation_plan(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Build the Ultralytics validation invocation as JSON-serializable data."""

    dataset = _section(config, "dataset")
    model = _section(config, "model")
    outputs = _section(config, "output")
    defaults = training_defaults(config)
    data_yaml = args.data_yaml or dataset.get("data_yaml", "")
    if not data_yaml:
        raise ValueError("missing data yaml path")
    return {
        "weights": args.weights or str(model.get("weights", "yolov8n.pt")),
        "data": str(data_yaml),
        "batch": args.batch or defaults["batch"],
        "imgsz": args.imgsz or defaults["imgsz"],
        "device": args.device,
        "project": str(args.project or outputs.get("report_dir", "reports/yolov8n")),
        "name": args.name,
    }


def prediction_plan(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Build the small-sample prediction/error-analysis plan."""

    dataset_root = _path_from(
        config,
        "dataset",
        "root",
        "data/processed/object_detection_accessibility_merged",
    )
    model = _section(config, "model")
    outputs = _section(config, "output")
    source = args.source or (dataset_root / "images" / "val")
    output_csv = args.output_csv or Path(
        str(Path(str(outputs.get("report_dir", "reports/yolov8n"))) / "errors.csv")
    )
    return {
        "weights": str(args.weights or model.get("weights", "yolov8n.pt")),
        "source": str(source),
        "output_csv": str(output_csv),
        "max_samples": args.max_samples,
        "conf": args.conf,
        "project": str(args.project or outputs.get("report_dir", "reports/yolov8n")),
        "name": args.name,
    }


def ablation_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a small dry-run-safe ablation matrix."""

    defaults = training_defaults(config)
    return [
        {"variant": "baseline", **defaults},
        {"variant": "no_mosaic", **defaults, "mosaic": 0.0},
        {"variant": "no_mixup", **defaults, "mixup": 0.0},
        {"variant": "lower_lr", **defaults, "lr0": 0.001},
    ]


def import_ultralytics_yolo() -> Any:
    """Import Ultralytics lazily so dry-run tests do not require runtime setup."""

    from ultralytics import YOLO  # type: ignore[import-untyped]

    return YOLO


def execute_train(plan: dict[str, Any]) -> None:
    """Run one Ultralytics YOLO training command."""

    yolo_class = import_ultralytics_yolo()
    model = yolo_class(plan["weights"])
    model.train(
        data=plan["data"],
        epochs=plan["epochs"],
        batch=plan["batch"],
        imgsz=plan["imgsz"],
        lr0=plan["lr0"],
        mosaic=plan["mosaic"],
        mixup=plan["mixup"],
        multi_scale=plan["multi_scale"],
        device=plan["device"],
        project=plan["project"],
        name=plan["name"],
    )


def execute_evaluate(plan: dict[str, Any]) -> None:
    """Run one Ultralytics YOLO validation command."""

    yolo_class = import_ultralytics_yolo()
    model = yolo_class(plan["weights"])
    model.val(
        data=plan["data"],
        batch=plan["batch"],
        imgsz=plan["imgsz"],
        device=plan["device"],
        project=plan["project"],
        name=plan["name"],
    )


def execute_predict_errors(plan: dict[str, Any]) -> None:
    """Run predictions on a small validation sample and write a CSV summary."""

    source = Path(str(plan["source"]))
    images = prediction_images(source, int(plan["max_samples"]))
    if not images:
        raise ValueError(f"no prediction images found: {source}")
    yolo_class = import_ultralytics_yolo()
    model = yolo_class(plan["weights"])
    results = model.predict(
        source=[str(path) for path in images],
        conf=plan["conf"],
        save=True,
        project=plan["project"],
        name=plan["name"],
    )
    output_csv = Path(str(plan["output_csv"]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "predicted_classes",
                "confidences",
                "output_artifact_path",
            ],
        )
        writer.writeheader()
        for image_path, result in zip(images, results, strict=True):
            boxes = getattr(result, "boxes", None)
            class_values = boxes.cls.tolist() if boxes is not None else []
            confidence_values = boxes.conf.tolist() if boxes is not None else []
            writer.writerow(
                {
                    "image_path": str(image_path),
                    "predicted_classes": " ".join(
                        str(int(value)) for value in class_values
                    ),
                    "confidences": " ".join(
                        f"{float(value):.6f}" for value in confidence_values
                    ),
                    "output_artifact_path": str(getattr(result, "save_dir", "")),
                }
            )


def prediction_images(source: Path, max_samples: int) -> list[Path]:
    """Return a deterministic small set of image paths for prediction."""

    if max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if not source.exists():
        return []
    return [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ][:max_samples]


def qat_guard_report() -> dict[str, Any]:
    """Return the Phase 0 QAT guard decision."""

    return {
        "supported": False,
        "method": "torch.quantization.prepare_qat on Ultralytics YOLOv8n",
        "reason": (
            "Direct eager-mode prepare_qat is not treated as supported for "
            "Ultralytics YOLOv8n in Phase 0 without a proven compatibility path."
        ),
        "action": "Keep true INT8 QAT comparison open; do not fabricate fake-QAT runs.",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the YOLOv8n CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("describe")

    data_yaml = subparsers.add_parser("write-data-yaml")
    data_yaml.add_argument("--dataset-root", type=Path)
    data_yaml.add_argument("--classes", type=Path)
    data_yaml.add_argument("--output", type=Path)
    data_yaml.add_argument("--execute", action="store_true")

    train = subparsers.add_parser("train")
    add_train_like_args(train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--data-yaml", type=Path)
    evaluate.add_argument("--weights")
    evaluate.add_argument("--batch", type=int)
    evaluate.add_argument("--imgsz", type=int)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--project", type=Path)
    evaluate.add_argument("--name", default="eval")
    evaluate.add_argument("--execute", action="store_true")

    predict = subparsers.add_parser("predict-errors")
    predict.add_argument("--source", type=Path)
    predict.add_argument("--weights")
    predict.add_argument("--output-csv", type=Path)
    predict.add_argument("--max-samples", type=int, default=25)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--project", type=Path)
    predict.add_argument("--name", default="error_analysis")
    predict.add_argument("--execute", action="store_true")

    ablate = subparsers.add_parser("ablate")
    add_train_like_args(ablate)
    ablate.add_argument("--matrix-output", type=Path)

    subparsers.add_parser("qat")
    return parser


def add_train_like_args(parser: argparse.ArgumentParser) -> None:
    """Add common training arguments to a subparser."""

    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--weights")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--mixup", type=float)
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name", default="train")
    parser.add_argument("--execute", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the YOLOv8n CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "describe":
        print(json.dumps(architecture_summary(), indent=2))
        return 0

    if args.command == "write-data-yaml":
        dataset_root = args.dataset_root or _path_from(
            config,
            "dataset",
            "root",
            "data/processed/object_detection_accessibility_merged",
        )
        classes_path = args.classes or _path_from(
            config,
            "dataset",
            "classes",
            "data/processed/object_detection_accessibility_merged/classes.txt",
        )
        output = args.output or _path_from(
            config,
            "dataset",
            "data_yaml",
            "data/processed/object_detection_accessibility_merged/yolov8n_data.yaml",
        )
        print(
            json.dumps(
                write_data_yaml(dataset_root, classes_path, output, args.execute),
                indent=2,
            )
        )
        return 0

    if args.command == "train":
        plan = train_plan(args, config)
        if args.execute:
            execute_train(plan)
        print(json.dumps({"dry_run": not args.execute, "train": plan}, indent=2))
        return 0

    if args.command == "evaluate":
        plan = evaluation_plan(args, config)
        if args.execute:
            execute_evaluate(plan)
        print(json.dumps({"dry_run": not args.execute, "evaluate": plan}, indent=2))
        return 0

    if args.command == "predict-errors":
        plan = prediction_plan(args, config)
        if args.execute:
            execute_predict_errors(plan)
        print(
            json.dumps({"dry_run": not args.execute, "predict_errors": plan}, indent=2)
        )
        return 0

    if args.command == "ablate":
        matrix = ablation_matrix(config)
        plans = []
        for variant in matrix:
            variant_args = argparse.Namespace(**vars(args))
            variant_args.name = f"{args.name}_{variant['variant']}"
            variant_args.lr0 = variant["lr0"]
            variant_args.mosaic = variant["mosaic"]
            variant_args.mixup = variant["mixup"]
            variant_args.multi_scale = variant["multi_scale"]
            plans.append(
                {
                    "variant": variant["variant"],
                    "train": train_plan(variant_args, config),
                }
            )
        report = {"dry_run": not args.execute, "ablation": plans}
        if args.matrix_output:
            args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        if args.execute:
            for plan in plans:
                execute_train(plan["train"])
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "qat":
        print(json.dumps(qat_guard_report(), indent=2))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
