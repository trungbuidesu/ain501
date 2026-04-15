"""Evaluate COCO-pretrained YOLO on accessibility val with COCO→subset class remap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]
from PIL import Image

from src.training.eval.yolov8_coco_remap import (
    ImageEval,
    build_coco_id_to_subset_id,
    load_subset_names_from_data_yaml,
    list_val_images,
    map50_macro,
    parse_yolo_gt_label,
    resolve_dataset_root,
    write_eval_json,
    yolo_labels_path,
)
from src.utils.device import resolve_device


def resolve_predict_device(requested: str) -> str:
    if requested.strip().lower() == "auto":
        return str(resolve_device("auto"))
    return requested


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required") from exc
    return YOLO


def run(
    *,
    data_yaml: Path,
    weights: str,
    imgsz: int,
    conf: float,
    device: str,
    batch_predict: int,
    max_images: int | None,
    output_json: Path,
) -> dict[str, Any]:
    device_str = resolve_predict_device(device)
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("data yaml must be a mapping")
    dataset_root = resolve_dataset_root(data_yaml, raw)
    val_rel = str(raw.get("val", "images/val"))
    subset_names = load_subset_names_from_data_yaml(data_yaml)
    nc = len(subset_names)
    coco_to_subset = build_coco_id_to_subset_id(subset_names)

    images = list_val_images(dataset_root, val_rel)
    if max_images is not None:
        images = images[: max(0, max_images)]
    if not images:
        raise ValueError("no validation images found")

    YOLO = import_yolo()
    model = YOLO(weights)

    samples: list[ImageEval] = []
    for start in range(0, len(images), max(1, batch_predict)):
        chunk = images[start : start + batch_predict]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device_str,
            verbose=False,
        )
        for image_path, result in zip(chunk, results, strict=True):
            with Image.open(image_path) as im:
                w, h = im.size
            label_path = yolo_labels_path(Path(image_path), dataset_root, val_rel)
            gt_cls, gt_xyxy = parse_yolo_gt_label(label_path, w, h)

            boxes = getattr(result, "boxes", None)
            pred_xyxy_list: list[list[float]] = []
            pred_conf_list: list[float] = []
            pred_subset_cls: list[int] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cfs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(np.int64)
                for row, cf, cl in zip(xyxy, cfs, clss, strict=True):
                    coco_id = int(cl)
                    if coco_id not in coco_to_subset:
                        continue
                    pred_subset_cls.append(int(coco_to_subset[coco_id]))
                    pred_conf_list.append(float(cf))
                    pred_xyxy_list.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

            if pred_subset_cls:
                p_cls = np.array(pred_subset_cls, dtype=np.int64)
                p_conf = np.array(pred_conf_list, dtype=np.float64)
                p_box = np.array(pred_xyxy_list, dtype=np.float64)
            else:
                p_cls = np.zeros((0,), dtype=np.int64)
                p_conf = np.zeros((0,), dtype=np.float64)
                p_box = np.zeros((0, 4), dtype=np.float64)

            samples.append(
                ImageEval(
                    gt_cls=gt_cls,
                    gt_xyxy=gt_xyxy,
                    pred_cls=p_cls,
                    pred_conf=p_conf,
                    pred_xyxy=p_box,
                )
            )

    m_ap50, per_ap, per_ngt = map50_macro(samples, nc, iou_threshold=0.5)
    write_eval_json(
        output_json,
        m_ap50=m_ap50,
        per_class=per_ap,
        n_gt_per_class=per_ngt,
        subset_names=subset_names,
        coco_to_subset=coco_to_subset,
        weights=weights,
        data_yaml=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        device=device_str,
        num_images=len(images),
    )
    return {
        "mAP50": m_ap50,
        "path": str(output_json),
        "num_images": len(images),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data/processed/object_detection_accessibility_merged/yolov8n_data.yaml"),
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Min confidence for predictions (val-style; use low value for mAP)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device string: auto (Arc XPU > CUDA > CPU), cpu, cuda, xpu, 0, ...",
    )
    parser.add_argument("--batch-predict", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/yolov8n/accessibility_coco_remap_eval.json"),
    )
    args = parser.parse_args(argv)
    try:
        summary = run(
            data_yaml=args.data_yaml,
            weights=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            batch_predict=args.batch_predict,
            max_images=args.max_images,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
