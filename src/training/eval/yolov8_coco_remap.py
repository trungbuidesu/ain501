"""COCO (Ultralytics 80-class) → accessibility subset class remap and mAP50."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml  # type: ignore[import-untyped]

# Order must match Ultralytics `coco.yaml` / YOLOv8 COCO pretrained head.
COCO80_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def coco80_index(name: str) -> int:
    """Return YOLO/COCO class index for a COCO category name."""

    try:
        return COCO80_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown COCO class name: {name!r}") from exc


def build_coco_id_to_subset_id(subset_names: list[str]) -> dict[int, int]:
    """Map pretrained COCO class id → local subset id (0..nc-1) by *name* match."""

    out: dict[int, int] = {}
    for subset_id, name in enumerate(subset_names):
        coco_id = coco80_index(name)
        if coco_id in out and out[coco_id] != subset_id:
            raise ValueError(f"COCO id {coco_id} mapped twice")
        out[coco_id] = subset_id
    return out


def load_subset_names_from_data_yaml(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    names = data.get("names")
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"missing names: list[str] in {path}")
    return list(names)


def resolve_dataset_root(data_yaml: Path, raw: dict[str, Any], *, cwd: Path | None = None) -> Path:
    """Resolve `path` like Ultralytics: relative to current working directory (repo root)."""

    root = raw.get("path")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("data yaml must set 'path' to dataset root")
    p = Path(root)
    if p.is_absolute():
        return p.resolve()
    base = cwd if cwd is not None else Path.cwd()
    return (base / p).resolve()


def list_val_images(dataset_root: Path, val_rel: str) -> list[Path]:
    val_dir = (dataset_root / val_rel).resolve()
    if not val_dir.is_dir():
        raise ValueError(f"val image dir missing: {val_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: list[Path] = []
    for p in val_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def yolo_labels_path(image_path: Path, dataset_root: Path, val_rel: str) -> Path:
    """Mirror `images/.../x.jpg` → `labels/.../x.txt` under the same split."""

    val_img_root = (dataset_root / val_rel).resolve()
    rel = image_path.relative_to(val_img_root)
    parts = Path(val_rel).parts
    if parts and parts[0] == "images":
        label_parts = ("labels",) + parts[1:]
    else:
        label_parts = ("labels", parts[-1] if parts else "val")
    return (dataset_root.joinpath(*label_parts) / rel.with_suffix(".txt")).resolve()


def parse_yolo_gt_label(path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cls_ids int (n,), boxes_xyxy (n,4)) in pixel coords; skip invalid lines."""

    if not path.is_file():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    cls_list: list[int] = []
    box_list: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        cls_list.append(c)
        box_list.append([x1, y1, x2, y2])
    if not cls_list:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    return np.array(cls_list, dtype=np.int64), np.array(box_list, dtype=np.float64)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, a (n,4), b (m,4)."""

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 0.0)
    area_b = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 0.0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-16)


def ap_coco(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under interpolated precision-recall curve (COCO-style)."""

    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass(frozen=True)
class ImageEval:
    """Per-image GT and predictions in subset class ids (0..nc-1)."""

    gt_cls: np.ndarray
    gt_xyxy: np.ndarray
    pred_cls: np.ndarray
    pred_conf: np.ndarray
    pred_xyxy: np.ndarray


def average_precision_one_class(
    samples: list[ImageEval],
    class_index: int,
    iou_threshold: float,
) -> tuple[float, int]:
    """Greedy IoU matching sorted by confidence; return AP, n_gt."""

    gt_by_img: list[np.ndarray] = []
    n_gt = 0
    for s in samples:
        mask = s.gt_cls == class_index
        gtb = s.gt_xyxy[mask]
        gt_by_img.append(gtb)
        n_gt += int(gtb.shape[0])

    preds: list[tuple[int, float, np.ndarray]] = []
    for img_idx, s in enumerate(samples):
        m = s.pred_cls == class_index
        idxs = np.flatnonzero(m)
        for j in idxs:
            preds.append(
                (
                    img_idx,
                    float(s.pred_conf[int(j)]),
                    s.pred_xyxy[int(j)].astype(np.float64),
                )
            )

    preds.sort(key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)
    matched: set[tuple[int, int]] = set()

    for i, (img_idx, _conf, box) in enumerate(preds):
        gtb = gt_by_img[img_idx]
        if gtb.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = box_iou_xyxy(box[None, :], gtb)[0]
        j_local = int(np.argmax(ious))
        best = float(ious[j_local])
        if best < iou_threshold:
            fp[i] = 1.0
            continue
        pair = (img_idx, j_local)
        if pair in matched:
            fp[i] = 1.0
            continue
        matched.add(pair)
        tp[i] = 1.0

    if n_gt == 0:
        return 0.0, 0

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-16)
    return ap_coco(recall, precision), n_gt


def map50_macro(
    samples: list[ImageEval],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Mean AP@0.5 over classes that have at least one GT box."""

    per_ap: dict[str, float] = {}
    per_ngt: dict[str, int] = {}
    aps: list[float] = []
    for c in range(num_classes):
        ap, ng = average_precision_one_class(samples, c, iou_threshold)
        key = str(c)
        per_ap[key] = ap
        per_ngt[key] = ng
        if ng > 0:
            aps.append(ap)
    mean_ap = float(np.mean(aps)) if aps else 0.0
    return mean_ap, per_ap, per_ngt


def write_eval_json(
    path: Path,
    *,
    m_ap50: float,
    per_class: dict[str, float],
    n_gt_per_class: dict[str, int],
    subset_names: list[str],
    coco_to_subset: dict[int, int],
    weights: str,
    data_yaml: str,
    imgsz: int,
    conf: float,
    device: str,
    num_images: int,
) -> None:
    name_by_subset = {str(i): subset_names[i] for i in range(len(subset_names))}
    payload = {
        "kind": "evaluate_coco_remap",
        "mAP50": m_ap50,
        "mAP50_per_class": per_class,
        "n_gt_per_class": n_gt_per_class,
        "class_names_by_subset_id": name_by_subset,
        "coco_class_id_to_subset_id": {str(k): v for k, v in sorted(coco_to_subset.items())},
        "weights": weights,
        "data_yaml": data_yaml,
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
        "num_val_images": num_images,
        "iou_threshold": 0.5,
        "note": "mAP50 computed after mapping COCO pred class ids to subset ids by name (no label files changed).",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
