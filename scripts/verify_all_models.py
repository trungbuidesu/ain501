"""Comprehensive model and agent verification without webcam interaction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import yaml  # type: ignore[import-untyped]

from src.agents.action_agent import ActionAgent
from src.agents.object_agent import ObjectAgent
from src.agents.ocr_agent import OcrAgent
from src.agents.yolo_postprocess import decode_yolo_predictions
from src.caption.template_engine import SpatialTemplateEngine
from src.capture.types import FramePacket
from src.rag.knowledge_base import KnowledgeBase


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return raw


def _input_for_meta(meta: Any) -> np.ndarray:
    shape = list(getattr(meta, "shape", []) or [])
    final: list[int] = []
    for i, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            final.append(dim)
        else:
            final.append(128 if i == 1 else 1)
    if not final:
        final = [1, 3, 224, 224]
    return np.random.rand(*final).astype(np.float32)


@dataclass
class HealthRow:
    model: str
    path: str
    exists: str
    loadable: str
    input_ok: str
    output_ok: str
    values_ok: str


def _dtype_to_numpy(dtype_str: str) -> Any:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(dtype_str, np.float32)


def _feed_for_session(sess: Any) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for meta in sess.get_inputs():
        shape = list(getattr(meta, "shape", []) or [])
        final: list[int] = []
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                final.append(dim)
            else:
                final.append(128 if i == 1 else 1)
        if not final:
            final = [1]
        np_dtype = _dtype_to_numpy(str(getattr(meta, "type", "tensor(float)")))
        if np.issubdtype(np_dtype, np.integer):
            arr = np.random.randint(0, 1000, size=final, dtype=np_dtype)
        elif np_dtype == np.bool_:
            arr = np.random.randint(0, 2, size=final, dtype=np.int8).astype(np.bool_)
        else:
            arr = np.random.rand(*final).astype(np_dtype)
        feed[meta.name] = arr
    return feed


def _health_check(name: str, model_path: Path, expected_rank: int | None) -> HealthRow:
    exists = model_path.is_file()
    if not exists:
        return HealthRow(name, str(model_path), "N", "N", "N", "N", "N")
    try:
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception:
        return HealthRow(name, str(model_path), "Y", "N", "N", "N", "N")
    try:
        in_meta = sess.get_inputs()[0]
        feed = _feed_for_session(sess)
        x = feed[in_meta.name]
        input_ok = "Y" if expected_rank is None or x.ndim == expected_rank else "N"
        out = sess.run(None, feed)
        output_ok = "Y" if len(out) > 0 else "N"
        vals = np.asarray(out[0])
        values_ok = (
            "Y"
            if vals.size > 0 and np.isfinite(vals).all() and not np.all(vals == 0)
            else "N"
        )
        return HealthRow(
            name,
            str(model_path),
            "Y",
            "Y",
            input_ok,
            output_ok,
            values_ok,
        )
    except Exception:
        return HealthRow(name, str(model_path), "Y", "Y", "N", "N", "N")


def _coco_paths(root: Path) -> tuple[Path, Path]:
    ann_candidates = [
        root / "data/raw/coco/annotations/instances_val2017.json",
        root / "data/external/coco2017/annotations/instances_val2017.json",
    ]
    img_candidates = [
        root / "data/raw/coco/val2017",
        root / "data/external/coco2017/val2017",
    ]
    ann = next((p for p in ann_candidates if p.is_file()), ann_candidates[-1])
    img = next((p for p in img_candidates if p.is_dir()), img_candidates[-1])
    return ann, img


def _select_images_by_class(
    ann_path: Path,
    target_classes: list[str],
) -> list[tuple[str, int, str]]:
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = payload.get("categories", [])
    anns = payload.get("annotations", [])
    imgs = payload.get("images", [])
    name_to_cid = {
        str(c["name"]): int(c["id"])
        for c in cats
        if "name" in c and "id" in c
    }
    id_to_file = {
        int(i["id"]): str(i["file_name"])
        for i in imgs
        if "id" in i and "file_name" in i
    }
    img_to_cids: dict[int, set[int]] = {}
    img_cid_area: dict[tuple[int, int], float] = {}
    for a in anns:
        img_id = int(a.get("image_id", -1))
        cid = int(a.get("category_id", -1))
        if img_id < 0 or cid < 0:
            continue
        img_to_cids.setdefault(img_id, set()).add(cid)
        area = float(a.get("area", 0.0))
        key = (img_id, cid)
        img_cid_area[key] = img_cid_area.get(key, 0.0) + max(0.0, area)
    selected: list[tuple[str, int, str]] = []
    used_files: set[str] = set()
    for cls in target_classes:
        cid = name_to_cid.get(cls)
        if cid is None:
            continue
        file_name = ""
        fallback = ""
        ranked_ids = sorted(
            [
                img_id
                for img_id, cids in img_to_cids.items()
                if cid in cids and img_id in id_to_file
            ],
            key=lambda img_id: (
                len(img_to_cids.get(img_id, set())),
                -img_cid_area.get((img_id, cid), 0.0),
            ),
        )
        for img_id in ranked_ids:
            candidate = id_to_file[img_id]
            if not fallback:
                fallback = candidate
            if candidate not in used_files:
                file_name = candidate
                break
        if not file_name and fallback:
            file_name = fallback
        if file_name:
            used_files.add(file_name)
            selected.append((file_name, cid, cls))
    return selected


def _gt_labels_for_image(payload: dict[str, Any], file_name: str) -> set[str]:
    cats = payload.get("categories", [])
    anns = payload.get("annotations", [])
    imgs = payload.get("images", [])
    cid_to_name = {
        int(c["id"]): str(c["name"])
        for c in cats
        if "id" in c and "name" in c
    }
    file_to_id = {
        str(i["file_name"]): int(i["id"])
        for i in imgs
        if "id" in i and "file_name" in i
    }
    img_id = file_to_id.get(file_name)
    if img_id is None:
        return set()
    out: set[str] = set()
    for a in anns:
        if int(a.get("image_id", -1)) != img_id:
            continue
        cid = int(a.get("category_id", -1))
        if cid in cid_to_name:
            out.add(cid_to_name[cid])
    return out


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    print(line)
    print(sep)
    for r in rows:
        cells = " | ".join(r[i].ljust(widths[i]) for i in range(len(headers)))
        print("| " + cells + " |")


def main() -> int:
    root = _repo_root()
    cfg = _load_yaml(root / "configs/app.yaml")

    models_cfg = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
    yolo_path = root / str(models_cfg.get("yolo_int8", "models/yolov8n.onnx"))
    mobilenet_path = root / str(
        models_cfg.get("mobilenet_fp32", "models/mobilenetv3_small_fp32.onnx")
    )
    minilm_path = root / str(
        cfg.get("rag", {}).get("model_path", "models/minilm_l6.onnx")
    )
    movinet_path = root / "models/movinet_a0_int8.onnx"
    scene_path = root / "models/scene_classifier_int8.onnx"

    health = [
        _health_check("yolov8n", yolo_path, 4),
        _health_check("movinet_a0", movinet_path, 5),
        _health_check("mobilenetv3", mobilenet_path, 4),
        _health_check("minilm_l6", minilm_path, None),
        _health_check("scene_classifier", scene_path, 4),
    ]

    # YOLO output format verification
    yolo_parser_ok = False
    if yolo_path.is_file():
        sess = ort.InferenceSession(str(yolo_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        x = np.random.randn(*_input_for_meta(inp).shape).astype(np.float32)
        raw = sess.run(None, {inp.name: x})[0]
        boxes, scores, class_ids = decode_yolo_predictions(raw, conf_threshold=0.1)
        yolo_parser_ok = bool(
            boxes.ndim == 2 and scores.ndim == 1 and class_ids.ndim == 1
        )

    # YOLO class mapping verification
    ann_path, img_dir = _coco_paths(root)
    ann_payload = json.loads(ann_path.read_text(encoding="utf-8"))
    target = ["person", "car", "dog", "chair", "bicycle", "bus", "traffic light"]
    selected = _select_images_by_class(ann_path, target)

    obj = ObjectAgent(model_path=yolo_path, conf_threshold=0.25, allowed_labels=[])
    mapping_rows: list[list[str]] = []
    mapping_pass = 0
    mapping_total = 0
    for file_name, _, target_label in selected:
        img_path = img_dir / file_name
        if not img_path.is_file():
            continue
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = obj.detect(frame_rgb)
        pred_labels = sorted({str(x.get("label", "")) for x in pred})
        gt_labels = sorted(_gt_labels_for_image(ann_payload, file_name))
        x, _ = obj.preprocess(frame_rgb)
        raw = obj._session.run(None, {obj._input_name: x})[0]  # type: ignore[attr-defined]
        _, raw_scores, raw_ids = decode_yolo_predictions(raw, conf_threshold=0.001)
        top_idx = (
            np.argsort(-raw_scores)[:5]
            if raw_scores.size
            else np.array([], dtype=int)
        )
        raw_ids_top = [int(raw_ids[i]) for i in top_idx] if top_idx.size else []
        raw_scores_top = [float(raw_scores[i]) for i in top_idx] if top_idx.size else []
        match = target_label in set(gt_labels) and target_label in set(pred_labels)
        mapping_rows.append(
            [
                f"{file_name} ({target_label})",
                ",".join(gt_labels)[:80],
                ",".join(pred_labels)[:80],
                ",".join(str(x) for x in raw_ids_top),
                "Y" if match else "N",
            ]
        )
        if not match:
            print(
                "[yolo-mapping-mismatch] "
                f"image={file_name} target={target_label} "
                f"raw_shape={list(np.asarray(raw).shape)} "
                f"raw_top_ids={raw_ids_top} "
                f"raw_top_scores={raw_scores_top} "
                "decode_path=decode_yolo_predictions -> class argmax from cls logits"
            )
        mapping_total += 1
        mapping_pass += int(match)

    # Agent smoke tests
    smoke_rows: list[list[str]] = []
    smoke_pass = 0
    smoke_total = 0

    def add_smoke(name: str, test: str, result: str) -> None:
        nonlocal smoke_pass, smoke_total
        smoke_rows.append([name, test, result])
        if result in {"Y", "N"}:
            smoke_total += 1
            smoke_pass += int(result == "Y")

    person_image = next((x[0] for x in selected if x[2] == "person"), "")
    if person_image:
        p = img_dir / person_image
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        ok_obj = False
        if img_bgr is not None:
            frame_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dets = obj.detect(frame_rgb)
            ok_obj = any(str(d.get("label", "")) == "person" for d in dets)
        add_smoke(
            "ObjectAgent",
            "detect person from COCO image",
            "Y" if ok_obj else "N",
        )
    else:
        add_smoke("ObjectAgent", "detect person from COCO image", "N")
    try:
        rng = np.random.default_rng(0)
        frames = [
            FramePacket(
                data=rng.integers(0, 255, size=(172, 172, 3), dtype=np.uint8),
                t_mono=0.0,
                source="file",
                frame_id=i,
            )
            for i in range(6)
        ]
        action = ActionAgent(model_path=movinet_path)
        out = action.predict(frames)
        add_smoke(
            "ActionAgent",
            "predict random clip",
            "Y" if bool(out.get("action")) else "N",
        )
    except Exception:
        add_smoke("ActionAgent", "predict random clip", "N")

    try:
        ocr = OcrAgent(confidence_threshold=0.5)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        out = ocr.read_text(img)
        add_smoke(
            "OCRAgent",
            "read text no crash",
            "Y" if isinstance(out, list) else "N",
        )
    except Exception as exc:
        print(f"[ocr-smoke] failed: {exc}")
        add_smoke("OCRAgent", "read text no crash", "N")

    try:
        rag_cfg = cfg.get("rag", {}) if isinstance(cfg.get("rag"), dict) else {}
        kb = KnowledgeBase(
            model_path=root / str(rag_cfg.get("model_path", "models/minilm_l6.onnx")),
            knowledge_dir=root / str(rag_cfg.get("knowledge_dir", "data/knowledge")),
            top_k=int(rag_cfg.get("top_k", 3)),
            min_similarity=float(rag_cfg.get("min_similarity", 0.3)),
        )
        hits = kb.search("car front close")
        ok = bool(hits and len(hits) > 0 and float(hits[0].score) > 0.3)
        add_smoke("RAG", "search car front close", "Y" if ok else "N")
    except Exception:
        add_smoke("RAG", "search car front close", "N")

    try:
        engine = SpatialTemplateEngine()
        det = [
            {
                "label": "person",
                "confidence": 0.8,
                "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 300.0},
            }
        ]
        events = engine.render(det, frame_width=640, frame_height=480)
        ok = bool(events and ("person" in events[0].text or "người" in events[0].text))
        add_smoke("TemplateEngine", "generate caption", "Y" if ok else "N")
    except Exception:
        add_smoke("TemplateEngine", "generate caption", "N")

    # Final report
    print("========== MODEL VERIFICATION REPORT ==========")
    print("\n[ONNX Health Check]")
    _print_table(
        ["Model", "Path", "Exists", "Loadable", "Input OK", "Output OK", "Values OK"],
        [
            [
                r.model,
                r.path,
                r.exists,
                r.loadable,
                r.input_ok,
                r.output_ok,
                r.values_ok,
            ]
            for r in health
        ],
    )

    print("\n[YOLO Class Mapping]")
    _print_table(
        ["Image", "GT labels", "Predicted labels", "Raw class_ids", "Match"],
        mapping_rows if mapping_rows else [["-", "-", "-", "-", "N"]],
    )

    print("\n[Agent Smoke Tests]")
    _print_table(["Agent", "Test", "Result"], smoke_rows)

    checks_total = len(health) + mapping_total + smoke_total + 1
    onnx_pass = sum(
        int(
            r.exists == "Y"
            and r.loadable == "Y"
            and r.input_ok == "Y"
            and r.output_ok == "Y"
            and r.values_ok == "Y"
        )
        for r in health
    )
    checks_pass = onnx_pass + mapping_pass + smoke_pass + int(yolo_parser_ok)
    print(f"\nOVERALL: {checks_pass}/{checks_total} passed")
    return 0 if checks_pass == checks_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
