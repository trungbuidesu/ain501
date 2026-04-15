"""Verify runtime model/data artifacts are present and loadable."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort
import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    path: Path
    required: bool
    expected_rank: int
    expected_tail: tuple[int, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid yaml mapping: {path}")
    return payload


def _resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (root / p).resolve()


def _input_shape_ok(shape: list[Any], expected_rank: int, expected_tail: tuple[int, ...]) -> bool:
    if len(shape) != expected_rank:
        return False
    if not expected_tail:
        return True
    tail = shape[-len(expected_tail) :]
    for dim, exp in zip(tail, expected_tail, strict=True):
        if isinstance(dim, int) and dim != exp:
            return False
    return True


def _session_input_shape(path: Path) -> list[Any]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return list(sess.get_inputs()[0].shape)


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_specs(root: Path, cfg: dict[str, Any]) -> list[ModelSpec]:
    models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
    tier1 = cfg.get("tier1", {}) if isinstance(cfg.get("tier1"), dict) else {}
    router = tier1.get("router", {}) if isinstance(tier1.get("router"), dict) else {}
    rag = cfg.get("rag", {}) if isinstance(cfg.get("rag"), dict) else {}

    tier1_enabled = bool(tier1.get("enabled", False))
    rag_enabled = bool(rag.get("enabled", False))

    return [
        ModelSpec(
            key="yolo_int8",
            path=_resolve(root, str(models.get("yolo_int8", "models/yolov8n_int8.onnx"))),
            required=True,
            expected_rank=4,
            expected_tail=(3, 640, 640),
        ),
        ModelSpec(
            key="mobilenet_fp32",
            path=_resolve(
                root,
                str(models.get("mobilenet_fp32", "models/mobilenetv3_small_fp32.onnx")),
            ),
            required=True,
            expected_rank=4,
            expected_tail=(3, 224, 224),
        ),
        ModelSpec(
            key="minilm_l6",
            path=_resolve(root, str(rag.get("model_path", "models/minilm_l6.onnx"))),
            required=rag_enabled,
            expected_rank=2,
            expected_tail=(),
        ),
        ModelSpec(
            key="scene_classifier_int8",
            path=_resolve(
                root,
                str(router.get("model_path", "models/scene_classifier_int8.onnx")),
            ),
            required=tier1_enabled,
            expected_rank=4,
            expected_tail=(3, 160, 160),
        ),
        ModelSpec(
            key="movinet_a0_int8",
            path=_resolve(root, str(tier1.get("movinet_int8", "models/movinet_a0_int8.onnx"))),
            required=tier1_enabled,
            expected_rank=5,
            expected_tail=(3, 6, 172, 172),
        ),
    ]


def verify_runtime_data(root: Path, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    rag = cfg.get("rag", {}) if isinstance(cfg.get("rag"), dict) else {}
    rag_enabled = bool(rag.get("enabled", False))
    knowledge_dir = _resolve(root, str(rag.get("knowledge_dir", "data/knowledge")))
    if not rag_enabled:
        return True, [f"[skip] rag disabled, skip knowledge check ({knowledge_dir})"]
    if not knowledge_dir.is_dir():
        return False, [f"[fail] missing knowledge dir: {knowledge_dir}"]
    files = sorted(knowledge_dir.glob("*.json"))
    if not files:
        return False, [f"[fail] no knowledge json files: {knowledge_dir}"]
    lines = [f"[ok] knowledge json files: {len(files)} in {knowledge_dir}"]
    lines.extend(f"  - {f.relative_to(root)}" for f in files)
    return True, lines


def verify_models(root: Path, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    ok_all = True
    lines: list[str] = []
    for spec in build_specs(root, cfg):
        if not spec.path.is_file():
            if spec.required:
                ok_all = False
                lines.append(f"[fail] {spec.key}: missing required file {spec.path}")
            else:
                lines.append(f"[warn] {spec.key}: optional file missing {spec.path}")
            continue

        try:
            shape = _session_input_shape(spec.path)
            shape_ok = _input_shape_ok(shape, spec.expected_rank, spec.expected_tail)
            if not shape_ok:
                ok_all = False
                lines.append(
                    f"[fail] {spec.key}: shape mismatch got={shape} "
                    f"expected_rank={spec.expected_rank} expected_tail={spec.expected_tail}"
                )
                continue
            lines.append(
                f"[ok] {spec.key}: {spec.path.relative_to(root)} "
                f"({ _size_mb(spec.path):.2f} MB) shape={shape}"
            )
        except Exception as exc:
            if spec.required:
                ok_all = False
                lines.append(f"[fail] {spec.key}: load error {exc}")
            else:
                lines.append(f"[warn] {spec.key}: optional load error {exc}")
    return ok_all, lines


def verify_piper_metadata(root: Path, cfg: dict[str, Any]) -> tuple[bool, str]:
    models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
    piper = _resolve(
        root,
        str(
            models.get(
                "piper_model_path",
                "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
            )
        ),
    )
    meta = Path(str(piper) + ".json")
    if piper.is_file() and meta.is_file():
        return True, f"[ok] piper metadata: {meta.relative_to(root)}"
    if piper.is_file() and not meta.is_file():
        return False, f"[fail] piper metadata missing: {meta}"
    return True, f"[warn] piper model missing (optional): {piper}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app.yaml"),
        help="Path to app config yaml.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = args.config if args.config.is_absolute() else (root / args.config)
    cfg = _load_yaml(cfg_path)

    print(f"[info] config={cfg_path}")
    ok_models, model_lines = verify_models(root, cfg)
    for line in model_lines:
        print(line)

    ok_data, data_lines = verify_runtime_data(root, cfg)
    for line in data_lines:
        print(line)

    ok_piper, piper_line = verify_piper_metadata(root, cfg)
    print(piper_line)

    all_ok = ok_models and ok_data and ok_piper
    print(f"[result] {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
