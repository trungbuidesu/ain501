"""Build models_release.zip with runtime model artifacts only."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import onnxruntime as ort

RUNTIME_MODEL_FILES: tuple[str, ...] = (
    "models/yolov8n_int8.onnx",
    "models/mobilenetv3_small_fp32.onnx",
    "models/minilm_l6.onnx",
    "models/scene_classifier_int8.onnx",
    "models/movinet_a0_int8.onnx",
    "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
    "models/tts/piper/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
)


def build_zip(repo_root: Path, output_zip: Path, *, strict: bool) -> int:
    found: list[Path] = []
    missing: list[Path] = []
    for rel in RUNTIME_MODEL_FILES:
        p = (repo_root / rel).resolve()
        if p.is_file():
            found.append(p)
        else:
            missing.append(p)

    if strict and missing:
        for p in missing:
            print(f"[missing] {p.relative_to(repo_root)}")
        return 2

    print("[verify] runtime files")
    total_bytes = 0
    for p in found:
        rel = p.relative_to(repo_root)
        size_mb = p.stat().st_size / (1024 * 1024)
        total_bytes += p.stat().st_size
        if p.suffix.lower() == ".onnx":
            try:
                _ = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                status = "ok"
            except Exception as exc:
                print(f"[fail] {rel} ({size_mb:.2f} MB) load_error={exc}")
                return 3
        else:
            status = "ok(meta)"
        print(f"[{status}] {rel} ({size_mb:.2f} MB)")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in found:
            zf.write(p, arcname=str(p.relative_to(repo_root)).replace("\\", "/"))

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)
    print(
        f"[ok] wrote {output_zip} ({size_mb:.2f} MB) "
        f"files={len(found)} source_total={total_mb:.2f} MB"
    )
    if missing:
        print("[warn] missing files:")
        for p in missing:
            print(f"  - {p.relative_to(repo_root)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models_release.zip"),
        help="Output zip path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any runtime file is missing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = args.output if args.output.is_absolute() else (repo_root / args.output)
    return build_zip(repo_root, out, strict=bool(args.strict))


if __name__ == "__main__":
    raise SystemExit(main())
