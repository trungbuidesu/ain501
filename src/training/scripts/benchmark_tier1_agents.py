"""Generate Tier 1 CPU benchmark summary into ``reports/phase3/tier1_benchmark.md``.

Run from repo root::

    python -m src.training.scripts.benchmark_tier1_agents

Requires ONNX artifacts where referenced; skips missing models with a note.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> int:
    root = _repo_root()
    out_dir = root / "reports" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "tier1_benchmark.md"
    lines: list[str] = [
        "# Tier 1 CPU benchmark (agents)",
        "",
        "Targets: action < 15 ms, OCR < 30 ms, face/pose < 15 ms (p50, same process).",
        "",
    ]

    payload: dict[str, object] = {}

    movinet = root / "models" / "movinet_a0_int8.onnx"
    if movinet.is_file():
        from src.agents.action_agent import ActionAgent, benchmark_action_agent_latency

        try:
            action_agent = ActionAgent(strict_artifacts=True)
        except FileNotFoundError:
            action_agent = None
        if action_agent is not None:
            bench = benchmark_action_agent_latency(action_agent, samples=30)
            lines.append("## MoViNet-A0 INT8")
            lines.append("")
            lines.append("| Metric | Value | Meets <15ms? |")
            lines.append("| --- | --- | --- |")
            lat = f"{bench['latency_ms_p50']:.2f} / {bench['latency_ms_p95']:.2f}"
            ok = "yes" if bench["meets_target"] else "no"
            lines.append(f"| p50 / p95 (ms) | {lat} | {ok} |")
            lines.append("")
            payload["movinet_a0_int8"] = bench
        else:
            lines.append("## MoViNet-A0 INT8")
            lines.append("")
            lines.append("*Skipped: ActionAgent could not load model.*")
            lines.append("")
    else:
        lines.append("## MoViNet-A0 INT8")
        lines.append("")
        lines.append(f"*Skipped: missing `{movinet.as_posix()}`*")
        lines.append("")

    ocr_path = root / "reports" / "prebuilt_week5" / "_synth_images"
    test_img = next(ocr_path.glob("*.png"), None) if ocr_path.is_dir() else None
    if test_img is not None:
        try:
            import numpy as np
            from PIL import Image

            from src.agents.ocr_agent import OcrAgent, benchmark_ocr_agent_latency

            rgb = np.asarray(Image.open(test_img).convert("RGB"))
            ocr_agent = OcrAgent(confidence_threshold=0.2, grid=1)
            ob = benchmark_ocr_agent_latency(ocr_agent, rgb, samples=10)
            lines.append("## PaddleOCR (sample image)")
            lines.append("")
            lines.append("| Metric | Value | Meets <30ms? |")
            lines.append("| --- | --- | --- |")
            lat = f"{ob['latency_ms_p50']:.2f} / {ob['latency_ms_p95']:.2f}"
            ok = "yes" if ob["meets_target"] else "no"
            lines.append(f"| p50 / p95 (ms) | {lat} | {ok} |")
            lines.append("")
            payload["paddleocr"] = ob
        except Exception as exc:
            lines.append("## PaddleOCR")
            lines.append("")
            lines.append(f"*Skipped: {exc}*")
            lines.append("")
    else:
        lines.append("## PaddleOCR")
        lines.append("")
        lines.append(
            "*Skipped: no sample image under `reports/prebuilt_week5/_synth_images/`.*"
        )
        lines.append("")

    try:
        import numpy as np

        from src.agents.face_pose_agent import (
            FacePoseAgent,
            benchmark_face_pose_latency,
        )

        face_pose_agent = FacePoseAgent(strict_artifacts=False)
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        fb = benchmark_face_pose_latency(face_pose_agent, rgb, samples=15)
        lines.append("## MediaPipe face + pose")
        lines.append("")
        lines.append("| Metric | Value | Meets <15ms? |")
        lines.append("| --- | --- | --- |")
        lat = f"{fb['latency_ms_p50']:.2f} / {fb['latency_ms_p95']:.2f}"
        ok = "yes" if fb["meets_target"] else "no"
        lines.append(f"| p50 / p95 (ms) | {lat} | {ok} |")
        lines.append("")
        payload["mediapipe_face_pose"] = fb
    except Exception as exc:
        lines.append("## MediaPipe face + pose")
        lines.append("")
        lines.append(f"*Skipped: {exc}*")
        lines.append("")

    (out_dir / "tier1_agents_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
