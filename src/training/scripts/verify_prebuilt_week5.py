"""Week 5 pre-built model verification: PaddleOCR, MediaPipe, TTS.

Run from repo root::

    python -m src.training.scripts.verify_prebuilt_week5 paddleocr --synthetic
    python -m src.training.scripts.verify_prebuilt_week5 mediapipe --synthetic
    python -m src.training.scripts.verify_prebuilt_week5 tts
    python -m src.training.scripts.verify_prebuilt_week5 all --synthetic

Artifacts default to ``reports/prebuilt_week5/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPORT_DIR = Path("reports/prebuilt_week5")
MEDIAPIPE_MODEL_DIR = REPORT_DIR / "_mediapipe_models"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
SYNTH_IMAGE_DIR = REPORT_DIR / "_synth_images"
UTTERANCES_PATH = Path("data/prebuilt_week5/tts_utterances.json")
TTS_CONFIG = Path("configs/datasets/tts_piper_accessibility.yaml")


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def _ensure_report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mediapipe_task_model_path(url: str, filename: str) -> Path:
    MEDIAPIPE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MEDIAPIPE_MODEL_DIR / filename
    if not path.exists():
        print("Downloading", filename, "…", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def _synthetic_text_images(out_dir: Path, n: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = [
        "HELLO",
        "EXIT",
        "STOP",
        "OPEN",
        "AIN501",
        "TextOCR",
        "Room 101",
        "WC",
        "PULL",
        "PUSH",
        "SALE 50%",
        "CAFE",
        "INFO",
        "LANE2",
        "NO PARKING",
        "Tickets",
        "Gate A",
        "Baggage",
        "Arrivals",
        "Departures",
    ]
    for i in range(n):
        text = labels[i % len(labels)]
        w, h = 360, 100
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except OSError:
            font = ImageFont.load_default()
        draw.text((12, 28), f"{text} {i}", fill=(0, 0, 0), font=font)
        p = out_dir / f"synth_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def _collect_images(image_dir: Path | None, synthetic: bool, limit: int) -> list[Path]:
    if synthetic:
        return _synthetic_text_images(SYNTH_IMAGE_DIR, limit)
    if image_dir is None:
        raise ValueError("Provide --image-dir or use --synthetic")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in exts],
        key=lambda x: str(x),
    )
    return files[:limit]


def _paddleocr_lines_from_predict_output(result: Any) -> list[dict[str, Any]]:
    """Normalize PaddleOCR output: legacy nested list or PaddleX ``OCRResult`` list."""

    if not result:
        return []
    # Legacy PaddleOCR 2.x: [ [ [box, (text, score)], ... ] ]
    if isinstance(result, list) and result and isinstance(result[0], list):
        page = result[0]
        if page and isinstance(page[0], (list, tuple)) and len(page[0]) == 2:
            lines_out: list[dict[str, Any]] = []
            for entry in page:
                box, rec = entry
                if isinstance(rec, (list, tuple)) and len(rec) == 2:
                    text, score = rec[0], rec[1]
                else:
                    continue
                lines_out.append(
                    {"box": box, "text": str(text), "score": float(score)},
                )
            return lines_out
    # PaddleOCR 3.x / PaddleX: list of OCRResult (mapping-like) per input
    if isinstance(result, list) and result:
        page = result[0]
        try:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            polys = page.get("rec_polys") or page.get("dt_polys")
        except (KeyError, TypeError):
            return []
        lines_out = []
        for i, text in enumerate(texts):
            box: Any
            if polys is not None and i < len(polys):
                poly = polys[i]
                box = poly.tolist() if hasattr(poly, "tolist") else poly
            else:
                box = []
            sc = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            lines_out.append({"box": box, "text": str(text), "score": sc})
        return lines_out
    return []


def _paddle_version_lines() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for mod in ("paddle", "paddleocr"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", None)
            out[mod] = str(ver) if ver is not None else None
        except Exception as exc:
            out[mod] = f"import_error: {exc}"
    return out


def cmd_paddleocr(args: argparse.Namespace) -> int:
    _ensure_report_dir()
    # Windows + oneDNN: avoid NotImplementedError in some Paddle+PaddleX builds.
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "false")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        _write_json(
            REPORT_DIR / "paddleocr_benchmark_cpu.json",
            {
                "status": "failed",
                "error": str(exc),
                "hint": "Install PaddlePaddle CPU wheel for your OS/Python, then paddleocr.",
            },
        )
        print("PaddleOCR import failed:", exc, file=sys.stderr)
        return 1

    images = _collect_images(args.image_dir, args.synthetic, args.limit)
    try:
        ocr = PaddleOCR(
            lang=args.lang,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except Exception as exc:
        _write_json(
            REPORT_DIR / "paddleocr_benchmark_cpu.json",
            {
                "status": "failed",
                "error": str(exc),
                "hint": (
                    "Install paddlepaddle for your OS/Python. On Windows CPU, "
                    "set PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=false if you hit oneDNN errors."
                ),
            },
        )
        print("PaddleOCR init failed:", exc, file=sys.stderr)
        return 1

    sample_path = REPORT_DIR / "paddleocr_samples.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []

    with sample_path.open("w", encoding="utf-8") as sink:
        for img_path in images:
            t0 = time.perf_counter()
            try:
                raw = ocr.predict(str(img_path))
            except Exception as exc:
                _write_json(
                    REPORT_DIR / "paddleocr_benchmark_cpu.json",
                    {
                        "status": "failed",
                        "error": str(exc),
                        "image": str(img_path),
                        "hint": "Try: set PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=false",
                    },
                )
                print("PaddleOCR predict failed:", exc, file=sys.stderr)
                return 1
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            lines_out = _paddleocr_lines_from_predict_output(raw)
            row = {
                "image_id": str(img_path.as_posix()),
                "source": "synthetic" if args.synthetic else "user_dir",
                "lines": lines_out,
                "elapsed_ms": round(elapsed_ms, 3),
            }
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    lat_sorted = sorted(latencies)
    warmup = max(0, args.warmup)
    iters = max(1, args.bench_iters)
    bench_lat: list[float] = []
    if images:
        target = images[0]
        for _ in range(warmup):
            _ = ocr.predict(str(target))
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = ocr.predict(str(target))
            bench_lat.append((time.perf_counter() - t0) * 1000.0)
    bench_lat.sort()
    payload = {
        "status": "ok",
        "paddle_stack": _paddle_version_lines(),
        "hardware_note": os.environ.get("COMPUTERNAME", "unknown"),
        "image_count": len(images),
        "lang": args.lang,
        "ocr_model_note": "PaddleOCR 3.x / PaddleX pipeline (lang-driven det+rec). "
        "MKLDNN default off via PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT on Windows when needed.",
        "per_image_latency_ms": {
            "p50": round(_percentile(lat_sorted, 50), 3),
            "p95": round(_percentile(lat_sorted, 95), 3),
            "samples": len(latencies),
        },
        "benchmark_repeat_on_first_image": {
            "warmup": warmup,
            "iterations": iters,
            "p50_ms": round(_percentile(bench_lat, 50), 3),
            "p95_ms": round(_percentile(bench_lat, 95), 3),
        },
        "rss_mb_approx": _rss_mb(),
        "samples_jsonl": str(sample_path.as_posix()),
    }
    _write_json(REPORT_DIR / "paddleocr_benchmark_cpu.json", payload)
    print("Wrote", sample_path, "and", REPORT_DIR / "paddleocr_benchmark_cpu.json")
    return 0


def _face_emotion_proxy(
    landmarks: Sequence[tuple[float, float, float]],
) -> dict[str, Any]:
    def d(i: int, j: int) -> float:
        xi, yi, _ = landmarks[i]
        xj, yj, _ = landmarks[j]
        return float(np.hypot(xi - xj, yi - yj))

    face_w = d(33, 263)
    mouth_open = d(13, 14) / max(face_w, 1e-6)
    mouth_width = d(61, 291) / max(face_w, 1e-6)
    label = "neutral"
    if mouth_open > 0.12:
        label = "open_mouth"
    elif mouth_width > 0.55:
        label = "smile_like"
    return {
        "mouth_open_ratio": round(mouth_open, 4),
        "mouth_width_ratio": round(mouth_width, 4),
        "heuristic_label": label,
        "note": "Heuristic proxy from mesh geometry; not a FER+ emotion classifier.",
    }


def _pose_label_tasks(lm_list: Sequence[Any]) -> dict[str, Any]:
    """Rule-based pose from 33 BlazePose landmarks (MediaPipe Tasks API)."""

    if len(lm_list) < 33:
        return {
            "rule_label": "unknown",
            "torso_vertical_span": 0.0,
            "arms_up": False,
            "thresholds_note": "Need 33 pose landmarks.",
        }

    def yv(idx: int) -> tuple[float, float]:
        m = lm_list[idx]
        y = float(m.y or 0.0)
        v = float(m.visibility) if m.visibility is not None else 1.0
        return y, v

    ls_y, _ = yv(11)
    rs_y, _ = yv(12)
    lh_y, _ = yv(23)
    rh_y, _ = yv(24)
    lw_y, lw_v = yv(15)
    rw_y, rw_v = yv(16)
    sh_y = (ls_y + rs_y) / 2.0
    hip_y = (lh_y + rh_y) / 2.0
    torso = float(hip_y - sh_y)
    arms_up = lw_v > 0.5 and rw_v > 0.5 and lw_y < ls_y - 0.03 and rw_y < rs_y - 0.03
    if arms_up:
        lbl = "arms_up"
    elif torso < 0.08:
        lbl = "sit_like"
    else:
        lbl = "stand_like"
    return {
        "rule_label": lbl,
        "torso_vertical_span": round(torso, 4),
        "arms_up": bool(arms_up),
        "thresholds_note": "sit_like if torso<0.08; arms_up if wrists above shoulders with visibility>0.5.",
    }


def _synthetic_face_pose_images(out_dir: Path, n: int) -> list[Path]:
    """Simple colored blobs as stand-ins when no portrait images exist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        img = Image.new("RGB", (256, 256), color=(40, 40, 60))
        draw = ImageDraw.Draw(img)
        draw.ellipse((88, 60, 168, 140), fill=(220, 190, 170))
        draw.rectangle((100, 140, 156, 220), fill=(90, 90, 120))
        p = out_dir / f"synth_pose_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def cmd_mediapipe(args: argparse.Namespace) -> int:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, PoseLandmarker

    _ensure_report_dir()
    if args.synthetic:
        images = _synthetic_face_pose_images(REPORT_DIR / "_synth_face", args.limit)
    else:
        images = _collect_images(args.image_dir, False, args.limit)

    try:
        face_path = _mediapipe_task_model_path(
            FACE_LANDMARKER_URL,
            "face_landmarker.task",
        )
        pose_path = _mediapipe_task_model_path(
            POSE_LANDMARKER_LITE_URL,
            "pose_landmarker_lite.task",
        )
        face_lm = FaceLandmarker.create_from_model_path(str(face_path))
        pose_lm = PoseLandmarker.create_from_model_path(str(pose_path))
    except Exception as exc:
        _write_json(
            REPORT_DIR / "mediapipe_benchmark_cpu.json",
            {
                "status": "failed",
                "error": str(exc),
                "hint": "Network required once to download .task models into reports/prebuilt_week5/_mediapipe_models/",
            },
        )
        print("MediaPipe setup failed:", exc, file=sys.stderr)
        return 1

    sample_path = REPORT_DIR / "mediapipe_face_pose_samples.jsonl"
    face_lat: list[float] = []
    pose_lat: list[float] = []

    try:
        with sample_path.open("w", encoding="utf-8") as sink:
            for img_path in images:
                rgb = np.asarray(Image.open(img_path).convert("RGB"))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t0 = time.perf_counter()
                fres = face_lm.detect(mp_image)
                face_lat.append((time.perf_counter() - t0) * 1000.0)
                t1 = time.perf_counter()
                pres = pose_lm.detect(mp_image)
                pose_lat.append((time.perf_counter() - t1) * 1000.0)

                has_face = bool(fres.face_landmarks)
                face_payload: dict[str, Any] = {"detected": has_face}
                if has_face:
                    lms = fres.face_landmarks[0]
                    tuples = [
                        (float(p.x or 0.0), float(p.y or 0.0), float(p.z or 0.0))
                        for p in lms
                    ]
                    face_payload["landmark_count"] = len(tuples)
                    if getattr(args, "full_landmarks", False):
                        face_payload["landmarks_normalized_xyz"] = [
                            {"x": x, "y": y, "z": z} for x, y, z in tuples
                        ]
                    else:
                        face_payload["landmarks_omitted"] = True
                    face_payload["emotion_proxy"] = _face_emotion_proxy(tuples)
                else:
                    face_payload["landmark_count"] = 0

                pose_list = pres.pose_landmarks[0] if pres.pose_landmarks else []
                pose_payload: dict[str, Any] = {"detected": bool(pose_list)}
                if pose_list:
                    pose_payload["landmark_count"] = len(pose_list)
                    pose_payload["keypoints"] = [
                        {
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": lm.visibility,
                        }
                        for lm in pose_list
                    ]
                    pose_payload["pose_rules"] = _pose_label_tasks(pose_list)
                else:
                    pose_payload["landmark_count"] = 0

                row = {
                    "image_id": str(img_path.as_posix()),
                    "face_ms": round(face_lat[-1], 3),
                    "pose_ms": round(pose_lat[-1], 3),
                    "face": face_payload,
                    "pose": pose_payload,
                }
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

        def bench_summary(xs: list[float]) -> dict[str, Any]:
            s = sorted(xs)
            return {
                "p50_ms": round(_percentile(s, 50), 3),
                "p95_ms": round(_percentile(s, 95), 3),
                "n": len(xs),
            }

        payload = {
            "status": "ok",
            "mediapipe_version": getattr(mp, "__version__", "unknown"),
            "api": "tasks FaceLandmarker + PoseLandmarker (IMAGE)",
            "coordinate_system": "normalized 0-1, origin top-left; z relative depth",
            "face_landmark_schema": "478 points when detected (face_landmarker.task)",
            "pose_landmark_schema": "33 BlazePose keypoints (pose_landmarker_lite.task)",
            "face_latency_ms": bench_summary(face_lat),
            "pose_latency_ms": bench_summary(pose_lat),
            "rss_mb_approx": _rss_mb(),
            "samples_jsonl": str(sample_path.as_posix()),
        }
        _write_json(REPORT_DIR / "mediapipe_benchmark_cpu.json", payload)
        print("Wrote", sample_path, "and", REPORT_DIR / "mediapipe_benchmark_cpu.json")
        return 0
    finally:
        face_lm.close()
        pose_lm.close()


def _write_tts_compare_md(path: Path) -> None:
    body = """# TTS engine comparison (Week 5)

Subjective quality and latency depend on voice pack and CPU. Fill measured numbers after running `python -m src.training.scripts.verify_prebuilt_week5 tts`.

| Engine | Quality (typical) | Latency | Model size | Offline | License | Windows notes |
| --- | --- | --- | --- | --- | --- | --- |
| Piper | Strong neural TTS | Low (ONNXRuntime) | ~15–60 MB / voice | Yes | GPL | Primary path in this repo; CLI `piper`. |
| Kokoro | Strong (when available) | Medium | Varies | Often | Check upstream | May require extra runtime or non-Windows builds; verify before production. |
| pyttsx3 | Robotic (SAPI) | Very low | None (OS voice) | Yes | MIT | Baseline fallback; poor prosody. |

## Interrupt behavior

- **Piper CLI** (`synthesize_with_piper`): one subprocess per utterance. To interrupt, terminate the subprocess or use an app-level queue that stops scheduling new synth jobs.
- **pyttsx3**: engine-specific; often `stop()` on the driver if exposed.
- **Kokoro**: depends on integration; treat like Piper (process or stream cancel).

Document what your app implements; library alone does not guarantee mid-utterance cut-in.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def cmd_tts(args: argparse.Namespace) -> int:
    from src.training.data.core import load_yaml
    from src.training.data.tts.piper import synthesize_with_piper, tts_voice_entries

    _ensure_report_dir()
    _write_tts_compare_md(REPORT_DIR / "tts_compare.md")

    if not TTS_CONFIG.exists():
        print("Missing", TTS_CONFIG, file=sys.stderr)
        return 1

    config = load_yaml(TTS_CONFIG)
    entries = tts_voice_entries(config)
    out_root = Path(str(config["paths"]["validation_output"])) / "week5"
    out_root.mkdir(parents=True, exist_ok=True)

    utterances: list[dict[str, str]] = []
    if UTTERANCES_PATH.exists():
        utterances = json.loads(UTTERANCES_PATH.read_text(encoding="utf-8")).get(
            "utterances", []
        )

    timings: list[dict[str, Any]] = []
    for u in utterances:
        lang = u.get("lang", "en")
        text = u.get("text", "")
        voice = next(
            (e for e in entries if e.get("language", "").startswith(lang)), None
        )
        if voice is None:
            voice = entries[0] if entries else None
        if voice is None:
            timings.append({"text": text, "lang": lang, "error": "no_voice_in_config"})
            continue
        model_path = Path(str(voice["model_path"]))
        wav_path = out_root / f"{lang}_{len(timings):02d}.wav"
        row: dict[str, Any] = {
            "lang": lang,
            "voice_id": voice.get("id"),
            "text": text,
            "wav": str(wav_path.as_posix()),
        }
        if not model_path.exists():
            row["error"] = "missing_model"
            row["model_path"] = str(model_path)
            timings.append(row)
            continue
        t0 = time.perf_counter()
        try:
            synthesize_with_piper(
                model_path=model_path,
                output_wav=wav_path,
                text=text,
            )
            row["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            with wave.open(str(wav_path), "rb") as wf:
                row["wav_frames"] = wf.getnframes()
                row["sample_rate_hz"] = wf.getframerate()
        except (subprocess.CalledProcessError, OSError, wave.Error) as exc:
            row["error"] = str(exc)
        timings.append(row)

    interrupt_note = {
        "scenario": "long_utterance_vs_short_alert",
        "piper_cli_behavior": "Each call blocks until WAV written; interrupt by killing subprocess or skipping queue.",
        "test_status": "documented_only",
    }

    bench = {
        "status": "ok",
        "utterance_count": len(utterances),
        "timings": timings,
        "interrupt": interrupt_note,
        "rss_mb_approx": _rss_mb(),
    }
    _write_json(REPORT_DIR / "piper_benchmark_cpu.json", bench)
    print(
        "Wrote",
        REPORT_DIR / "tts_compare.md",
        "and",
        REPORT_DIR / "piper_benchmark_cpu.json",
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
        ("paddleocr", cmd_paddleocr),
        ("mediapipe", cmd_mediapipe),
        ("tts", cmd_tts),
    ]
    rc = 0
    for name, fn in steps:
        print("===", name, "===")
        r = fn(args)
        if r != 0:
            rc = r
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--image-dir", type=Path, default=None)
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--limit", type=int, default=20)

    po = sub.add_parser("paddleocr", help="PaddleOCR lite OCR samples + benchmark")
    add_common(po)
    po.add_argument("--lang", default="en")
    po.add_argument("--warmup", type=int, default=2)
    po.add_argument("--bench-iters", type=int, default=10)
    po.set_defaults(func=cmd_paddleocr)

    mp = sub.add_parser("mediapipe", help="Face mesh + pose samples + benchmark")
    add_common(mp)
    mp.add_argument(
        "--full-landmarks",
        action="store_true",
        help="Include all 478 face points in JSONL (large files)",
    )
    mp.set_defaults(func=cmd_mediapipe)

    tt = sub.add_parser("tts", help="Piper samples + comparison markdown")
    tt.set_defaults(func=cmd_tts)

    all_p = sub.add_parser(
        "all", help="Run paddleocr, mediapipe, tts with shared flags"
    )
    add_common(all_p)
    all_p.add_argument("--lang", default="en")
    all_p.add_argument("--warmup", type=int, default=2)
    all_p.add_argument("--bench-iters", type=int, default=10)
    all_p.add_argument(
        "--full-landmarks",
        action="store_true",
    )
    all_p.set_defaults(func=cmd_all)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
