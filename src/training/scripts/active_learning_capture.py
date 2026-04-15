"""Copy frames or images into the active-learning inbox with sidecar metadata."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_INBOX = Path("data/active_learning/inbox")


def load_prediction(raw: str | None, path: Path | None) -> dict[str, Any]:
    if path is not None:
        if not path.is_file():
            raise ValueError(f"prediction file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    if raw:
        return json.loads(raw)
    return {}


def ingest_one(
    image_path: Path,
    inbox: Path,
    *,
    model_id: str,
    confidence: float,
    prediction: dict[str, Any],
) -> tuple[Path, Path]:
    if not image_path.is_file():
        raise ValueError(f"image not found: {image_path}")
    inbox.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    safe_stem = f"{ts:.3f}_{image_path.stem}".replace(" ", "_")
    dest = inbox / f"{safe_stem}{image_path.suffix.lower() or '.jpg'}"
    shutil.copy2(image_path, dest)
    meta = {
        "timestamp": ts,
        "source_image": str(image_path.resolve()),
        "model_id": model_id,
        "confidence": confidence if not math.isnan(confidence) else None,
        "prediction": prediction,
    }
    meta_path = inbox / f"{safe_stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest, meta_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", dest="images", help="Image file (repeatable)")
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--confidence", type=float, default=float("nan"))
    parser.add_argument("--prediction-json", type=str, default="", help="Inline JSON object string")
    parser.add_argument("--prediction-file", type=Path, help="Path to JSON prediction payload")
    args = parser.parse_args(argv)
    if not args.images:
        print("need at least one --image", file=sys.stderr)
        return 2
    try:
        prediction = load_prediction(args.prediction_json or None, args.prediction_file)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"invalid prediction: {exc}", file=sys.stderr)
        return 2
    for img in args.images:
        dest, meta = ingest_one(
            img,
            args.inbox,
            model_id=args.model_id,
            confidence=args.confidence,
            prediction=prediction,
        )
        print(f"wrote {dest} {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
