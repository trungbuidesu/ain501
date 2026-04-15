"""Merge human annotations into a scene-router style manifest CSV (MVP)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_OUT = Path("data/active_learning/merged/scene_append.csv")


def read_annotations(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"annotations not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_to_scene_manifest(
    ann_rows: list[dict[str, str]],
    *,
    source: str,
    default_split: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, row in enumerate(ann_rows):
        p = (row.get("path") or "").strip()
        label = (row.get("label_correct") or row.get("label") or "").strip()
        if not p or not label:
            raise ValueError(f"row {i + 1}: need path and label_correct")
        split = (row.get("split") or default_split).strip()
        src = (row.get("source") or source).strip()
        out.append({"path": p, "label": label, "source": src, "split": split})
    return out


def write_manifest(rows: list[dict[str, str]], path: Path, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.is_file() else "w"
    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        if mode == "w" or not path.stat().st_size:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="CSV: path,label_correct[,split,source,notes]")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--append", action="store_true", help="Append rows if file exists")
    parser.add_argument("--source", default="active_learning")
    parser.add_argument("--default-split", default="train")
    args = parser.parse_args(argv)
    try:
        ann = read_annotations(args.annotations)
        rows = rows_to_scene_manifest(
            ann,
            source=args.source,
            default_split=args.default_split,
        )
        write_manifest(rows, args.output_csv, append=args.append)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(
        "Next: concatenate with your full manifest or train with "
        "`python -m src.training.scripts.scene_classifier train --manifest-csv ...`",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
