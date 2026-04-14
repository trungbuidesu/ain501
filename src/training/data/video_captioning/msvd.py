# -*- coding: utf-8 -*-
"""MSVD video-captioning helpers.

The Phase 0 pipeline keeps video captioning as Tier 2/deferred, but this
module provides a small normalizer so config validation, notebooks, and future
loaders can agree on one `video_id -> captions[]` shape when MSVD data is
available locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_msvd_captions(path: Path) -> dict[str, list[str]]:
    """Normalize MSVD-style annotations into `video_id -> captions[]`.

    Accepts either a top-level list, a mapping with an `annotations` list, or a
    mapping keyed by `video_id`. Each row may store text under `captions`,
    `caption`, or `sentence`; `video_path` falls back to its filename stem.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    annotations: Any
    if isinstance(data, dict):
        annotations = data.get("annotations", data)
    else:
        annotations = data
    if isinstance(annotations, dict):
        annotations = [
            {"video_id": video_id, "captions": captions}
            for video_id, captions in annotations.items()
        ]
    if not isinstance(annotations, list):
        raise ValueError("Expected MSVD annotations list")

    captions_by_video: dict[str, list[str]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        video_id = annotation.get(
            "video_id",
            annotation.get("image_id", annotation.get("youtube_id")),
        )
        if video_id is None and annotation.get("video_path") is not None:
            video_id = Path(str(annotation["video_path"])).stem
        raw_captions = annotation.get(
            "captions",
            annotation.get("caption", annotation.get("sentence")),
        )
        if video_id is None or raw_captions is None:
            continue
        if isinstance(raw_captions, list):
            captions_by_video.setdefault(str(video_id), []).extend(
                str(caption) for caption in raw_captions
            )
        else:
            captions_by_video.setdefault(str(video_id), []).append(str(raw_captions))
    return captions_by_video
