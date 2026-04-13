# -*- coding: utf-8 -*-
"""Wrapper cho bước lập kế hoạch download dataset Phase 0."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from src.training.scripts.data_utils import main as data_utils_main


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy subcommand `download-plan` với tham số CLI được truyền vào."""
    args = ["download-plan"]
    if argv:
        args.extend(argv)
    return data_utils_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
