"""Wrapper mỏng chuyển tiếp sang ``data_utils`` subcommand ``download-plan``.

Dùng khi muốn một entrypoint tên gọn (ví dụ script hoặc doc) mà không gõ đủ
``python -m src.training.scripts.data_utils download-plan ...``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from src.training.scripts.data_utils import main as data_utils_main


def main(argv: Sequence[str] | None = None) -> int:
    """Gọi ``data_utils.main`` với tiền tố ``download-plan`` và ``argv`` thêm vào.

    Returns:
        Mã thoát do ``data_utils.main`` trả về (thường ``0`` sau khi in kế hoạch).
    """
    args = ["download-plan"]
    if argv:
        args.extend(argv)
    return data_utils_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
