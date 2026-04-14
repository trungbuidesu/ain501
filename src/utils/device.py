"""Tiện ích phát hiện và chọn thiết bị tính toán (XPU / CUDA / CPU).

Dùng chung cho toàn bộ project thay vì viết lại logic detect ở từng module.
"""

from __future__ import annotations

import torch


def arc_xpu_is_available() -> bool:
    """Trả về ``True`` nếu Intel Arc XPU backend được phát hiện trong env.

    Kiểm tra theo thứ tự:
    1. ``torch.xpu`` API tồn tại (build XPU).
    2. Ít nhất một thiết bị XPU khả dụng.
    3. Tên thiết bị đầu tiên chứa chuỗi ``arc`` (không phân biệt hoa thường).
    """
    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return False
        return "arc" in torch.xpu.get_device_name(0).casefold()
    except RuntimeError:
        return False


def resolve_device(requested: str = "auto") -> torch.device:
    """Phân giải tên thiết bị thành ``torch.device`` phù hợp.

    Thứ tự ưu tiên khi ``requested="auto"``:

    1. Intel Arc XPU (nếu ``arc_xpu_is_available()``).
    2. CUDA (nếu ``torch.cuda.is_available()``).
    3. CPU (fallback cuối cùng).

    Khi ``requested`` là ``"cpu"``, ``"cuda"``, ``"xpu"`` hoặc device string
    tường minh khác (ví dụ ``"xpu:0"``), giá trị đó được dùng trực tiếp mà
    không qua fallback chain.

    Args:
        requested: Tên thiết bị yêu cầu hoặc ``"auto"`` để tự phát hiện.

    Returns:
        ``torch.device`` đã được phân giải.
    """
    requested = requested.lower().strip()
    if requested != "auto":
        return torch.device(requested)
    if arc_xpu_is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
