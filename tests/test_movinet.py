"""Unit tests cho pipeline MoViNet-A0."""

from pathlib import Path

import pytest
import torch

from src.training.data.action_recognition.clip_dataset import VideoClipDataset
from src.training.models.video.movinet import MoViNetA0Backbone


def test_movinet_backbone_structure():
    """Kiểm tra cấu trúc và output shape của MoViNetA0Backbone."""
    num_classes = 10
    model = MoViNetA0Backbone(num_classes=num_classes, causal=True)
    model.eval()

    # Input: [Batch=1, C=3, T=6, H=172, W=172]
    dummy_input = torch.randn(1, 3, 6, 172, 172)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        1,
        num_classes,
    ), f"Expected shape (1, {num_classes}), got {output.shape}"


def test_movinet_causal_streaming():
    """Kiểm tra tính năng streaming (causal) và clean_activation_buffers."""
    model = MoViNetA0Backbone(num_classes=5, causal=True)
    model.eval()

    clip1 = torch.randn(1, 3, 2, 172, 172)
    clip2 = torch.randn(1, 3, 2, 172, 172)

    # Reset buffers
    model.clean_activation_buffers()
    out1 = model(clip1)
    out2 = model(clip2)

    assert out1.shape == (1, 5)
    assert out2.shape == (1, 5)

    # Sau khi clean, inference trên clip2 phải khác (vì mất context từ clip1)
    model.clean_activation_buffers()
    out2_fresh = model(clip2)

    # Lưu ý: Với random weights, out2 và out2_fresh chắc chắn khác nhau
    # do temporal context
    diff = torch.abs(out2 - out2_fresh).sum().item()
    assert (
        diff > 1e-6
    ), "Streaming context seems not working or buffers not cleaned properly"


@pytest.mark.skipif(
    not Path("data/processed/action_accessibility/manifest.csv").exists(),
    reason="Manifest file not found",
)
def test_clip_dataset():
    """Kiểm tra VideoClipDataset với dữ liệu thực tế."""
    manifest_path = Path("data/processed/action_accessibility/manifest.csv")
    class_names = ["BabyCrawling", "Basketball"]  # Giả sử có 2 class này

    dataset = VideoClipDataset(
        manifest_path=manifest_path,
        class_names=class_names,
        n_clip_frames=4,
        resolution=172,
        is_train=False,
    )

    # Chỉ test item đầu tiên
    try:
        clip, label = dataset[0]
        # Dataset giờ trả về uint8 [C, T, H, W]
        assert clip.dtype == torch.uint8
        assert clip.shape == (3, 4, 172, 172)
        assert isinstance(label, int)
    except Exception as e:
        pytest.fail(f"Dataset access failed: {e}")


def test_movinet_xpu_compatibility():
    """Kiểm tra khả năng tương thích với XPU nếu có."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    device = torch.device("xpu")
    model = MoViNetA0Backbone(num_classes=5).to(device)
    dummy_input = torch.randn(1, 3, 4, 172, 172).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.device.type == "xpu"
    assert output.shape == (1, 5)
