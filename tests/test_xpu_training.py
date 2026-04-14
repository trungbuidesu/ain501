"""Sanity check nhanh: train model nhỏ trên Intel Arc A770 qua XPU."""

import time

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.device import arc_xpu_is_available


@pytest.mark.skipif(
    not arc_xpu_is_available(),
    reason="Intel Arc XPU backend is not available in this environment",
)
def test_xpu_training() -> None:
    """Train model nhỏ trên XPU và xác minh loss ở mức hợp lý."""
    device = torch.device("xpu")
    print(f"Device: {torch.xpu.get_device_name(0)}")

    # Model nhỏ cho sanity check.
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Dữ liệu tổng hợp.
    x = torch.randn(256, 64, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # Vòng lặp huấn luyện.
    start = time.perf_counter()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3d} | Loss: {loss.item():.4f}")

    elapsed = time.perf_counter() - start
    print(f"Training 20 epochs took {elapsed:.3f}s on XPU")

    # Xác minh loss ở mức kỳ vọng.
    assert loss.item() < 2.5, f"Loss too high: {loss.item()}"
    print("✓ XPU training sanity check PASSED")


if __name__ == "__main__":
    test_xpu_training()
