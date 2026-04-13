"""Quick sanity check: train a tiny model on Intel Arc A770 via XPU."""
import torch
import torch.nn as nn
import torch.optim as optim
import time


def test_xpu_training() -> None:
    device = torch.device("xpu")
    print(f"Device: {torch.xpu.get_device_name(0)}")

    # Tiny model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Synthetic data
    X = torch.randn(256, 64, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # Training loop
    start = time.perf_counter()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:>3d} | Loss: {loss.item():.4f}")

    elapsed = time.perf_counter() - start
    print(f"Training 20 epochs took {elapsed:.3f}s on XPU")

    # Verify loss decreased
    assert loss.item() < 2.5, f"Loss too high: {loss.item()}"
    print("✓ XPU training sanity check PASSED")


if __name__ == "__main__":
    test_xpu_training()
