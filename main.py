#!/usr/bin/env python3
"""
main.py — Entry point for the real-time ML visualization tool.

Architecture Overview
=====================

    ┌──────────────────────┐         ┌──────────────────────┐
    │   Training Thread    │         │    UI Thread          │
    │                      │         │   (daemon)            │
    │  model.forward()     │         │                       │
    │       │              │  write  │  LiveDashboard        │
    │  ┌────▼────┐         │ ──────► │    │                  │
    │  │ Forward  │────┐   │         │    ├── Loss curve     │
    │  │  Hook    │    │   │         │    ├── LR schedule    │
    │  └─────────┘    │   │         │    ├── Grad norms     │
    │                  ▼   │         │    └── Act. stats     │
    │  ┌─────────┐  Monitor│  read   │                       │
    │  │Backward │──►Data  │ ◄────── │  (polls every 250ms)  │
    │  │  Hook   │  Store  │         │                       │
    │  └─────────┘ (RLock) │         │                       │
    │                      │         │                       │
    │  loss.backward()     │         │                       │
    │  optimizer.step()    │         │                       │
    │  monitor.on_step_end │         │                       │
    └──────────────────────┘         └──────────────────────┘

Usage
-----
    python main.py                       # CPU
    python main.py --device mps          # Apple Silicon GPU
    python main.py --epochs 20 --lr 0.01 # custom hyperparams
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from monitor import TrainingMonitor, MonitorDataStore
from viz import LiveDashboard
from models.demo_cnn import DemoCNN
from models.mnist_cnn import MnistCNN


# ──────────────────────────────────────────────────────────────────────
# Training logic — runs on its own thread, fully separated from the UI
# ──────────────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    monitor: TrainingMonitor,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    epochs: int,
) -> None:
    """Standard training loop — the monitor hooks are invisible here."""
    model.train()
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)                    # ← forward hooks fire
            loss = criterion(output, target)
            loss.backward()                         # ← backward hooks fire
            optimizer.step()

            # Record scalars — the ONLY explicit coupling with the monitor.
            monitor.on_step_end(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch,
                accuracy=_batch_accuracy(output, target),
            )

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch}/{epochs}  "
                    f"Step {monitor.step:>5d}  "
                    f"Loss {loss.item():.4f}"
                )

        scheduler.step()

    monitor.detach()
    print("\n✅  Training complete.")


def _batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


# ──────────────────────────────────────────────────────────────────────
# CLI & wiring
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time ML Training Monitor")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Which dataset / model pair to train",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Record hooks every N steps (increase for big models)",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable the live dashboard (headless mode)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"🖥  Device: {device}")

    # ---- data -----------------------------------------------------------
    if args.dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        model = MnistCNN(num_classes=10).to(device)
        print(f"📦  Dataset: MNIST  |  Model: MnistCNN")
    else:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_set = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        model = DemoCNN(num_classes=10).to(device)
        print(f"📦  Dataset: CIFAR-10  |  Model: DemoCNN")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(args.device != "cpu"),
    )

    # ---- optimiser ------------------------------------------------------
    # ---- optimiser ------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- monitoring stack -----------------------------------------------
    store = MonitorDataStore(max_steps=5000)
    monitor = TrainingMonitor(
        model,
        store,
        sample_every=args.sample_every,
    )
    monitor.attach()

    # ---- launch dashboard on a background thread ------------------------
    if not args.no_viz:
        dashboard = LiveDashboard(store, refresh_ms=300)
        dashboard.launch_in_thread()
        time.sleep(0.5)  # let the window initialise before training starts

    # ---- training thread ------------------------------------------------
    train_thread = threading.Thread(
        target=train,
        args=(model, monitor, device, train_loader, optimizer, scheduler, criterion, args.epochs),
        name="train-loop",
    )
    train_thread.start()

    if args.no_viz:
        train_thread.join()
    else:
        # Keep the main thread alive for Matplotlib event loop.
        # When the training finishes, the daemon viz thread exits automatically.
        train_thread.join()
        print("Close the dashboard window to exit.")
        # Block until user closes the plot window
        try:
            while threading.active_count() > 1:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInterrupted — shutting down.")


if __name__ == "__main__":
    main()
