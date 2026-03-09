"""
Standard CNN for MNIST — designed for the TrainingMonitor dashboard.

Architecture
------------
    Input (1 × 28 × 28)
      │
      ├─ Conv2d(1 → 32, 3×3, pad=1) + BN + ReLU
      ├─ Conv2d(32 → 32, 3×3, pad=1) + BN + ReLU + MaxPool(2)   → 14×14
      │
      ├─ Conv2d(32 → 64, 3×3, pad=1) + BN + ReLU
      ├─ Conv2d(64 → 64, 3×3, pad=1) + BN + ReLU + MaxPool(2)   → 7×7
      │
      ├─ Flatten → 64 × 7 × 7 = 3136
      ├─ Linear(3136 → 128) + ReLU + Dropout(0.3)
      └─ Linear(128 → 10)

The model is deliberately kept small (≈ 120 k params) so that training
finishes in a few minutes on CPU and produces clearly visible gradient /
activation dynamics on the dashboard.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """A standard CNN for MNIST digit classification (10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # ── Block 1 ──────────────────────────────────────────
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # ── Block 2 ──────────────────────────────────────────
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        # ── Classifier ───────────────────────────────────────
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = F.max_pool2d(x, 2)

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.max_pool2d(x, 2)

        # Classifier
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
