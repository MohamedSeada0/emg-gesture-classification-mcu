#!/usr/bin/env python3
"""Smaller CNN for UCI EMG gesture classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network_Small(nn.Module):
    """Reduced CNN tailored for resource-limited deployment.

    This version keeps the same conceptual structure as the original Network_XL but uses a
    much smaller channel width and FC layer size.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 8, (5, 3), padding=(4, 2))
        self.conv2_1 = nn.Conv2d(8, 16, (5, 3), padding=(4, 2))
        self.conv3 = nn.Conv2d(16, 32, (5, 3), padding=(4, 2))

        self.pool = nn.MaxPool2d((3, 1))
        self.BN1 = nn.BatchNorm2d(8)
        self.BN2 = nn.BatchNorm2d(16)
        self.BN3 = nn.BatchNorm2d(32)
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout2d()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 52, 8)
            dummy = self.conv1_1(dummy)
            dummy = self.BN1(dummy)
            dummy = self.prelu(dummy)
            dummy = self.pool(dummy)
            dummy = self.conv2_1(dummy)
            dummy = self.BN2(dummy)
            dummy = self.prelu(dummy)
            dummy = self.drop(dummy)
            dummy = self.pool(dummy)
            dummy = self.conv3(dummy)
            dummy = self.BN3(dummy)
            dummy = self.prelu(dummy)
            dummy = self.drop(dummy)
            dummy = self.pool(dummy)
            flat_dim = int(torch.flatten(dummy, 1).shape[1])
            print(f"[Network_Small] inferred fc1 input features = {flat_dim}")
            assert flat_dim > 0, "Flattened feature size must be positive. Check input shape or layers."

        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.BN1(self.conv1_1(x)))
        x = self.pool(x)
        x = self.prelu(self.BN2(self.conv2_1(x)))
        x = self.drop(x)
        x = self.pool(x)
        x = self.prelu(self.BN3(self.conv3(x)))
        x = self.drop(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        return x  # raw logits; CrossEntropyLoss handles softmax internally


class Network_Small_Quantizable(nn.Module):
    """Network_Small copy using ReLU activations for FX static quantization."""

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 8, (5, 3), padding=(4, 2))
        self.conv2_1 = nn.Conv2d(8, 16, (5, 3), padding=(4, 2))
        self.conv3 = nn.Conv2d(16, 32, (5, 3), padding=(4, 2))

        self.pool = nn.MaxPool2d((3, 1))
        self.BN1 = nn.BatchNorm2d(8)
        self.BN2 = nn.BatchNorm2d(16)
        self.BN3 = nn.BatchNorm2d(32)
        self.prelu = nn.ReLU()
        self.drop = nn.Dropout2d()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 52, 8)
            dummy = self.conv1_1(dummy)
            dummy = self.BN1(dummy)
            dummy = self.prelu(dummy)
            dummy = self.pool(dummy)
            dummy = self.conv2_1(dummy)
            dummy = self.BN2(dummy)
            dummy = self.prelu(dummy)
            dummy = self.drop(dummy)
            dummy = self.pool(dummy)
            dummy = self.conv3(dummy)
            dummy = self.BN3(dummy)
            dummy = self.prelu(dummy)
            dummy = self.drop(dummy)
            dummy = self.pool(dummy)
            flat_dim = int(torch.flatten(dummy, 1).shape[1])
            print(f"[Network_Small_Quantizable] inferred fc1 input features = {flat_dim}")
            assert flat_dim > 0, "Flattened feature size must be positive. Check input shape or layers."

        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.BN1(self.conv1_1(x)))
        x = self.pool(x)
        x = self.prelu(self.BN2(self.conv2_1(x)))
        x = self.drop(x)
        x = self.pool(x)
        x = self.prelu(self.BN3(self.conv3(x)))
        x = self.drop(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.prelu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = Network_Small(6)
    dummy = torch.randn(1, 1, 52, 8)
    out = model(dummy)
    print(f"dummy output shape: {tuple(out.shape)}")
    print(f"total params: {sum(p.numel() for p in model.parameters())}")
