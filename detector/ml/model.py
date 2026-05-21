"""MLP model for the ours_mlp detector.

Small architecture by design: ~10 input features × tiny hidden layers.
With only ~45 training fixtures, anything bigger would overfit.

  input (FEATURE_DIM=10) -> 16 -> ReLU -> 8 -> ReLU -> 1 -> sigmoid

Trainable params: 10*16 + 16 + 16*8 + 8 + 8*1 + 1 = 313 parameters.
That's small enough to fit comfortably with 45 training samples and
permit the model to learn non-linear combinations the rules can't.

Also bundles per-feature mean / std for normalisation. Stored on the
module so the inference adapter doesn't need a separate preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .features import FEATURE_DIM


@dataclass
class FeatureNormaliser:
    mean: torch.Tensor  # (FEATURE_DIM,)
    std: torch.Tensor   # (FEATURE_DIM,)

    @classmethod
    def from_features(cls, X: torch.Tensor) -> "FeatureNormaliser":
        mean = X.mean(dim=0)
        std = X.std(dim=0).clamp(min=1e-6)
        return cls(mean=mean, std=std)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std


class OursMlp(nn.Module):
    """input(10) -> 16 -> 8 -> 1 logit"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.normaliser: Optional[FeatureNormaliser] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.normaliser is not None:
            X = self.normaliser(X)
        return self.net(X).squeeze(-1)

    def set_normaliser(self, normaliser: FeatureNormaliser) -> None:
        self.normaliser = normaliser
