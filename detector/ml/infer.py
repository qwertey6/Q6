"""Inference path for the q6_mlp detector.

Loads the trained model on first call, caches it; exposes
predict_mlp_verdict(video_path, profile) -> (verdict_str, score).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from .features import extract_features, FEATURE_DIM
from .model import OursMlp, FeatureNormaliser


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = REPO_ROOT / "detector" / "ml" / "checkpoints" / "q6_mlp.pt"
DEFAULT_THRESHOLD = 0.5


_MODEL: Optional[OursMlp] = None


def _load() -> OursMlp:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"q6_mlp checkpoint missing at {CKPT_PATH}. "
            f"Run `python3 -m detector.ml.train` first."
        )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = OursMlp()
    model.load_state_dict(ckpt["model_state"])
    model.set_normaliser(FeatureNormaliser(
        mean=torch.tensor(ckpt["normaliser_mean"], dtype=torch.float32),
        std=torch.tensor(ckpt["normaliser_std"], dtype=torch.float32),
    ))
    model.eval()
    _MODEL = model
    return _MODEL


def predict_mlp_verdict(video_path: Path,
                         profile: str = "WCAG2.2-SC2.3.1",
                         threshold: float = DEFAULT_THRESHOLD
                         ) -> Tuple[str, float]:
    """Returns (verdict_str, raw_probability)."""
    model = _load()
    feats = extract_features(video_path, profile=profile)
    with torch.no_grad():
        logit = model(torch.from_numpy(feats).unsqueeze(0))
        prob = float(torch.sigmoid(logit).item())
    verdict = "FAIL" if prob >= threshold else "PASS"
    return verdict, prob
