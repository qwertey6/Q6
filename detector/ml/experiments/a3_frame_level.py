"""A3: Frame-level training on OURS-extended.

Instead of one (features, label) pair per fixture, derive per-frame
labels from the generation parameters (which frames are "in a
hazardous burst") and train a frame-level classifier. Aggregate
frame predictions to fixture verdict via max-over-frames.

Hazard frame label heuristic, per category (using the analytical
properties of how the fixture was generated):

  - fps_sweep / codec_roundtrip / boundary_precision / color_space:
    the entire video is either above or below the WCAG count threshold;
    every frame in a "fail" fixture is hazardous, every frame in a
    "pass" fixture is non-hazardous. Frame label = fixture label.
  - area_boundary: same -- whole-video area exceeds threshold or not.
  - false_positive_battery: PASS, all frames non-hazardous.

So for OURS-extended, frame label = fixture label everywhere -- the
generation didn't produce sub-fixture temporal variation.

This means: A3 turns 45 fixtures into ~45 * mean_frames ≈ ~5,000
training samples, all sharing the fixture label. The MLP can now learn
per-frame discriminators (windowed_count, flash_area at THIS frame)
rather than max-aggregated summary stats.

We hold out 20% of OURS-extended fixtures (not frames) for val to
avoid label leakage; test on TRACE fixtures aggregated to verdict.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..dataset import REPO_ROOT, _iter_fixtures
from .common import (ExperimentResult, confusion, print_result, save_result)


PER_FRAME_FEATURE_NAMES = (
    "lum_transitions",
    "red_transitions",
    "flash_area",
    "lum_transitions_minus_3",     # signed margin to WCAG threshold
    "lum_transitions_minus_6",
    "flash_area_minus_0p25",
    "fps",
    "log_n_frames",
)
PER_FRAME_FEATURE_DIM = len(PER_FRAME_FEATURE_NAMES)


def _extract_per_frame_features(video_path: Path,
                                  profile: str = "WCAG2.2-SC2.3.1"
                                  ) -> tuple[np.ndarray, float]:
    """Return per-frame features (N, PER_FRAME_FEATURE_DIM) and fps."""
    from detector import analyze, CV2_CC_BACKEND  # type: ignore
    res = analyze(video_path, profile=profile, cc_backend=CV2_CC_BACKEND)
    if not res.per_frame:
        return np.zeros((0, PER_FRAME_FEATURE_DIM), dtype=np.float32), float(res.fps)
    n_frames = float(len(res.per_frame))
    log_n = math.log10(max(n_frames, 1.0))
    rows = []
    for f in res.per_frame:
        rows.append([
            float(f.lum_transitions),
            float(f.red_transitions),
            float(f.flash_area),
            float(f.lum_transitions) - 3.0,
            float(f.lum_transitions) - 6.0,
            float(f.flash_area) - 0.25,
            float(res.fps),
            log_n,
        ])
    return np.array(rows, dtype=np.float32), float(res.fps)


class FrameMlp(nn.Module):
    """input(PER_FRAME_FEATURE_DIM) -> 16 -> 8 -> 1 logit per frame."""
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(PER_FRAME_FEATURE_DIM),
                                  requires_grad=False)
        self.std = nn.Parameter(torch.ones(PER_FRAME_FEATURE_DIM),
                                 requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(PER_FRAME_FEATURE_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, X):
        return self.net((X - self.mean) / self.std).squeeze(-1)


def run(seed: int = 0, val_frac: float = 0.2, epochs: int = 200,
        lr: float = 3e-3, weight_decay: float = 1e-3) -> ExperimentResult:
    print("A3: per-frame training on OURS-extended")

    # Gather per-fixture per-frame features + fixture labels.
    train_pool = []
    for video_path, fid, label in _iter_fixtures(
            "expected_wcag2_2", sources_in=("OURS-extended",)):
        try:
            feats, fps = _extract_per_frame_features(video_path)
        except Exception as e:
            print(f"  skip train {fid}: {e}")
            continue
        train_pool.append((feats, label, fid))
    print(f"  OURS-extended fixtures: {len(train_pool)}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_pool))
    n_val = max(1, int(round(val_frac * len(train_pool))))
    val_pool = [train_pool[i] for i in perm[:n_val]]
    tr_pool  = [train_pool[i] for i in perm[n_val:]]

    # Concatenate frame-level training data; each frame inherits fixture label.
    X_tr_frames = np.concatenate([fp[0] for fp in tr_pool]) if tr_pool else np.zeros((0, PER_FRAME_FEATURE_DIM), dtype=np.float32)
    y_tr_frames = np.concatenate([
        np.full(len(fp[0]), fp[1], dtype=np.int64) for fp in tr_pool
    ]) if tr_pool else np.zeros((0,), dtype=np.int64)
    print(f"  train frames: {len(X_tr_frames)} (positive: {int((y_tr_frames==1).sum())})")

    # Per-feature norm from training set.
    mean = X_tr_frames.mean(axis=0)
    std = X_tr_frames.std(axis=0); std[std < 1e-6] = 1.0

    torch.manual_seed(seed)
    model = FrameMlp()
    with torch.no_grad():
        model.mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        model.std.copy_(torch.from_numpy(std.astype(np.float32)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(X_tr_frames)
    ytr_t = torch.from_numpy(y_tr_frames.astype(np.float32))
    for epoch in range(epochs):
        model.train()
        logits = model(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()

    # Fixture verdict = max frame probability >= 0.5
    def fixture_verdict(frames_np: np.ndarray) -> int:
        if len(frames_np) == 0:
            return 0
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(frames_np))).numpy()
        return int(probs.max() >= 0.5)

    # Val: per OURS-extended held-out fixture
    val_y_true = np.array([fp[1] for fp in val_pool])
    val_y_pred = np.array([fixture_verdict(fp[0]) for fp in val_pool])

    # Test: TRACE
    test_y_true = []
    test_y_pred = []
    for video_path, fid, label in _iter_fixtures(
            "expected_wcag2_2", source_prefix="pse-test-media"):
        try:
            feats, _fps = _extract_per_frame_features(video_path)
        except Exception:
            continue
        test_y_true.append(label)
        test_y_pred.append(fixture_verdict(feats))
    test_y_true = np.array(test_y_true)
    test_y_pred = np.array(test_y_pred)
    print(f"  test fixtures (TRACE): {len(test_y_true)}")

    result = ExperimentResult(
        approach="A3",
        method="frame_level_mlp_max_aggregation",
        config={"seed": seed, "val_frac": val_frac, "epochs": epochs,
                "lr": lr, "weight_decay": weight_decay,
                "n_train_frames": int(len(X_tr_frames))},
        train_scores={"loss": float(loss.item())},
        val_scores=confusion(val_y_true, val_y_pred),
        test_scores=confusion(test_y_true, test_y_pred),
        notes="frame label = fixture label (gen params don't vary "
              "sub-fixture-level); max-prob aggregation",
    )
    save_result(result)
    print_result(result)
    return result


if __name__ == "__main__":
    run()
