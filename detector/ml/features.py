"""Feature extraction for the q6_mlp detector.

Strategy: run the classical Q6 detector to get its per-frame trace, then
compute summary statistics. The MLP learns to combine the classical
signals into a verdict rather than the rule-based threshold function.

Features (FEATURE_NAMES, in order, length FEATURE_DIM):
  - max_lum_window:        max windowed luminance-transition count
  - max_red_window:        max windowed red-transition count
  - max_flash_area:        max per-frame fraction of pixels firing
  - mean_flash_area:       mean per-frame flash area over the video
  - frac_frames_area_gt25: fraction of frames with flash_area > 0.25
  - frac_frames_lum_gt6:   fraction of frames with windowed lum count > 6
  - frac_frames_lum_gt3:   fraction of frames with windowed lum count > 3
  - frac_frames_red_gt6:   fraction of frames with windowed red count > 6
  - fps:                   video frame rate
  - log_n_frames:          log10 of total frame count (captures duration)

10 features. Chosen to be linear-and-monotone-friendly so a small MLP can
learn the rules + non-linear interactions on top.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np


FEATURE_NAMES: Sequence[str] = (
    "max_lum_window",
    "max_red_window",
    "max_flash_area",
    "mean_flash_area",
    "frac_frames_area_gt25",
    "frac_frames_lum_gt6",
    "frac_frames_lum_gt3",
    "frac_frames_red_gt6",
    "fps",
    "log_n_frames",
)
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(video_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> np.ndarray:
    """Run the classical detector on a video and return a (FEATURE_DIM,)
    float32 feature vector summarising its per-frame trace.

    Raises IOError if the video can't be opened.
    """
    # Lazy import so the harness adapter can import this module without
    # bringing torch / cv2 in until inference time.
    from detector import analyze, CV2_CC_BACKEND  # type: ignore

    res = analyze(video_path, profile=profile, cc_backend=CV2_CC_BACKEND)
    if not res.per_frame:
        # First-frame-only video or empty: return zeros (model treats as PASS).
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    lum_counts = np.array([f.lum_transitions for f in res.per_frame], dtype=np.float32)
    red_counts = np.array([f.red_transitions for f in res.per_frame], dtype=np.float32)
    flash_area = np.array([f.flash_area for f in res.per_frame], dtype=np.float32)
    n_frames = float(len(res.per_frame))

    feats = np.array([
        float(lum_counts.max()) if lum_counts.size else 0.0,
        float(red_counts.max()) if red_counts.size else 0.0,
        float(flash_area.max()) if flash_area.size else 0.0,
        float(flash_area.mean()) if flash_area.size else 0.0,
        float((flash_area > 0.25).mean()) if flash_area.size else 0.0,
        float((lum_counts > 6).mean()) if lum_counts.size else 0.0,
        float((lum_counts > 3).mean()) if lum_counts.size else 0.0,
        float((red_counts > 6).mean()) if red_counts.size else 0.0,
        float(res.fps),
        float(math.log10(max(n_frames, 1.0))),
    ], dtype=np.float32)
    return feats
