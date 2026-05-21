"""Adapter for `flickerfilter` (Hu/Kim/Jo/Han) -- the existing ML-using
PSE detector with available source code.

Flickerfilter is an sklearn ElasticNetCV regression trained on hand-engineered
features over 64x48 HSV per-frame diff time series. It's the only published
PSE detector we found that actually uses ML (Flikcer / Kaya 2025 / FlashGuard
are all classical despite the marketing language). Including it here gives
us an ML baseline for our own learned detectors to be compared against.

Upstream: https://github.com/taehyoungjo/flickerfilter
Algorithm (their analysis/test_analysis.ipynb, cell 11):
    1. Downsample each frame to 64x48 (INTER_NEAREST)
    2. Convert BGR -> HSV (the upstream code calls COLOR_RGB2HSV after
       cv2.VideoCapture; in cv2 that's actually BGR->HSV; we preserve
       their behaviour for fidelity)
    3. Per-frame absolute pixel diff in each H/S/V channel
    4. Per-frame median of the diff (one scalar per channel)
    5. scipy.signal.find_peaks(width=8) on each channel's time series
    6. Per-channel feature triple: (n_peaks, mean, median)
    7. 9-dim feature vector -> sklearn ElasticNetCV -> scalar score
    8. Threshold at FLICKER_FILTER_THRESHOLD -> FAIL/PASS

Model file: NOT shipped in this repo (no license granted on the upstream).
Fetch manually from:
    https://github.com/taehyoungjo/flickerfilter/blob/main/analysis/model.joblib
and place at: harness/adapters/flicker_filter_data/model.joblib

The model was pickled with sklearn 0.19.1 (circa 2018). Modern sklearn
needs the import-path shim below to load it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from harness.schema import NormalizedResult


TOOL = "flicker_filter"
SUPPORTED_PROFILES = [
    "WCAG2.2-SC2.3.1",
    "Trace24",
    "ITU-R-BT.1702",
    "Ofcom-GN2-Annex1",
]
# The model has a single output that's independent of standards profile.
# Same verdict is reported under each profile so per-standard scoring can
# compare it against per-standard labels.
PROFILE_AFFECTS_BEHAVIOR = False


_MODEL_PATH = Path(__file__).parent / "flicker_filter_data" / "model.joblib"
_MODEL_SOURCE_URL = (
    "https://github.com/taehyoungjo/flickerfilter"
    "/raw/refs/heads/main/analysis/model.joblib"
)

# Classification threshold on the model's regression output. Flickerfilter's
# ElasticNet predicts a scalar in roughly [0, 0.5] (their training set was
# ~30% epileptic-tagged YouTube videos with 0/1 labels). Their Chrome
# extension's threshold lives in a hosted Flask backend that isn't in the
# public repo, so we default to 0.5 (the standard regression-as-classifier
# cutoff) and document the choice. Override via FLICKER_FILTER_THRESHOLD.
_THRESHOLD = float(os.environ.get("FLICKER_FILTER_THRESHOLD", "0.5"))

# scipy.signal.find_peaks parameter from their cell 11.
_PEAK_WIDTH = 8

# Downsample target from their pipeline.
_W, _H = 64, 48


def _load_model():
    """Load the sklearn model, with the import-path compat shim. Cached at
    module level after the first successful load."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"flickerfilter model not found at {_MODEL_PATH}.\n"
            f"Download from {_MODEL_SOURCE_URL} and place it there.\n"
            f"(Not committed to this repo; upstream license not granted.)"
        )
    # The model was pickled with sklearn 0.19.1 (~2018), which had the
    # coordinate_descent submodule under sklearn.linear_model.* publicly.
    # Modern sklearn moved it to the private _coordinate_descent path.
    import sklearn.linear_model._coordinate_descent as _cd
    sys.modules.setdefault("sklearn.linear_model.coordinate_descent", _cd)
    import joblib
    return joblib.load(_MODEL_PATH)


_MODEL = None  # lazy-loaded on first run()


def _flickerfilter_version() -> str:
    """Best-effort version string: model file mtime + sklearn version."""
    try:
        import sklearn
        if _MODEL_PATH.exists():
            mtime = int(_MODEL_PATH.stat().st_mtime)
            return f"upstream-2018+sklearn{sklearn.__version__}+model-mtime-{mtime}"
        return f"upstream-2018+sklearn{sklearn.__version__}+model-missing"
    except Exception:
        return "unavailable"


_VERSION = _flickerfilter_version()


def _extract_features(video_path: Path) -> np.ndarray:
    """Compute the 9-dim feature vector from a video, exactly as in
    flickerfilter analysis/test_analysis.ipynb cell 11."""
    from scipy import signal as scipy_signal

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise IOError(f"video has no frames: {video_path}")

    buf = np.empty((frame_count, _H, _W, 3), dtype=np.uint8)
    fc = 0
    while fc < frame_count:
        ok, frame = cap.read()
        if not ok or frame is None:
            if fc > 0:
                buf[fc] = buf[fc - 1]  # repeat last frame on read failure (their behaviour)
            else:
                buf[fc] = 0
        else:
            small = cv2.resize(frame, (_W, _H), interpolation=cv2.INTER_NEAREST)
            buf[fc] = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)  # preserves their bug/choice
        fc += 1
    cap.release()

    # int16 to avoid uint8 underflow in diff
    diffs = np.abs(buf[1:].astype(np.int16) - buf[:-1].astype(np.int16))
    diffs = diffs.reshape(-1, _W * _H, 3)
    diffs = np.median(diffs, axis=1)  # shape (n_frames-1, 3), one HSV triple per frame

    h_peaks = scipy_signal.find_peaks(diffs[:, 0], width=_PEAK_WIDTH)[0]
    s_peaks = scipy_signal.find_peaks(diffs[:, 1], width=_PEAK_WIDTH)[0]
    v_peaks = scipy_signal.find_peaks(diffs[:, 2], width=_PEAK_WIDTH)[0]
    n_peaks = np.array([len(h_peaks), len(s_peaks), len(v_peaks)], dtype=np.float64)
    average = np.asarray(np.average(diffs, axis=0), dtype=np.float64)
    median = np.asarray(np.median(diffs, axis=0), dtype=np.float64)

    return np.concatenate([n_peaks, average, median])


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    """Run flickerfilter on a video, return harness-normalised result."""
    global _MODEL
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=0.0,
            raw_output_path="",
            standard_profile=profile,
            unsupported_reason="flickerfilter operates on video only.",
        ).to_dict()

    t0 = time.perf_counter()
    if _MODEL is None:
        try:
            _MODEL = _load_model()
        except FileNotFoundError as e:
            return NormalizedResult(
                fixture_id=fixture_path.name, verdict="ERROR",
                tool=TOOL, tool_version=_VERSION,
                runtime_seconds=time.perf_counter() - t0,
                raw_output_path="",
                standard_profile=profile,
                error_message=str(e),
            ).to_dict()

    try:
        features = _extract_features(fixture_path).reshape(1, -1)
        score = float(_MODEL.predict(features)[0])
    except Exception as e:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message=f"flickerfilter inference failed: {e}",
        ).to_dict()

    verdict = "FAIL" if score >= _THRESHOLD else "PASS"
    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=verdict,
        failed_dimensions=["luminance"] if verdict == "FAIL" else [],
        first_fail_timestamp=None,  # model doesn't localise
        tool=TOOL, tool_version=_VERSION,
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=profile,
        score=float(score),
    ).to_dict()
