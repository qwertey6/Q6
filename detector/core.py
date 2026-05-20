"""detector/core.py -- JAX-accelerated PSE detector.

Per-frame compute runs as ``jax.jit``-compiled tensor ops on whatever
backend JAX has selected (Metal on Apple Silicon, CUDA on NVIDIA, CPU
elsewhere). cv2 is still used for frame I/O and for the rare
connected-components call on the hazard mask (no JAX equivalent; the
call is gated by an early-exit on the cheap ``max()`` over windowed
counts).

Algorithm unchanged from the classical version (THRESHOLDS.md). The
shape of the implementation is:

  - One ``jit``-compiled function fuses the entire per-frame compute
    into a single kernel: sRGB → relative luminance, saturated red,
    region-mean ΔL (via local-mean convolution), per-axis accumulator
    + anchor gate, sliding-window count update.
  - Region-mean ΔL is approximated as a windowed average rather than
    computed via connected components. For large uniform flash regions
    (which is what TRACE / production content looks like) the two are
    functionally equivalent; the convolution form is jit-friendly and
    much faster than a boundary-crossing CC call per frame.
  - Sliding window is a list of ``window_frames`` jnp arrays plus a
    write-index pointer; the running count is maintained by jit as a
    pure functional update.

This file kept the same public API (``analyze``, ``Result``, ``PerFrame``,
``Profile``, ``PROFILES``, ``detect_static_pattern_hazard``); harness
adapters don't need to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

import cv2  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np


# Suppress JAX's metal-backend chatter on import.
import os as _os
_os.environ.setdefault("JAX_PLATFORM_NAME", _os.environ.get("JAX_PLATFORM_NAME", ""))


# --- Constants (sourced from detector/THRESHOLDS.md) -----------------------

# sRGB → linear LUT. 256 entries (input is discrete uint8), so this is
# exact, not an approximation. Stored as a jnp array for jit'd lookup.
def _build_srgb_lin_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for i in range(256):
        v = i / 255.0
        lut[i] = v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return lut
_SRGB_LIN_LUT_NP = _build_srgb_lin_lut()
_SRGB_LIN_LUT = jnp.asarray(_SRGB_LIN_LUT_NP)


@dataclass(frozen=True)
class Profile:
    name: str
    general_flash_max_per_second: int = 3
    general_flash_luminance_delta: float = 0.1
    general_flash_darker_bound: float = 0.8
    area_fraction_limit: float = 0.25
    area_pixels_limit: int = 10_000_000  # effectively disabled by default
    red_flash_max_per_second: int = 3
    red_sat_delta: int = 20
    red_sat_min: int = 80
    sliding_window_seconds: float = 1.0
    absolute_flashes_per_second_cap: Optional[int] = None
    pattern_hazard_enabled: bool = False
    pattern_min_bars: int = 5
    pattern_min_area_fraction: float = 0.40


# Reference visual-field rectangle (Harding / CRS FCS Implementation
# Guide convention: 341×256 px = 10° of central vision; 25% of that =
# 21,824 px is the WCAG-strict hazard threshold).
_REF_RECT_W      = 341
_REF_RECT_H      = 256
_REF_RECT_AREA   = _REF_RECT_W * _REF_RECT_H
_WCAG_AREA_LIMIT = int(round(0.25 * _REF_RECT_AREA))


PROFILES: dict[str, Profile] = {
    "WCAG2.2-SC2.3.1": Profile(
        name="WCAG2.2-SC2.3.1",
        area_fraction_limit=0.25,
        area_pixels_limit=_WCAG_AREA_LIMIT,
        pattern_hazard_enabled=True,
    ),
    "WCAG2.2-classic": Profile(
        name="WCAG2.2-classic",
        area_fraction_limit=1.0,
        area_pixels_limit=_REF_RECT_AREA,
        pattern_hazard_enabled=True,
    ),
    "ITU-R-BT.1702":    Profile(name="ITU-R-BT.1702",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Ofcom-GN2-Annex1": Profile(name="Ofcom-GN2-Annex1",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Trace24":          Profile(name="Trace24",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "NAB-J":            Profile(name="NAB-J",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  absolute_flashes_per_second_cap=5,
                                  pattern_hazard_enabled=True),
}


# --- Result types ----------------------------------------------------------

@dataclass
class PerFrame:
    frame: int
    timestamp: float
    lum_transitions: int
    red_transitions: int
    flash_area: float
    pattern_risk: float = 0.0


@dataclass
class Result:
    verdict: str
    failed_dimensions: list[str] = field(default_factory=list)
    first_fail_timestamp: Optional[float] = None
    per_frame: list[PerFrame] = field(default_factory=list)
    profile_name: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    n_frames: int = 0


# --- JAX state types (immutable; jit-friendly) -----------------------------

class AxisState(NamedTuple):
    acc: jnp.ndarray            # HxW float32, signed accumulator
    anchor: jnp.ndarray         # HxW float32, signal value at last fire
    window_counts: jnp.ndarray  # HxW int16, running sum of fires in window


# --- Pixel-feature kernels (jit'd) -----------------------------------------

@jax.jit
def _srgb_to_relative_luminance(bgr_uint8: jnp.ndarray) -> jnp.ndarray:
    """L in [0,1] from BGR uint8 image, using the precomputed LUT."""
    b_lin = _SRGB_LIN_LUT[bgr_uint8[..., 0]]
    g_lin = _SRGB_LIN_LUT[bgr_uint8[..., 1]]
    r_lin = _SRGB_LIN_LUT[bgr_uint8[..., 2]]
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


@jax.jit
def _saturated_red(bgr_uint8: jnp.ndarray) -> jnp.ndarray:
    """R - max(G, B), clamped to [0, 255], as float32."""
    b = bgr_uint8[..., 0].astype(jnp.int16)
    g = bgr_uint8[..., 1].astype(jnp.int16)
    r = bgr_uint8[..., 2].astype(jnp.int16)
    return jnp.clip(r - jnp.maximum(g, b), 0, 255).astype(jnp.float32)


# --- Region-mean ΔL (local-mean approximation, jit'd) ----------------------

def _regional_delta(delta_np: np.ndarray,
                    intensity_threshold: float) -> np.ndarray:
    """CC-based region-mean ΔL (numpy/cv2). For each connected component
    of "active" pixels (|delta| > threshold/4) with area >= 100, replace
    each pixel's ΔL with the component's mean ΔL. Inactive pixels keep
    their original (small) delta.

    Stays on the numpy side because cv2.connectedComponentsWithStats has
    no jnp equivalent. One numpy↔jnp boundary crossing per frame; cheap
    relative to the per-frame JAX compute that follows. An early version
    tried a pure-JAX local-mean approximation via separable box blur,
    but it under-counted on TRACE's wcagc fixtures (kernel-window mean
    differs from true-region mean near boundaries); CC is exact.
    """
    active = np.abs(delta_np) > intensity_threshold * 0.25
    if not active.any():
        return delta_np
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return delta_np
    out = delta_np.copy()
    for li in range(1, n_labels):
        if int(stats[li, cv2.CC_STAT_AREA]) < 100:
            continue
        region_mask = labels == li
        out[region_mask] = float(delta_np[region_mask].mean())
    return out


# --- Per-axis step (jit'd) -------------------------------------------------

@jax.jit
def _axis_step_lum(state: AxisState, signal: jnp.ndarray,
                    delta: jnp.ndarray,
                    threshold: float, darker_bound: float):
    """One frame of per-axis work for luminance: accumulator update +
    reset, anchor advance, return (new_state, fired_mask). ``delta`` is
    the pre-computed (region-mean) ΔL vs the previous frame."""
    new_acc = state.acc + delta
    crossed = (new_acc >= threshold) | (new_acc <= -threshold)
    # Darker-bound gate: min(L_current, L_anchor) < darker_bound. When
    # darker_bound >= 1.0 (effectively disabled) the gate passes everything.
    gate_ok = jnp.minimum(signal, state.anchor) < darker_bound
    fired = crossed & gate_ok
    new_acc = jnp.where(fired, 0.0, new_acc)
    new_anchor = jnp.where(fired, signal, state.anchor)
    return AxisState(new_acc, new_anchor, state.window_counts), fired


@jax.jit
def _axis_step_red(state: AxisState, signal: jnp.ndarray,
                    delta: jnp.ndarray,
                    threshold: float, sat_min: float):
    """Red axis: same shape; the gate is the saturated-red minimum (the
    LARGER of the two endpoints must clear sat_min)."""
    new_acc = state.acc + delta
    crossed = (new_acc >= threshold) | (new_acc <= -threshold)
    gate_ok = jnp.maximum(signal, state.anchor) >= sat_min
    fired = crossed & gate_ok
    new_acc = jnp.where(fired, 0.0, new_acc)
    new_anchor = jnp.where(fired, signal, state.anchor)
    return AxisState(new_acc, new_anchor, state.window_counts), fired


@jax.jit
def _window_update(window_counts: jnp.ndarray, new_fired: jnp.ndarray,
                   evicted: jnp.ndarray) -> jnp.ndarray:
    """Incremental update: add new fires, subtract evicted fires."""
    return window_counts + new_fired.astype(jnp.int16) - evicted.astype(jnp.int16)


# --- Region-aware area decision (numpy/cv2 boundary; called rarely) --------

def _region_exceeds_area(hazard_mask_np: np.ndarray, n_pixels: int,
                         fraction_limit: float, pixels_limit: int) -> bool:
    """True iff any connected component of ``hazard_mask`` exceeds either
    the screen-fraction limit or the absolute pixel-area limit. Numpy-side
    (cv2 has no jnp equivalent); called only when the cheap max-check
    early-exit fails."""
    if not hazard_mask_np.any():
        return False
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        hazard_mask_np.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return False
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return (max_area > pixels_limit) or ((max_area / n_pixels) > fraction_limit)


# --- Static spatial-pattern detection (ITU-R BT.1702 §3) -------------------

def detect_static_pattern_hazard(image_path: Path,
                                  profile: str = "ITU-R-BT.1702"
                                  ) -> tuple[bool, dict]:
    """Return (is_hazardous, info) for a still image under ``profile``."""
    p = PROFILES[profile]
    if not p.pattern_hazard_enabled:
        return False, {"reason": "pattern_hazard disabled in profile",
                       "profile": profile}

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, {"reason": "image unreadable", "path": str(image_path)}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=max(15, (min(h, w) // 32) | 1), C=2,
    )

    def _bar_count_and_band(profile_1d: np.ndarray) -> tuple[int, tuple[int, int]]:
        sign = (profile_1d > profile_1d.mean()).astype(np.int8)
        diffs = np.abs(np.diff(sign))
        n_transitions = int(diffs.sum())
        if n_transitions == 0:
            return 0, (0, 0)
        nz = np.nonzero(diffs)[0]
        return n_transitions // 2, (int(nz[0]), int(nz[-1]))

    v_bars, (vy0, vy1) = _bar_count_and_band(binarized.mean(axis=0))
    h_bars, (hx0, hx1) = _bar_count_and_band(binarized.mean(axis=1))

    if v_bars >= h_bars:
        bar_count = v_bars
        coverage_px = (vy1 - vy0 + 1) * h
        dominant_axis = "vertical"
    else:
        bar_count = h_bars
        coverage_px = (hx1 - hx0 + 1) * w
        dominant_axis = "horizontal"
    coverage_frac = coverage_px / float(w * h)

    is_hazard = (bar_count >= p.pattern_min_bars and
                 coverage_frac > p.pattern_min_area_fraction)
    return is_hazard, {
        "bar_count": int(bar_count),
        "vertical_bars": int(v_bars),
        "horizontal_bars": int(h_bars),
        "dominant_axis": dominant_axis,
        "coverage_area_fraction": float(coverage_frac),
        "min_bars_threshold": int(p.pattern_min_bars),
        "min_area_threshold": float(p.pattern_min_area_fraction),
    }


# --- Main analyzer ---------------------------------------------------------

def _initial_axis_state(h: int, w: int) -> AxisState:
    return AxisState(
        acc=jnp.zeros((h, w), dtype=jnp.float32),
        anchor=jnp.zeros((h, w), dtype=jnp.float32),
        window_counts=jnp.zeros((h, w), dtype=jnp.int16),
    )


def analyze(video_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> Result:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    p = PROFILES[profile]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release()
        raise IOError(f"video has no usable fps metadata: {video_path}")

    n_pixels = max(1, width * height)
    window_frames = max(1, int(round(p.sliding_window_seconds * fps)))

    # Profile-derived static thresholds (closed over by jit; the jit cache
    # is keyed on these as Python floats, so each profile compiles once).
    lum_threshold = float(p.general_flash_luminance_delta)
    lum_darker_bound = float(p.general_flash_darker_bound)
    red_threshold = float(p.red_sat_delta)
    red_sat_min = float(p.red_sat_min)
    lum_count_thresh = 2 * p.general_flash_max_per_second
    red_count_thresh = 2 * p.red_flash_max_per_second
    abs_cap = (2 * p.absolute_flashes_per_second_cap
               if p.absolute_flashes_per_second_cap is not None else None)

    # Per-axis state, plus circular buffer of fired masks for window
    # eviction. Buffers are Python lists of jnp arrays; the running
    # `window_counts` inside each AxisState is the maintained sum.
    lum_state = _initial_axis_state(height, width)
    red_state = _initial_axis_state(height, width)
    zero_fired = jnp.zeros((height, width), dtype=jnp.bool_)
    lum_window = [zero_fired] * window_frames
    red_window = [zero_fired] * window_frames
    write_idx = 0

    prev_L: Optional[jnp.ndarray] = None
    prev_R: Optional[jnp.ndarray] = None
    per_frame: list[PerFrame] = []
    failed_dims: set[str] = set()
    first_fail_ts: Optional[float] = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        timestamp = (frame_idx - 1) / fps

        frame_j = jnp.asarray(frame)  # numpy uint8 → jax (zero-copy on CPU)
        L = _srgb_to_relative_luminance(frame_j)
        R = _saturated_red(frame_j)

        if prev_L is None:
            prev_L, prev_R = L, R
            lum_state = lum_state._replace(anchor=L)
            red_state = red_state._replace(anchor=R)
            per_frame.append(PerFrame(frame=frame_idx, timestamp=timestamp,
                                       lum_transitions=0, red_transitions=0,
                                       flash_area=0.0))
            continue

        # Region-mean ΔL on numpy/cv2 side. One boundary crossing per
        # frame; cheap relative to the per-frame JAX compute.
        lum_delta = jnp.asarray(_regional_delta(
            np.asarray(L - prev_L), lum_threshold))
        red_delta = jnp.asarray(_regional_delta(
            np.asarray(R - prev_R), red_threshold))

        # Per-axis update (fully jit'd).
        lum_state, lum_fired = _axis_step_lum(
            lum_state, L, lum_delta, lum_threshold, lum_darker_bound)
        red_state, red_fired = _axis_step_red(
            red_state, R, red_delta, red_threshold, red_sat_min)

        # Incremental window update (jit'd). Evict the mask in the
        # current write slot (which is the oldest entry once the window
        # is full; before that it's an all-zero placeholder).
        new_lum_counts = _window_update(
            lum_state.window_counts, lum_fired, lum_window[write_idx])
        new_red_counts = _window_update(
            red_state.window_counts, red_fired, red_window[write_idx])
        lum_state = lum_state._replace(window_counts=new_lum_counts)
        red_state = red_state._replace(window_counts=new_red_counts)
        lum_window[write_idx] = lum_fired
        red_window[write_idx] = red_fired
        write_idx = (write_idx + 1) % window_frames

        # Early-exit hazard test. Max-check is cheap; CC only if it could
        # possibly trigger.
        lum_max = int(new_lum_counts.max())
        red_max = int(new_red_counts.max())

        if "luminance" not in failed_dims and lum_max > lum_count_thresh:
            mask_np = np.asarray(new_lum_counts > lum_count_thresh)
            if _region_exceeds_area(mask_np, n_pixels,
                                     p.area_fraction_limit, p.area_pixels_limit):
                failed_dims.add("luminance")
                if first_fail_ts is None: first_fail_ts = timestamp
        if "red" not in failed_dims and red_max > red_count_thresh:
            mask_np = np.asarray(new_red_counts > red_count_thresh)
            if _region_exceeds_area(mask_np, n_pixels,
                                     p.area_fraction_limit, p.area_pixels_limit):
                failed_dims.add("red")
                if first_fail_ts is None: first_fail_ts = timestamp
        if abs_cap is not None and "count" not in failed_dims:
            if lum_max > abs_cap or red_max > abs_cap:
                failed_dims.add("count")
                if first_fail_ts is None: first_fail_ts = timestamp

        per_frame.append(PerFrame(
            frame=frame_idx, timestamp=timestamp,
            lum_transitions=lum_max,
            red_transitions=red_max,
            flash_area=float(max(lum_fired.sum(), red_fired.sum())) / n_pixels,
        ))
        prev_L, prev_R = L, R

    cap.release()
    return Result(
        verdict="FAIL" if failed_dims else "PASS",
        failed_dimensions=sorted(failed_dims),
        first_fail_timestamp=first_fail_ts,
        per_frame=per_frame,
        profile_name=p.name,
        fps=float(fps),
        width=width,
        height=height,
        n_frames=frame_idx,
    )


# --- CLI -------------------------------------------------------------------

def _main_cli(argv: list[str]) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="Run our PSE detector on a video.")
    ap.add_argument("video", type=Path, nargs="?",
                    help="Input video. Optional only when --self-test is set.")
    ap.add_argument("--profile", default="WCAG2.2-SC2.3.1", choices=list(PROFILES))
    ap.add_argument("--per-frame-csv", type=Path, default=None,
                    help="Optional path to write a per-frame CSV.")
    ap.add_argument("--self-test", action="store_true",
                    help="Quick smoke self-test on a tiny synthetic video.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    res = analyze(args.video, args.profile)
    print(json.dumps({
        "video":                str(args.video),
        "profile":              res.profile_name,
        "verdict":              res.verdict,
        "failed_dimensions":    res.failed_dimensions,
        "first_fail_timestamp": res.first_fail_timestamp,
        "fps":                  res.fps,
        "n_frames":             res.n_frames,
        "backend":              str(jax.default_backend()),
    }, indent=2))

    if args.per_frame_csv:
        import csv
        from harness.schema import PER_FRAME_CSV_HEADER
        args.per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_frame_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(PER_FRAME_CSV_HEADER)
            for pf in res.per_frame:
                w.writerow([pf.frame, pf.lum_transitions, pf.red_transitions,
                            f"{pf.flash_area:.6f}", f"{pf.pattern_risk:.4f}"])
    return 0


def _self_test() -> int:
    """Smoke test on two synthetic clips labelled analytically."""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="detector_selftest_"))
    try:
        passing = tmp / "two_hz.mp4"
        failing = tmp / "five_hz.mp4"
        for path, hz in ((passing, 2.0), (failing, 5.0)):
            writer = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      30.0, (320, 240))
            for fi in range(int(2 * 30)):
                state = int((fi / 30.0) * 2 * hz) % 2
                img = np.full((240, 320, 3), 255 if state else 0, dtype=np.uint8)
                writer.write(img)
            writer.release()
        r_pass = analyze(passing)
        r_fail = analyze(failing)
        print(f"self-test (jax backend={jax.default_backend()}): "
              f"2Hz -> {r_pass.verdict} (expect PASS); "
              f"5Hz -> {r_fail.verdict} (expect FAIL); dims={r_fail.failed_dimensions}")
        return 0 if (r_pass.verdict == "PASS" and r_fail.verdict == "FAIL") else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
