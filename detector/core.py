"""detector/core.py -- implementation of our PSE detector.

Per detector/THRESHOLDS.md. Each numeric constant references the clause
it implements. No constant is tuned against benchmark labels.

Entry point: ``analyze(video_path, profile) -> Result``.

Algorithm
---------
Per frame:

1. Compute relative-luminance map L[y,x] in [0, 1] from sRGB via a
   precomputed 256-entry LUT (the input is discrete uint8 per channel,
   so the LUT is mathematically exact, not an approximation).
2. Compute saturated-red map R_sat[y,x] = max(R - max(G, B), 0).
3. For each axis (luminance, red), run one ``_axis_step``:
   - Compute per-pixel ΔL (or ΔR) vs previous frame.
   - Replace per-pixel Δ with the region-mean Δ of its connected
     component on the |Δ| > frac × threshold mask (handles codec-noise
     fragmentation; aligns with the standard's "mean over the region"
     intent).
   - Accumulate; on threshold crossing, gate by the WCAG darker-bound
     (or red saturated-min) anchor check. Only reset accumulator +
     advance anchor for pixels that pass the gate.
   - Incrementally update a HxW int16 running sum of fires within the
     1-second sliding window (add new fires, subtract evicted).
4. For each axis, hazard test: if windowed-count max > threshold AND
   any connected component of (windowed_count > threshold) exceeds the
   area limit, mark axis FAIL.

Each axis has one ``AxisState`` (per-pixel buffers) and one
``AxisConfig`` (thresholds + anchor gate). The two axes share
``_axis_step``; the loop body is therefore short.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2  # type: ignore
import numpy as np


# --- Constants (sourced from detector/THRESHOLDS.md) -----------------------

# sRGB → linear LUT (WCAG 2.2 Relative Luminance definition). Input is
# 8-bit per channel, so a 256-entry table is mathematically exact.
_SRGB_LIN_LUT: np.ndarray = np.empty(256, dtype=np.float32)
for _i in range(256):
    _v = _i / 255.0
    _SRGB_LIN_LUT[_i] = _v / 12.92 if _v <= 0.03928 else ((_v + 0.055) / 1.055) ** 2.4


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


# --- Pixel-feature helpers -------------------------------------------------

def _srgb_to_relative_luminance(bgr_uint8: np.ndarray) -> np.ndarray:
    """Relative luminance L in [0, 1] from an HxWx3 BGR uint8 image.

    Uses a precomputed 256-entry LUT for the sRGB → linear step (exact,
    since input is discrete uint8), then the WCAG-cited coefficients.
    """
    b_lin = _SRGB_LIN_LUT[bgr_uint8[..., 0]]
    g_lin = _SRGB_LIN_LUT[bgr_uint8[..., 1]]
    r_lin = _SRGB_LIN_LUT[bgr_uint8[..., 2]]
    return (0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin).astype(np.float32)


def _saturated_red(bgr_uint8: np.ndarray) -> np.ndarray:
    """Harding saturated-red value, R - max(G, B), clamped to [0, 255]."""
    b = bgr_uint8[..., 0].astype(np.int16)
    g = bgr_uint8[..., 1].astype(np.int16)
    r = bgr_uint8[..., 2].astype(np.int16)
    sat = r - np.maximum(g, b)
    np.clip(sat, 0, 255, out=sat)
    return sat.astype(np.float32)


# --- Region-level ΔL aggregation -------------------------------------------

def _regional_delta(delta: np.ndarray,
                    intensity_threshold: float,
                    active_threshold_frac: float,
                    min_region_area_px: int) -> np.ndarray:
    """Replace per-pixel ``delta`` with the mean delta of each pixel's
    connected component in the active mask (|delta| > frac × threshold).

    Pixels not in any significant component keep their original delta.
    Aligns the per-pixel signal with the standards' region-mean intent
    and resists codec-noise fragmentation.
    """
    active = np.abs(delta) > intensity_threshold * active_threshold_frac
    if not active.any():
        return delta
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return delta
    out = delta.copy()
    for li in range(1, n_labels):
        if int(stats[li, cv2.CC_STAT_AREA]) < min_region_area_px:
            continue
        region_mask = labels == li
        out[region_mask] = float(delta[region_mask].mean())
    return out


# --- Region-aware area decision --------------------------------------------

def _region_exceeds_area(hazard_mask: np.ndarray, n_pixels: int,
                         fraction_limit: float, pixels_limit: int) -> bool:
    """True iff any connected component of ``hazard_mask`` exceeds EITHER
    the screen-fraction limit OR the absolute pixel-area limit."""
    if not hazard_mask.any():
        return False
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        hazard_mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return False
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return (max_area > pixels_limit) or ((max_area / n_pixels) > fraction_limit)


# --- Per-axis pipeline -----------------------------------------------------

AnchorGate = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class AxisConfig:
    """Algorithmic parameters for one axis (luminance or red)."""
    threshold: float                       # accumulator fires above this magnitude
    count_per_second_limit: int            # WCAG-style "more than N flashes/sec"
    active_threshold_frac: float = 0.25    # for _regional_delta
    region_min_area_px: int = 100          # for _regional_delta
    anchor_gate: Optional[AnchorGate] = None  # None = always allow fire


@dataclass
class AxisState:
    """Per-pixel state for one axis, mutated in place each frame."""
    acc: np.ndarray            # signed accumulator (HxW float32)
    anchor: np.ndarray         # signal value at last successful fire (HxW float32)
    window: deque              # deque of per-frame fired masks (bool HxW)
    window_counts: np.ndarray  # incremental running sum of fires in window (HxW int16)

    @classmethod
    def initial(cls, h: int, w: int, window_frames: int) -> "AxisState":
        return cls(
            acc=np.zeros((h, w), dtype=np.float32),
            anchor=np.zeros((h, w), dtype=np.float32),
            window=deque(maxlen=window_frames),
            window_counts=np.zeros((h, w), dtype=np.int16),
        )


def _axis_step(state: AxisState, prev_signal: np.ndarray,
               signal: np.ndarray, cfg: AxisConfig) -> np.ndarray:
    """Advance one axis by one frame. Mutates ``state`` in place; returns
    the boolean mask of pixels that fired this frame (post-gate)."""
    delta = signal - prev_signal
    delta = _regional_delta(delta, cfg.threshold,
                            cfg.active_threshold_frac, cfg.region_min_area_px)

    state.acc += delta
    crossed = (state.acc >= cfg.threshold) | (state.acc <= -cfg.threshold)

    if cfg.anchor_gate is not None:
        fired = crossed & cfg.anchor_gate(signal, state.anchor)
    else:
        fired = crossed

    # Reset and advance anchor ONLY for fired pixels (not for pixels that
    # crossed but failed the gate -- otherwise codec-smeared transitions
    # discard their residual mid-flight).
    state.acc[fired] = 0.0
    state.anchor[fired] = signal[fired]

    # Incremental sliding-window count: subtract the evicted frame's
    # fires (if window is full) before appending the new fires.
    if len(state.window) == state.window.maxlen:
        state.window_counts -= state.window[0].astype(np.int16)
    state.window.append(fired)
    state.window_counts += fired.astype(np.int16)

    return fired


def _axis_hazard(state: AxisState, count_threshold: int, n_pixels: int,
                 fraction_limit: float, pixels_limit: int) -> bool:
    """True iff this axis is in a hazardous state THIS frame: some pixel
    has windowed_count > count_threshold AND that hazardous region
    exceeds the area limit. Early-exit on the cheap max() check."""
    if int(state.window_counts.max()) <= count_threshold:
        return False
    hazard_mask = state.window_counts > count_threshold
    return _region_exceeds_area(hazard_mask, n_pixels, fraction_limit, pixels_limit)


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

    # Per-axis configuration. The gates close over Profile values; both
    # axes' gates use the symmetric "darker endpoint of the flash pair"
    # logic, the luminance version with a min threshold, the red version
    # with a max threshold (we want the brighter red value to clear
    # Harding's saturated-red minimum).
    lum_gate = (
        (lambda L_cur, L_anch: np.minimum(L_cur, L_anch) < p.general_flash_darker_bound)
        if p.general_flash_darker_bound < 1.0 else None
    )
    red_gate = (
        (lambda R_cur, R_anch: np.maximum(R_cur, R_anch) >= p.red_sat_min)
        if p.red_sat_min > 0 else None
    )
    lum_cfg = AxisConfig(
        threshold=p.general_flash_luminance_delta,
        count_per_second_limit=p.general_flash_max_per_second,
        anchor_gate=lum_gate,
    )
    red_cfg = AxisConfig(
        threshold=float(p.red_sat_delta),
        count_per_second_limit=p.red_flash_max_per_second,
        anchor_gate=red_gate,
    )
    lum_state = AxisState.initial(height, width, window_frames)
    red_state = AxisState.initial(height, width, window_frames)

    lum_count_thresh = 2 * lum_cfg.count_per_second_limit
    red_count_thresh = 2 * red_cfg.count_per_second_limit
    abs_cap = (2 * p.absolute_flashes_per_second_cap
               if p.absolute_flashes_per_second_cap is not None else None)

    prev_L: Optional[np.ndarray] = None
    prev_R: Optional[np.ndarray] = None
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

        L = _srgb_to_relative_luminance(frame)
        R = _saturated_red(frame)

        if prev_L is None:
            prev_L, prev_R = L, R
            lum_state.anchor[:] = L
            red_state.anchor[:] = R
            per_frame.append(PerFrame(frame=frame_idx, timestamp=timestamp,
                                       lum_transitions=0, red_transitions=0,
                                       flash_area=0.0))
            continue

        lum_fired = _axis_step(lum_state, prev_L, L, lum_cfg)
        red_fired = _axis_step(red_state, prev_R, R, red_cfg)

        if "luminance" not in failed_dims and _axis_hazard(
                lum_state, lum_count_thresh, n_pixels,
                p.area_fraction_limit, p.area_pixels_limit):
            failed_dims.add("luminance")
            if first_fail_ts is None: first_fail_ts = timestamp
        if "red" not in failed_dims and _axis_hazard(
                red_state, red_count_thresh, n_pixels,
                p.area_fraction_limit, p.area_pixels_limit):
            failed_dims.add("red")
            if first_fail_ts is None: first_fail_ts = timestamp
        if abs_cap is not None and "count" not in failed_dims:
            if (int(lum_state.window_counts.max()) > abs_cap or
                int(red_state.window_counts.max()) > abs_cap):
                failed_dims.add("count")
                if first_fail_ts is None: first_fail_ts = timestamp

        per_frame.append(PerFrame(
            frame=frame_idx, timestamp=timestamp,
            lum_transitions=int(lum_state.window_counts.max()),
            red_transitions=int(red_state.window_counts.max()),
            flash_area=max(float(lum_fired.sum()), float(red_fired.sum())) / n_pixels,
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
    """Smoke test on two synthetic videos labelled analytically (not from
    any benchmark fixture)."""
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
        print(f"self-test: 2Hz -> {r_pass.verdict} (expect PASS); "
              f"5Hz -> {r_fail.verdict} (expect FAIL); dims={r_fail.failed_dimensions}")
        return 0 if (r_pass.verdict == "PASS" and r_fail.verdict == "FAIL") else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
