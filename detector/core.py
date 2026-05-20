"""detector/core.py — implementation of our PSE detector.

Per detector/THRESHOLDS.md. Each numeric constant references the clause it
implements. No constant is tuned against benchmark labels.

Top-level entry point: ``analyze(video_path, profile) -> Result``.

Algorithm sketch
----------------
For each frame we compute:
  * relative-luminance map ``L[y, x]`` in [0, 1] using the sRGB→linear
    transform from WCAG 2.2 ("Relative Luminance" definition);
  * saturated-red map ``R_sat[y, x] = max(R - max(G, B), 0)`` in [0, 255].

Then, for each consecutive pair of frames f, f+1, we maintain a per-pixel
*accumulator* per axis (luminance and red). Whenever the accumulator
crosses the per-pixel threshold (`±GENERAL_FLASH_LUMINANCE_DELTA` for
luminance, `±RED_SAT_DELTA` for red), we record a transition at that
pixel and reset the accumulator. The number of opposing transitions in a
1-second window divided by 2 gives the per-pixel flash count.

The *flashing-area* at any moment is the fraction of pixels that
transitioned in the most recent frame. We track its rolling max over the
1-second window; if that max ever exceeds `AREA_FRACTION_LIMIT` AND the
per-frame mean flash count in the windowed pixels exceeded the per-second
cap, the sequence FAILs on that axis at that timestamp.

This is faithful to the standards' "more than 3 flashes per second" (count
axis) combined with "more than 25% of any 10° visual field" (area axis)
combined with the intensity threshold (the accumulator's threshold
*is* the intensity threshold). All three axes must concur for FAIL.

The implementation is intentionally simple and direct so a reviewer can
trace each computation to a clause in THRESHOLDS.md.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np


# --- Constants (sourced from detector/THRESHOLDS.md) ------------------------

# sRGB→relative-luminance transform (WCAG 2.2 "Relative Luminance").
_SRGB_LINEAR_THRESHOLD = 0.03928
_SRGB_LINEAR_DIVISOR   = 12.92
_SRGB_GAMMA_OFFSET     = 0.055
_SRGB_GAMMA_DIVISOR    = 1.055
_SRGB_GAMMA            = 2.4
_LUM_COEFFS            = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)  # BGR order

# Per-profile constants. Defaults are WCAG 2.2 SC 2.3.1; per detector/THRESHOLDS.md
# the in-scope profiles share the same flash count and intensity numerics.

@dataclass(frozen=True)
class Profile:
    name: str
    general_flash_max_per_second: int = 3   # WCAG 2.2 SC 2.3.1
    general_flash_luminance_delta: float = 0.1   # WCAG 2.2 SC 2.3.1
    general_flash_darker_bound: float = 0.8      # WCAG 2.2 SC 2.3.1 exception
    area_fraction_limit: float = 0.25            # WCAG 2.2 SC 2.3.1 normative wording
    # Absolute pixel-area limit. Defaults to "effectively disabled". The
    # WCAG2.2-classic profile sets this to 341*256 = 87,296 px (the
    # Harding / Cambridge Research Systems FCS reference rectangle that
    # WCAG 2.2 SC 2.3.1's Understanding document points to). A connected
    # hazardous region exceeding EITHER the fraction OR the pixel limit
    # is treated as area-axis-FAIL. OQ-4 is resolved by using a profile
    # variant that switches to this stricter pixel-area limit, NOT by
    # tuning the default.
    area_pixels_limit: int = 10_000_000          # effectively disabled by default
    red_flash_max_per_second: int = 3            # WCAG 2.2 SC 2.3.1
    red_sat_delta: int = 20                      # Harding / IRIS-equivalent
    red_sat_min: int = 80                        # Harding minimum
    sliding_window_seconds: float = 1.0          # WCAG 2.2 SC 2.3.1
    # J-BA absolute cap on flashes/sec regardless of area.
    absolute_flashes_per_second_cap: Optional[int] = None
    # ITU-R BT.1702 / Ofcom / NAB-J pattern hazard enabled?
    pattern_hazard_enabled: bool = False
    # Pattern hazard thresholds (ITU-R BT.1702 numerics).
    pattern_min_bars: int = 5            # ≥ 5 light/dark bars
    pattern_min_area_fraction: float = 0.40  # covering > 40% of visible area


# Reference visual-field rectangle from WCAG 2.2 SC 2.3.1 Understanding
# (the Harding / Cambridge Research Systems FCS Implementation Guide
# convention: 341×256 px at 1024×768 reference, representing 10° of
# central visual field at typical viewing distance).
_REF_RECT_W      = 341
_REF_RECT_H      = 256
_REF_RECT_AREA   = _REF_RECT_W * _REF_RECT_H          # 87,296 px (the full 10° field)
_WCAG_AREA_LIMIT = int(round(0.25 * _REF_RECT_AREA))  # 21,824 px = 25% of 10° field


PROFILES: dict[str, Profile] = {
    # WCAG 2.2 SC 2.3.1 — strict reading of the Understanding document.
    # "More than 25% of any 10° of visual field on the screen" with the
    # 10° field operationalized as the 341×256 reference rectangle
    # (Harding / CRS FCS Implementation Guide). Threshold = ~21,824 px,
    # which is the WCAG-strict area gate. Note this is STRICTER than the
    # parenthetical "about 25% of the screen" wording in the same clause.
    # See OQ-4 in detector/THRESHOLDS.md for the full discussion.
    "WCAG2.2-SC2.3.1": Profile(
        name="WCAG2.2-SC2.3.1",
        area_fraction_limit=0.25,          # parenthetical fallback for very large canvases
        area_pixels_limit=_WCAG_AREA_LIMIT,
        pattern_hazard_enabled=True,
    ),
    # WCAG2.2-classic — the Harding-rectangle reading (less strict than
    # WCAG-strict above). Kept as a separate profile so callers can
    # explicitly select either reading; the report shows both.
    "WCAG2.2-classic": Profile(
        name="WCAG2.2-classic",
        area_fraction_limit=1.0,            # disable fraction-form
        area_pixels_limit=_REF_RECT_AREA,    # the full 341×256 rectangle
        pattern_hazard_enabled=True,
    ),
    # Broadcast profiles use the Harding-classic (less strict) reading.
    "ITU-R-BT.1702":    Profile(name="ITU-R-BT.1702",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Ofcom-GN2-Annex1": Profile(name="Ofcom-GN2-Annex1",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Trace24":          Profile(name="Trace24",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    # NAB-J: Harding-classic area + absolute count cap.
    "NAB-J":            Profile(name="NAB-J",
                                  area_pixels_limit=_REF_RECT_AREA,
                                  absolute_flashes_per_second_cap=5,
                                  pattern_hazard_enabled=True),
}


# --- Result types -----------------------------------------------------------

@dataclass
class PerFrame:
    frame: int                  # 1-based to align with IRIS convention
    timestamp: float            # seconds
    lum_transitions: int        # cumulative count up to this frame (over window)
    red_transitions: int
    flash_area: float           # area-fraction in [0, 1] of pixels actively transitioning this frame
    pattern_risk: float = 0.0


@dataclass
class Result:
    verdict: str                # "PASS" | "FAIL"
    failed_dimensions: list[str] = field(default_factory=list)
    first_fail_timestamp: Optional[float] = None
    per_frame: list[PerFrame] = field(default_factory=list)
    profile_name: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    n_frames: int = 0


# --- sRGB → relative luminance ---------------------------------------------

def _srgb_to_relative_luminance(bgr_uint8: np.ndarray) -> np.ndarray:
    """Compute relative luminance map L in [0, 1] from an HxWx3 BGR uint8 image.

    Follows the WCAG 2.2 Relative Luminance definition exactly. Vectorized
    with numpy; runs at ~real-time on modern hardware.
    """
    rgb = bgr_uint8.astype(np.float32) / 255.0
    # sRGB → linear; use np.where for the piecewise function.
    low = rgb / _SRGB_LINEAR_DIVISOR
    high = ((rgb + _SRGB_GAMMA_OFFSET) / _SRGB_GAMMA_DIVISOR) ** _SRGB_GAMMA
    linear = np.where(rgb <= _SRGB_LINEAR_THRESHOLD, low, high)
    # BGR coefficients (we kept the BGR order from cv2).
    L = linear @ _LUM_COEFFS
    return L.astype(np.float32)


def _saturated_red(bgr_uint8: np.ndarray) -> np.ndarray:
    """Harding saturated-red value, R - max(G, B), clamped to [0, 255]."""
    b = bgr_uint8[..., 0].astype(np.int16)
    g = bgr_uint8[..., 1].astype(np.int16)
    r = bgr_uint8[..., 2].astype(np.int16)
    sat = r - np.maximum(g, b)
    np.clip(sat, 0, 255, out=sat)
    return sat.astype(np.int16)


# --- Per-pixel transition accumulator --------------------------------------
#
# For each pixel we maintain a running signed accumulator on each axis.
# When the absolute value of the accumulator crosses the per-pixel threshold
# (intensity threshold for that axis), we count a transition at that pixel
# in the direction of the accumulator's sign and reset.
#
# This implements the "opposing-change" rule (a flash is a *pair* of
# transitions in opposing directions) at pixel granularity: a sustained
# drift in one direction can only contribute one transition before reset.
# The 1-second windowed count of transitions at a pixel divided by 2 is
# that pixel's flash rate.

def _update_transitions(acc: np.ndarray,
                        delta: np.ndarray,
                        threshold: float) -> np.ndarray:
    """Update the accumulator and return a boolean mask of pixels that
    completed an opposing-transition step this frame."""
    acc += delta
    # Where |acc| has crossed the threshold, count a transition and reset.
    fired_pos = acc >=  threshold
    fired_neg = acc <= -threshold
    fired = fired_pos | fired_neg
    # Reset accumulator at fired pixels.
    acc[fired] = 0.0
    return fired


# --- Region-aware area decision --------------------------------------------

def _region_exceeds_area(hazard_mask: np.ndarray, n_pixels: int,
                         fraction_limit: float, pixels_limit: int) -> bool:
    """True iff any single connected component of ``hazard_mask`` exceeds
    EITHER the screen-fraction limit OR the absolute pixel-area limit.

    This implements the "the flashing region" wording of the standards:
    we look for a *contiguous* hazardous region, not the union of
    disconnected hazardous pixels. Many small independently-flashing
    regions (each below threshold) do NOT compose into a single hazard.

    cv2.connectedComponentsWithStats is O(n_pixels) and returns per-label
    area in stats[:, cv2.CC_STAT_AREA] in pixels. Label 0 is background.
    """
    if not hazard_mask.any():
        return False
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        hazard_mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return False
    # stats[0] is background; skip it.
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    max_area_frac = max_area / n_pixels
    return (max_area > pixels_limit) or (max_area_frac > fraction_limit)


# --- Static spatial-pattern detection (ITU-R BT.1702 §3) -------------------
#
# A "regular pattern" hazard is ≥ pattern_min_bars approximately-equally-
# spaced light/dark bars covering more than pattern_min_area_fraction of
# the visible area. We detect this on a single still frame via:
#   1. Convert to grayscale L map.
#   2. Adaptive threshold → binary.
#   3. Sum along each axis → 1D intensity profiles (rows, cols).
#   4. Count zero-crossings of the row/col profile relative to its mean.
#      ≥ 2*N zero-crossings means at least N bar transitions.
#   5. Estimate covered area as the bounding box of the high-variance band.
#
# This is intentionally simple. The IRIS approach uses Hough lines plus
# circular-pattern detection (their CircularExpectedResults dir). The
# simple-bar detection here handles the bulk of the test images
# (stripes.png, ChessBoard.png, HorizontalLines.jpg, Diagonals.png).
# Open question OQ-3 in THRESHOLDS.md now narrows to: more sophisticated
# patterns (kaleidoscopes, sacred-geometry overlays) may still be missed.

def detect_static_pattern_hazard(image_path: Path,
                                  profile: str = "ITU-R-BT.1702"
                                  ) -> tuple[bool, dict]:
    """Return (is_hazardous, info) for a still image under ``profile``.

    ``info`` contains diagnostic numbers: bar_count (max of horizontal/
    vertical), covered_area_fraction, and the axis that dominated.
    """
    p = PROFILES[profile]
    if not p.pattern_hazard_enabled:
        return False, {"reason": "pattern_hazard disabled in profile",
                       "profile": profile}

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, {"reason": "image unreadable", "path": str(image_path)}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive threshold isolates light/dark bars without committing to a
    # global mean (handles vignetted or non-uniform-background patterns).
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=max(15, (min(h, w) // 32) | 1),  # odd
        C=2,
    )

    # Count bar transitions along each axis. A "transition" is a flip in
    # the binarized signal across the axis-projected mean.
    def _bar_count_and_band(profile_1d: np.ndarray) -> tuple[int, tuple[int, int]]:
        m = profile_1d.mean()
        sign = (profile_1d > m).astype(np.int8)
        diffs = np.abs(np.diff(sign))
        n_transitions = int(diffs.sum())
        # The "band" containing the pattern: the region of the projection
        # where the signal is meaningfully oscillating. Approximate by the
        # min..max indices of the high-variance window.
        if n_transitions == 0:
            return 0, (0, 0)
        nonzero = np.nonzero(diffs)[0]
        return n_transitions // 2, (int(nonzero[0]), int(nonzero[-1]))

    row_profile = binarized.mean(axis=1)  # length h: vertical-bar detector
    col_profile = binarized.mean(axis=0)  # length w: horizontal-bar detector
    v_bars, (vy0, vy1) = _bar_count_and_band(col_profile)
    h_bars, (hx0, hx1) = _bar_count_and_band(row_profile)

    # Area coverage estimate: the bounding box of the high-variance band
    # on the dominant axis, projected over the other axis (assume the
    # pattern fills the orthogonal direction). This over-estimates for
    # diagonals, which is conservative for FAIL.
    if v_bars >= h_bars:
        bar_count = v_bars
        # vertical bars: pattern spans full height, between vy0..vy1 cols
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


# --- Main analyzer ----------------------------------------------------------

def analyze(video_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> Result:
    """Open ``video_path`` and run the PSE analysis under the given profile.

    Returns a Result with verdict + failed dimensions + per-frame trace.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    p = PROFILES[profile]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if fps <= 0:
        cap.release()
        raise IOError(f"video has no usable fps metadata: {video_path}")

    n_pixels = max(1, width * height)
    window_frames = max(1, int(round(p.sliding_window_seconds * fps)))

    # Per-pixel accumulators (float32 saves memory).
    lum_acc = np.zeros((height, width), dtype=np.float32)
    red_acc = np.zeros((height, width), dtype=np.float32)

    # Previous-frame maps (for diffs).
    prev_L: Optional[np.ndarray] = None
    prev_R: Optional[np.ndarray] = None
    # For the WCAG "darker image" exemption: we track a per-pixel previous L
    # so we can check that the darker of the two endpoints is < 0.8.
    # The check is approximate at pixel granularity; we evaluate it on
    # the *contributing pixels* at transition time.

    # Sliding-window per-pixel transition counts (for the 1-second window).
    lum_trans_window: deque[np.ndarray] = deque(maxlen=window_frames)
    red_trans_window: deque[np.ndarray] = deque(maxlen=window_frames)

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
            prev_L = L
            prev_R = R
            per_frame.append(PerFrame(frame=frame_idx, timestamp=timestamp,
                                       lum_transitions=0, red_transitions=0,
                                       flash_area=0.0))
            continue

        lum_delta = L - prev_L                                # signed, in luminance units
        red_delta = R.astype(np.float32) - prev_R.astype(np.float32)  # signed, in [-255, 255]

        # Update accumulators and find pixels that fired a transition this frame.
        lum_fired = _update_transitions(lum_acc, lum_delta,
                                         p.general_flash_luminance_delta)
        red_fired = _update_transitions(red_acc, red_delta, float(p.red_sat_delta))

        # WCAG "darker image must be < 0.8" exception: at fired pixels,
        # the *darker* endpoint of the pair-of-opposing-changes must be
        # < 0.8 for the flash to count. We approximate: a fired pixel
        # counts only if min(L, prev_L) at that pixel < darker_bound.
        if p.general_flash_darker_bound < 1.0:
            darker_ok = np.minimum(L, prev_L) < p.general_flash_darker_bound
            lum_fired &= darker_ok

        # Red flash: also require the larger of the two saturated-red values
        # to be ≥ RED_SAT_MIN (Harding minimum). Otherwise the "transition"
        # is between two near-zero saturated-red values and isn't a red flash.
        if p.red_sat_min > 0:
            larger_red = np.maximum(R, prev_R)
            red_ok = larger_red >= p.red_sat_min
            red_fired &= red_ok

        # Update windowed transitions.
        lum_trans_window.append(lum_fired)
        red_trans_window.append(red_fired)

        # Per-frame area-fraction = pixels that fired this frame / total.
        lum_area_frac_now = float(lum_fired.sum()) / n_pixels
        red_area_frac_now = float(red_fired.sum()) / n_pixels
        flash_area_now = max(lum_area_frac_now, red_area_frac_now)

        # Per-pixel transition counts in the rolling window.
        lum_counts = np.sum(np.stack(lum_trans_window, axis=0).astype(np.int16), axis=0)
        red_counts = np.sum(np.stack(red_trans_window, axis=0).astype(np.int16), axis=0)

        # "More than 3 flashes" = more than 6 transitions (two transitions/flash).
        lum_hazard_mask = lum_counts > (2 * p.general_flash_max_per_second)
        red_hazard_mask = red_counts > (2 * p.red_flash_max_per_second)

        # Region-aware area decision (replaces the naive screen-fraction sum).
        # Standards talk about "the flashing region" — singular, contiguous,
        # over a 10° visual field. We extract connected components per axis
        # and check whether ANY single region exceeds the area threshold
        # (fraction OR absolute pixel limit, whichever is configured).
        # Multi-region content with many small independently-flashing areas
        # that individually fall under the limit is correctly PASSed.
        if "luminance" not in failed_dims and _region_exceeds_area(
                lum_hazard_mask, n_pixels, p.area_fraction_limit, p.area_pixels_limit):
            failed_dims.add("luminance")
            if first_fail_ts is None: first_fail_ts = timestamp
        if "red" not in failed_dims and _region_exceeds_area(
                red_hazard_mask, n_pixels, p.area_fraction_limit, p.area_pixels_limit):
            failed_dims.add("red")
            if first_fail_ts is None: first_fail_ts = timestamp

        # J-BA absolute cap (if profile enables it).
        if p.absolute_flashes_per_second_cap is not None:
            cap_trans = 2 * p.absolute_flashes_per_second_cap
            # Any pixel exceeding the absolute cap fails regardless of area.
            if bool(np.any(lum_counts > cap_trans)) or bool(np.any(red_counts > cap_trans)):
                if "count" not in failed_dims:
                    failed_dims.add("count")
                    if first_fail_ts is None: first_fail_ts = timestamp

        per_frame.append(PerFrame(
            frame=frame_idx, timestamp=timestamp,
            lum_transitions=int(lum_counts.max()) if lum_counts.size else 0,
            red_transitions=int(red_counts.max()) if red_counts.size else 0,
            flash_area=flash_area_now,
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


# --- CLI --------------------------------------------------------------------

def _main_cli(argv: list[str]) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="Run our PSE detector on a video.")
    ap.add_argument("video", type=Path, nargs="?",
                    help="Input video. Optional only when --self-test is set.")
    ap.add_argument("--profile", default="WCAG2.2-SC2.3.1", choices=list(PROFILES))
    ap.add_argument("--per-frame-csv", type=Path, default=None,
                    help="Optional path to write a per-frame CSV matching the harness contract.")
    ap.add_argument("--self-test", action="store_true",
                    help="Quick smoke self-test on a tiny synthetic video.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    res = analyze(args.video, args.profile)
    out = {
        "video":               str(args.video),
        "profile":             res.profile_name,
        "verdict":             res.verdict,
        "failed_dimensions":   res.failed_dimensions,
        "first_fail_timestamp": res.first_fail_timestamp,
        "fps":                 res.fps,
        "n_frames":            res.n_frames,
    }
    print(json.dumps(out, indent=2))

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
    """Synthesize a tiny full-screen alternating clip and confirm verdicts.

    This is a smoke test only — it intentionally uses two synthetic clips
    whose label derives from generation parameters analytically. It does
    NOT use any benchmark fixture; running it does not violate the
    no-tuning-against-labels rule.
    """
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="detector_selftest_"))
    try:
        passing = tmp / "two_hz.mp4"      # 2 flashes/sec → PASS
        failing = tmp / "five_hz.mp4"     # 5 flashes/sec → FAIL
        for path, hz in ((passing, 2.0), (failing, 5.0)):
            writer = cv2.VideoWriter(str(path),
                                      cv2.VideoWriter_fourcc(*"mp4v"),
                                      30.0, (320, 240))
            for fi in range(int(2 * 30)):  # 2 seconds at 30fps
                t = fi / 30.0
                state = int(t * 2 * hz) % 2
                img = np.full((240, 320, 3), 255 if state else 0, dtype=np.uint8)
                writer.write(img)
            writer.release()
        r_pass = analyze(passing)
        r_fail = analyze(failing)
        print(f"self-test: 2Hz → {r_pass.verdict} (expect PASS); "
              f"5Hz → {r_fail.verdict} (expect FAIL); dims={r_fail.failed_dimensions}")
        ok = (r_pass.verdict == "PASS") and (r_fail.verdict == "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))


# Allow ``python -m detector`` as a convenient alias for the CLI.
def __main_module__() -> None:  # pragma: no cover
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
