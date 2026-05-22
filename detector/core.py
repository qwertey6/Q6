"""detector/core.py -- Q6's standards-grounded PSE detector.

Implements per-frame photosensitive-epilepsy hazard detection from the
text of the standards (WCAG 2.2 SC 2.3.1, Trace24, ITU-R BT.1702, Ofcom
GN2 Annex 1, NAB-J, ISO 9241-391). Every numeric threshold is justified
in detector/THRESHOLDS.md by a clause citation; nothing is tuned against
benchmark labels.

Entry point: ``analyze(video_path, profile, cc_backend=None) -> Result``.

Architecture (per frame):

  numpy uint8 frame
    → torch tensor on auto-selected device (MPS / CUDA / CPU)
    → relative-luminance map L (sRGB via 256-entry LUT) + saturated-red map R
    → region-mean ΔL on both channels (via the CCBackend, cv2 by default)
    → per-axis state update (accumulator + anchor-based WCAG darker-bound
      gate + no-reset-on-fail sequencing -- see THRESHOLDS.md OQ-5)
    → incremental 1-second windowed transition counts
    → hazard-region extraction (union-CC, per-region class tagging,
      severity / mitigation / counterfactual)

Each frame emits PerFrame.hazard_regions (list[HazardRegion]) with bbox,
classes, severity, mitigation hints, and standards-clause citations. At
fixture level, Result carries the verdict + a continuous severity score
+ per-axis aggregates (fail intervals, peak counts, peak areas).

This file is the orchestration layer. The data types and constants live
in sibling modules:

  * detector/profiles.py    -- Profile + PROFILES dict
  * detector/regions.py     -- HazardRegion + mitigation/counterfactual builders
  * detector/cc_backends.py -- CCBackend interface + cv2/tensor impls
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2  # type: ignore
import numpy as np
import torch

from .profiles import Profile, PROFILES
from .regions import HazardRegion, make_hazard_region
from .cc_backends import CCBackend, CV2_CC_BACKEND, TENSOR_CC_BACKEND, default_cc_backend


# --- Device selection ------------------------------------------------------

def _select_device() -> torch.device:
    """Pick the best device available. Respect TORCH_DEVICE env var override
    for benchmarking ('cpu', 'mps', 'cuda')."""
    forced = os.environ.get("TORCH_DEVICE", "").lower()
    if forced in ("cpu", "mps", "cuda"):
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _select_device()


# --- sRGB → linear LUT (exact for 8-bit input) ----------------------------

def _build_srgb_lin_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for i in range(256):
        v = i / 255.0
        lut[i] = v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return lut
_SRGB_LIN_LUT_NP = _build_srgb_lin_lut()
_SRGB_LIN_LUT = torch.from_numpy(_SRGB_LIN_LUT_NP).to(DEVICE)


# --- Result types ----------------------------------------------------------
#
# Multi-resolution output structure:
#
#   Result                                 fixture-level summary
#     verdict / score / failed_dimensions  binary + continuous summaries
#     per_axis: dict[axis_name, PerAxisResult]
#                                          per-axis verdict, fail intervals,
#                                          peak values, margins to threshold
#     per_frame: list[PerFrame]            time-resolved trace
#       per_frame[t].hazard_regions: list[HazardRegion]
#                                          spatially-resolved hazards
#
# Backward-compat: all original fields are preserved; new fields default
# to empty/zero so older consumers keep working.

@dataclass
class PerAxisResult:
    """Per-axis fixture-level summary derived from the per-frame regions."""
    name: str                                          # luminance | red | pattern | count
    verdict: str                                       # PASS | FAIL
    score: float = 0.0                                 # peak / threshold; >= 1 ⇒ FAIL
    max_windowed_count: int = 0
    max_hazard_area_px: int = 0
    max_hazard_area_frac: float = 0.0
    first_fail_timestamp: Optional[float] = None
    last_fail_timestamp: Optional[float] = None
    fail_intervals: list[tuple[float, float]] = field(default_factory=list)
    margin_to_threshold: float = 0.0                   # signed; positive = above


@dataclass
class PerFrame:
    frame: int
    timestamp: float
    lum_transitions: int
    red_transitions: int
    flash_area: float
    pattern_risk: float = 0.0
    hazard_regions: list[HazardRegion] = field(default_factory=list)


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
    # Enriched fields (default-zeroed for back-compat with older consumers).
    score: float = 0.0                                 # max over per_axis scores
    per_axis: dict[str, PerAxisResult] = field(default_factory=dict)
    standards_evaluated: list[str] = field(default_factory=list)
    per_standard_verdict: dict[str, str] = field(default_factory=dict)


# --- Per-axis state (mutable; torch supports in-place ops) ----------------

class AxisState:
    """Per-pixel state for one axis (luminance or red), persists across the
    per-frame loop. Mutated in place each frame for efficiency."""
    __slots__ = ("acc", "anchor", "window_counts")

    def __init__(self, h: int, w: int):
        self.acc = torch.zeros((h, w), dtype=torch.float32, device=DEVICE)
        self.anchor = torch.zeros((h, w), dtype=torch.float32, device=DEVICE)
        self.window_counts = torch.zeros((h, w), dtype=torch.int16, device=DEVICE)


# --- Pixel-feature kernels (torch.compile'd in steady state) --------------

_USE_COMPILE = os.environ.get("TORCH_COMPILE", "1") != "0"


def _srgb_to_relative_luminance_impl(bgr_uint8: torch.Tensor) -> torch.Tensor:
    """L in [0,1] from BGR uint8 tensor, via 256-entry LUT."""
    b_lin = _SRGB_LIN_LUT[bgr_uint8[..., 0].long()]
    g_lin = _SRGB_LIN_LUT[bgr_uint8[..., 1].long()]
    r_lin = _SRGB_LIN_LUT[bgr_uint8[..., 2].long()]
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _saturated_red_impl(bgr_uint8: torch.Tensor) -> torch.Tensor:
    """R - max(G, B), clamped to [0, 255], as float32."""
    b = bgr_uint8[..., 0].to(torch.int16)
    g = bgr_uint8[..., 1].to(torch.int16)
    r = bgr_uint8[..., 2].to(torch.int16)
    return torch.clamp(r - torch.maximum(g, b), 0, 255).to(torch.float32)


_srgb_to_relative_luminance = (
    torch.compile(_srgb_to_relative_luminance_impl, mode="reduce-overhead")
    if _USE_COMPILE else _srgb_to_relative_luminance_impl
)
_saturated_red = (
    torch.compile(_saturated_red_impl, mode="reduce-overhead")
    if _USE_COMPILE else _saturated_red_impl
)


# --- Per-axis step (accumulator + anchor gate; torch.compile'd) -----------

def _axis_step_lum_impl(acc: torch.Tensor, anchor: torch.Tensor,
                         signal: torch.Tensor, delta: torch.Tensor,
                         threshold: float, darker_bound: float):
    """Pure functional axis step (returns new acc, new anchor, fired).
    Implements the WCAG "darker image must be < 0.80" exception via the
    anchor: compare current L against the anchor (L at last successful
    fire) rather than the previous frame's L. This handles codec-smeared
    transitions correctly -- see THRESHOLDS.md OQ-5."""
    new_acc = acc + delta
    crossed = (new_acc >= threshold) | (new_acc <= -threshold)
    gate_ok = torch.minimum(signal, anchor) < darker_bound
    fired = crossed & gate_ok
    zero = torch.zeros_like(new_acc)
    new_acc = torch.where(fired, zero, new_acc)
    new_anchor = torch.where(fired, signal, anchor)
    return new_acc, new_anchor, fired


def _axis_step_red_impl(acc: torch.Tensor, anchor: torch.Tensor,
                         signal: torch.Tensor, delta: torch.Tensor,
                         threshold: float, sat_min: float):
    """Same shape as luminance, but the gate checks the LARGER (not smaller)
    of the two endpoints against the Harding minimum -- the saturated-red
    transition only counts if the brighter endpoint is genuinely red."""
    new_acc = acc + delta
    crossed = (new_acc >= threshold) | (new_acc <= -threshold)
    gate_ok = torch.maximum(signal, anchor) >= sat_min
    fired = crossed & gate_ok
    zero = torch.zeros_like(new_acc)
    new_acc = torch.where(fired, zero, new_acc)
    new_anchor = torch.where(fired, signal, anchor)
    return new_acc, new_anchor, fired


_axis_step_lum_compiled = (
    torch.compile(_axis_step_lum_impl, mode="reduce-overhead")
    if _USE_COMPILE else _axis_step_lum_impl
)
_axis_step_red_compiled = (
    torch.compile(_axis_step_red_impl, mode="reduce-overhead")
    if _USE_COMPILE else _axis_step_red_impl
)


def _axis_step_lum(state: AxisState, signal: torch.Tensor,
                   delta: torch.Tensor,
                   threshold: float, darker_bound: float) -> torch.Tensor:
    """Wraps the compiled pure-function form; writes back into state."""
    new_acc, new_anchor, fired = _axis_step_lum_compiled(
        state.acc, state.anchor, signal, delta, threshold, darker_bound)
    state.acc = new_acc
    state.anchor = new_anchor
    return fired


def _axis_step_red(state: AxisState, signal: torch.Tensor,
                   delta: torch.Tensor,
                   threshold: float, sat_min: float) -> torch.Tensor:
    new_acc, new_anchor, fired = _axis_step_red_compiled(
        state.acc, state.anchor, signal, delta, threshold, sat_min)
    state.acc = new_acc
    state.anchor = new_anchor
    return fired


# --- Static spatial-pattern detection (ITU-R BT.1702 §3) -------------------

def detect_static_pattern_hazard(image_path: Path,
                                  profile: str = "ITU-R-BT.1702"
                                  ) -> tuple[bool, dict]:
    """Detect "regular pattern" hazards (stripes, gratings) in a single
    still image, per ITU-R BT.1702 §3."""
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

def analyze(video_path: Path, profile: str = "WCAG2.2-SC2.3.1",
            cc_backend: Optional[CCBackend] = None) -> Result:
    """Analyze a video for PSE hazards under the given profile.

    ``cc_backend`` selects the connected-components implementation
    (defaults to env var Q6_CC_BACKEND, then to cv2). Pass
    CV2_CC_BACKEND or TENSOR_CC_BACKEND explicitly to override.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {sorted(PROFILES)}")
    p = PROFILES[profile]
    if cc_backend is None:
        cc_backend = default_cc_backend()

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

    lum_threshold = float(p.general_flash_luminance_delta)
    lum_darker_bound = float(p.general_flash_darker_bound)
    red_threshold = float(p.red_sat_delta)
    red_sat_min = float(p.red_sat_min)
    lum_count_thresh = 2 * p.general_flash_max_per_second
    red_count_thresh = 2 * p.red_flash_max_per_second
    abs_cap = (2 * p.absolute_flashes_per_second_cap
               if p.absolute_flashes_per_second_cap is not None else None)

    lum_state = AxisState(height, width)
    red_state = AxisState(height, width)
    zero_fired = torch.zeros((height, width), dtype=torch.bool, device=DEVICE)
    lum_window = [zero_fired.clone() for _ in range(window_frames)]
    red_window = [zero_fired.clone() for _ in range(window_frames)]
    write_idx = 0

    prev_L: Optional[torch.Tensor] = None
    prev_R: Optional[torch.Tensor] = None
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

        frame_t = torch.from_numpy(frame).to(DEVICE)  # uint8 HxWx3
        L = _srgb_to_relative_luminance(frame_t)
        R = _saturated_red(frame_t)

        if prev_L is None:
            prev_L, prev_R = L, R
            lum_state.anchor = L.clone()
            red_state.anchor = R.clone()
            per_frame.append(PerFrame(frame=frame_idx, timestamp=timestamp,
                                       lum_transitions=0, red_transitions=0,
                                       flash_area=0.0))
            continue

        # Region-mean ΔL via the selected CC backend.
        lum_delta = cc_backend.regional_delta(L - prev_L, lum_threshold)
        red_delta = cc_backend.regional_delta(R - prev_R, red_threshold)

        # Per-axis update (mutates state).
        lum_fired = _axis_step_lum(
            lum_state, L, lum_delta, lum_threshold, lum_darker_bound)
        red_fired = _axis_step_red(
            red_state, R, red_delta, red_threshold, red_sat_min)

        # Incremental window update.
        lum_state.window_counts += lum_fired.to(torch.int16)
        lum_state.window_counts -= lum_window[write_idx].to(torch.int16)
        red_state.window_counts += red_fired.to(torch.int16)
        red_state.window_counts -= red_window[write_idx].to(torch.int16)
        lum_window[write_idx] = lum_fired
        red_window[write_idx] = red_fired
        write_idx = (write_idx + 1) % window_frames

        lum_max = int(lum_state.window_counts.max().item())
        red_max = int(red_state.window_counts.max().item())

        # Per-frame hazard region extraction. Only fires when either axis
        # has at least one pixel over its count threshold -- the cheap
        # early-exit keeps the streaming-hot-path performant.
        frame_hazard_regions: list[HazardRegion] = []
        lum_above = lum_max > lum_count_thresh
        red_above = red_max > red_count_thresh
        if lum_above or red_above:
            lum_haz_np = (lum_state.window_counts > lum_count_thresh
                          ).cpu().numpy() if lum_above else None
            red_haz_np = (red_state.window_counts > red_count_thresh
                          ).cpu().numpy() if red_above else None
            if lum_haz_np is not None and red_haz_np is not None:
                union_mask = lum_haz_np | red_haz_np
            elif lum_haz_np is not None:
                union_mask = lum_haz_np
            else:
                union_mask = red_haz_np  # type: ignore[assignment]
            if union_mask is not None and union_mask.any():
                lum_counts_np = lum_state.window_counts.cpu().numpy()
                red_counts_np = red_state.window_counts.cpu().numpy()
                n_labels, labels, stats, centroids = (
                    cv2.connectedComponentsWithStats(
                        union_mask.astype(np.uint8), connectivity=8))
                for li in range(1, n_labels):
                    area = int(stats[li, cv2.CC_STAT_AREA])
                    if area < int(p.area_pixels_limit * 0.05):  # 5% of limit; debounce noise
                        continue
                    x0 = int(stats[li, cv2.CC_STAT_LEFT])
                    y0 = int(stats[li, cv2.CC_STAT_TOP])
                    w_box = int(stats[li, cv2.CC_STAT_WIDTH])
                    h_box = int(stats[li, cv2.CC_STAT_HEIGHT])
                    region_mask = (labels == li)
                    per_class_peak: dict[str, int] = {}
                    if lum_haz_np is not None:
                        per_class_peak["luminance"] = int(
                            lum_counts_np[region_mask].max())
                    if red_haz_np is not None:
                        per_class_peak["red"] = int(
                            red_counts_np[region_mask].max())
                    if abs_cap is not None:
                        per_class_peak["count"] = max(
                            per_class_peak.get("luminance", 0),
                            per_class_peak.get("red", 0))
                    region = make_hazard_region(
                        bbox=(x0, y0, x0 + w_box, y0 + h_box),
                        area_px=area,
                        centroid=(float(centroids[li][0]),
                                   float(centroids[li][1])),
                        per_class_peak=per_class_peak,
                        profile=p,
                    )
                    if region.classes:  # at least one axis triggered
                        frame_hazard_regions.append(region)

        # Aggregate per-frame regions into the fixture-level failed_dims.
        for region in frame_hazard_regions:
            for cls in region.classes:
                if region.area_px > p.area_pixels_limit or \
                   (region.area_px / n_pixels) > p.area_fraction_limit:
                    failed_dims.add(cls)
                    if first_fail_ts is None: first_fail_ts = timestamp
        if abs_cap is not None and "count" not in failed_dims:
            if lum_max > abs_cap or red_max > abs_cap:
                failed_dims.add("count")
                if first_fail_ts is None: first_fail_ts = timestamp

        per_frame.append(PerFrame(
            frame=frame_idx, timestamp=timestamp,
            lum_transitions=lum_max,
            red_transitions=red_max,
            flash_area=float(max(lum_fired.sum().item(),
                                  red_fired.sum().item())) / n_pixels,
            hazard_regions=frame_hazard_regions,
        ))
        prev_L, prev_R = L, R

    cap.release()

    # Build per-axis fixture-level aggregates from the per-frame regions.
    per_axis = _aggregate_per_axis(per_frame, p, n_pixels, failed_dims)
    fixture_score = max((a.score for a in per_axis.values()), default=0.0)

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
        score=fixture_score,
        per_axis=per_axis,
        standards_evaluated=[p.name],
        per_standard_verdict={p.name: ("FAIL" if failed_dims else "PASS")},
    )


def _aggregate_per_axis(per_frame: list[PerFrame], profile: Profile,
                          n_pixels: int,
                          failed_dims: set[str]) -> dict[str, PerAxisResult]:
    """Walk the per-frame hazard regions and build a PerAxisResult for
    each hazard class that appears. ``verdict`` reflects the COMBINED
    (count + area + intensity) per-axis outcome from ``failed_dims``;
    ``score`` is the per-axis count-severity (peak windowed count over
    the count threshold) which can be >= 1 even when verdict is PASS
    (e.g. count crossed but the hazardous region was too small to meet
    the area limit -- diagnostically useful)."""
    axes_seen: set[str] = set()
    for f in per_frame:
        for region in f.hazard_regions:
            axes_seen.update(region.classes)

    out: dict[str, PerAxisResult] = {}
    for axis in sorted(axes_seen):
        if axis == "luminance":
            count_limit = 2 * profile.general_flash_max_per_second
        elif axis == "red":
            count_limit = 2 * profile.red_flash_max_per_second
        elif axis == "count" and profile.absolute_flashes_per_second_cap is not None:
            count_limit = 2 * profile.absolute_flashes_per_second_cap
        else:
            count_limit = 0
        max_count = 0
        max_area = 0
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        intervals: list[tuple[float, float]] = []
        in_fail = False
        interval_start: Optional[float] = None
        for f in per_frame:
            this_frame_has = False
            for region in f.hazard_regions:
                if axis not in region.classes:
                    continue
                this_frame_has = True
                region_peak = int(round(region.severity.get(axis, 0.0)
                                         * count_limit))
                max_count = max(max_count, region_peak)
                max_area = max(max_area, region.area_px)
                if first_ts is None: first_ts = f.timestamp
                last_ts = f.timestamp
            if this_frame_has and not in_fail:
                interval_start = f.timestamp
                in_fail = True
            elif not this_frame_has and in_fail:
                intervals.append((interval_start or 0.0, f.timestamp))
                in_fail = False
                interval_start = None
        if in_fail and interval_start is not None:
            intervals.append((interval_start, per_frame[-1].timestamp
                              if per_frame else interval_start))
        score = (float(max_count) / float(count_limit)) if count_limit > 0 else 0.0
        out[axis] = PerAxisResult(
            name=axis,
            verdict="FAIL" if axis in failed_dims else "PASS",
            score=score,
            max_windowed_count=max_count,
            max_hazard_area_px=max_area,
            max_hazard_area_frac=float(max_area) / max(n_pixels, 1),
            first_fail_timestamp=first_ts,
            last_fail_timestamp=last_ts,
            fail_intervals=intervals,
            margin_to_threshold=float(max_count - count_limit) if count_limit > 0 else 0.0,
        )
    return out


# --- CLI -------------------------------------------------------------------

def _main_cli(argv: list[str]) -> int:
    import argparse, json
    ap = argparse.ArgumentParser(description="Run Q6's PSE detector on a video.")
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
        "score":                res.score,
        "failed_dimensions":    res.failed_dimensions,
        "first_fail_timestamp": res.first_fail_timestamp,
        "fps":                  res.fps,
        "n_frames":             res.n_frames,
        "backend":              str(DEVICE),
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
        print(f"self-test (torch device={DEVICE}): "
              f"2Hz -> {r_pass.verdict} (expect PASS); "
              f"5Hz -> {r_fail.verdict} (expect FAIL); dims={r_fail.failed_dimensions}")
        return 0 if (r_pass.verdict == "PASS" and r_fail.verdict == "FAIL") else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
