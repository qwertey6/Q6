"""detector/core.py -- PyTorch-accelerated PSE detector.

PyTorch port of the classical algorithm, structured to allow direct
comparison against the JAX version on `main`. The algorithm is
identical (region-mean ΔL via CC on cv2/numpy, anchor-based darker
bound, no-reset-on-fail, incremental sliding window); only the tensor
framework changes.

Backend selection: MPS on Apple Silicon (M-series), CUDA on NVIDIA, CPU
elsewhere. PyTorch's MPS backend is materially more mature than
jax-metal (the JAX Apple-GPU backend currently fails at module import
on the M4 Max), which is the main reason this branch exists -- to find
out whether we can get useful GPU acceleration that JAX couldn't give
us today.

Pipeline shape (same as JAX version):
  numpy uint8 frame
    -> torch.from_numpy + .to(DEVICE)
    -> srgb_to_relative_luminance via 256-entry LUT
    -> saturated_red
    -> [numpy] regional_delta via cv2 connected components
    -> axis_step (accumulator + anchor gate + reset)
    -> incremental window count update
    -> [numpy/cv2] hazard CC -- only when the cheap max-check fires
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F


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


# --- Constants (sourced from detector/THRESHOLDS.md) -----------------------

def _build_srgb_lin_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for i in range(256):
        v = i / 255.0
        lut[i] = v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return lut
_SRGB_LIN_LUT_NP = _build_srgb_lin_lut()
_SRGB_LIN_LUT = torch.from_numpy(_SRGB_LIN_LUT_NP).to(DEVICE)


@dataclass(frozen=True)
class Profile:
    name: str
    general_flash_max_per_second: int = 3
    general_flash_luminance_delta: float = 0.1
    general_flash_darker_bound: float = 0.8
    area_fraction_limit: float = 0.25
    area_pixels_limit: int = 10_000_000
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


# --- Per-axis state (mutable; torch supports in-place ops) ----------------

class AxisState:
    __slots__ = ("acc", "anchor", "window_counts")

    def __init__(self, h: int, w: int):
        self.acc = torch.zeros((h, w), dtype=torch.float32, device=DEVICE)
        self.anchor = torch.zeros((h, w), dtype=torch.float32, device=DEVICE)
        self.window_counts = torch.zeros((h, w), dtype=torch.int16, device=DEVICE)


# --- Pixel-feature kernels -------------------------------------------------

# Whether to use torch.compile (PyTorch 2.x TorchInductor). Compile cost
# is paid once per shape/profile pair; benefits the steady-state loop.
# Set TORCH_COMPILE=0 in env to disable for benchmarking.
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


# --- Native-tensor connected components ------------------------------------
#
# Eliminates the per-frame cv2/numpy boundary crossing (which was ~5-10 ms
# per frame on MPS due to tensor → CPU → numpy → CPU → MPS transfers).
#
# Algorithm: path-doubling label propagation. Each active pixel starts
# with its own (1-indexed) linear-index label; each iteration takes the
# min of (self, each 4-neighbor's label) where both endpoints are active.
# Naïve 1-pixel-per-iteration propagation converges in O(diameter)
# iterations -- 2998 iters for 1080p worst case, far too slow. We instead
# walk through shifts of 1, 2, 4, 8, ..., 1024 (covers up to 1920 px
# paths), then do a small convergence pass at distance 1 to clean up.
# This is O(log(diameter)) iterations.
#
# Path-doubling with "endpoint must be active" is correct for DENSE
# active regions (which is what our flash regions are). For sparse
# active masks with thin connecting paths, the long jumps could
# erroneously merge components that aren't actually connected; but
# our active masks are the union of contiguous flash regions, never
# sparse.

_CC_SHIFT_SCHEDULE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
_CC_CONVERGE_ITERS = 4  # nearest-neighbour cleanup after path-doubling


def _tensor_cc_step(labels: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    """One direction of label propagation along ``dim`` by ``abs(shift)``.

    Pure-functional: uses F.pad + slice instead of torch.roll + in-place
    zeroing, so the whole thing fuses cleanly under torch.compile and
    avoids materialising intermediates the wrap-around case would
    otherwise need.
    """
    abs_shift = abs(shift)
    h = labels.shape[0]
    w = labels.shape[1]
    if dim == 0:
        if shift > 0:
            # value at row (i-shift) appears at row i
            shifted = F.pad(labels, (0, 0, abs_shift, 0))[:h]
        else:
            shifted = F.pad(labels, (0, 0, 0, abs_shift))[abs_shift:]
    else:
        if shift > 0:
            shifted = F.pad(labels, (abs_shift, 0))[:, :w]
        else:
            shifted = F.pad(labels, (0, abs_shift))[:, abs_shift:]
    both_active = (labels > 0) & (shifted > 0)
    return torch.where(both_active, torch.minimum(labels, shifted), labels)


def _tensor_cc_impl(active: torch.Tensor) -> torch.Tensor:
    """Pure-tensor connected component labeling via path-doubling.

    Returns int32 label map of the same shape as ``active``; 0 = inactive,
    positive integers = component labels (each component labeled with the
    linear index of its smallest-indexed active pixel + 1).
    """
    h = active.shape[0]
    w = active.shape[1]
    flat_idx = (torch.arange(h * w, dtype=torch.int32, device=active.device)
                 .view(h, w) + 1)
    labels = torch.where(active, flat_idx, torch.zeros_like(flat_idx))
    # Path-doubling pass.
    for shift in _CC_SHIFT_SCHEDULE:
        labels = _tensor_cc_step(labels, dim=0, shift=shift)
        labels = _tensor_cc_step(labels, dim=0, shift=-shift)
        labels = _tensor_cc_step(labels, dim=1, shift=shift)
        labels = _tensor_cc_step(labels, dim=1, shift=-shift)
    # Convergence pass at distance 1 -- ensures any irregular boundaries
    # get cleaned up that the path-doubling missed.
    for _ in range(_CC_CONVERGE_ITERS):
        labels = _tensor_cc_step(labels, dim=0, shift=1)
        labels = _tensor_cc_step(labels, dim=0, shift=-1)
        labels = _tensor_cc_step(labels, dim=1, shift=1)
        labels = _tensor_cc_step(labels, dim=1, shift=-1)
    return labels


# Compiled version: TorchInductor fuses the ~60 path-doubling tensor ops
# into a small number of kernels. First call per shape is slow (~seconds
# of compile time); steady-state is fast.
_tensor_cc = (
    torch.compile(_tensor_cc_impl, mode="reduce-overhead", dynamic=False)
    if _USE_COMPILE else _tensor_cc_impl
)


# --- Two interchangeable CC backends behind a small DI surface -----------
#
# The `tensor` backend keeps everything on-device but currently hits three
# known-pathological MPS ops (see Q6 issue tracker / pytorch issues #97310,
# #141789, #149325): torch.bincount, torch.unique(return_counts=True), and
# eager torch.roll. On M-series Macs these make a single frame take 10-20s
# each. Once upstream PyTorch fixes those, this backend becomes the right
# default; for now we ship the cv2 backend as default and keep the tensor
# backend code path live (good for CPU; passes on the regression panel)
# behind a Q6_CC_BACKEND=tensor opt-in.
#
# Both backends present the same signature: tensors in, tensors out.
# Backends are selected via Q6_CC_BACKEND env var ("cv2" or "tensor") OR
# by passing cc_backend= to analyze().


def _regional_delta_cv2(delta: torch.Tensor,
                         intensity_threshold: float) -> torch.Tensor:
    """cv2-backed regional ΔL. One numpy↔torch boundary crossing per
    call; cv2's C connected-components is fast and well-tuned. Proven
    correct on the full regression panel."""
    delta_np = delta.cpu().numpy()
    active = np.abs(delta_np) > intensity_threshold * 0.25
    if not active.any():
        return delta
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return delta
    out = delta_np.copy()
    for li in range(1, n_labels):
        if int(stats[li, cv2.CC_STAT_AREA]) < 100:
            continue
        region_mask = labels == li
        out[region_mask] = float(delta_np[region_mask].mean())
    return torch.from_numpy(out).to(delta.device)


def _regional_delta_tensor(delta: torch.Tensor,
                            intensity_threshold: float) -> torch.Tensor:
    """Native-tensor region-mean ΔL. Currently slow on MPS due to upstream
    PyTorch issues with bincount / unique on this backend (see block
    comment above). Works correctly on CPU; included for future use when
    the MPS ops are fixed and as a CPU-only option."""
    active = torch.abs(delta) > intensity_threshold * 0.25
    if not bool(active.any().item()):
        return delta

    labels = _tensor_cc(active)
    flat_labels = labels.view(-1).long()
    flat_delta = delta.view(-1)
    n_buckets = labels.numel() + 1

    areas = torch.bincount(flat_labels, minlength=n_buckets)
    sums = torch.zeros(n_buckets, dtype=delta.dtype, device=delta.device)
    sums.scatter_add_(0, flat_labels, flat_delta)
    safe_areas = areas.clamp(min=1).to(delta.dtype)
    means = torch.where(areas >= 100, sums / safe_areas, torch.zeros_like(sums))

    means_at_pixel = means[flat_labels].view(delta.shape)
    use_mean = (areas[flat_labels] >= 100).view(delta.shape)
    return torch.where(use_mean, means_at_pixel, delta)


def _region_exceeds_area_cv2(hazard_mask: torch.Tensor, n_pixels: int,
                              fraction_limit: float,
                              pixels_limit: int) -> bool:
    """cv2-backed area test. Tensor in, Python bool out."""
    mask_np = hazard_mask.cpu().numpy()
    if not mask_np.any():
        return False
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_np.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return False
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return (max_area > pixels_limit) or ((max_area / n_pixels) > fraction_limit)


def _region_exceeds_area_tensor(hazard_mask: torch.Tensor, n_pixels: int,
                                 fraction_limit: float,
                                 pixels_limit: int) -> bool:
    """Native-tensor area test. Same MPS caveat as _regional_delta_tensor."""
    if not bool(hazard_mask.any().item()):
        return False
    labels = _tensor_cc(hazard_mask)
    flat_labels = labels.view(-1).long()
    n_buckets = labels.numel() + 1
    areas = torch.bincount(flat_labels, minlength=n_buckets)
    areas[0] = 0
    max_area = int(areas.max().item())
    return (max_area > pixels_limit) or ((max_area / n_pixels) > fraction_limit)


@dataclass(frozen=True)
class CCBackend:
    """Dependency-injection seam for connected-component operations.
    Bundles the two cc-dependent ops behind a stable signature so the
    analyze loop is agnostic to which implementation is in use."""
    name: str
    regional_delta: "Callable[[torch.Tensor, float], torch.Tensor]"
    region_exceeds_area: "Callable[[torch.Tensor, int, float, int], bool]"


CV2_CC_BACKEND = CCBackend(
    name="cv2",
    regional_delta=_regional_delta_cv2,
    region_exceeds_area=_region_exceeds_area_cv2,
)
TENSOR_CC_BACKEND = CCBackend(
    name="tensor",
    regional_delta=_regional_delta_tensor,
    region_exceeds_area=_region_exceeds_area_tensor,
)
_CC_BACKENDS = {b.name: b for b in (CV2_CC_BACKEND, TENSOR_CC_BACKEND)}


def _default_cc_backend() -> CCBackend:
    """Resolved at every call (not import-time) so tests can flip the env
    var between runs without re-importing the module."""
    name = os.environ.get("Q6_CC_BACKEND", "cv2")
    if name not in _CC_BACKENDS:
        raise ValueError(
            f"unknown Q6_CC_BACKEND={name!r}; "
            f"known: {sorted(_CC_BACKENDS)}"
        )
    return _CC_BACKENDS[name]


# --- Per-axis step ---------------------------------------------------------

def _axis_step_lum_impl(acc: torch.Tensor, anchor: torch.Tensor,
                         signal: torch.Tensor, delta: torch.Tensor,
                         threshold: float, darker_bound: float):
    """Pure functional axis step (returns new acc, new anchor, fired).
    Pure-function form so torch.compile can JIT the entire step."""
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


# (Region-area test is now native-tensor; see _region_exceeds_area_tensor
#  above. cv2 stays only for static-pattern detection and image I/O.)


# --- Static spatial-pattern detection (ITU-R BT.1702 §3) -------------------

def detect_static_pattern_hazard(image_path: Path,
                                  profile: str = "ITU-R-BT.1702"
                                  ) -> tuple[bool, dict]:
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
        cc_backend = _default_cc_backend()

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

        # Hazard tests.
        lum_max = int(lum_state.window_counts.max().item())
        red_max = int(red_state.window_counts.max().item())

        if "luminance" not in failed_dims and lum_max > lum_count_thresh:
            hazard_mask = lum_state.window_counts > lum_count_thresh
            if cc_backend.region_exceeds_area(
                    hazard_mask, n_pixels,
                    p.area_fraction_limit, p.area_pixels_limit):
                failed_dims.add("luminance")
                if first_fail_ts is None: first_fail_ts = timestamp
        if "red" not in failed_dims and red_max > red_count_thresh:
            hazard_mask = red_state.window_counts > red_count_thresh
            if cc_backend.region_exceeds_area(
                    hazard_mask, n_pixels,
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
            flash_area=float(max(lum_fired.sum().item(),
                                  red_fired.sum().item())) / n_pixels,
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
