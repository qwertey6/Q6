"""detector/cc_backends.py -- connected-component backends for region work.

A ``CCBackend`` bundles the two connected-component operations the
detector needs each frame:

  - ``regional_delta(delta, threshold) -> delta``: replace each pixel's
    ΔL with the mean ΔL of its connected component on the active mask;
    pixels not in a significant component keep their original delta.
  - ``region_exceeds_area(mask, n_pixels, frac_limit, px_limit) -> bool``:
    return True iff any connected component of ``mask`` exceeds the
    area limit.

Two implementations live side-by-side:

  - **cv2** (default): uses ``cv2.connectedComponentsWithStats``, the
    industry-standard scanline labeller. One numpy↔torch boundary
    crossing per call, but the C implementation is fast and well-tuned.
    Proven correct on the full regression panel.

  - **tensor** (opt-in): pure-torch path-doubling label propagation.
    Keeps everything on-device. Currently blocked on MPS by three
    upstream PyTorch perf issues -- ``torch.bincount``,
    ``torch.unique(return_counts=True)``, and eager ``torch.roll`` are
    all 100-1000× slower than they should be on MPS. See
    ``detector/ml/SANITY_CHECKS.md`` for the diagnostic. Works on CPU,
    just slower than cv2. Set ``Q6_CC_BACKEND=tensor`` to opt in.

Selection: ``CCBackend`` instances at module level (``CV2_CC_BACKEND``
and ``TENSOR_CC_BACKEND``); resolve the default via
``_default_cc_backend()`` which reads ``Q6_CC_BACKEND``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F


# Whether to use torch.compile for the tensor CC path. Set
# TORCH_COMPILE=0 in env to disable for benchmarking. (The cv2 backend
# doesn't use this; it's used by the tensor backend and -- via re-import
# in detector/core.py -- by the per-frame pixel kernels.)
_USE_COMPILE = os.environ.get("TORCH_COMPILE", "1") != "0"


# --- Tensor CC (path-doubling label propagation) ---------------------------
#
# Algorithm: each active pixel starts with its own (1-indexed) linear-index
# label; each iteration takes min(self, each 4-neighbor's label) where
# both endpoints are active. Naïve 1-pixel-per-iter propagation converges
# in O(diameter) iters -- 2998 iters for 1080p worst case, too slow. We
# walk shifts of 1, 2, 4, 8, ..., 1024 (covers up to 1920 px paths) then
# do a small convergence pass at distance 1 to clean up. O(log(diameter))
# iters total.

_CC_SHIFT_SCHEDULE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
_CC_CONVERGE_ITERS = 4


def _tensor_cc_step(labels: torch.Tensor, dim: int, shift: int) -> torch.Tensor:
    """One direction of label propagation along ``dim`` by ``abs(shift)``."""
    abs_shift = abs(shift)
    h = labels.shape[0]
    w = labels.shape[1]
    if dim == 0:
        if shift > 0:
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
    """Pure-tensor connected component labeling via path-doubling."""
    h = active.shape[0]
    w = active.shape[1]
    flat_idx = (torch.arange(h * w, dtype=torch.int32, device=active.device)
                 .view(h, w) + 1)
    labels = torch.where(active, flat_idx, torch.zeros_like(flat_idx))
    for shift in _CC_SHIFT_SCHEDULE:
        labels = _tensor_cc_step(labels, dim=0, shift=shift)
        labels = _tensor_cc_step(labels, dim=0, shift=-shift)
        labels = _tensor_cc_step(labels, dim=1, shift=shift)
        labels = _tensor_cc_step(labels, dim=1, shift=-shift)
    for _ in range(_CC_CONVERGE_ITERS):
        labels = _tensor_cc_step(labels, dim=0, shift=1)
        labels = _tensor_cc_step(labels, dim=0, shift=-1)
        labels = _tensor_cc_step(labels, dim=1, shift=1)
        labels = _tensor_cc_step(labels, dim=1, shift=-1)
    return labels


# Compiled version: TorchInductor fuses the ~60 path-doubling tensor ops
# into a small number of kernels. First call per shape is slow (compile);
# steady-state is fast.
_tensor_cc = (
    torch.compile(_tensor_cc_impl, mode="reduce-overhead", dynamic=False)
    if _USE_COMPILE else _tensor_cc_impl
)


# --- Backend implementations ----------------------------------------------

def _regional_delta_cv2(delta: torch.Tensor,
                         intensity_threshold: float) -> torch.Tensor:
    """cv2-backed regional ΔL. One numpy↔torch boundary crossing per call."""
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
    """Native-tensor region-mean ΔL. Currently slow on MPS (upstream perf)."""
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
    """cv2-backed area test."""
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
    """Native-tensor area test."""
    if not bool(hazard_mask.any().item()):
        return False
    labels = _tensor_cc(hazard_mask)
    flat_labels = labels.view(-1).long()
    n_buckets = labels.numel() + 1
    areas = torch.bincount(flat_labels, minlength=n_buckets)
    areas[0] = 0
    max_area = int(areas.max().item())
    return (max_area > pixels_limit) or ((max_area / n_pixels) > fraction_limit)


# --- DI surface ------------------------------------------------------------

@dataclass(frozen=True)
class CCBackend:
    """Dependency-injection seam for connected-component operations.
    Bundles the two CC-dependent ops behind a stable signature so the
    analyze loop is agnostic to which implementation is in use."""
    name: str
    regional_delta: Callable[[torch.Tensor, float], torch.Tensor]
    region_exceeds_area: Callable[[torch.Tensor, int, float, int], bool]


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


def default_cc_backend() -> CCBackend:
    """Resolved at every call (not import-time) so tests can flip the env
    var between runs without re-importing the module."""
    name = os.environ.get("Q6_CC_BACKEND", "cv2")
    if name not in _CC_BACKENDS:
        raise ValueError(
            f"unknown Q6_CC_BACKEND={name!r}; "
            f"known: {sorted(_CC_BACKENDS)}"
        )
    return _CC_BACKENDS[name]
