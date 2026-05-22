"""Q6's canonical hazard visualization: per-class swimlanes.

For each fixture, draw one horizontal lane per hazard class
(luminance / red / count / pattern). Each lane has:

  * a left header strip with the class name, a colored PASS/FAIL chip,
    and a one-line summary (peak severity, number of fail intervals)
  * a right body with the fail intervals drawn as continuous bars on
    a shared time axis

Why this shape:

  - PSE-safe by construction. Bars are continuous spans (cannot
    flicker), lane backgrounds are muted tints (no high-contrast
    adjacencies), no fine repeating patterns.
  - "Detect if there's an error" is unambiguous. The colored verdict
    chip + lightly-tinted lane background make a FAIL visually loud
    even when the underlying data is one frame wide.
  - "When temporally are the risks" is the x-axis directly.
  - Class differentiation via lane position + color so a viewer never
    confuses one class for another.
  - Severity encoded inside the bar via fill darkness while bar height
    stays uniform, so a fixture with one severe FAIL and a fixture
    with one marginal FAIL look visually distinguishable.

The module previously contained a 3x3 spatial × time heatmap; the
function name ``render_heatmap`` is preserved so external callers
(``report/fixture_report.py``, anything that imports from this module)
keep working.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


# Per-class palette. Muted but distinguishable; no saturated red as a
# flash element.
CLASS_COLORS: dict[str, str] = {
    "luminance": "#c88840",   # warm amber
    "red":       "#b65c5c",   # desaturated brick
    "count":     "#7060a0",   # muted purple
    "pattern":   "#508080",   # muted teal
}
PASS_COLOR = "#1a7f3c"
FAIL_COLOR = "#a8222b"
SAFE_GRID_COLOR = "#dddddd"


# --- Helpers ---------------------------------------------------------------

def _as_dict(obj: Any) -> Any:
    """Recursively convert dataclasses/frozenset to plain dict/list."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _as_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


def _present_classes(result_dict: dict) -> list[str]:
    """Classes that appear anywhere in the result. Always include
    luminance + red so a clean-PASS fixture still renders two lanes
    instead of an empty chart."""
    seen: set[str] = set()
    for f in result_dict.get("per_frame", []):
        for region in f.get("hazard_regions", []):
            seen.update(region.get("classes", []))
    seen.update({"luminance", "red"})
    # Canonical ordering across charts.
    return [c for c in ("luminance", "red", "pattern", "count") if c in seen]


def _per_class_severity_series(result_dict: dict, classes: list[str]):
    """For each class, per-frame peak severity (0 if not triggered)."""
    import numpy as np
    per_frame = result_dict.get("per_frame", [])
    n_frames = len(per_frame)
    out = {cls: np.zeros(n_frames, dtype=np.float32) for cls in classes}
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            for cls, sev in (region.get("severity") or {}).items():
                if cls in out and sev > out[cls][fi]:
                    out[cls][fi] = float(sev)
    return out


def _severity_to_alpha(severity: float) -> float:
    """1.0 (just-FAIL) → 0.45, 2.0+ → 1.0. Keeps a bare-minimum FAIL
    visible while making severe events visually heavier."""
    s = max(1.0, float(severity))
    return min(1.0, 0.45 + 0.45 * (s - 1.0))


# --- Main entry point ------------------------------------------------------

def render_heatmap(result: Any, out_path: Path) -> Path:
    """Render and save the per-class swimlane chart. Returns ``out_path``.

    Public API preserved for back-compat with the older module shape.
    """
    r = _as_dict(result)
    per_frame = r.get("per_frame", [])
    if not per_frame:
        return _render_empty(out_path)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = float(r.get("fps", 0.0)) or 1.0
    duration = float(r.get("n_frames", 0)) / max(fps, 0.001)
    failed_dims = set(r.get("failed_dimensions") or [])
    overall_verdict = r.get("verdict", "?")
    classes = _present_classes(r)
    per_axis = r.get("per_axis") or {}
    severity = _per_class_severity_series(r, classes)

    n_lanes = len(classes)
    fig = plt.figure(figsize=(max(11.5, duration * 1.6 + 2.5),
                                max(2.6, 1.0 * n_lanes + 1.0)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.2], wspace=0.02)
    ax_head = fig.add_subplot(gs[0, 0])
    ax_body = fig.add_subplot(gs[0, 1])

    # --- Headers (left column) ---
    ax_head.set_xlim(0, 1)
    ax_head.set_ylim(-0.5, n_lanes - 0.5)
    ax_head.set_xticks([]); ax_head.set_yticks([])
    for spine in ax_head.spines.values():
        spine.set_visible(False)

    for ai, cls in enumerate(classes):
        y = n_lanes - 1 - ai
        info = per_axis.get(cls, {})
        intervals = info.get("fail_intervals", []) or []
        peak_sev = float(info.get("score", 0.0))
        class_failed = cls in failed_dims
        lane_color = CLASS_COLORS.get(cls, "#888")
        chip_color = FAIL_COLOR if class_failed else PASS_COLOR
        chip_text = f"{cls.upper()}: {'FAIL' if class_failed else 'PASS'}"

        ax_head.axhspan(y - 0.45, y + 0.45, color=lane_color, alpha=0.07)
        ax_head.text(0.05, y + 0.18, chip_text,
                      va="center", ha="left",
                      fontsize=10.5, fontweight="bold", color="white",
                      bbox=dict(facecolor=chip_color, edgecolor="none",
                                 pad=4, boxstyle="round,pad=0.4"))
        if class_failed:
            sub = (f"peak severity {peak_sev:.2f}  •  "
                   f"{len(intervals)} interval{'s' if len(intervals) != 1 else ''}")
        else:
            sub = "no intervals above threshold"
        ax_head.text(0.05, y - 0.25, sub,
                      va="center", ha="left", fontsize=8.5, color="#555")

    # --- Lane bodies (right column) ---
    ax_body.set_xlim(0, max(duration, 0.5))
    ax_body.set_ylim(-0.5, n_lanes - 0.5)
    ax_body.set_yticks([])
    ax_body.set_xlabel("time (s)")
    ax_body.grid(axis="x", color=SAFE_GRID_COLOR, linewidth=0.5)
    ax_body.set_axisbelow(True)

    min_bar_width = max(duration * 0.012, 1 / max(fps, 1.0))
    for ai, cls in enumerate(classes):
        y = n_lanes - 1 - ai
        info = per_axis.get(cls, {})
        intervals = info.get("fail_intervals", []) or []
        lane_color = CLASS_COLORS.get(cls, "#888")
        ax_body.axhspan(y - 0.45, y + 0.45, color=lane_color, alpha=0.07)
        sev_series = severity[cls]
        for start, end in intervals:
            width = max(end - start, min_bar_width)
            i0 = max(0, int(start * fps))
            i1 = min(len(sev_series), max(i0 + 1, int(end * fps)))
            peak_in_interval = (float(sev_series[i0:i1].max())
                                if i1 > i0 else 1.0)
            alpha = _severity_to_alpha(peak_in_interval)
            ax_body.barh(y, width, left=start, height=0.55,
                          color=lane_color, alpha=alpha,
                          edgecolor=FAIL_COLOR, linewidth=1.0, zorder=3)

    # Strong horizontal lane dividers (swimlane style).
    for y in range(n_lanes - 1):
        ax_body.axhline(y + 0.5, color="#aaa", linewidth=1.0)
        ax_head.axhline(y + 0.5, color="#aaa", linewidth=1.0)

    overall_color = FAIL_COLOR if overall_verdict == "FAIL" else PASS_COLOR
    fig.suptitle(f"Q6 — per-class hazard swimlanes  "
                  f"(overall: {overall_verdict}, profile: {r.get('profile_name','?')})",
                  color=overall_color, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _render_empty(out_path: Path) -> Path:
    """Placeholder PNG for fixtures with no per-frame data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, "(no per-frame data)",
            ha="center", va="center", fontsize=14, color="#888")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _main_cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Render Q6's per-class swimlane chart.")
    ap.add_argument("video", type=Path, help="Input video.")
    ap.add_argument("--profile", default="WCAG2.2-SC2.3.1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from detector import analyze
    result = analyze(args.video, args.profile)
    out_path = args.out or (Path(__file__).resolve().parents[1] / "report"
                              / "out" / (args.video.stem + "_swimlanes.png"))
    render_heatmap(result, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
