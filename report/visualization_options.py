"""Generate all four candidate hazard-visualization styles for visual
comparison. Run once; produces ``report/out/viz_comparison.html`` plus
the underlying PNGs.

Options (see chat for rationale):

  A. Iterated heatmap -- same 3x3 spatial × time grid as the existing
     post-fix heatmap, but with a muted sequential colormap and stronger
     row separators. Minimal change.
  B. Event-based Gantt -- one row per detected hazard interval, each
     interval drawn as a continuous horizontal bar. Fundamentally
     unflickerable; reads like a project Gantt.
  C. Annotated thumbnails -- one thumbnail per hazardous region,
     extracted at the region's peak-severity frame, with bbox + class +
     severity + timestamp annotated on the frame itself.
  D. Hybrid -- peak-frame thumbnail at top, single-row Gantt below.

All four are PSE-safe by construction (no flicker, no high-contrast
adjacencies, no repeating fine patterns).

Run:
  PYTHONPATH=. python3 -m report.visualization_options
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "report" / "out"

# Sample fixtures: one with a small spatially-localized hazard, one
# with a full-screen sustained hazard. Both well-understood.
TEST_FIXTURES = [
    ("trace_f002f038",
        REPO_ROOT / "corpus" / "generated" / "pse-test-media"
                  / "wcagc_30fps_area01" / "f002f038.mp4"),
    ("q6_60fps_fail_31hz",
        REPO_ROOT / "corpus" / "generated" / "Q6-extended"
                  / "fps_sweep" / "60fps_fail_31hz.mp4"),
]


# Muted accent palette (no saturated red; viridis-derived).
ACCENT_BAR_COLOR    = "#cc6633"   # warm but desaturated
ACCENT_LABEL_COLOR  = "#222222"
SAFE_GRID_COLOR     = "#dddddd"


def _result_to_dict(result: Any) -> dict:
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return result


def _hazard_intervals(result: dict) -> list[dict]:
    """Flatten per_axis.fail_intervals into a list of {start, end, axis,
    severity_peak} dicts, deduplicating overlapping intervals across
    axes (a single hazard event that trips both luminance and red shows
    as one row with multi-class label)."""
    out: list[dict] = []
    for axis_name, axis in (result.get("per_axis") or {}).items():
        for start, end in axis.get("fail_intervals", []):
            out.append({
                "start": float(start), "end": float(end),
                "axis": axis_name,
                "severity_peak": float(axis.get("score", 0.0)),
            })
    # Merge overlaps across axes -- if two axes' intervals overlap,
    # combine into one with comma-joined axis labels.
    out.sort(key=lambda x: x["start"])
    merged: list[dict] = []
    for ev in out:
        if merged and ev["start"] <= merged[-1]["end"]:
            merged[-1]["end"]     = max(merged[-1]["end"], ev["end"])
            merged[-1]["axis"]    = merged[-1]["axis"] + "/" + ev["axis"]
            merged[-1]["severity_peak"] = max(merged[-1]["severity_peak"],
                                                  ev["severity_peak"])
        else:
            merged.append(dict(ev))
    return merged


def _all_hazard_regions(result: dict) -> list[tuple[int, float, dict]]:
    """All hazard regions across all frames as (frame_idx, timestamp,
    region_dict). Deduplicated by bbox -- one entry per spatial region."""
    seen: set[tuple] = set()
    out: list[tuple[int, float, dict]] = []
    for f in result.get("per_frame", []):
        for region in f.get("hazard_regions", []):
            bbox = tuple(region.get("bbox", ()))
            if bbox in seen:
                continue
            seen.add(bbox)
            out.append((f.get("frame", 0), f.get("timestamp", 0.0), region))
    return out


def _extract_frame(video_path: Path, timestamp_s: float) -> np.ndarray | None:
    """Return the BGR frame closest to the given timestamp, or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target = max(0, int(round(timestamp_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _draw_bbox(ax, bbox: tuple, label: str, severity: float,
                 color: str = ACCENT_BAR_COLOR) -> None:
    x0, y0, x1, y1 = bbox
    rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=2.5, edgecolor=color,
                              facecolor=color, alpha=0.18)
    ax.add_patch(rect)
    ax.text(x0 + 6, y0 + 22,
            f"{label}\nsev={severity:.2f}",
            color="white", fontsize=10,
            bbox=dict(facecolor=color, alpha=0.85, edgecolor="none",
                       pad=4, boxstyle="round,pad=0.3"))


# --- Option A: iterated heatmap -------------------------------------------

def render_option_a(result_dict: dict, _video_path: Path, out_path: Path,
                      bucket_ms: int = 200) -> Path:
    """3x3 spatial × time grid with sequential muted colormap, time
    max-pooled into 200ms buckets. Same shape as current heatmap, gentler
    palette."""
    width  = result_dict.get("width", 0)
    height = result_dict.get("height", 0)
    per_frame = result_dict.get("per_frame", [])
    fps = float(result_dict.get("fps", 0.0)) or 1.0
    n_frames = len(per_frame)
    if n_frames == 0 or width == 0 or height == 0:
        return _empty_png(out_path, "Option A: heatmap (no data)")

    n_buckets = 9
    # Per-frame per-bucket severity (matches the existing heatmap logic).
    cell_w = width / 3
    cell_h = height / 3

    def buckets_for(bbox):
        x0, y0, x1, y1 = bbox
        c0 = max(0, min(2, int(x0 // cell_w)))
        c1 = max(0, min(2, int(x1 // cell_w)))
        r0 = max(0, min(2, int(y0 // cell_h)))
        r1 = max(0, min(2, int(y1 // cell_h)))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                yield r * 3 + c

    heat = np.zeros((n_buckets, n_frames), dtype=np.float32)
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            peak = max((region.get("severity") or {}).values(), default=0.0)
            if peak <= 0:
                continue
            for b in buckets_for(tuple(region.get("bbox", (0, 0, 0, 0)))):
                if peak > heat[b, fi]:
                    heat[b, fi] = peak

    # Temporal max-pool into 200ms buckets.
    frames_per_bucket = max(1, int(round((bucket_ms / 1000.0) * fps)))
    n_out = (n_frames + frames_per_bucket - 1) // frames_per_bucket
    pooled = np.zeros((n_buckets, n_out), dtype=np.float32)
    for j in range(n_out):
        s = j * frames_per_bucket
        e = min(n_frames, s + frames_per_bucket)
        if e > s:
            pooled[:, j] = heat[:, s:e].max(axis=1)
    duration = n_frames / fps
    extent = [0.0, duration, n_buckets - 0.5, -0.5]

    vmax = max(2.0, float(pooled.max()) if pooled.size else 1.0)
    fig, ax = plt.subplots(figsize=(max(8, duration * 1.5), 4.5))
    im = ax.imshow(pooled, aspect="auto", extent=extent, origin="upper",
                    cmap="YlOrBr", vmin=0.0, vmax=vmax,
                    interpolation="nearest")
    bucket_labels = ["top-L", "top-C", "top-R",
                     "mid-L", "mid-C", "mid-R",
                     "bot-L", "bot-C", "bot-R"]
    ax.set_yticks(range(n_buckets))
    ax.set_yticklabels(bucket_labels, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spatial bucket (3x3 grid)")
    ax.set_title("Option A -- iterated heatmap")
    for y in (2.5, 5.5):
        ax.axhline(y, color=SAFE_GRID_COLOR, linewidth=0.6, linestyle="-")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("peak severity (>= 1 = FAIL)")
    fig.text(0.01, 0.01,
              f"Temporal smoothing: {bucket_ms}ms max-pool. "
              f"Source video runs at {fps:.1f}fps.",
              fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option B: event-based Gantt ------------------------------------------

def render_option_b(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """One row per hazard interval, drawn as a continuous bar from start
    to end. No flicker possible because each bar is one continuous span."""
    intervals = _hazard_intervals(result_dict)
    duration = float(result_dict.get("n_frames", 0)) / max(
        float(result_dict.get("fps", 0.0)) or 1.0, 0.001)
    if not intervals:
        return _empty_png(out_path, "Option B: no hazardous intervals")

    n_rows = len(intervals)
    fig, ax = plt.subplots(figsize=(max(8, duration * 1.5),
                                      max(2.5, 0.6 + 0.45 * n_rows)))
    for i, ev in enumerate(intervals):
        y = n_rows - 1 - i
        bar_width = max(ev["end"] - ev["start"], duration * 0.005)
        ax.barh(y, bar_width, left=ev["start"], height=0.6,
                color=ACCENT_BAR_COLOR, edgecolor="#7a3a14")
        ax.text(ev["start"] + bar_width + duration * 0.005, y,
                f"{ev['axis']}  (sev {ev['severity_peak']:.2f})",
                va="center", fontsize=10, color=ACCENT_LABEL_COLOR)
    ax.set_xlim(0, max(duration, 0.5))
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"interval {n_rows - i}" for i in range(n_rows)], fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_title("Option B -- event-based Gantt (one row per interval)")
    ax.grid(axis="x", color=SAFE_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option C: annotated thumbnails per region ----------------------------

def render_option_c(result_dict: dict, video_path: Path, out_path: Path,
                      max_thumbnails: int = 4) -> Path:
    """One thumbnail per detected hazardous region, at the region's peak
    timestamp, with bbox + class + severity + timestamp drawn on it."""
    regions = _all_hazard_regions(result_dict)
    if not regions:
        return _empty_png(out_path, "Option C: no hazardous regions")
    regions = regions[:max_thumbnails]

    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5),
                              squeeze=False)
    for i, (frame_idx, ts, region) in enumerate(regions):
        ax = axes[0, i]
        frame = _extract_frame(video_path, ts)
        if frame is None:
            ax.text(0.5, 0.5, "(frame extraction failed)",
                    ha="center", va="center")
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        bbox = tuple(region.get("bbox", (0, 0, 0, 0)))
        classes = sorted(region.get("classes", []))
        severity_peak = max((region.get("severity") or {}).values(), default=0.0)
        _draw_bbox(ax, bbox, ",".join(classes), severity_peak)
        ax.set_title(f"t = {ts:.2f}s   (frame {frame_idx})",
                      fontsize=10, color=ACCENT_LABEL_COLOR)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Option C -- annotated thumbnails (one per region)",
                   fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option D: hybrid (peak thumbnail + Gantt strip) ----------------------

def render_option_d(result_dict: dict, video_path: Path, out_path: Path) -> Path:
    """Peak-frame thumbnail at top with all bboxes drawn at that frame;
    single-row Gantt strip below showing all hazardous intervals."""
    intervals = _hazard_intervals(result_dict)
    if not intervals:
        return _empty_png(out_path, "Option D: no hazardous intervals")
    peak_ts = float(result_dict.get("first_fail_timestamp", 0.0) or 0.0)
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1])

    # Top: peak-frame thumbnail with bboxes
    ax_top = fig.add_subplot(gs[0])
    frame = _extract_frame(video_path, peak_ts)
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax_top.imshow(rgb)
        # Draw every region active at this frame (or closest).
        for f in result_dict.get("per_frame", []):
            if abs(f.get("timestamp", 0.0) - peak_ts) < 0.05:
                for region in f.get("hazard_regions", []):
                    classes = sorted(region.get("classes", []))
                    severity_peak = max(
                        (region.get("severity") or {}).values(), default=0.0)
                    _draw_bbox(ax_top, tuple(region.get("bbox", (0, 0, 0, 0))),
                                ",".join(classes), severity_peak)
                break
        ax_top.set_title(f"peak hazard at t = {peak_ts:.2f}s",
                          fontsize=11, color=ACCENT_LABEL_COLOR)
    else:
        ax_top.text(0.5, 0.5, "(frame extraction failed)", ha="center")
    ax_top.set_xticks([]); ax_top.set_yticks([])

    # Bottom: single-row Gantt
    duration = float(result_dict.get("n_frames", 0)) / max(
        float(result_dict.get("fps", 0.0)) or 1.0, 0.001)
    ax_bot = fig.add_subplot(gs[1])
    for ev in intervals:
        bar_width = max(ev["end"] - ev["start"], duration * 0.005)
        ax_bot.barh(0, bar_width, left=ev["start"], height=0.6,
                     color=ACCENT_BAR_COLOR, edgecolor="#7a3a14")
    ax_bot.set_xlim(0, max(duration, 0.5))
    ax_bot.set_ylim(-0.4, 0.4)
    ax_bot.set_yticks([])
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_title("hazardous intervals", fontsize=10)
    ax_bot.grid(axis="x", color=SAFE_GRID_COLOR, linewidth=0.5)
    ax_bot.set_axisbelow(True)

    fig.suptitle("Option D -- hybrid (peak thumbnail + interval strip)",
                   fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option E: multi-scale hazard density ---------------------------------

from matplotlib.colors import LinearSegmentedColormap

# Soft white-to-charcoal palette. The max value (#444 rather than #000)
# keeps a fully-active cell from being a hard high-contrast adjacency
# against a neighbouring zero cell -- the chart itself stays calm even
# when the underlying signal is intense.
_SOFTGREY_CMAP = LinearSegmentedColormap.from_list(
    "softgrey", ["#ffffff", "#dddddd", "#888888", "#444444"]
)

# Per-class palette for Options F, G, H. Muted but distinguishable; no
# saturated red as a flash element. Each class also has its own
# white-to-class-color cmap for density panels.
CLASS_COLORS = {
    "luminance": "#c88840",   # warm amber
    "red":       "#b65c5c",   # desaturated brick
    "count":     "#7060a0",   # muted purple
    "pattern":   "#508080",   # muted teal
}
CLASS_CMAPS = {
    cls: LinearSegmentedColormap.from_list(cls, ["#ffffff", color])
    for cls, color in CLASS_COLORS.items()
}
PASS_COLOR = "#1a7f3c"
FAIL_COLOR = "#a8222b"


def render_option_e(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """Multi-scale density: each row shows the rolling fraction of frames
    in the trailing window-W that contained ANY hazard region. Window
    sizes ascend (coarsening) from bottom to top -- like a flame chart
    where leaves are below.

    PSE-safety: grayscale colormap softened so max = charcoal not black;
    bilinear interpolation across cells smooths hard edges; rows show
    progressively coarser temporal aggregation so no row can flicker.
    """
    per_frame = result_dict.get("per_frame", [])
    fps = float(result_dict.get("fps", 0.0)) or 1.0
    n_frames = len(per_frame)
    if n_frames == 0:
        return _empty_png(out_path, "Option E: no data")
    duration = n_frames / fps

    # Boolean: did this frame have at least one hazard region?
    has_hazard = np.array(
        [1 if f.get("hazard_regions") else 0 for f in per_frame],
        dtype=np.int32,
    )
    cum = np.concatenate([[0], np.cumsum(has_hazard)])  # cum[t] = count in [0, t)

    # Window sizes (seconds). Pick a fixed ladder that's meaningful for
    # PSE: 33ms (~ 1 frame at 30fps), 100ms, 333ms (≈ 3-flashes/sec
    # window), 1s (the WCAG count window), 5s (sustained-hazard sense).
    # Drop any window that's larger than the source duration; pin the
    # smallest to at least one frame.
    candidate_windows_s = [1/30, 0.1, 1/3, 1.0, 5.0]
    windows_s = [max(1 / fps, W) for W in candidate_windows_s
                  if W <= duration * 1.0]
    if not windows_s:
        windows_s = [1 / fps]
    # Deduplicate while preserving order
    seen: set[float] = set()
    windows_s = [w for w in windows_s
                 if (round(w, 4) not in seen and not seen.add(round(w, 4)))]

    # For each window, compute the rolling per-frame fraction.
    rows: list[np.ndarray] = []
    for W in windows_s:
        W_frames = max(1, int(round(W * fps)))
        # Rolling count via cumsum: count[t] = cum[t+1] - cum[max(0, t-W+1)]
        starts = np.maximum(0, np.arange(n_frames) - W_frames + 1)
        counts = cum[np.arange(n_frames) + 1] - cum[starts]
        rows.append(counts.astype(np.float32) / W_frames)

    # Bottom row = finest window; topmost row = coarsest. User asked for
    # bottom-up coarsening, so reverse the row stack.
    rows_stacked = np.stack(list(reversed(rows)))   # shape (n_windows, n_frames)

    fig, ax = plt.subplots(
        figsize=(max(8, duration * 1.6),
                 max(2.4, 0.55 * len(rows_stacked) + 1.2)),
    )
    im = ax.imshow(
        rows_stacked, aspect="auto",
        extent=[0.0, duration, len(rows_stacked) - 0.5, -0.5],
        cmap=_SOFTGREY_CMAP, vmin=0.0, vmax=1.0,
        interpolation="bilinear",  # smooth, no hard pixel edges
    )

    # Window labels on the y-axis.
    def _label(w: float) -> str:
        if w < 1.0:
            return f"{w * 1000:.0f} ms"
        return f"{w:.1f} s" if w < 10 else f"{w:.0f} s"

    ax.set_yticks(range(len(rows_stacked)))
    ax.set_yticklabels([_label(w) for w in reversed(windows_s)], fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("window size\n(coarser ↑   finer ↓)", fontsize=9)
    ax.set_title("Option E -- multi-scale hazard density")
    # Soft separator lines between rows.
    for y in range(len(rows_stacked) - 1):
        ax.axhline(y + 0.5, color=SAFE_GRID_COLOR, linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("fraction of window with any hazard region")
    fig.text(0.01, 0.01,
              "Soft greyscale: full charcoal = fully-active window; "
              "white = no hazards in that window. Bilinear interpolation; "
              "no row has a temporal resolution finer than 1 frame.",
              fontsize=8, color="#666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _empty_png(out_path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=12, color="#888")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Helpers shared by Options F, G, H ------------------------------------

def _per_class_severity_series(result_dict: dict, classes: list[str]
                                 ) -> dict[str, np.ndarray]:
    """For each class, return a per-frame max-severity timeseries (0 where
    the class didn't trigger in that frame). Uses region.severity[cls]
    so we get continuous values, not just on/off."""
    per_frame = result_dict.get("per_frame", [])
    n_frames = len(per_frame)
    out = {cls: np.zeros(n_frames, dtype=np.float32) for cls in classes}
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            for cls, sev in (region.get("severity") or {}).items():
                if cls in out and sev > out[cls][fi]:
                    out[cls][fi] = float(sev)
    return out


def _per_class_presence_series(result_dict: dict, classes: list[str]
                                 ) -> dict[str, np.ndarray]:
    """For each class, return a per-frame binary indicator (1 = at least
    one region triggered that class in that frame)."""
    per_frame = result_dict.get("per_frame", [])
    n_frames = len(per_frame)
    out = {cls: np.zeros(n_frames, dtype=np.int32) for cls in classes}
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            for cls in region.get("classes", []):
                if cls in out:
                    out[cls][fi] = 1
    return out


def _present_classes(result_dict: dict) -> list[str]:
    """Classes that appear anywhere in this result. Falls back to a
    default pair so empty results still render a recognisable chart."""
    seen: set[str] = set()
    for f in result_dict.get("per_frame", []):
        for region in f.get("hazard_regions", []):
            seen.update(region.get("classes", []))
    # Always at least show luminance + red so a clean PASS still produces
    # a chart that says "all classes PASS" instead of being empty.
    seen.update({"luminance", "red"})
    # Preserve a consistent ordering across charts.
    canonical = [c for c in ("luminance", "red", "pattern", "count") if c in seen]
    return canonical


def _verdict_chip(ax, text: str, color: str, x: float = 0.0, y: float = 1.04) -> None:
    """Place a colored PASS/FAIL chip near the top-left of an axes."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="white",
            bbox=dict(facecolor=color, alpha=1.0, edgecolor="none",
                       pad=4, boxstyle="round,pad=0.4"))


# --- Option F: per-class multi-scale density ------------------------------

def render_option_f(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """Per-class flame chart: one band per hazard class, each band shows
    rolling hazard density at three temporal windows (33ms / 333ms / 1s).
    Each class-band has its own colormap so the bands are visually
    distinct without spatial overlap. Verdict label per class is colored
    PASS-green or FAIL-red so a single-blip FAIL is unmissable."""
    per_frame = result_dict.get("per_frame", [])
    fps = float(result_dict.get("fps", 0.0)) or 1.0
    n_frames = len(per_frame)
    if n_frames == 0:
        return _empty_png(out_path, "Option F: no data")
    duration = n_frames / fps
    failed_dims = set(result_dict.get("failed_dimensions") or [])
    overall_verdict = result_dict.get("verdict", "?")
    classes = _present_classes(result_dict)
    presence = _per_class_presence_series(result_dict, classes)

    windows_s = [w for w in (1 / 30, 1 / 3, 1.0) if w <= duration]
    if not windows_s:
        windows_s = [1 / fps]

    fig, axes = plt.subplots(
        len(classes), 1,
        figsize=(max(8.5, duration * 1.6),
                 max(2.8, 1.4 * len(classes) + 0.5)),
        squeeze=False, sharex=True,
    )
    for ai, cls in enumerate(classes):
        ax = axes[ai, 0]
        cum = np.concatenate([[0], np.cumsum(presence[cls])])
        rows = []
        for W in windows_s:
            W_frames = max(1, int(round(W * fps)))
            starts = np.maximum(0, np.arange(n_frames) - W_frames + 1)
            counts = cum[np.arange(n_frames) + 1] - cum[starts]
            rows.append(counts.astype(np.float32) / W_frames)
        rows_stacked = np.stack(list(reversed(rows)))
        cmap = CLASS_CMAPS.get(cls, _SOFTGREY_CMAP)
        ax.imshow(rows_stacked, aspect="auto",
                   extent=[0.0, duration, len(rows_stacked) - 0.5, -0.5],
                   cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="bilinear")

        class_failed = cls in failed_dims
        chip_color = FAIL_COLOR if class_failed else PASS_COLOR
        chip_text = f"{cls.upper()}: {'FAIL' if class_failed else 'PASS'}"
        _verdict_chip(ax, chip_text, chip_color)

        def _label(w: float) -> str:
            return f"{w * 1000:.0f}ms" if w < 1 else f"{w:.1f}s"
        ax.set_yticks(range(len(rows_stacked)))
        ax.set_yticklabels([_label(w) for w in reversed(windows_s)],
                            fontsize=8)
        ax.set_ylabel("window", fontsize=9)
        for y in range(len(rows_stacked) - 1):
            ax.axhline(y + 0.5, color=SAFE_GRID_COLOR, linewidth=0.4)
    axes[-1, 0].set_xlabel("time (s)")

    overall_color = FAIL_COLOR if overall_verdict == "FAIL" else PASS_COLOR
    fig.suptitle(f"Option F -- per-class multi-scale density "
                  f"(overall: {overall_verdict})",
                  color=overall_color, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option G: per-class severity line plot -------------------------------

def render_option_g(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """Per-class severity over time as line plot. Y = peak windowed
    count / threshold for that class; X = time. Horizontal dashed line
    at y=1 marks the threshold. Above-threshold region for each line is
    shaded in the class color so single-blip FAILs are loudly visible.
    Verdict chip per class in the legend."""
    per_frame = result_dict.get("per_frame", [])
    fps = float(result_dict.get("fps", 0.0)) or 1.0
    n_frames = len(per_frame)
    if n_frames == 0:
        return _empty_png(out_path, "Option G: no data")
    duration = n_frames / fps
    failed_dims = set(result_dict.get("failed_dimensions") or [])
    overall_verdict = result_dict.get("verdict", "?")
    classes = _present_classes(result_dict)
    severity = _per_class_severity_series(result_dict, classes)
    times = np.arange(n_frames) / fps

    fig, ax = plt.subplots(figsize=(max(8.5, duration * 1.6), 4.5))
    ymax_data = max(
        (s.max() for s in severity.values()), default=1.0)
    ymax = max(2.0, float(ymax_data) * 1.1)

    # Highlight every above-threshold region in light red for "danger zone"
    # then draw per-class lines on top.
    ax.axhspan(1.0, ymax, color="#fce8e8", alpha=0.5, zorder=0)
    ax.axhline(1.0, color="#a8222b", linestyle="--", linewidth=1.2,
                label="threshold (FAIL line)")
    for cls in classes:
        color = CLASS_COLORS.get(cls, "#888")
        sev = severity[cls]
        class_failed = cls in failed_dims
        suffix = " FAIL" if class_failed else " PASS"
        ax.plot(times, sev, color=color, linewidth=2.2,
                 label=f"{cls}{suffix}", zorder=2)
        above = sev >= 1.0
        if above.any():
            ax.fill_between(times, 1.0, sev, where=above,
                             color=color, alpha=0.45, zorder=1)
            # Also mark each fail moment with a small triangle so a 1-frame
            # blip can't be missed: a thin spike line.
            fail_idx = np.where(above)[0]
            if len(fail_idx) > 0:
                # Draw a thin vertical line at each fail frame, capped at
                # the peak severity
                for i in fail_idx:
                    ax.vlines(times[i], 1.0, sev[i], color=color,
                                linewidth=0.5, alpha=0.6, zorder=1)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("severity (peak count / threshold;  1.0 = FAIL line)")
    ax.grid(True, axis="y", alpha=0.3)

    overall_color = FAIL_COLOR if overall_verdict == "FAIL" else PASS_COLOR
    ax.set_title(f"Option G -- per-class severity over time "
                  f"(overall: {overall_verdict})",
                  color=overall_color, fontweight="bold")
    leg = ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    # Color FAILing legend entries red, PASSing entries green
    for text in leg.get_texts():
        if "FAIL" in text.get_text():
            text.set_color(FAIL_COLOR)
            text.set_fontweight("bold")
        elif "PASS" in text.get_text():
            text.set_color(PASS_COLOR)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option H: per-class Gantt rows ---------------------------------------

def render_option_h(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """One row per hazard class; each fail interval drawn as a continuous
    bar in that class's row. Light tint behind any FAILing class makes
    even a one-frame interval unmissable. Class verdict + peak severity
    + interval count on the left so a viewer can never mistake a single
    blip for "no big deal."""
    duration = float(result_dict.get("n_frames", 0)) / max(
        float(result_dict.get("fps", 0.0)) or 1.0, 0.001)
    failed_dims = set(result_dict.get("failed_dimensions") or [])
    overall_verdict = result_dict.get("verdict", "?")
    classes = _present_classes(result_dict)
    per_axis = result_dict.get("per_axis") or {}

    fig, ax = plt.subplots(
        figsize=(max(9.0, duration * 1.6),
                 max(2.4, 0.85 * len(classes) + 1.4)),
    )

    for ai, cls in enumerate(classes):
        y = len(classes) - 1 - ai
        info = per_axis.get(cls, {})
        intervals = info.get("fail_intervals", []) or []
        color = CLASS_COLORS.get(cls, "#888")
        class_failed = cls in failed_dims

        # Background tint for failing classes -- impossible to miss.
        if class_failed:
            ax.axhspan(y - 0.45, y + 0.45, color=color, alpha=0.10)

        # Minimum bar width so a one-frame interval still draws visibly.
        min_bar_width = max(duration * 0.012, 1 / max(
            float(result_dict.get("fps", 0.0)) or 1.0, 1.0))
        for start, end in intervals:
            width = max(end - start, min_bar_width)
            ax.barh(y, width, left=start, height=0.55,
                     color=color, edgecolor=FAIL_COLOR, linewidth=1.2,
                     zorder=3)

        # Per-class label on the left side of the chart.
        verdict_text = "FAIL" if class_failed else "PASS"
        verdict_color = FAIL_COLOR if class_failed else PASS_COLOR
        peak_sev = info.get("score", 0.0)
        n_int = len(intervals)
        if class_failed:
            sub = (f"sev {peak_sev:.2f}  •  "
                   f"{n_int} interval{'s' if n_int != 1 else ''}")
        else:
            sub = "no intervals above threshold"
        ax.text(-duration * 0.012, y + 0.18, f"{cls}",
                va="center", ha="right", fontsize=11,
                fontweight="bold", color="#222")
        ax.text(-duration * 0.012, y - 0.18,
                f"{verdict_text}   {sub}",
                va="center", ha="right", fontsize=8.5,
                color=verdict_color,
                fontweight="bold" if class_failed else "normal")

    ax.set_yticks([])
    ax.set_xlim(0, max(duration, 0.5))
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", color=SAFE_GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    overall_color = FAIL_COLOR if overall_verdict == "FAIL" else PASS_COLOR
    ax.set_title(f"Option H -- per-class fail intervals "
                  f"(overall: {overall_verdict})",
                  color=overall_color, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- Option I: per-class swimlanes ----------------------------------------

def _severity_to_alpha(severity: float) -> float:
    """Map a severity score to a fill alpha. 1.0 (just-FAIL) → 0.45,
    2.0+ (severe) → 1.0. Keeps a bare-minimum FAIL still visible while
    making severe events visually heavier."""
    s = max(1.0, float(severity))
    return min(1.0, 0.45 + 0.45 * (s - 1.0))


def render_option_i(result_dict: dict, _video_path: Path, out_path: Path) -> Path:
    """Per-class swimlane chart. Each hazard class is a horizontal lane
    with:
      * a left header strip containing the class name + PASS/FAIL chip +
        summary (peak severity, interval count)
      * a right body containing fail intervals drawn as bars on the
        shared time axis
    Severity is encoded inside each bar via fill darkness so the bar
    height stays uniform (visually quiet) but a severe event reads as
    heavier than a marginal one. Lanes are separated by strong dividers
    and lightly class-tinted backgrounds.

    PSE-safety: no flicker possible (bars are continuous spans); class
    backgrounds and bars are muted, no saturated hues; lane dividers are
    horizontal not vertical, so no chance of a striped pattern.
    """
    duration = float(result_dict.get("n_frames", 0)) / max(
        float(result_dict.get("fps", 0.0)) or 1.0, 0.001)
    failed_dims = set(result_dict.get("failed_dimensions") or [])
    overall_verdict = result_dict.get("verdict", "?")
    classes = _present_classes(result_dict)
    per_axis = result_dict.get("per_axis") or {}

    # Layout: header column wide enough for "PEAK SEVERITY x.xx" + chip,
    # body fills the rest.
    n_lanes = len(classes)
    fig = plt.figure(figsize=(max(11.5, duration * 1.6 + 2.5),
                                max(2.6, 1.0 * n_lanes + 1.0)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.2], wspace=0.02)
    ax_head = fig.add_subplot(gs[0, 0])
    ax_body = fig.add_subplot(gs[0, 1], sharey=None)

    # --- Left: headers ---
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
        chip_color = FAIL_COLOR if class_failed else PASS_COLOR
        chip_text = f"{cls.upper()}: {'FAIL' if class_failed else 'PASS'}"
        # Lane background tint extends into the header for visual continuity
        # with the body lane.
        lane_color = CLASS_COLORS.get(cls, "#888")
        ax_head.axhspan(y - 0.45, y + 0.45, color=lane_color, alpha=0.07)
        # Chip
        ax_head.text(0.05, y + 0.18, chip_text,
                      va="center", ha="left",
                      fontsize=10.5, fontweight="bold", color="white",
                      bbox=dict(facecolor=chip_color, edgecolor="none",
                                 pad=4, boxstyle="round,pad=0.4"))
        # Summary line under the chip
        if class_failed:
            sub = f"peak severity {peak_sev:.2f}  •  {len(intervals)} interval{'s' if len(intervals) != 1 else ''}"
        else:
            sub = "no intervals above threshold"
        ax_head.text(0.05, y - 0.25, sub,
                      va="center", ha="left",
                      fontsize=8.5, color="#555")

    # --- Right: lane bodies ---
    ax_body.set_xlim(0, max(duration, 0.5))
    ax_body.set_ylim(-0.5, n_lanes - 0.5)
    ax_body.set_yticks([])
    ax_body.set_xlabel("time (s)")
    ax_body.grid(axis="x", color=SAFE_GRID_COLOR, linewidth=0.5)
    ax_body.set_axisbelow(True)

    # Per-axis severity timeseries so we can darken the bar at its peak.
    severity = _per_class_severity_series(result_dict, classes)

    min_bar_width = max(duration * 0.012,
                         1 / max(float(result_dict.get("fps", 0.0)) or 1.0, 1.0))

    for ai, cls in enumerate(classes):
        y = n_lanes - 1 - ai
        info = per_axis.get(cls, {})
        intervals = info.get("fail_intervals", []) or []
        class_failed = cls in failed_dims
        lane_color = CLASS_COLORS.get(cls, "#888")
        ax_body.axhspan(y - 0.45, y + 0.45, color=lane_color, alpha=0.07)

        sev_series = severity[cls]
        fps_local = float(result_dict.get("fps", 0.0)) or 1.0
        for start, end in intervals:
            width = max(end - start, min_bar_width)
            i0 = max(0, int(start * fps_local))
            i1 = min(len(sev_series), max(i0 + 1, int(end * fps_local)))
            peak_in_interval = float(sev_series[i0:i1].max()) if i1 > i0 else 1.0
            alpha = _severity_to_alpha(peak_in_interval)
            ax_body.barh(y, width, left=start, height=0.55,
                          color=lane_color, alpha=alpha,
                          edgecolor=FAIL_COLOR, linewidth=1.0, zorder=3)
        # Severity lives in the header summary, not on every bar -- a
        # fixture with 50 intervals would otherwise be unreadable.

    # Strong horizontal dividers between lanes (BPMN-style swimlane look).
    for y in range(n_lanes - 1):
        ax_body.axhline(y + 0.5, color="#aaa", linewidth=1.0)
        ax_head.axhline(y + 0.5, color="#aaa", linewidth=1.0)

    overall_color = FAIL_COLOR if overall_verdict == "FAIL" else PASS_COLOR
    fig.suptitle(f"Option I -- per-class swimlanes (overall: {overall_verdict})",
                   color=overall_color, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --- HTML comparison page -------------------------------------------------

from string import Template

_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>Q6 -- hazard visualization options</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:1200px;margin:24px auto;padding:0 24px;color:#222;line-height:1.5}
  h1{margin-bottom:4px}
  .lede{color:#555;font-size:.95em;margin-bottom:24px}
  .opt-block{margin:36px 0;padding:16px;border:1px solid #ddd;border-radius:6px}
  .opt-block h2{margin:0 0 4px 0;font-size:1.2em}
  .opt-block .pros-cons{display:grid;grid-template-columns:1fr 1fr;gap:12px;
                          margin:8px 0 16px 0;font-size:.92em}
  .pros{background:#f0f7e8;padding:8px 12px;border-radius:4px}
  .cons{background:#fce8e8;padding:8px 12px;border-radius:4px}
  .pros h3,.cons h3{margin:0 0 4px 0;font-size:.95em}
  .fixture-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
  .fixture-row figure{margin:0}
  .fixture-row img{width:100%;border:1px solid #ddd;border-radius:4px}
  .fixture-row figcaption{font-size:.85em;color:#666;margin-top:4px;text-align:center}
</style>
</head><body>

<h1>Q6 -- hazard visualization options</h1>
<p class='lede'>Four candidate visualization styles, each rendered on two
test fixtures: <code>trace_f002f038</code> (small spatially-localized hazard,
~1.4s) and <code>q6_60fps_fail_31hz</code> (full-screen sustained 31Hz hazard,
~3s).
All four are PSE-safe by construction (no flicker, no high-contrast
adjacencies, no fine repeating patterns).</p>

$blocks

</body></html>
""")

_BLOCK_TEMPLATE = Template("""
<div class='opt-block'>
  <h2>$title</h2>
  <p>$description</p>
  <div class='pros-cons'>
    <div class='pros'><h3>Pros</h3>$pros</div>
    <div class='cons'><h3>Cons</h3>$cons</div>
  </div>
  <div class='fixture-row'>
    <figure>
      <img src='$img_small' alt='$title on small-region fixture'>
      <figcaption>trace_f002f038 (small spatial hazard, ~1.4s burst)</figcaption>
    </figure>
    <figure>
      <img src='$img_full' alt='$title on full-screen fixture'>
      <figcaption>q6_60fps_fail_31hz (full-screen 31Hz, ~3s sustained)</figcaption>
    </figure>
  </div>
</div>
""")


def _build_html(blocks_html: str) -> str:
    return _HTML_TEMPLATE.safe_substitute(blocks=blocks_html)


OPTION_META = [
    ("A", "Iterated heatmap",
        "3x3 spatial × time grid; 200ms time buckets via max-pool; "
        "sequential muted YlOrBr colormap (no saturated red); softer "
        "row separators. Minimal change from current.",
        "Familiar shape. Preserves the where+when answer in one chart.",
        "Spatial resolution is coarse (3x3 buckets; can't show actual bbox). "
        "Cells fill a row at a time for full-screen flashes, which still "
        "looks blocky."),
    ("B", "Event-based Gantt",
        "One row per detected hazard interval; each interval drawn as one "
        "continuous bar from its start to its end timestamp. Continuous "
        "spans cannot flicker.",
        "Reads like a project Gantt. No flicker possible by construction. "
        "Multiple intervals are visually distinct. Class label per row.",
        "Doesn't show <i>where</i> in the frame the hazard sits -- just "
        "<i>when</i>. Most useful as a complement to a thumbnail."),
    ("C", "Annotated thumbnails",
        "One frame per detected region, extracted at the region's peak-"
        "severity timestamp, with bbox + class + severity + frame number "
        "annotated on top.",
        "Most communicative -- a content creator sees exactly what to fix. "
        "No temporal pattern at all. Spatial precision is exact (not bucketed).",
        "Doesn't show temporal density. If the hazard recurs 50 times, "
        "this shows it once. Best as a complement, not standalone."),
    ("D", "Hybrid (thumbnail + Gantt strip)",
        "Peak-frame thumbnail with all bboxes drawn at the peak moment, "
        "plus a single-row Gantt strip below showing every interval over "
        "the video duration. Combines C's spatial precision with B's "
        "temporal coverage.",
        "Strongest single image -- shows where (bboxes), what (class "
        "labels), when (Gantt strip). PSE-safe by construction.",
        "Most rendering work; needs frame extraction. If hazards span "
        "multiple non-overlapping spatial regions and don't co-occur at "
        "the peak moment, only the Gantt captures that."),
    ("E", "Multi-scale density",
        "Flame-chart-style: each row is the rolling fraction of frames "
        "containing any hazard, viewed through a different temporal "
        "window (33ms / 100ms / 333ms / 1s / 5s; clamped to video "
        "duration). Bottom row = finest, top = coarsest. Soft greyscale "
        "with bilinear smoothing -- charcoal at max, never pure black.",
        "Reveals the same hazard at multiple scales: a brief burst lights "
        "up only the finest rows; a sustained problem darkens every row. "
        "No spatial dimension to lose, no flicker possible (smoothed "
        "across windows). Reads like a temporal-density flame chart.",
        "Doesn't show <i>where</i> spatially. Multi-region hazards aren't "
        "distinguished -- two simultaneous hazards in different parts of "
        "the frame look the same as one. Pairs naturally with C/D for "
        "the where-info."),
    ("F", "Per-class multi-scale density",
        "Like E but rows grouped by class. Three class-bands stacked "
        "vertically (luminance / red / count), each with its own "
        "muted colormap (amber / desat-brick / muted-purple). Per-class "
        "PASS/FAIL chip top-left of each band.",
        "Densest information per pixel. Class differentiation via row "
        "position <i>and</i> colormap. Verdict chip is unmissable: a "
        "FAIL class shows a red chip even if its data band is otherwise "
        "near-empty.",
        "More vertical real estate per chart. Multi-class fixtures get "
        "tall."),
    ("G", "Per-class severity line",
        "Multi-line plot. Y = severity ratio (peak count / threshold for "
        "that class). X = time. Lines colored by class. Horizontal "
        "dashed line at y=1.0 = FAIL line. Above-1.0 region tinted "
        "light red as the 'danger zone'; per-line above-threshold area "
        "filled in the class colour. Each FAIL frame additionally drawn "
        "as a thin spike so single-blip FAILs are loudly visible.",
        "Most direct answer to 'when temporally are the risks?' Severity "
        "is on the y-axis, time on the x-axis, class via colour. Anyone "
        "with statistics-chart literacy parses it in seconds. Threshold "
        "line + danger-zone shading + per-frame spikes make it impossible "
        "to read a brief excursion above 1.0 as harmless.",
        "Lines can occlude each other if multiple classes peak at the "
        "same moments. Less visually striking than F."),
    ("I", "Per-class swimlanes",
        "BPMN-style swimlane diagram. Each hazard class gets its own "
        "horizontal lane with a header strip on the left (class name + "
        "PASS/FAIL chip + summary) and a wide body on the right where "
        "fail intervals are drawn as bars on the shared time axis. "
        "Lanes are visually separated by strong dividers and lightly "
        "class-tinted backgrounds. Severity encoded inside each bar "
        "via fill darkness (1.0 = light, 2.0+ = saturated).",
        "Reads instantly: each class's behaviour is a self-contained "
        "lane. Lane backgrounds + verdict chips make a FAIL in any lane "
        "impossible to miss; lanes that PASS clearly say so. Severity "
        "encoded inside the bars recovers the magnitude info that plain "
        "Gantt loses. Most polished single-pane shape.",
        "Loses temporal-density detail at the sub-bar level (a long bar "
        "doesn't say whether the class was continuously hazardous or "
        "just had peaks). For that, pair with F or E."),
    ("H", "Per-class Gantt rows",
        "Y = hazard classes (one row per class). X = time. Each fail "
        "interval drawn as a continuous bar in that class's row. Light "
        "background tint behind any FAILing class; minimum bar width "
        "applied so a one-frame interval still draws visibly; class "
        "label on the left side shows PASS/FAIL + peak severity + "
        "interval count.",
        "Quietest of the three. Reads at a glance. Class differentiation "
        "via row position. Background tint makes 'this class FAILed' "
        "impossible to miss even if the interval is one frame wide.",
        "Loses severity magnitude (a slightly-above-threshold blip looks "
        "identical to a 3x-threshold sustained event). Pair with G for "
        "magnitude information."),
]


def main() -> int:
    print("rendering visualization options...")
    blocks: list[str] = []
    # Need detector results for each test fixture.
    from detector import analyze
    fixture_results: dict[str, tuple[Path, dict]] = {}
    for name, path in TEST_FIXTURES:
        if not path.exists():
            print(f"  SKIP {name}: fixture missing at {path}")
            continue
        print(f"  analyzing {name}...")
        r = analyze(path)
        fixture_results[name] = (path, _result_to_dict(r))

    for letter, title, description, pros, cons in OPTION_META:
        renderer = {
            "A": render_option_a,
            "B": render_option_b,
            "C": render_option_c,
            "D": render_option_d,
            "E": render_option_e,
            "F": render_option_f,
            "G": render_option_g,
            "H": render_option_h,
            "I": render_option_i,
        }[letter]
        rendered: dict[str, Path] = {}
        for name, (path, r) in fixture_results.items():
            out_path = OUT_DIR / f"viz_{letter}_{name}.png"
            try:
                renderer(r, path, out_path)
                rendered[name] = out_path
                print(f"  Option {letter} on {name} -> {out_path.name}")
            except Exception as e:
                print(f"  Option {letter} on {name} FAILED: {e}", file=sys.stderr)
        img_small = (rendered.get("trace_f002f038") or Path(
                       "viz_missing.png")).name
        img_full = (rendered.get("q6_60fps_fail_31hz") or Path(
                      "viz_missing.png")).name
        blocks.append(_BLOCK_TEMPLATE.safe_substitute(
            title=f"Option {letter} -- {title}",
            description=description,
            pros=pros, cons=cons,
            img_small=img_small, img_full=img_full,
        ))

    html_path = OUT_DIR / "viz_comparison.html"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_build_html("\n".join(blocks)))
    print(f"\ncomparison page -> {html_path}")
    print(f"open in browser: open '{html_path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
