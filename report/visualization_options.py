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
