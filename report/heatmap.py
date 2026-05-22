"""Spatial-temporal hazard heatmap.

Renders a PNG where the x-axis is time and the y-axis is a coarse
spatial grid (default 3x3 → 9 buckets, listed top-to-bottom,
left-to-right). Each cell's colour is the peak hazard severity in that
spatial bucket within a small time window.

**Why this matters for PSE-safe rendering:** a naïve plot with one
column per video frame reproduces the source's flicker frequency in
the visualization itself -- a heatmap of a 31 Hz hazardous video
becomes a 31 Hz stripe pattern, which is the same hazard we're trying
to warn about. To prevent that, we aggregate frames into wider time
buckets (default 200 ms, well below 5 Hz) using max-pool, so the
semantic "this region was hazardous at this time" is preserved without
inheriting the source's flicker. A subtitle on every rendered chart
documents the smoothing window so readers know it's intentional.

Bucket order in the y-axis (top→bottom):
  0: top-left      1: top-center     2: top-right
  3: middle-left   4: middle-center  5: middle-right
  6: bottom-left   7: bottom-center  8: bottom-right
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


GRID_SIZE = 3  # 3x3 spatial grid
N_BUCKETS = GRID_SIZE * GRID_SIZE
_BUCKET_LABELS = [
    "top-L", "top-C", "top-R",
    "mid-L", "mid-C", "mid-R",
    "bot-L", "bot-C", "bot-R",
]

# Width of the temporal max-pool bucket. 200 ms keeps any visible
# alternation in the rendered chart below 5 Hz (worst case: 2.5 Hz),
# safely below the 3-flash-per-second WCAG count threshold. Reducing
# this risks the visualization becoming a PSE hazard itself.
_DEFAULT_BUCKET_MS = 200


def _as_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _as_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


def _bbox_buckets(bbox: tuple[int, int, int, int],
                    width: int, height: int) -> list[int]:
    """Return the list of spatial-grid bucket indices that bbox intersects."""
    x0, y0, x1, y1 = bbox
    cell_w = width / GRID_SIZE
    cell_h = height / GRID_SIZE
    col0 = max(0, min(GRID_SIZE - 1, int(x0 // cell_w)))
    col1 = max(0, min(GRID_SIZE - 1, int(x1 // cell_w)))
    row0 = max(0, min(GRID_SIZE - 1, int(y0 // cell_h)))
    row1 = max(0, min(GRID_SIZE - 1, int(y1 // cell_h)))
    out: list[int] = []
    for r in range(row0, row1 + 1):
        for c in range(col0, col1 + 1):
            out.append(r * GRID_SIZE + c)
    return out


def _max_pool_time(per_frame_heat, fps: float, bucket_ms: int):
    """Aggregate per-frame heat (shape: N_BUCKETS x n_frames) into wider
    time buckets via max-pool. Returns (pooled_heat, bucket_duration_s).

    The PSE-safety reason: rendering one column per video frame
    reproduces the source's flicker frequency in the visualization
    itself. A 31 Hz hazard video becomes a 31 Hz stripe pattern --
    the same kind of hazard we're warning about. Pooling to wider
    time buckets defeats this by guaranteeing visible alternation in
    the chart stays below ~5 Hz.
    """
    import numpy as np
    if fps <= 0 or per_frame_heat.size == 0:
        return per_frame_heat, 1.0 / max(fps, 1.0)
    n_frames = per_frame_heat.shape[1]
    bucket_dur_s = bucket_ms / 1000.0
    n_time_buckets = max(1, int(np.ceil(n_frames / (fps * bucket_dur_s))))
    pooled = np.zeros((per_frame_heat.shape[0], n_time_buckets),
                       dtype=per_frame_heat.dtype)
    for fi in range(n_frames):
        bi = min(int(fi / (fps * bucket_dur_s)), n_time_buckets - 1)
        np.maximum(pooled[:, bi], per_frame_heat[:, fi], out=pooled[:, bi])
    return pooled, bucket_dur_s


def render_heatmap(result: Any, out_path: Path,
                     bucket_ms: int = _DEFAULT_BUCKET_MS) -> Path:
    """Render and save the spatial-temporal heatmap. Returns out_path."""
    r = _as_dict(result)
    width = r.get("width", 0)
    height = r.get("height", 0)
    per_frame = r.get("per_frame", [])
    if not per_frame or width == 0 or height == 0:
        return _render_empty(out_path)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = len(per_frame)
    per_frame_heat = np.zeros((N_BUCKETS, n_frames), dtype=np.float32)
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            bbox = tuple(region.get("bbox", (0, 0, 0, 0)))
            severity_dict = region.get("severity", {}) or {}
            peak_sev = max(severity_dict.values(), default=0.0)
            if peak_sev <= 0:
                continue
            for b in _bbox_buckets(bbox, width, height):
                if peak_sev > per_frame_heat[b, fi]:
                    per_frame_heat[b, fi] = peak_sev

    fps = float(r.get("fps", 0.0)) or 1.0
    duration = n_frames / fps

    # PSE-safe rendering: max-pool to wider time buckets so the chart
    # itself can't reproduce the source's flicker frequency.
    heat, bucket_dur_s = _max_pool_time(per_frame_heat, fps, bucket_ms)
    extent = [0, duration, N_BUCKETS - 0.5, -0.5]

    fig, ax = plt.subplots(figsize=(max(8, duration * 1.5), 4.5))
    vmax = max(2.0, float(heat.max()) if heat.size else 1.0)
    # Sequential colormap with no white→bright-red transition; the bold
    # bicolour cmaps (hot_r in particular) create high-contrast edges
    # at hazard onset that read as flicker.
    im = ax.imshow(heat, aspect="auto", extent=extent, origin="upper",
                    cmap="Reds", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(N_BUCKETS))
    ax.set_yticklabels(_BUCKET_LABELS, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spatial bucket (3x3 grid)")
    ax.set_title(f"Q6 — {r.get('verdict','?')} — {Path(r.get('profile_name','?')).name} "
                  f"— score {r.get('score', 0.0):.3f}", fontsize=11)
    # Footer noting the PSE-safe smoothing.
    ax.text(0.0, -0.18,
            f"chart temporally aggregated to {bucket_ms} ms buckets "
            f"(max-pool) so the visualization itself stays below ~5 Hz "
            f"and does not reproduce the source's flicker frequency",
            transform=ax.transAxes, fontsize=8, color="#666",
            verticalalignment="top")

    # Horizontal grid lines between rows of the 3x3 (also de-emphasised
    # to avoid creating their own visual flicker against the colormap).
    for y in (2.5, 5.5):
        ax.axhline(y, color="#bbb", linewidth=0.4, linestyle="--")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("peak severity (≥1 = FAIL)")

    fig.tight_layout()
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
    ap = argparse.ArgumentParser(description="Render spatial-temporal hazard heatmap PNG.")
    ap.add_argument("video", type=Path, help="Input video.")
    ap.add_argument("--profile", default="WCAG2.2-SC2.3.1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from detector import analyze
    result = analyze(args.video, args.profile)
    out_path = args.out or (Path(__file__).resolve().parents[1] / "report"
                              / "out" / (args.video.stem + "_heatmap.png"))
    render_heatmap(result, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
