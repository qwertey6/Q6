"""Spatial-temporal hazard heatmap.

Renders a PNG where the x-axis is time and the y-axis is a coarse
spatial grid (default 3x3 → 9 buckets, listed top-to-bottom,
left-to-right). Each cell's colour is the peak hazard severity in that
spatial bucket at that frame, across all hazard classes.

Reads a detector ``Result`` (dataclass or dict) and writes a PNG. Used
to give a one-glance answer to "where and when does this video have
hazardous regions?"

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


def render_heatmap(result: Any, out_path: Path) -> Path:
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
    heat = np.zeros((N_BUCKETS, n_frames), dtype=np.float32)
    for fi, f in enumerate(per_frame):
        for region in f.get("hazard_regions", []):
            bbox = tuple(region.get("bbox", (0, 0, 0, 0)))
            severity_dict = region.get("severity", {}) or {}
            peak_sev = max(severity_dict.values(), default=0.0)
            if peak_sev <= 0:
                continue
            for b in _bbox_buckets(bbox, width, height):
                if peak_sev > heat[b, fi]:
                    heat[b, fi] = peak_sev

    fps = float(r.get("fps", 0.0)) or 1.0
    duration = n_frames / fps
    extent = [0, duration, N_BUCKETS - 0.5, -0.5]

    fig, ax = plt.subplots(figsize=(max(8, duration * 1.5), 4.5))
    vmax = max(2.0, float(heat.max()) if heat.size else 1.0)
    im = ax.imshow(heat, aspect="auto", extent=extent, origin="upper",
                    cmap="hot_r", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(N_BUCKETS))
    ax.set_yticklabels(_BUCKET_LABELS, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("spatial bucket (3x3 grid)")
    ax.set_title(f"Q6 — {r.get('verdict','?')} — {Path(r.get('profile_name','?')).name} "
                  f"— score {r.get('score', 0.0):.3f}", fontsize=11)

    # Horizontal grid lines between rows of the 3x3.
    for y in (2.5, 5.5):
        ax.axhline(y, color="#888", linewidth=0.5, linestyle="--")

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
