"""Per-fixture HTML report generator.

Given a detector ``Result`` (or a JSON dict in the same shape), produce
a self-contained HTML file with:

  * top-line verdict, score, profile, fps/duration
  * per-axis summary table (verdict, score, max count, peak area, intervals)
  * per-region cards (bbox, area, classes, severity, confidence band,
    mitigation hints, standards clauses, counterfactual)
  * per-frame trace (svg sparkline of windowed counts + flash area)
  * spatial-temporal heatmap reference (rendered separately, embedded via <img>)

Usage:
    from report.fixture_report import render_fixture_report
    html = render_fixture_report(result, video_path)
    Path("report/out/f038.html").write_text(html)
"""

from __future__ import annotations

import dataclasses
import html
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _as_dict(obj: Any) -> Any:
    """Convert dataclasses / nested structures to plain dict/list. Tolerant
    of frozenset (HazardRegion.classes) and tuples."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _as_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    if isinstance(obj, frozenset):
        return sorted(obj)
    return obj


def _fmt(x: Any, n: int = 3) -> str:
    if x is None: return "—"
    if isinstance(x, float): return f"{x:.{n}f}"
    return str(x)


def _band_color(band: str) -> str:
    return {"marginal": "#f5a623",
            "clear":    "#e25c5c",
            "severe":   "#a8222b"}.get(band, "#888")


def _per_axis_rows(per_axis: dict) -> str:
    if not per_axis:
        return "<tr><td colspan='6'><em>no axes triggered</em></td></tr>"
    rows = []
    for axis_name in sorted(per_axis):
        a = per_axis[axis_name]
        intervals = ", ".join(f"{s:.2f}–{e:.2f}s" for s, e in a.get("fail_intervals", []))
        if not intervals: intervals = "—"
        verdict_color = "#a8222b" if a.get("verdict") == "FAIL" else "#1a7f3c"
        rows.append(
            f"<tr><td><b>{html.escape(axis_name)}</b></td>"
            f"<td style='color:{verdict_color}'><b>{a.get('verdict','?')}</b></td>"
            f"<td>{_fmt(a.get('score'))}</td>"
            f"<td>{a.get('max_windowed_count', 0)}</td>"
            f"<td>{a.get('max_hazard_area_px', 0)} px ({_fmt(a.get('max_hazard_area_frac'))})</td>"
            f"<td>{intervals}</td></tr>"
        )
    return "\n".join(rows)


def _region_card(region: dict, region_idx: int) -> str:
    bbox = region.get("bbox", (0, 0, 0, 0))
    classes = sorted(region.get("classes", []))
    severity_items = sorted(region.get("severity", {}).items(),
                              key=lambda kv: -kv[1])
    severity_str = ", ".join(
        f"<span class='sev-pill'>{html.escape(k)}: {_fmt(v)}</span>"
        for k, v in severity_items)
    band = region.get("confidence_band", "marginal")
    band_color = _band_color(band)
    mit_html = ""
    for m in region.get("mitigation", []):
        alts = "".join(f"<li>{html.escape(a)}</li>"
                       for a in m.get("alternatives", []))
        mit_html += (
            f"<div class='mitigation'>"
            f"<div class='m-axis'>{html.escape(m.get('axis',''))}</div>"
            f"<div>current: <code>{m.get('current','?')} {html.escape(m.get('unit',''))}</code>"
            f", limit: <code>{m.get('limit','?')}</code></div>"
            f"<div class='m-suggest'>{html.escape(m.get('suggestion',''))}</div>"
            f"<ul class='m-alts'>{alts}</ul></div>"
        )
    clauses = region.get("standards_clauses", [])
    clauses_html = ""
    for c in clauses:
        clauses_html += (
            f"<li><a href='{html.escape(c.get('url',''))}'>{html.escape(c.get('standard','?'))}</a>"
            f" — {html.escape(c.get('clause','?'))}</li>"
        )
    cf = region.get("counterfactual", {})
    cf_edits = "".join(f"<li>{html.escape(e)}</li>"
                       for e in cf.get("minimal_edits", []))
    return (
        f"<div class='region-card' style='border-left-color:{band_color}'>"
        f"<div class='r-header'>"
        f"<span class='r-idx'>region #{region_idx + 1}</span>"
        f"<span class='r-band' style='background:{band_color}'>{html.escape(band)}</span>"
        f"</div>"
        f"<div class='r-meta'>"
        f"bbox: <code>({bbox[0]}, {bbox[1]}) → ({bbox[2]}, {bbox[3]})</code> "
        f"&nbsp;&nbsp; area: <code>{region.get('area_px', 0)} px</code> "
        f"&nbsp;&nbsp; centroid: <code>({region.get('centroid', (0,0))[0]:.0f}, "
        f"{region.get('centroid', (0,0))[1]:.0f})</code>"
        f"</div>"
        f"<div class='r-classes'>classes: {', '.join(f'<code>{html.escape(c)}</code>' for c in classes)}</div>"
        f"<div class='r-severity'>{severity_str}</div>"
        f"<details><summary>mitigation</summary>{mit_html}</details>"
        f"<details><summary>standards clauses</summary><ul>{clauses_html}</ul></details>"
        f"<details><summary>counterfactual</summary><ul>{cf_edits}</ul></details>"
        f"</div>"
    )


def _summarize_per_frame(per_frame: list[dict]) -> dict:
    """Aggregate per-frame data for the trace summary."""
    n = len(per_frame)
    if not n:
        return {"n_frames": 0, "max_lum": 0, "max_red": 0, "max_area": 0.0,
                "n_with_regions": 0}
    return {
        "n_frames": n,
        "max_lum": max((f.get("lum_transitions", 0) for f in per_frame), default=0),
        "max_red": max((f.get("red_transitions", 0) for f in per_frame), default=0),
        "max_area": max((f.get("flash_area", 0.0) for f in per_frame), default=0.0),
        "n_with_regions": sum(1 for f in per_frame if f.get("hazard_regions")),
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>Q6 fixture report — {fixture_name}</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:980px;margin:24px auto;padding:0 24px;color:#222;line-height:1.5}}
  h1{{margin-bottom:4px}}
  .verdict{{display:inline-block;padding:6px 14px;border-radius:6px;
            font-weight:700;font-size:1.1em;color:#fff}}
  .v-PASS{{background:#1a7f3c}}
  .v-FAIL{{background:#a8222b}}
  .meta-grid{{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;
              margin:8px 0 24px;font-size:.95em}}
  .meta-grid b{{color:#555}}
  table{{border-collapse:collapse;width:100%;margin:8px 0;font-size:.92em}}
  th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}}
  th{{background:#f6f6f6}}
  .region-card{{border:1px solid #ddd;border-left:6px solid;padding:12px 14px;
                margin:10px 0;border-radius:4px;background:#fafafa}}
  .r-header{{display:flex;justify-content:space-between;align-items:center;
             margin-bottom:6px}}
  .r-idx{{font-weight:600;color:#444}}
  .r-band{{color:#fff;padding:2px 8px;border-radius:3px;font-size:.85em;
           text-transform:uppercase;letter-spacing:.5px}}
  .r-meta,.r-classes,.r-severity{{font-size:.92em;margin:4px 0}}
  .sev-pill{{display:inline-block;background:#eef;border:1px solid #ccd;
             padding:2px 8px;border-radius:10px;margin-right:6px;font-size:.85em}}
  .mitigation{{border-left:3px solid #69a;padding:6px 10px;margin:8px 0;
               background:#f0f4f8}}
  .m-axis{{font-weight:600;color:#369}}
  .m-suggest{{margin:4px 0}}
  details{{margin:6px 0;padding:4px 0}}
  summary{{cursor:pointer;color:#369;font-size:.92em}}
  code{{background:#eef;padding:1px 5px;border-radius:3px;font-size:.9em}}
  footer{{margin-top:30px;padding-top:14px;border-top:1px solid #ddd;
          color:#888;font-size:.85em}}
</style>
</head><body>

<h1>{fixture_name}</h1>
<div>
  <span class="verdict v-{verdict}">{verdict}</span>
  &nbsp; score: <b>{score}</b> &nbsp;
  profile: <code>{profile_name}</code>
</div>

<div class='meta-grid'>
  <b>fixture</b><span><code>{fixture_path}</code></span>
  <b>resolution</b><span>{width} × {height}</span>
  <b>fps</b><span>{fps}</span>
  <b>frames</b><span>{n_frames}</span>
  <b>duration</b><span>{duration}</span>
  <b>first fail</b><span>{first_fail}</span>
  <b>failed axes</b><span>{failed_dims}</span>
</div>

<h2>Per-axis summary</h2>
<table>
  <tr><th>axis</th><th>verdict</th><th>score</th><th>max windowed count</th>
      <th>peak area</th><th>fail intervals</th></tr>
  {per_axis_rows}
</table>

<h2>Hazardous regions ({n_regions} total across {n_frames_with_regions} frames)</h2>
{regions_block}

<h2>Per-frame trace</h2>
<table>
  <tr><th>metric</th><th>value</th></tr>
  <tr><td>max windowed luminance count</td><td>{trace_max_lum}</td></tr>
  <tr><td>max windowed red count</td><td>{trace_max_red}</td></tr>
  <tr><td>max flash area (fraction)</td><td>{trace_max_area}</td></tr>
  <tr><td>frames with hazard regions</td><td>{n_frames_with_regions} / {n_frames}</td></tr>
</table>

<h2>Spatial-temporal heatmap</h2>
{heatmap_block}

<footer>
  Generated by Q6 (q6 PSE benchmark). Detector profile: {profile_name}.<br>
  Standards references on individual regions link to the normative text.
</footer>

</body></html>
"""


def render_fixture_report(result: Any, fixture_path: Path | str = "",
                            heatmap_path: Path | str | None = None) -> str:
    """Render a per-fixture HTML report. ``result`` may be a Result
    dataclass instance or an equivalent dict. If ``heatmap_path`` is
    given, its relative path is embedded as an <img>."""
    r = _as_dict(result)
    per_frame = r.get("per_frame", [])
    per_axis = r.get("per_axis", {})

    # Collect all hazard regions across frames (with origin frame number).
    all_regions: list[tuple[int, float, dict]] = []
    for f in per_frame:
        for region in f.get("hazard_regions", []):
            all_regions.append((f.get("frame", 0), f.get("timestamp", 0.0), region))
    # Deduplicate by bbox (rough dedup -- consecutive frames often have
    # overlapping regions for the same hazard). Show the FIRST occurrence.
    seen_bboxes: set[tuple] = set()
    unique_regions = []
    for frame_idx, ts, region in all_regions:
        bbox_key = tuple(region.get("bbox", ()))
        if bbox_key in seen_bboxes:
            continue
        seen_bboxes.add(bbox_key)
        unique_regions.append((frame_idx, ts, region))

    regions_block_parts = []
    for i, (frame_idx, ts, region) in enumerate(unique_regions):
        regions_block_parts.append(
            f"<div class='region-origin'>first seen at frame {frame_idx} "
            f"(<code>t = {ts:.3f}s</code>)</div>")
        regions_block_parts.append(_region_card(region, i))
    regions_block = "".join(regions_block_parts) or "<em>no hazardous regions detected</em>"

    fps = r.get("fps", 0.0) or 0.0
    n_frames = r.get("n_frames", 0)
    duration = f"{(n_frames / fps):.2f}s" if fps > 0 else "—"

    trace = _summarize_per_frame(per_frame)

    if heatmap_path:
        heatmap_block = (f"<img src='{html.escape(str(heatmap_path))}' "
                          f"alt='spatial-temporal heatmap' "
                          f"style='max-width:100%;border:1px solid #ddd'>")
    else:
        heatmap_block = "<em>heatmap not generated</em>"

    return _HTML_TEMPLATE.format(
        fixture_name=html.escape(Path(str(fixture_path)).name or "fixture"),
        fixture_path=html.escape(str(fixture_path)),
        verdict=html.escape(r.get("verdict", "?")),
        score=_fmt(r.get("score", 0.0)),
        profile_name=html.escape(r.get("profile_name", "?")),
        width=r.get("width", 0),
        height=r.get("height", 0),
        fps=_fmt(fps, 2),
        n_frames=n_frames,
        duration=duration,
        first_fail=_fmt(r.get("first_fail_timestamp"), 3),
        failed_dims=", ".join(r.get("failed_dimensions", [])) or "—",
        per_axis_rows=_per_axis_rows(per_axis),
        n_regions=len(unique_regions),
        n_frames_with_regions=trace["n_with_regions"],
        regions_block=regions_block,
        trace_max_lum=trace["max_lum"],
        trace_max_red=trace["max_red"],
        trace_max_area=_fmt(trace["max_area"]),
        heatmap_block=heatmap_block,
    )


def _main_cli(argv: list[str]) -> int:
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="Render a per-fixture HTML report.")
    ap.add_argument("video", type=Path, help="Input video.")
    ap.add_argument("--profile", default="WCAG2.2-SC2.3.1")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML path (default: report/out/<fixture>.html)")
    args = ap.parse_args(argv)

    from detector import analyze
    from report.heatmap import render_heatmap
    result = analyze(args.video, args.profile)
    out_path = args.out or (REPO_ROOT / "report" / "out"
                              / (args.video.stem + ".html"))
    heatmap_path = out_path.with_suffix(".heatmap.png")
    render_heatmap(result, heatmap_path)
    html_text = render_fixture_report(result, args.video,
                                       heatmap_path=heatmap_path.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text)
    print(f"wrote {out_path}")
    print(f"wrote {heatmap_path}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main_cli(sys.argv[1:]))
