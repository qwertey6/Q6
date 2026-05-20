#!/usr/bin/env python3
"""report/generate_report.py -- compute scored-data sections of the
HTML report and inject them into the template.

Architectural shape:

    report/template.html   -- the static structure + prose, with
                              ``$placeholders`` filled in by this script
                              (uses ``string.Template`` so the CSS's
                              `{}`s don't collide with substitution).
    report/style.css       -- the report's CSS, inlined into the
                              output for self-containedness (single
                              HTML file is easy to email/share).
    report/generate_report.py  -- THIS file. Computes the dynamic
                              fragments (tables, lists) from
                              ``results/scores/scores.json`` +
                              ``corpus/MANIFEST.csv`` and substitutes
                              them into the template. No prose lives
                              here. Edit ``template.html`` for narrative
                              changes; edit this file only when the
                              SHAPE of computed data changes.

Reads:
  * results/scores/scores.json  (from harness/scoring.py)
  * corpus/MANIFEST.csv          (for the known-but-excluded table)
  * report/template.html         (HTML structure + prose with $placeholders)
  * report/style.css             (inlined into output)

Writes:
  * report/out/index.html         (the human-readable report)
  * report/out/scores.csv         (mirror of harness's scores.csv)
  * report/out/scores.json        (mirror)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from string import Template


REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent


# Map each standard-slug column to the profile name a tool should have
# been run with to produce a sound score against that standard's labels.
# Inverse of harness.scoring.PROFILE_TO_STANDARD_SLUG (canonicalised).
STANDARD_TO_CANONICAL_PROFILE = {
    "wcag2.2-sc2.3.1": "WCAG2.2-SC2.3.1",
    "trace24":         "Trace24",
    "itu-r-bt.1702":   "ITU-R-BT.1702",
    "ofcom-gn2":       "Ofcom-GN2-Annex1",
    "nab-j":           "NAB-J",
    "iso9241-391":     None,  # not implemented by any of our profiles
}


# --- Helpers ---------------------------------------------------------------

def _adapter_names(scores: dict) -> list[str]:
    """Extract unique adapter names from per_tool keys formatted
    ``adapter@profile``."""
    names: set[str] = set()
    for k in scores["per_tool"]:
        names.add(k.split("@", 1)[0])
    return sorted(names)


def _fmt(x, places=3):
    if x is None: return "&mdash;"
    try:
        return f"{float(x):.{places}f}"
    except Exception:
        return html.escape(str(x))


def _tool_row(tool: str, b: dict) -> str:
    return (
        f"<tr><td><code>{html.escape(tool)}</code></td>"
        f"<td class='num'>{_fmt(b.get('mcc'))}</td>"
        f"<td class='num'>{_fmt(b.get('recall'))}</td>"
        f"<td class='num'>{_fmt(b.get('specificity'))}</td>"
        f"<td class='num{' fn' if b.get('fn', 0) > 0 else ''}'>{b.get('fn', 0)}</td>"
        f"<td class='num'>{b.get('fp', 0)}</td>"
        f"<td class='num'>{b.get('error', 0)}</td>"
        f"<td class='num'>{b.get('unsupported', 0)}</td>"
        f"<td class='num'>{b.get('fixture_count', 0)}</td>"
        "</tr>"
    )


def _summary_table(scores: dict, source_filter: str) -> str:
    rows = []
    for tool in sorted(scores["per_tool"]):
        b = scores["per_tool"][tool].get(f"source:{source_filter}")
        if not b: continue
        rows.append(_tool_row(tool, b))
    if not rows:
        return f"<p><em>No data for source filter {html.escape(source_filter)!r}.</em></p>"
    return (
        "<table class='lede'>"
        "<thead><tr><th>Tool (adapter@profile)</th><th>MCC</th><th>Recall</th><th>Specificity</th>"
        "<th>FN <span class='warn'>(missed&nbsp;hazards)</span></th><th>FP</th>"
        "<th>ERROR</th><th>UNSUPPORTED</th><th>N (scored)</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _per_standard_table(scores: dict, source_filter: str) -> str:
    """Cross-cut: rows are adapters, columns are standards. Each cell
    uses the result from ``<adapter>@<profile-that-matches-this-standard>``."""
    standards = ["wcag2.2-sc2.3.1", "itu-r-bt.1702", "ofcom-gn2", "trace24", "nab-j", "iso9241-391"]
    header_cells = "".join(f"<th>{html.escape(s)}</th>" for s in standards)
    rows = []
    for adapter in _adapter_names(scores):
        cells = [f"<td><code>{html.escape(adapter)}</code></td>"]
        for std in standards:
            profile = STANDARD_TO_CANONICAL_PROFILE.get(std)
            tool_key = f"{adapter}@{profile}" if profile else adapter
            tool_bucket_map = scores["per_tool"].get(tool_key, {})
            b = tool_bucket_map.get(f"source+standard:{source_filter}/{std}")
            if not b or not b.get("fixture_count"):
                cells.append("<td class='num'>&mdash;</td>")
            else:
                mcc = b.get("mcc")
                fn = b.get("fn", 0)
                fp = b.get("fp", 0)
                n = b.get("fixture_count", 0)
                fn_marker = " &#9888;" if fn > 0 else ""
                title = f"profile={profile} N={n} FN={fn} FP={fp}"
                cells.append(
                    f"<td class='num' title='{html.escape(title)}'>"
                    f"{_fmt(mcc)}{fn_marker}</td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='per-standard'>"
        f"<thead><tr><th>Tool</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class='caption'>MCC per standard. "
        "Each cell uses the tool's run under the canonical profile for "
        "that standard. &#9888; = FN&nbsp;&gt;&nbsp;0 in that slice.</p>"
    )


def _list_fixtures(scores: dict, key: str, fixture_key: str, limit: int = 20) -> str:
    blocks = []
    for tool in sorted(scores["per_tool"]):
        b = scores["per_tool"][tool].get(key)
        if not b: continue
        lst = b.get(fixture_key, [])
        if not lst: continue
        items = "".join(f"<li><code>{html.escape(p)}</code></li>" for p in lst[:limit])
        more = "" if len(lst) <= limit else f" <em>(and {len(lst) - limit} more)</em>"
        blocks.append(
            f"<details><summary><strong>{html.escape(tool)}</strong> &mdash; {len(lst)} fixtures</summary>"
            f"<ul>{items}</ul>{more}</details>"
        )
    return "".join(blocks) or "<p><em>None.</em></p>"


def _known_but_excluded(manifest_csv: Path) -> str:
    rows = []
    with manifest_csv.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["type"] != "excluded-tool":
                continue
            reason = r["notes"].replace("excluded; reason=", "")
            rows.append(
                f"<tr><td>{html.escape(r['source'])}</td>"
                f"<td>{html.escape(r['license'])}</td>"
                f"<td><a href='{html.escape(r['path'])}'>{html.escape(r['path'])}</a></td>"
                f"<td>{html.escape(reason)}</td></tr>"
            )
    return (
        "<table class='excluded'><thead><tr><th>Tool</th><th>License</th>"
        "<th>Upstream</th><th>Reason for exclusion</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _per_axis_deep_dives(scores: dict) -> str:
    """Per-fps + per-codec slices, collapsed to one row per adapter
    (using the canonical WCAG profile) so the table stays readable."""
    blocks = ["<h2>5. Per-axis deep dives</h2>"]

    canonical_profile = STANDARD_TO_CANONICAL_PROFILE["wcag2.2-sc2.3.1"]

    # 5a. Frame-rate sensitivity.
    fps_rates = [24, 25, 30, 50, 60, 90, 120]
    head = ("<tr><th>Tool</th>" +
            "".join(f"<th>{r}&nbsp;fps</th>" for r in fps_rates) + "</tr>")
    rows = []
    for adapter in _adapter_names(scores):
        tool_key = f"{adapter}@{canonical_profile}"
        bucket_map = scores["per_tool"].get(tool_key, {})
        cells = [f"<td><code>{html.escape(adapter)}</code></td>"]
        for r in fps_rates:
            b = bucket_map.get(f"fps:{r}")
            n = b.get("fixture_count", 0) if b else 0
            mcc_val = b.get("mcc") if b else None
            cells.append(f"<td class='num' title='N={n}'>{_fmt(mcc_val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    blocks.append(
        "<h3>5a. Frame-rate sensitivity (MCC by fps; OURS-extended fps_sweep)</h3>"
        f"<table>{head}{''.join(rows)}</table>"
        "<p class='caption'>Each row uses the adapter's WCAG-profile run. "
        "Empty cells mean no fixtures at that fps in the corpus.</p>"
    )

    # 5b. Codec stability.
    blocks.append(
        "<h3>5b. Codec stability (OURS-extended codec_roundtrip)</h3>"
        "<p>The same logical content was re-encoded via H.264 CRF18/CRF28, "
        "ProRes&nbsp;422, VP9. A tool whose verdicts <em>differ</em> across "
        "encodes of identical seed content is unstable on real-world media. "
        "Per-codec MCC:</p>"
    )
    codecs = ["mp4v", "h264_crf18", "h264_crf28", "prores422", "vp9_crf32"]
    head = ("<tr><th>Tool</th>" +
            "".join(f"<th>{html.escape(c)}</th>" for c in codecs) + "</tr>")
    rows = []
    for adapter in _adapter_names(scores):
        tool_key = f"{adapter}@{canonical_profile}"
        bucket_map = scores["per_tool"].get(tool_key, {})
        cells = [f"<td><code>{html.escape(adapter)}</code></td>"]
        for c in codecs:
            b = bucket_map.get(f"codec:{c}")
            n = b.get("fixture_count", 0) if b else 0
            mcc_val = b.get("mcc") if b else None
            cells.append(f"<td class='num' title='N={n}'>{_fmt(mcc_val)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    blocks.append(
        f"<table>{head}{''.join(rows)}</table>"
        "<p class='caption'>Each row uses the adapter's WCAG-profile run.</p>"
    )

    return "".join(blocks)


# --- Main ------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate the comparative HTML report.")
    ap.add_argument("--scores",   type=Path, default=REPO_ROOT / "results" / "scores")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "corpus" / "MANIFEST.csv")
    ap.add_argument("--out",      type=Path, default=REPO_ROOT / "report" / "out")
    ap.add_argument("--template", type=Path, default=THIS_DIR / "template.html")
    ap.add_argument("--css",      type=Path, default=THIS_DIR / "style.css")
    args = ap.parse_args(argv)

    scores_path = args.scores / "scores.json"
    if not scores_path.exists():
        raise SystemExit(f"missing {scores_path}; run harness/scoring.py first")
    scores = json.loads(scores_path.read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    # Mirror raw files into the report bundle for self-containedness.
    for f in ("scores.json", "scores.csv"):
        src = args.scores / f
        if src.exists():
            (args.out / f).write_text(src.read_text())

    placeholders = {
        "css":                     args.css.read_text(),
        "per_std_upstream":        _per_standard_table(scores, "upstream"),
        "per_std_extended":        _per_standard_table(scores, "OURS-extended"),
        "upstream_block":          _summary_table(scores, "upstream"),
        "extended_block":          _summary_table(scores, "OURS-extended"),
        "missed_hazards_upstream": _list_fixtures(scores, "source:upstream",       "fn_fixtures"),
        "missed_hazards_extended": _list_fixtures(scores, "source:OURS-extended",  "fn_fixtures"),
        "false_alarms_upstream":   _list_fixtures(scores, "source:upstream",       "fp_fixtures"),
        "false_alarms_extended":   _list_fixtures(scores, "source:OURS-extended",  "fp_fixtures"),
        "excluded":                _known_but_excluded(args.manifest),
        "deep":                    _per_axis_deep_dives(scores),
    }

    template = Template(args.template.read_text())
    # safe_substitute won't crash if the template has stray $-prefixed
    # tokens we haven't filled (e.g., $foo in a literal context). It
    # also won't error on extras, which we don't have.
    html_doc = template.safe_substitute(placeholders)

    (args.out / "index.html").write_text(html_doc)
    print(f"report: wrote {args.out / 'index.html'}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
