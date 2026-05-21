"""Re-derive per-standard labels for OURS-extended fixtures from their
generation parameters and category. The original build_extended_corpus.py
filled `expected_label` (a fallback) but left the per-standard columns
empty; this script propagates those labels into the per-standard
columns where the derivation is safely universal.

Mapping logic:
  - Count-axis fixtures (fps_sweep, codec_roundtrip, boundary_precision,
    color_space): label is universal -- standards agree on the
    "> 3 flashes/sec" or "> 6 transitions/sec" threshold. Copy to all
    five per-standard columns.
  - false_positive_battery fixtures: designed to PASS under any standard
    (each tests robustness on a different axis). Copy to all five.
  - area_boundary fixtures: WCAG-strict uses the 25% / 21,824 px area
    threshold; Harding-classic / broadcast standards use the full
    341 x 256 = 87,296 px Harding rectangle. The two readings disagree
    on the 21,824 < area < 87,296 px range. Copy to expected_wcag2_2
    (and expected_trace24, which targets the strict reading); leave
    expected_ofcom2017, expected_itu_r1702_4, expected_iso9241_391
    empty for area_boundary so scoring doesn't punish a broadcast tool
    for the strict reading's verdicts.

Run from repo root:
    PYTHONPATH=. python3 corpus/relabel_ours_extended.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "corpus" / "MANIFEST.csv"

# Categories whose labels are universal across all five standards.
UNIVERSAL_CATEGORIES = {
    "fps_sweep",
    "codec_roundtrip",
    "boundary_precision",
    "color_space",
    "false_positive_battery",
}

# Categories whose labels apply only to WCAG-strict and Trace24.
STRICT_ONLY_CATEGORIES = {
    "area_boundary",
}

ALL_PER_STANDARD_COLS = (
    "expected_trace24",
    "expected_wcag2_2",
    "expected_ofcom2017",
    "expected_itu_r1702_4",
    "expected_iso9241_391",
)
STRICT_PER_STANDARD_COLS = (
    "expected_trace24",
    "expected_wcag2_2",
)


def _category_from_path(p: str) -> str:
    # corpus/generated/OURS-extended/<category>/<fixture>.mp4
    parts = p.split("/")
    try:
        i = parts.index("OURS-extended")
        return parts[i + 1]
    except (ValueError, IndexError):
        return ""


def main() -> int:
    with MANIFEST.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    n_updated = 0
    for row in rows:
        if row.get("source") != "OURS-extended":
            continue
        label = row.get("expected_label", "").strip()
        if label not in ("PASS", "FAIL"):
            continue
        category = _category_from_path(row.get("path", ""))
        if category in UNIVERSAL_CATEGORIES:
            cols = ALL_PER_STANDARD_COLS
        elif category in STRICT_ONLY_CATEGORIES:
            cols = STRICT_PER_STANDARD_COLS
        else:
            print(f"WARN: unknown OURS-extended category {category!r} "
                  f"for {row['path']}; skipping per-standard fill",
                  file=sys.stderr)
            continue
        for col in cols:
            if row.get(col, "").strip() == "":
                row[col] = label
                n_updated += 1

    with MANIFEST.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"updated {n_updated} per-standard label cells across "
          f"{sum(1 for r in rows if r.get('source')=='OURS-extended')} "
          f"OURS-extended fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
