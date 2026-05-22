"""Derive per-standard ground-truth labels analytically from each fixture's
known parameters, for standards TRACE doesn't ship labels for.

Why this exists: PSE detection's global industry has converged on
Harding-style test methodology, so regional standards (NAB-J, ARIB
TR-B25, ISO 9241-391, NHK guidelines, etc.) differ in numeric
thresholds but not in test corpus. The ground truth for a given
fixture under a given standard is therefore COMPUTABLE from the
fixture's known temporal/spatial parameters -- we don't need a
separate physical corpus per region.

This script fills MANIFEST.csv columns that the upstream corpus
authors didn't supply, using:

  * TRACE fixtures: parse `expected_detail_file` (the per-fixture JSON)
    -> follow `pattern[].temporal_color` reference -> count opposing
    luminance transitions analytically.
  * Q6-extended fixtures: parse `generation_params` JSON to read
    `transitions_per_sec` directly (we authored the generator; the
    ground truth is the input we asked for).

Standards filled:

  * expected_nabj: NAB-J / J-BA (Japan broadcasters) -- differs from
    WCAG/Ofcom only in the absolute count cap (> 5 flashes/sec
    = > 10 transitions/sec). Same area + intensity thresholds as
    Harding-classic.

Future-extensible: add another (name, derive_fn) entry to STANDARDS
to handle ARIB TR-B25, EBU R 168 etc. when their numerics are codified.

Run from repo root:
    PYTHONPATH=. python3 corpus/derive_per_standard_labels.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "corpus" / "MANIFEST.csv"

# Standards we're deriving labels for and the manifest columns they
# populate. We DON'T overwrite existing non-empty cells (TRACE-supplied
# labels stay authoritative).
TARGET_COLUMNS = ("expected_nabj",)


# --- sRGB → relative luminance (WCAG definition; same as detector/core.py)
def _srgb_to_relative_luminance_byte(b: int) -> float:
    v = b / 255.0
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def _l_from_rgba_row(r: int, g: int, b: int) -> float:
    """Compute WCAG relative luminance from an 8-bit RGB triple."""
    return (0.2126 * _srgb_to_relative_luminance_byte(r)
            + 0.7152 * _srgb_to_relative_luminance_byte(g)
            + 0.0722 * _srgb_to_relative_luminance_byte(b))


# --- TRACE temporal-CSV parsing -------------------------------------------

def _parse_trace_temporal_csv(csv_path: Path) -> list[float]:
    """Read a TRACE temporal_color CSV (rows like `1,r,g,b,a`) and return
    the per-frame luminance series."""
    L = []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # header
            if len(row) < 5:
                continue
            try:
                r, g, b = int(row[1]), int(row[2]), int(row[3])
            except ValueError:
                continue
            L.append(_l_from_rgba_row(r, g, b))
    return L


def _count_opposing_transitions_per_sec(L_series: list[float],
                                          fps: int,
                                          threshold: float = 0.10
                                          ) -> int:
    """Max windowed transition count over a 1-sec window. Uses the same
    accumulator semantics as detector.core (cross threshold in opposing
    direction = 1 transition). Returns transitions/sec at peak window."""
    if not L_series:
        return 0
    transitions: list[int] = []   # frame index where each fire happened
    acc = 0.0
    for i in range(1, len(L_series)):
        dL = L_series[i] - L_series[i - 1]
        acc += dL
        if acc >= threshold:
            transitions.append(i)
            acc = 0.0
        elif acc <= -threshold:
            transitions.append(i)
            acc = 0.0
    # Windowed count: maximum number of fires in any (fps)-frame window.
    if not transitions:
        return 0
    max_count = 0
    j = 0
    for i, fire_idx in enumerate(transitions):
        while j < len(transitions) and transitions[j] - fire_idx < fps:
            j += 1
        max_count = max(max_count, j - i)
    return max_count


# --- TRACE fixture: derive NAB-J label -------------------------------------

def _trace_fixture_json_path(fixture_path: str) -> Optional[Path]:
    """Map fixture path -> per-fixture JSON metadata path.

    fixture_path looks like
        corpus/generated/pse-test-media/<set>/<name>.mp4
    Matching JSON is
        corpus/sources/pse-test-media/video_creation/<set>/<name>.json
    """
    p = Path(fixture_path)
    parts = p.parts
    try:
        i = parts.index("pse-test-media")
    except ValueError:
        return None
    set_name = parts[i + 1]
    name = Path(parts[i + 2]).stem
    json_path = (REPO_ROOT / "corpus" / "sources" / "pse-test-media"
                 / "video_creation" / set_name / f"{name}.json")
    return json_path if json_path.exists() else None


def _derive_trace_nabj(manifest_row: dict) -> Optional[str]:
    """Apply NAB-J's > 5 flashes/sec (> 10 transitions/sec) absolute cap.

    Implementation note: NAB-J also has area + intensity axes, both
    matching Harding-classic. For TRACE fixtures, the area-axis
    decision is already encoded in expected_ofcom2017 (which uses the
    same Harding-classic area threshold). So:

      NAB-J FAIL iff (count > 10 transitions/sec)
                AND (area > Harding-classic limit)
                AND (intensity > 0.10)

      Equivalently: if Ofcom is the closest non-Japan broadcast
      analogue, NAB-J FAIL iff Ofcom would FAIL *and* count > 10/sec.
      If Ofcom PASSes, NAB-J PASSes too (NAB-J is looser on count and
      identical on area/intensity).
    """
    json_path = _trace_fixture_json_path(manifest_row.get("path", ""))
    if json_path is None:
        return None
    try:
        with json_path.open() as fh:
            meta = json.load(fh)
    except Exception:
        return None
    fps = int(meta.get("framerate", 30) or 30)
    patterns = meta.get("pattern") or []
    if not patterns:
        return None
    temporal_rel = patterns[0].get("temporal_color")
    if not temporal_rel:
        return None
    temporal_path = (json_path.parent / temporal_rel).resolve()
    if not temporal_path.exists():
        return None
    L_series = _parse_trace_temporal_csv(temporal_path)
    if not L_series:
        return None
    transitions_per_sec = _count_opposing_transitions_per_sec(L_series, fps)
    # NAB-J's count axis: > 10 transitions/sec = > 5 flashes/sec.
    count_above_nabj = transitions_per_sec > 10
    # The area/intensity decision: if Ofcom (Harding-classic) FAILs, both
    # are met. If Ofcom PASSes, NAB-J PASSes too.
    ofcom = (manifest_row.get("expected_ofcom2017", "") or "").strip().upper()
    if ofcom == "PASS":
        return "PASS"
    if ofcom == "FAIL":
        return "FAIL" if count_above_nabj else "PASS"
    return None   # unknown / no Ofcom label


# --- Q6-extended fixture: derive NAB-J label -------------------------------

def _derive_q6_extended_nabj(manifest_row: dict) -> Optional[str]:
    """Use the fixture's generation_params (we authored the synthesizer;
    ground truth is the input)."""
    raw = manifest_row.get("generation_params", "") or ""
    if not raw:
        return None
    try:
        params = json.loads(raw)
    except Exception:
        return None
    tps = params.get("transitions_per_sec")
    area_frac = params.get("area_fraction", 1.0)
    if tps is None:
        return None
    # Mirror the WCAG-strict thresholds where applicable; NAB-J is looser
    # on count (5/sec). For Q6-extended we know area_fraction directly.
    count_above_nabj = float(tps) > 10.0
    # Harding-classic area limit on a 1920x1080 canvas: 87,296 px
    # out of 2,073,600 pixels = ~4.21%. So any fixture with area_fraction
    # > 0.0421 fails the area axis.
    HARDING_AREA_FRAC = 87296 / (1920 * 1080)
    area_above_harding = float(area_frac) > HARDING_AREA_FRAC
    # Intensity assumed from the same generation: if generation_params
    # didn't restrict intensity, treat as above threshold.
    # (Our Q6-extended fixtures use BW pairs by default, so intensity is
    # well above 0.10 unless explicitly noted.)
    a_rgb = params.get("a_rgb", [0, 0, 0])
    b_rgb = params.get("b_rgb", [255, 255, 255])
    intensity_above = abs(_l_from_rgba_row(*a_rgb)
                            - _l_from_rgba_row(*b_rgb)) > 0.10
    if count_above_nabj and area_above_harding and intensity_above:
        return "FAIL"
    return "PASS"


def _derive_nabj(row: dict) -> Optional[str]:
    source = (row.get("source") or "").strip()
    if "pse-test-media" in source:
        return _derive_trace_nabj(row)
    if source == "Q6-extended":
        return _derive_q6_extended_nabj(row)
    # Other sources (EA/IRIS, Apple, Flikcer, Kaya) -- no derivation
    # method; their global expected_label remains the fallback.
    return None


# --- Main ------------------------------------------------------------------

def main() -> int:
    with MANIFEST.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "expected_nabj" not in fieldnames:
        # Insert next to the other expected_* columns for readability.
        try:
            after = fieldnames.index("expected_iso9241_391") + 1
        except ValueError:
            after = len(fieldnames)
        fieldnames.insert(after, "expected_nabj")
        for row in rows:
            row.setdefault("expected_nabj", "")
    n_filled = 0
    n_already = 0
    n_no_method = 0
    for row in rows:
        if (row.get("expected_nabj", "") or "").strip() in ("PASS", "FAIL"):
            n_already += 1
            continue
        label = _derive_nabj(row)
        if label is None:
            n_no_method += 1
            continue
        row["expected_nabj"] = label
        n_filled += 1
    with MANIFEST.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"expected_nabj: filled {n_filled} new, kept {n_already} existing, "
          f"{n_no_method} fixtures had no derivation method")
    # Distribution
    from collections import Counter
    dist = Counter(row.get("expected_nabj", "") or "<empty>" for row in rows)
    print(f"  distribution: {dict(dist)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
