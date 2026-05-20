#!/usr/bin/env python3
"""corpus/build_manifest.py — build corpus/MANIFEST.csv deterministically.

The manifest is the single source of truth for ground truth, joining
fixtures to:
  * their expected label (PASS / FAIL),
  * the specific standard clause that label derives from,
  * upstream provenance (commit hash, path-on-disk),
  * any per-fixture detail file (e.g. IRIS's *_RELATIVE.csv expected log).

This file is regenerated from upstream data and pins; never hand-edited.

Schema (matches brief §2.2 exactly):

    source, license, type, path, expected_label, expected_detail_file,
    standard_clause, frame_rate, color_space, dynamic_range, resolution,
    codec, generation_params, provenance_commit, notes

`expected_label` is PASS or FAIL. For aggregate "set" rows, we instead emit
one row per individual fixture (the per-set CSVs already give us that
granularity).

For TRACE fixtures, `standard_clause` is the *list of standards that apply
to this set* (joined by `;`). Scoring then knows which standards' thresholds
to expect-conformance-to for that fixture. Per-clause text for each standard
profile is documented in `detector/THRESHOLDS.md`.

No label is ever derived from "we ran the tools and voted." Every label
traces to either:
  * an upstream project's own ground-truth file (TRACE per-set CSVs;
    IRIS *_RELATIVE.csv frame-level logs), or
  * the parameters used to generate the fixture (our extended corpus).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


CORPUS_DIR = Path(__file__).resolve().parent
SOURCES = CORPUS_DIR / "sources"
GENERATED = CORPUS_DIR / "generated"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.csv"

# Pinned commits — kept in sync with environment.lock and PROVENANCE.md.
COMMITS = {
    "pse-test-media":         "edf799a15cc1a8817a58c0120a7b25b2b28a1932",
    "pseGuidelines":          "48d0c20f22a3333f64f444159b52c8c9eb097c71",
    "IRIS":                   "d96978ac1107f3463b77f69a9c1b1ec5d45291a0",
    "VideoFlashingReduction": "7357d2f347c8659cc5ab4804b1338cfb0e95f362",
    "IRIS-Unreal-Plugin":     "85311532a588d951b833a7b942234bcc9b578bd1",
}


# --- TRACE applicability matrix --------------------------------------------
# Parsed from corpus/sources/pse-test-media/video_creation/README.md.
# Hard-coded here so manifest generation is deterministic even if the
# upstream README is reformatted; pin must match the upstream README at
# the pinned commit. Verified against the markdown table at retrieval time.
#
# (set_name, applies_iso, applies_itu_r1702, applies_ofcom, applies_trace24,
#  applies_wcag2, applies_nab_j, n_tests, flash_type, fps, color, dynamic_range)
TRACE_SET_TABLE = [
    ("30fps_alternating_01",      True,  True,  True,  True,  True,  False, 16, "Luminance",        30, "sRGB", "SDR"),
    ("broadcast_30fps_01",        True,  True,  True,  False, False, False, 40, "Luminance",        30, "sRGB", "SDR"),
    ("broadcast_30fps_combo01",   True,  True,  True,  False, False, False, 14, "Red & Luminance",  30, "sRGB", "SDR"),
    ("broadcast_30fps_inf01",     True,  True,  True,  False, False, False, 10, "Luminance",        30, "sRGB", "SDR"),
    ("broadcast_30fps_inf02",     True,  True,  True,  False, False, False, 10, "Luminance",        30, "sRGB", "SDR"),
    ("broadcast_30fps_red01",     True,  True,  True,  False, False, False, 18, "Red",              30, "sRGB", "SDR"),
    ("broadcast_30fps_red02",     True,  True,  True,  False, False, False, 30, "Red",              30, "sRGB", "SDR"),
    ("trace24_30fps_01",          False, False, False, True,  False, False, 54, "Luminance",        30, "sRGB", "SDR"),
    ("trace24_30fps_combo01",     False, False, False, True,  False, False, 14, "Red & Luminance",  30, "sRGB", "SDR"),
    ("trace24_30fps_inf01",       False, False, False, True,  False, False, 16, "Luminance",        30, "sRGB", "SDR"),
    ("trace24_30fps_red01",       False, False, False, True,  False, False, 18, "Red",              30, "sRGB", "SDR"),
    ("trace24_30fps_red02",       False, False, False, True,  False, False, 30, "Red",              30, "sRGB", "SDR"),
    ("wcagc_30fps_area01",        False, False, False, False, True,  False, 12, "Luminance",        30, "sRGB", "SDR"),
    ("wcagc_30fps_area02",        False, False, False, False, True,  False, 12, "Luminance",        30, "sRGB", "SDR"),
    ("wcagc_30fps_area03",        False, False, False, False, True,  False, 12, "Red",              30, "sRGB", "SDR"),
]


def trace_set_applicable_standards(row) -> str:
    name, iso, itu, ofcom, trace24, wcag2, nab_j, *_ = row
    out = []
    if iso:     out.append("iso9241-391")
    if itu:     out.append("itu-r-bt.1702")
    if ofcom:   out.append("ofcom-gn2-annex1")
    if trace24: out.append("trace24")
    if wcag2:   out.append("wcag2.2-sc2.3.1")
    if nab_j:   out.append("nab-j")
    return ";".join(out)


# --- Standard-clause boilerplate (the "clause" portion of standard_clause)--
# For TRACE fixtures we cite the applicable standards' general-flash + red-flash
# + area + count thresholds collectively (TRACE sets are designed to exercise
# whichever axis the set name encodes). Per-axis precision lives in
# detector/THRESHOLDS.md.
TRACE_CLAUSE_BY_SET_PATTERN = {
    "alternating": "general flash counting / 3-flashes-per-second; per applicable standards",
    "broadcast":   "luminance and/or red flash count + area + intensity; per applicable broadcast standards (ITU-R BT.1702 / Ofcom GN2 Annex 1)",
    "trace24":     "Trace24 proposed thresholds (Jordan & Vanderheiden 2024); see pseGuidelines spec",
    "wcagc":       "WCAG 2.2 SC 2.3.1 'classic' area + luminance threshold (general flash; 25%-of-screen / 341x256 rect)",
}


def trace_set_clause(set_name: str, applicable: str) -> str:
    for key, clause in TRACE_CLAUSE_BY_SET_PATTERN.items():
        if key in set_name:
            return f"{clause} [applies: {applicable}]"
    return f"see pse-test-media/{set_name} per-set README [applies: {applicable}]"


# --- IRIS fixtures (manually inventoried; their expected logs are the GT) --
# Labels here are NOT vote-derived — they are read off IRIS's shipped
# *_RELATIVE.csv expected logs (presence of any non-zero
# FlashLuminanceFailedFrame / FlashRedFailedFrame / PatternFailedFrame ≡ FAIL).
# build_manifest reads each expected log and derives the label programmatically.
IRIS_VIDEO_DIR = "test/Iris.Tests/data/TestVideos"
IRIS_EXPECTED_DIR = "test/Iris.Tests/data/ExpectedVideoLogFiles"
IRIS_PATTERN_DIR = "test/Iris.Tests/data/TestImages/Patterns"


def derive_iris_video_label(expected_csv: Path) -> tuple[str, list[str], Optional[float]]:
    """Read an IRIS *_RELATIVE.csv expected log; derive PASS/FAIL + failed dims.

    Per the IRIS README, columns FlashLuminanceFailedFrame /
    FlashRedFailedFrame / PatternFailedFrame are nonzero on frames that
    constitute a fail. Their per-second flash count threshold is implemented
    in IRIS itself; for OUR manifest, the expected log is treated as the
    upstream "this is what IRIS says" ground truth for these 8 fixtures.
    """
    failed_dims: set[str] = set()
    first_fail_frame: Optional[int] = None
    fps_estimate: Optional[float] = None
    with expected_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                flash_lum = int(row["FlashLuminanceFailedFrame"])
                flash_red = int(row["FlashRedFailedFrame"])
                pattern   = int(row["PatternFailedFrame"])
                frame_idx = int(row["Frame"])
            except (KeyError, ValueError):
                continue
            if flash_lum:
                failed_dims.add("luminance")
                if first_fail_frame is None: first_fail_frame = frame_idx
            if flash_red:
                failed_dims.add("red")
                if first_fail_frame is None: first_fail_frame = frame_idx
            if pattern:
                failed_dims.add("pattern")
                if first_fail_frame is None: first_fail_frame = frame_idx
    label = "FAIL" if failed_dims else "PASS"
    # IRIS expected logs use a TimeStamp column we could parse for fps, but
    # frame rate is recorded only weakly there. Leave None; harness can probe.
    return label, sorted(failed_dims), None


# --- Manifest row dataclass ------------------------------------------------

MANIFEST_COLUMNS = [
    "source", "license", "type", "path",
    "expected_label", "expected_detail_file",
    "standard_clause", "frame_rate", "color_space", "dynamic_range",
    "resolution", "codec", "generation_params",
    "provenance_commit", "notes",
]


@dataclass
class Row:
    source: str
    license: str
    type: str
    path: str
    expected_label: str
    expected_detail_file: str = ""
    standard_clause: str = ""
    frame_rate: str = ""
    color_space: str = ""
    dynamic_range: str = ""
    resolution: str = ""
    codec: str = ""
    generation_params: str = ""
    provenance_commit: str = ""
    notes: str = ""

    def to_list(self) -> list[str]:
        return [getattr(self, c) for c in MANIFEST_COLUMNS]


# --- Row generators --------------------------------------------------------

def trace_rows() -> Iterable[Row]:
    """One row per TRACE-generated video, label from upstream per-set CSV."""
    base = SOURCES / "pse-test-media" / "video_creation"
    commit = COMMITS["pse-test-media"]
    for set_row in TRACE_SET_TABLE:
        set_name = set_row[0]
        flash_type = set_row[8]
        fps        = set_row[9]
        color      = set_row[10]
        dr         = set_row[11]
        set_dir = base / set_name
        applicable = trace_set_applicable_standards(set_row)
        clause = trace_set_clause(set_name, applicable)
        # Per-set ground-truth CSV.
        gt_csv = set_dir / f"{set_name}.csv"
        if not gt_csv.exists():
            # Some sets carry a different stem; fall back to first .csv.
            csvs = list(set_dir.glob("*.csv"))
            if not csvs:
                continue
            gt_csv = csvs[0]
        with gt_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                stem = r["filename"]
                # Two CSV shapes exist: simple "pass" column, or
                # combo-set "pass_luminance" + "pass_red". For combo sets,
                # the fixture is PASS only if BOTH axes pass (either axis
                # failing is a hazard).
                if "pass" in r:
                    upstream_pass = (r["pass"].strip().upper() == "TRUE")
                else:
                    pl = r.get("pass_luminance", "").strip().upper() == "TRUE"
                    pr = r.get("pass_red", "").strip().upper() == "TRUE"
                    upstream_pass = pl and pr
                limiting_dim  = r["dimension"].strip()  # area|saturation|count|FAIL or composite
                label = "PASS" if upstream_pass else "FAIL"
                # Generated video path lives under corpus/generated/.
                gen_path = GENERATED / "pse-test-media" / set_name / f"{stem}.mp4"
                # Path to the upstream per-test JSON that drives generation —
                # this is the "generation_params" the manifest cites so a
                # reviewer can re-derive the label analytically.
                gen_params_path = set_dir / f"{stem}.json"
                yield Row(
                    source="TRACE/pse-test-media",
                    license="BSD-3-Clause",
                    type="video",
                    path=str(gen_path.relative_to(CORPUS_DIR.parent)),
                    expected_label=label,
                    expected_detail_file=str(gt_csv.relative_to(CORPUS_DIR.parent)),
                    standard_clause=clause,
                    frame_rate=str(fps),
                    color_space=color,
                    dynamic_range=dr,
                    resolution="",       # filled at generation time
                    codec="",            # filled at generation time
                    generation_params=str(gen_params_path.relative_to(CORPUS_DIR.parent)),
                    provenance_commit=commit,
                    notes=(
                        f"set={set_name}; flash_type={flash_type}; "
                        f"upstream_limiting_dimension={limiting_dim}"
                    ),
                )


def iris_video_rows() -> Iterable[Row]:
    """One row per IRIS test video. Label derived from shipped expected log."""
    base = SOURCES / "IRIS"
    commit = COMMITS["IRIS"]
    video_dir = base / IRIS_VIDEO_DIR
    expected_dir = base / IRIS_EXPECTED_DIR
    if not video_dir.exists():
        return
    for vid in sorted(video_dir.glob("*.mp4")):
        stem = vid.stem
        exp = expected_dir / f"{stem}_RELATIVE.csv"
        if not exp.exists():
            continue
        label, failed_dims, _ = derive_iris_video_label(exp)
        yield Row(
            source="EA/IRIS",
            license="BSD-3-Clause",
            type="video",
            path=str(vid.relative_to(CORPUS_DIR.parent)),
            expected_label=label,
            expected_detail_file=str(exp.relative_to(CORPUS_DIR.parent)),
            standard_clause=(
                # IRIS implements ITU/Ofcom-flavored thresholds; their
                # shipped expected logs are the per-frame ground truth.
                "ITU-R BT.1702 / WCAG 2.2 SC 2.3.1 / Ofcom GN2 Annex 1 "
                "[as implemented per IRIS expected per-frame log]"
            ),
            frame_rate="",
            color_space="sRGB",
            dynamic_range="SDR",
            resolution="",
            codec="H.264",
            generation_params="",
            provenance_commit=commit,
            notes=(
                f"upstream_per_frame_log={exp.name}; "
                f"failed_dimensions_from_log={'+'.join(failed_dims) if failed_dims else 'none'}"
            ),
        )


def iris_pattern_rows() -> Iterable[Row]:
    """One row per IRIS pattern test image.

    IRIS ships circular- and line-detection expected results in
    sibling directories. A pattern fixture is FAIL if a matching file
    exists in CircularExpectedResults/ or LineExpectedResults/ (per
    IRIS's test convention: the presence of an expected-result file
    indicates a detected hazardous pattern).
    """
    base = SOURCES / "IRIS"
    commit = COMMITS["IRIS"]
    pat_dir = base / IRIS_PATTERN_DIR
    if not pat_dir.exists():
        return
    circ_expected = {p.name for p in (pat_dir / "CircularExpectedResults").glob("*")
                     if p.is_file()}
    line_expected = {p.name for p in (pat_dir / "LineExpectedResults").glob("*")
                     if p.is_file()}
    for img in sorted(p for p in pat_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        # IRIS uses .jpg and .png; expected-result files may have a different
        # extension (typically .png). Compare by stem.
        stem = img.stem
        # Find any expected file with same stem in either directory.
        def _stem_present(dirset: set[str]) -> bool:
            return any(Path(n).stem == stem for n in dirset)
        is_fail = _stem_present(circ_expected) or _stem_present(line_expected)
        # Build the expected_detail_file pointer to either match if present.
        detail = ""
        for d in ("CircularExpectedResults", "LineExpectedResults"):
            for n in (img.name, stem + ".png"):
                p = pat_dir / d / n
                if p.exists():
                    detail = str(p.relative_to(CORPUS_DIR.parent))
                    break
            if detail:
                break
        yield Row(
            source="EA/IRIS",
            license="BSD-3-Clause",
            type="image-pattern",
            path=str(img.relative_to(CORPUS_DIR.parent)),
            expected_label="FAIL" if is_fail else "PASS",
            expected_detail_file=detail,
            standard_clause=(
                "spatial pattern hazard (bold static stripes / regular patterns); "
                "WCAG 2.2 'Three Flashes or Below Threshold' contemplates pattern "
                "hazards via the same SC; IRIS implements explicit pattern detection."
            ),
            frame_rate="",
            color_space="",
            dynamic_range="",
            resolution="",
            codec="",
            generation_params="",
            provenance_commit=commit,
            notes="IRIS pattern fixture; label = presence of expected-result file in Circular/Line dir.",
        )


def apple_vfr_rows() -> Iterable[Row]:
    """One canonical row for the (triplicate-verified) Apple VFR demo clip."""
    base = SOURCES / "VideoFlashingReduction"
    canonical = base / "VideoFlashingReduction_MATLAB" / "TestContent" / "TestVideo.mp4"
    if not canonical.exists():
        return
    yield Row(
        source="Apple/VideoFlashingReduction",
        license="Apple Sample Code License",
        type="video",
        path=str(canonical.relative_to(CORPUS_DIR.parent)),
        expected_label="UNLABELED",  # Apple ships no per-fixture ground truth
        expected_detail_file="",
        standard_clause=(
            "no per-fixture ground truth shipped by Apple; sanity fixture only; "
            "not used for PASS/FAIL scoring"
        ),
        frame_rate="",
        color_space="",
        dynamic_range="",
        resolution="",
        codec="",
        generation_params="",
        provenance_commit=COMMITS["VideoFlashingReduction"],
        notes=(
            "Triplicate-byte-identical with Mathematica/Resources/movie.mp4 and "
            "Xcode/VideoFlashingReduction/Resources/movie.mp4; "
            "SHA-256 896551b3857a8096d0243046ce21655f858a1e3310d5cf8b43156504b071a25b. "
            "Used as a sanity-pass / smoke fixture; excluded from PASS/FAIL accuracy "
            "metrics in scoring."
        ),
    )


def excluded_tool_rows() -> Iterable[Row]:
    """Excluded tools: not fixtures per se, but recorded in manifest for the
    report's known-but-excluded table. type=excluded-tool, label=N/A."""
    excluded = [
        ("EA/IRIS-Unreal-Plugin",
         "BSD-3-Clause",
         "https://github.com/electronicarts/IRIS-Unreal-Plugin",
         "Not headless-runnable; requires Unreal Engine 5 runtime.",
         COMMITS["IRIS-Unreal-Plugin"]),
        ("TRACE D2 PSE analysis tool",
         "TBD (not yet released)",
         "https://trace.umd.edu/open-source-photosensitive-epilepsy-analysis-tool/",
         "Tool not yet publicly released as of 2026-05-19; no source repo published.",
         ""),
        ("Flikcer",
         "TBD (closed/SaaS)",
         "https://flikcerapp.com/",
         "Web app only; no published open-source library entry point.",
         ""),
        ("samfatu/pse-detection-correction",
         "NONE (no LICENSE file in upstream repo)",
         "https://github.com/samfatu/pse-detection-correction",
         "No LICENSE in repo; default = all rights reserved. Excluded pending license clarification.",
         ""),
        ("Carreira et al. 2025 PSE detection/correction",
         "TBD (paper only)",
         "https://link.springer.com/article/10.1007/s11760-025-04608-4",
         "Paper published; no public reference implementation linked.",
         ""),
    ]
    for name, lic, url, reason, commit in excluded:
        yield Row(
            source=name,
            license=lic,
            type="excluded-tool",
            path=url,
            expected_label="N/A",
            expected_detail_file="",
            standard_clause="",
            frame_rate="",
            color_space="",
            dynamic_range="",
            resolution="",
            codec="",
            generation_params="",
            provenance_commit=commit,
            notes=f"excluded; reason={reason}",
        )


# --- Main ------------------------------------------------------------------

def main() -> None:
    rows: list[Row] = []
    rows.extend(trace_rows())
    rows.extend(iris_video_rows())
    rows.extend(iris_pattern_rows())
    rows.extend(apple_vfr_rows())
    rows.extend(excluded_tool_rows())

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_COLUMNS)
        for r in rows:
            w.writerow(r.to_list())

    # Print a compact summary so the make step's stdout is meaningful.
    n_trace = sum(1 for r in rows if r.source.startswith("TRACE"))
    n_iris_v = sum(1 for r in rows if r.source == "EA/IRIS" and r.type == "video")
    n_iris_p = sum(1 for r in rows if r.source == "EA/IRIS" and r.type == "image-pattern")
    n_apple  = sum(1 for r in rows if r.source.startswith("Apple"))
    n_excl   = sum(1 for r in rows if r.type == "excluded-tool")
    print(f"MANIFEST.csv: {len(rows)} rows "
          f"(TRACE={n_trace}, IRIS_video={n_iris_v}, IRIS_pattern={n_iris_p}, "
          f"Apple={n_apple}, excluded={n_excl})")


if __name__ == "__main__":
    main()
