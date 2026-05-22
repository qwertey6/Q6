"""Adapter for our detector.

By design, this adapter goes through the same interface as every external
tool. It does NOT have privileged access to the corpus manifest or any
ground-truth labels. The runner passes only ``fixture_path``.
"""

from __future__ import annotations

import time
from pathlib import Path

from detector import analyze, detect_static_pattern_hazard
from harness.schema import NormalizedResult, PER_FRAME_CSV_HEADER


TOOL = "q6"
TOOL_VERSION = "0.2.0"   # detector v2: region-aware area + pattern detection

# Profiles this adapter supports for multi-profile harness runs. Our
# detector reads each as a Profile object from detector.PROFILES and
# actually changes its thresholds accordingly, so PROFILE_AFFECTS_BEHAVIOR
# is True -- the runner will invoke run() once per profile per fixture.
SUPPORTED_PROFILES = [
    "WCAG2.2-SC2.3.1",
    "WCAG2.2-classic",
    "ITU-R-BT.1702",
    "Ofcom-GN2-Annex1",
    "Trace24",
    "NAB-J",
    "ISO9241-391",
]
PROFILE_AFFECTS_BEHAVIOR = True


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1",
        per_frame_out: Path | None = None) -> dict:
    t0 = time.perf_counter()
    # Image fixtures (IRIS pattern PNGs) → static spatial-pattern detection
    # (ITU-R BT.1702 §3 / WCAG SC 2.3.1 pattern clause).
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            is_hazard, info = detect_static_pattern_hazard(fixture_path,
                                                          profile=profile)
        except Exception as e:
            return NormalizedResult(
                fixture_id=fixture_path.name,
                verdict="ERROR",
                tool=TOOL, tool_version=TOOL_VERSION,
                runtime_seconds=time.perf_counter() - t0,
                raw_output_path="",
                standard_profile=profile,
                error_message=f"{type(e).__name__}: {e}",
            ).to_dict()
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="FAIL" if is_hazard else "PASS",
            failed_dimensions=["pattern"] if is_hazard else [],
            first_fail_timestamp=0.0 if is_hazard else None,
            tool=TOOL, tool_version=TOOL_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
        ).to_dict()

    try:
        result = analyze(fixture_path, profile)
    except Exception as e:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="ERROR",
            tool=TOOL, tool_version=TOOL_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message=f"{type(e).__name__}: {e}",
        ).to_dict()

    per_frame_csv_path = ""
    if per_frame_out is not None:
        per_frame_out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with per_frame_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(PER_FRAME_CSV_HEADER)
            for pf in result.per_frame:
                w.writerow([pf.frame, pf.lum_transitions, pf.red_transitions,
                            f"{pf.flash_area:.6f}", f"{pf.pattern_risk:.4f}"])
        per_frame_csv_path = str(per_frame_out)

    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=result.verdict,
        failed_dimensions=result.failed_dimensions,
        first_fail_timestamp=result.first_fail_timestamp,
        per_frame_csv=per_frame_csv_path or None,
        tool=TOOL, tool_version=TOOL_VERSION,
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=result.profile_name,
        score=result.score,
        per_axis_scores={name: a.score for name, a in result.per_axis.items()},
    ).to_dict()
