"""Regression-panel test for Q6's classical detector.

A fixed set of fixtures with known-correct verdicts under
WCAG2.2-SC2.3.1. Every algorithm change in the detector loop must
preserve these verdicts. If a verdict here flips, either:

  (a) a real upstream-label is wrong and we have new evidence (rare;
      requires a separate writeup like the Jordan/OQ-5 incident); or
  (b) the change broke detection.

The 9 fixtures span:

  - TRACE wcagc 30fps area pairs (count-axis FAIL vs area-axis PASS)
  - TRACE alternating count-axis pairs
  - Q6-extended false-positive battery (tricky PASSes the detector
    must not over-flag)
  - Q6-extended fps_sweep (clear FAILs at known-failing flash rates)

Expected steady-state runtime: ~5–15s end-to-end on a modern laptop,
warmup-dominated. The test is skipped if the corpus isn't available
(make corpus / git LFS / external download).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# Each row: (label, expected_verdict, fixture_relative_path).
REGRESSION_PANEL: list[tuple[str, str, str]] = [
    # TRACE wcagc 30fps area pairs (one FAIL with count > area_limit and
    # one PASS where the area falls just under the WCAG-strict limit but
    # the count axis still trips at the same temporal rate; the OQ-5
    # incident landed here).
    ("TRACE wcagc f002f038",   "FAIL",
        "corpus/generated/pse-test-media/wcagc_30fps_area01/f002f038.mp4"),
    ("TRACE wcagc a002f038",   "PASS",
        "corpus/generated/pse-test-media/wcagc_30fps_area01/a002f038.mp4"),
    # TRACE alternating 30fps -- count-axis pair (full-screen, single
    # temporal pattern).
    ("TRACE alt f001f037",     "FAIL",
        "corpus/generated/pse-test-media/30fps_alternating_01/f001f037.mp4"),
    ("TRACE alt f001c037",     "PASS",
        "corpus/generated/pse-test-media/30fps_alternating_01/f001c037.mp4"),
    # Q6-extended false-positive battery (must NOT flag).
    ("Q6-extended tiny_area_fast",  "PASS",
        "corpus/generated/Q6-extended/false_positive_battery/tiny_area_fast_flash.mp4"),
    ("Q6-extended equiluminant",    "PASS",
        "corpus/generated/Q6-extended/false_positive_battery/equiluminant_chroma_swap.mp4"),
    # Q6-extended fps_sweep (extreme 31 Hz cases at various fps).
    ("Q6-extended 60fps_fail",      "FAIL",
        "corpus/generated/Q6-extended/fps_sweep/60fps_fail_31hz.mp4"),
    ("Q6-extended 60fps_pass",      "PASS",
        "corpus/generated/Q6-extended/fps_sweep/60fps_pass_14hz.mp4"),
    ("Q6-extended 120fps_fail",     "FAIL",
        "corpus/generated/Q6-extended/fps_sweep/120fps_fail_31hz.mp4"),
]


def _fixture_present(rel_path: str) -> bool:
    return (REPO_ROOT / rel_path).exists()


@pytest.mark.parametrize("label,expected_verdict,rel_path", REGRESSION_PANEL,
                          ids=[r[0] for r in REGRESSION_PANEL])
def test_regression_panel(label: str, expected_verdict: str, rel_path: str):
    """Each fixture must produce the expected verdict under
    WCAG2.2-SC2.3.1."""
    if not _fixture_present(rel_path):
        pytest.skip(f"fixture not materialized: {rel_path} "
                    f"(run `make corpus`)")
    from detector import analyze  # local import: keeps cold-start cheap
    result = analyze(REPO_ROOT / rel_path, profile="WCAG2.2-SC2.3.1")
    assert result.verdict == expected_verdict, (
        f"{label}: expected {expected_verdict}, got {result.verdict} "
        f"(score={result.score:.3f}, failed_dims={result.failed_dimensions})"
    )
