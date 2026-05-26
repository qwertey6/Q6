"""Adapter for Kaya et al. (2025) -- ``samfatu/pse-detection-correction``.

Upstream: https://github.com/samfatu/pse-detection-correction
Pinned commit: 0dd4eb79441b71ccc88f978d31a6ad1e0bd351b7
License: pending PR at samfatu/pse-detection-correction#9 (BSD-3-Clause).
         Prof. Aydın Kaya granted email permission for inclusion in
         this benchmark with appropriate citation/provenance until the
         license PR is merged.

The tool is described in the SIViP 2025 paper by Aydın Kaya et al. as
a WCAG SC 2.3.1 detector + corrector. We invoke only the detection
path: ``CustomVideo(path).analyse_video()`` runs the upstream's W3C
implementation, after which ``video.flashes`` lists detected flash
intervals as ``(class, start_frame, end_frame)`` tuples and
``video.flashing_frame_count`` reports the total flagged frames.

  - PASS iff no flash intervals were flagged
  - FAIL otherwise
  - Continuous score: fraction of frames flagged (used for AUROC)

Static images are not supported (the tool reads via cv2.VideoCapture).

We import the upstream code directly rather than running its CLI as a
subprocess -- this skips the (irrelevant for us) correction step,
avoids per-call cold-start, and lets us suppress the tool's verbose
stdout chatter. ``sys.path`` is augmented at first call to point at
the pinned local clone under ``corpus/sources/``.

Interpretation note: upstream's WCAG area axis is implemented in
``PhotosensitivitySafetyEngine/guidelines/w3c.py`` as
``area_averages_max(x, fragment_shape=(1/3, 1/3), threshold=0.25)``,
which flags a frame iff at least one non-overlapping 1/3 × 1/3 frame
cell has >= 25% of its pixels active. On 1920x1080 that's an effective
~57,600 px floor on contiguous active area, where WCAG-strict (the
TRACE label reading) puts the floor at ~21,824 px via a sliding 341x256
reference rectangle -- about 2.6x looser. The tool therefore
systematically misses small-localized hazards under TRACE's WCAG-strict
labels. See ``detector/ml/SANITY_CHECKS.md`` Check 5 for the full
verification and discussion.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

from harness.schema import NormalizedResult


TOOL = "samfatu_pse"
TOOL_VERSION = "0dd4eb79"   # short pinned-commit prefix


# Their tool implements WCAG SC 2.3.1; we only score it under that
# profile. (They also have an Ofcom guideline in
# PhotosensitivitySafetyEngine/guidelines/ofcom.py but it isn't wired
# into CustomVideo by default.)
SUPPORTED_PROFILES = ["WCAG2.2-SC2.3.1"]
PROFILE_AFFECTS_BEHAVIOR = False   # one-pass detector, profile-agnostic


_SOURCE_DIR = (Path(__file__).resolve().parents[2] / "corpus" / "sources"
                / "samfatu_pse-detection-correction")


def _ensure_on_path() -> bool:
    """Add the upstream clone to sys.path. Returns False if the source
    directory isn't materialised (no clone done)."""
    if not _SOURCE_DIR.exists():
        return False
    src = str(_SOURCE_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    return True


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    t0 = time.perf_counter()

    # Static images aren't supported (tool reads via cv2.VideoCapture).
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version=TOOL_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            unsupported_reason="samfatu_pse operates on video only.",
        ).to_dict()

    if not _ensure_on_path():
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version=TOOL_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                f"upstream clone missing at {_SOURCE_DIR.name}; "
                f"run corpus/fetch_sources.sh"
            ),
        ).to_dict()

    try:
        # Their module has top-level `print()` calls and progress prints;
        # capture them so the harness output stays readable.
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            # Imported lazily so the module isn't loaded on every adapter
            # discovery -- e.g. when the user only asks for `q6`.
            from custom_video import CustomVideo  # type: ignore
            video = CustomVideo(str(fixture_path))
            video.analyse_video()
            n_flashes = len(video.flashes) if video.flashes else 0
            flashing_count = int(video.flashing_frame_count)
            n_frames = int(video.frame_count)
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

    verdict = "FAIL" if n_flashes > 0 else "PASS"
    # Continuous score: fraction of frames the upstream tool flagged.
    # Bounded [0, 1]; enables AUROC computation against ground truth.
    score = float(flashing_count) / max(n_frames, 1)

    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=verdict,
        # Upstream emits flash intervals of class 'general' / 'red' /
        # 'both'. We don't translate per-interval class to per-fixture
        # failed_dimensions here; collapsing all to "luminance" on FAIL
        # is the conservative WCAG mapping (general flashes are
        # luminance-axis hazards).
        failed_dimensions=["luminance"] if verdict == "FAIL" else [],
        first_fail_timestamp=None,
        tool=TOOL, tool_version=TOOL_VERSION,
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=profile,
        score=score,
    ).to_dict()
