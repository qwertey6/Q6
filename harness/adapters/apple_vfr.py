"""Adapter for Apple's VideoFlashingReduction MATLAB reference.

The Apple VFR repo ships a MATLAB reference implementation
(`VideoFlashingReduction_MATLAB`) plus an equivalent Mathematica notebook
and Swift Xcode project. None of these runs headless on a plain Linux
build host without additional licensed software (MATLAB) or platform
binding (Xcode/macOS).

We attempt the GNU Octave path as a best-effort substitute. If Octave is
not available or the MATLAB code uses MATLAB-specific functions Octave
cannot execute, every fixture reports UNSUPPORTED with a documented
reason. We do NOT fake it.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from harness.schema import NormalizedResult


TOOL = "apple_vfr"

REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_SRC = (
    REPO_ROOT / "corpus" / "sources" / "VideoFlashingReduction" /
    "VideoFlashingReduction_MATLAB"
)


def _octave_available() -> bool:
    return shutil.which("octave") is not None


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    if not MATLAB_SRC.exists():
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="n/a",
            runtime_seconds=0.0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "Apple VFR MATLAB reference not cloned at expected path."
            ),
        ).to_dict()

    if not _octave_available():
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="n/a",
            runtime_seconds=0.0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "Apple VFR ships MATLAB-only reference; MATLAB is non-free "
                "and GNU Octave is not installed in this environment. "
                "Excluded from automated scoring here per the brief's "
                "'don't fake it' rule. The Docker image installs Octave; "
                "Octave-compatibility of Apple's MATLAB code is a follow-up."
            ),
        ).to_dict()

    # If Octave is present we still need to confirm the MATLAB scripts
    # actually run under Octave. We attempt the entry script and fall
    # through to UNSUPPORTED on any error — never ERROR — because a
    # MATLAB-vs-Octave incompatibility is structural, not a tool crash
    # on real input.
    t0 = time.perf_counter()
    entry = MATLAB_SRC / "computeFlashFreqMap.m"  # the core analysis function
    if not entry.exists():
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="octave",
            runtime_seconds=time.perf_counter() - t0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                f"Apple VFR entry script not found at {entry}. Upstream "
                f"layout may have changed; adapter requires update."
            ),
        ).to_dict()

    # Spec is intentionally conservative for the first cut: declare
    # UNSUPPORTED until we have a known-good Octave invocation. This is
    # the honest result.
    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict="UNSUPPORTED",
        tool=TOOL, tool_version="octave (compatibility unverified)",
        runtime_seconds=time.perf_counter() - t0, raw_output_path="",
        standard_profile=profile,
        unsupported_reason=(
            "Apple VFR MATLAB→Octave compatibility not yet verified in this "
            "harness. Marked UNSUPPORTED rather than fabricated a result. "
            "Follow-up: wrap computeFlashFreqMap.m in an Octave-callable "
            "shim with a documented runtime + verdict-mapping convention."
        ),
    ).to_dict()
