"""Adapter for EA IRIS (C++).

IRIS is built from `corpus/sources/IRIS` at the pinned commit via the
`macos-release` or `linux-release` cmake preset (with `BUILD_EXAMPLE_APP=ON`).
If the example-app binary is not present at adapter load time, every
fixture invocation reports UNSUPPORTED with a documented reason -- this
is the honest result for an environment that hasn't built IRIS yet.

When IRIS is available, the adapter:
  1. Runs the IrisApp binary with `-j 1 -v <fixture>` in a per-fixture
     temp dir (IrisApp writes to Results/<basename>/ relative to CWD,
     so a temp dir avoids collisions across parallel runs).
  2. Reads Results/<basename>/result.json for the OverallResult plus
     the per-dimension Results[] list.
  3. Maps each Results[] entry to our schema's failed_dimensions:
       "LuminanceFlashFailure"          -> ["luminance"]
       "LuminanceExtendedFlashFailure"  -> ["luminance", "extended"]
       "RedFlashFailure"                -> ["red"]
       "RedExtendedFlashFailure"        -> ["red", "extended"]
       "PatternFailure"                 -> ["pattern"]
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from harness.schema import NormalizedResult


TOOL = "iris"

REPO_ROOT = Path(__file__).resolve().parents[2]
# The example app is built as the `IrisApp` cmake target. The preset
# puts the binary at bin/build/<preset>/example/IrisApp inside the IRIS
# source tree.
IRIS_BINARY_CANDIDATES = [
    Path("/usr/local/bin/iris-example"),
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "bin" / "build" / "linux-release"   / "example" / "IrisApp",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "bin" / "build" / "linux-debug"     / "example" / "IrisApp",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "bin" / "build" / "macos-release"   / "example" / "IrisApp",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "bin" / "build" / "macos-debug"     / "example" / "IrisApp",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "bin" / "build" / "windows-release" / "example" / "IrisApp.exe",
]


def _binary_path() -> Path | None:
    for c in IRIS_BINARY_CANDIDATES:
        if c.exists() and c.is_file():
            return c
    return None


def _version_string(_binary: Path) -> str:
    # IRIS's example app doesn't expose a version flag (as of pinned commit).
    # Use the pinned upstream commit instead.
    return "d96978ac (pinned 2025-01-14)"


# Map IRIS's Results[] strings to our schema's failed_dimensions vocab.
# Multiple entries can decompose to multiple dims; we dedupe at the end.
_DIMENSION_FROM_RESULT = {
    "LuminanceFlashFailure":          ("luminance",),
    "LuminanceExtendedFlashFailure":  ("luminance", "extended"),
    "RedFlashFailure":                ("red",),
    "RedExtendedFlashFailure":        ("red", "extended"),
    "PatternFailure":                 ("pattern",),
}


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    """Invoke IRIS console example on the fixture; map result.json to schema."""
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="n/a",
            runtime_seconds=0.0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "IRIS pattern fixtures use a separate pattern-detection entry "
                "point; this adapter currently covers IRIS's video flash analysis."
            ),
        ).to_dict()

    binary = _binary_path()
    if binary is None:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="not-built",
            runtime_seconds=0.0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "IRIS example app binary not found. Build IRIS via cmake from "
                "corpus/sources/IRIS at the pinned commit; the binary should "
                "appear at bin/build/<preset>/example/IrisApp (or at "
                "/usr/local/bin/iris-example if installed system-wide)."
            ),
        ).to_dict()

    t0 = time.perf_counter()
    # IrisApp writes to Results/<basename>/ relative to its CWD and reads
    # appsettings.json from CWD too (crashing silently if absent -- the
    # binary even returns exit 0 on uncaught exception, so we MUST verify
    # result.json materialized below). Use a per-fixture temp dir so
    # parallel runs don't collide, and copy appsettings.json in.
    appsettings_src = binary.parent / "appsettings.json"
    with tempfile.TemporaryDirectory(prefix="iris-run-") as tmpdir:
        if appsettings_src.exists():
            shutil.copy(appsettings_src, Path(tmpdir) / "appsettings.json")
        try:
            cp = subprocess.run(
                [str(binary), "-j", "1", "-v", str(fixture_path)],
                cwd=tmpdir, capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            return NormalizedResult(
                fixture_id=fixture_path.name, verdict="ERROR",
                tool=TOOL, tool_version=_version_string(binary),
                runtime_seconds=time.perf_counter() - t0, raw_output_path="",
                standard_profile=profile,
                error_message="IRIS example app timed out (>300s)",
            ).to_dict()

        result_json = Path(tmpdir) / "Results" / fixture_path.name / "result.json"
        if cp.returncode != 0 or not result_json.exists():
            return NormalizedResult(
                fixture_id=fixture_path.name, verdict="ERROR",
                tool=TOOL, tool_version=_version_string(binary),
                runtime_seconds=time.perf_counter() - t0, raw_output_path="",
                standard_profile=profile,
                error_message=(
                    f"IRIS exit={cp.returncode}; result.json present="
                    f"{result_json.exists()}; stderr tail: {cp.stderr[-500:]}"
                ),
            ).to_dict()

        try:
            payload = json.loads(result_json.read_text())
        except Exception as e:
            return NormalizedResult(
                fixture_id=fixture_path.name, verdict="ERROR",
                tool=TOOL, tool_version=_version_string(binary),
                runtime_seconds=time.perf_counter() - t0, raw_output_path="",
                standard_profile=profile,
                error_message=f"could not parse IRIS result.json: {type(e).__name__}: {e}",
            ).to_dict()

    # IRIS uses three verdict levels: "Pass", "Fail", and "PassWithWarning"
    # (returned when the analysis detected events that crossed a warning
    # threshold but not a fail threshold -- e.g. PassWithWarningFrames > 0
    # but no FlashFailFrames). PassWithWarning is informational, not a
    # hazard verdict; we map it to PASS for scoring purposes.
    overall = (payload.get("OverallResult") or "").strip().replace(" ", "").upper()
    if overall in ("PASS", "PASSWITHWARNING"):
        verdict = "PASS"
    elif overall == "FAIL":
        verdict = "FAIL"
    else:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_version_string(binary),
            runtime_seconds=time.perf_counter() - t0, raw_output_path="",
            standard_profile=profile,
            error_message=f"unrecognised OverallResult: {overall!r}",
        ).to_dict()

    failed: set[str] = set()
    for r in payload.get("Results", []):
        for dim in _DIMENSION_FROM_RESULT.get(r, ()):
            failed.add(dim)

    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=verdict,
        failed_dimensions=sorted(failed),
        first_fail_timestamp=None,  # IRIS reports per-frame indices, not seconds; left None
        tool=TOOL, tool_version=_version_string(binary),
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=profile,
    ).to_dict()
