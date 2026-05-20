"""Adapter for EA IRIS (C++).

IRIS is built from `corpus/sources/IRIS` at the pinned commit. The build
requires cmake + ninja + vcpkg and is the responsibility of the
Dockerfile (and of an optional manual build outside Docker). If the
expected example-app binary is not present at adapter load time, every
fixture invocation reports UNSUPPORTED with a documented reason — this
is the *honest* result for an environment that hasn't built IRIS yet.

When IRIS is available, the adapter:
  1. Runs `iris_example -j 1 -v <fixture>` to get JSON output.
  2. Parses the JSON for flash/red/pattern verdicts.
  3. Maps to the normalized schema.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from harness.schema import NormalizedResult


TOOL = "iris"

REPO_ROOT = Path(__file__).resolve().parents[2]
# Conventional build output location (matches Dockerfile build step).
IRIS_BINARY_CANDIDATES = [
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "build" / "linux-release"  / "example" / "Iris.Example",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "build" / "release"        / "example" / "Iris.Example",
    REPO_ROOT / "corpus" / "sources" / "IRIS" / "out"   / "build" / "linux-release" / "example" / "Iris.Example",
]


def _binary_path() -> Path | None:
    for c in IRIS_BINARY_CANDIDATES:
        if c.exists() and c.is_file():
            return c
    return None


def _version_string(binary: Path) -> str:
    # IRIS's example app doesn't expose a version flag (as of pinned commit).
    # Use the pinned upstream commit instead.
    return "d96978ac (pinned)"


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    """Invoke IRIS console example on the fixture; map JSON output to schema."""
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version="n/a",
            runtime_seconds=0.0, raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "IRIS pattern fixtures use a separate pattern-detection entry "
                "point; this adapter currently covers IRIS's video flash analysis "
                "(matches the brief's primary scope)."
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
                "IRIS example app binary not found; IRIS must be built via cmake "
                "from corpus/sources/IRIS at the pinned commit (see Dockerfile). "
                "Local environments without the Docker build will see this "
                "tool as UNSUPPORTED."
            ),
        ).to_dict()

    t0 = time.perf_counter()
    cmd = [str(binary), "-j", "1", "-v", str(fixture_path)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_version_string(binary),
            runtime_seconds=time.perf_counter() - t0, raw_output_path="",
            standard_profile=profile,
            error_message="IRIS example app timed out (>300s)",
        ).to_dict()

    # IRIS writes results in a Results/ directory next to the binary; in
    # JSON mode the example app emits to stdout and/or to a sibling .json.
    raw = cp.stdout + "\n" + cp.stderr
    failed: list[str] = []
    first_fail: float | None = None
    try:
        # Try JSON-on-stdout path first; fall back to scanning Results dir.
        payload = json.loads(cp.stdout)
        if payload.get("luminance_failed"): failed.append("luminance")
        if payload.get("red_failed"):       failed.append("red")
        if payload.get("pattern_failed"):   failed.append("pattern")
        first_fail = payload.get("first_failure_timestamp")
    except Exception:
        # Heuristic fall-back parser on the example app's human-readable lines.
        if re.search(r"LuminanceFlashFail", raw):  failed.append("luminance")
        if re.search(r"RedFlashFail", raw):        failed.append("red")
        if re.search(r"PatternFail", raw):         failed.append("pattern")

    verdict = "FAIL" if failed else "PASS"
    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=verdict,
        failed_dimensions=failed,
        first_fail_timestamp=first_fail,
        tool=TOOL, tool_version=_version_string(binary),
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=profile,
    ).to_dict()
