"""Adapter for FFmpeg's `vf_photosensitivity` filter.

FFmpeg's photosensitivity filter is a *mitigation*, not a standards
detector. It reports per-frame "flash" lines on stderr when it
attenuates content. The filter has no PASS/FAIL concept and is not
written to any specific PSE standard.

We include it as a tool-under-test anyway — clearly labeled
"mitigation, non-conformant by design" — because it is informative to
show how a non-standards approach scores against a standards-grounded
corpus. We derive a proxy verdict: if the filter logged ANY flash event,
report FAIL; otherwise PASS. The proxy is the closest honest mapping
available; the report calls it a proxy and shows it separately from
standards-grounded tools.

References:
  * `ffmpeg -h filter=photosensitivity` (run at module import to capture
    the available options).
  * https://ffmpeg.org/ffmpeg-filters.html#photosensitivity
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from harness.schema import NormalizedResult


TOOL = "ffmpeg_photosensitivity"

# vf_photosensitivity has no standards-aware profile (it's a single
# heuristic mitigation filter). We declare profiles only so its single
# verdict gets cross-checked against each standard's labels.
SUPPORTED_PROFILES = [
    "WCAG2.2-SC2.3.1",
    "Trace24",
    "ITU-R-BT.1702",
    "Ofcom-GN2-Annex1",
]
PROFILE_AFFECTS_BEHAVIOR = False


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], check=True,
                             capture_output=True, text=True).stdout
        m = re.search(r"ffmpeg version (\S+)", out)
        return m.group(1) if m else "unknown"
    except Exception:
        return "unavailable"


_VERSION = _ffmpeg_version()


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    """Run vf_photosensitivity on the fixture, derive a proxy verdict.

    The filter writes "Detected at t=<time>" lines on stderr at default
    verbosity for every flash event. We count those lines.
    """
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=0.0,
            raw_output_path="",
            standard_profile=profile,
            unsupported_reason=(
                "vf_photosensitivity operates on video only; static image fixtures "
                "are out of its structural scope."
            ),
        ).to_dict()

    t0 = time.perf_counter()
    # bypass=1 means the filter still detects and logs but does not modify
    # output, which avoids generating a real output file we'd discard.
    # NB: ffmpeg's `photosensitivity` filter only emits its per-frame
    # detection diagnostics at `verbose` log level (not `info`). The
    # message format is `badness: <pre> -> <post> / <thresh> (<pct>% -
    # OK|EXCEEDED)` per frame; we count EXCEEDED frames.
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "verbose",
           "-i", str(fixture_path),
           "-vf", "photosensitivity=bypass=1",
           "-f", "null", "-"]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as e:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message=f"ffmpeg not found: {e}",
        ).to_dict()
    except subprocess.TimeoutExpired:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message="ffmpeg vf_photosensitivity timed out (>120s)",
        ).to_dict()

    stderr = cp.stderr or ""
    # ffmpeg's vf_photosensitivity emits one line per frame at verbose
    # log level, e.g.
    #   "[Parsed_photosensitivity_0 @ 0x...] badness: <pre> -> <post> /
    #     <thresh> (<pct>% - EXCEEDED|OK)"
    # CAVEAT: the filter measures generic temporal pixel variation, NOT
    # any PSE-specific property. On TRACE fixture pairs that differ
    # ONLY in area (e.g., f002 vs a002 -- same temporal pattern, just
    # different flash-region sizes), ffmpeg emits IDENTICAL EXCEEDED
    # counts. So this tool cannot discriminate area-axis hazards by
    # construction; it can only flag temporal variation density.
    # We use "fraction of frames EXCEEDED > 0.5" as a proxy verdict and
    # document the limitation in the report.
    n_ok = n_exceeded = 0
    first_exceeded_line_no: int | None = None
    for ln in stderr.splitlines():
        if "Parsed_photosensitivity" in ln and "badness:" in ln:
            if "EXCEEDED" in ln:
                if first_exceeded_line_no is None:
                    first_exceeded_line_no = n_ok + n_exceeded
                n_exceeded += 1
            elif "OK" in ln:
                n_ok += 1
    total = n_ok + n_exceeded
    exceeded_frac = (n_exceeded / total) if total > 0 else 0.0
    src_fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    src_fps = float(src_fps_m.group(1)) if src_fps_m else 30.0
    first_ts = (first_exceeded_line_no / src_fps
                if first_exceeded_line_no is not None and src_fps > 0
                else None)
    proxy_verdict = "FAIL" if exceeded_frac > 0.5 else "PASS"

    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=proxy_verdict,
        failed_dimensions=["luminance"] if proxy_verdict == "FAIL" else [],
        first_fail_timestamp=first_ts,
        tool=TOOL, tool_version=_VERSION,
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",  # we discard the null output
        standard_profile=profile,
        score=float(exceeded_frac),
    ).to_dict()
