"""Adapter for ours_mlp -- the MLP-on-classical-features detector.

This is the "Option A" learned detector: a small MLP that consumes
summary statistics from the classical Q6 detector's per-frame trace and
outputs a fixture-level PASS/FAIL verdict. Trained on the 45 OURS-extended
fixtures (synthetic, ground-truth from generation params); evaluated on
TRACE here in the harness.

Sister adapter: ours (classical), flicker_filter (existing-art ML).
Future sister: ours_cnn (Option C).
"""

from __future__ import annotations

import time
from pathlib import Path

from detector.ml.infer import predict_mlp_verdict, CKPT_PATH
from harness.schema import NormalizedResult


TOOL = "ours_mlp"
# Model is profile-independent (features include profile-config-derived
# threshold-crossing counts, but the same model evaluates across all
# profiles for now). Emit one verdict reused across the profile group.
SUPPORTED_PROFILES = [
    "WCAG2.2-SC2.3.1",
    "Trace24",
    "ITU-R-BT.1702",
    "Ofcom-GN2-Annex1",
]
PROFILE_AFFECTS_BEHAVIOR = False


def _version() -> str:
    if CKPT_PATH.exists():
        return f"ours_mlp+ckpt-mtime-{int(CKPT_PATH.stat().st_mtime)}"
    return "ours_mlp+no-ckpt"


_VERSION = _version()


def run(fixture_path: Path, profile: str = "WCAG2.2-SC2.3.1") -> dict:
    if fixture_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return NormalizedResult(
            fixture_id=fixture_path.name,
            verdict="UNSUPPORTED",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=0.0,
            raw_output_path="",
            standard_profile=profile,
            unsupported_reason="ours_mlp operates on video only.",
        ).to_dict()

    t0 = time.perf_counter()
    try:
        verdict, _prob = predict_mlp_verdict(fixture_path, profile=profile)
    except FileNotFoundError as e:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message=str(e),
        ).to_dict()
    except Exception as e:
        return NormalizedResult(
            fixture_id=fixture_path.name, verdict="ERROR",
            tool=TOOL, tool_version=_VERSION,
            runtime_seconds=time.perf_counter() - t0,
            raw_output_path="",
            standard_profile=profile,
            error_message=f"ours_mlp inference failed: {e}",
        ).to_dict()

    return NormalizedResult(
        fixture_id=fixture_path.name,
        verdict=verdict,
        failed_dimensions=["luminance"] if verdict == "FAIL" else [],
        first_fail_timestamp=None,
        tool=TOOL, tool_version=_VERSION,
        runtime_seconds=time.perf_counter() - t0,
        raw_output_path="",
        standard_profile=profile,
    ).to_dict()
