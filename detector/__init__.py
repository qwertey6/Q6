"""Our PSE detector.

Implemented from the *text* of the standards, not from benchmark labels.
Every numeric constant is justified in detector/THRESHOLDS.md by a clause
citation. The held-out upstream-labeled corpus is an acceptance check we
run at milestone end — not a tuning loop.

Public entry point: ``detector.analyze(video_path, profile) -> Result``.
"""

from .core import (  # noqa: F401
    analyze, Result, Profile, PROFILES,
    detect_static_pattern_hazard,
    CV2_CC_BACKEND, TENSOR_CC_BACKEND, CCBackend,
)
