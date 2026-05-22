"""Q6's PSE detector.

Implemented from the *text* of the standards, not from benchmark labels.
Every numeric constant is justified in detector/THRESHOLDS.md by a clause
citation. The held-out upstream-labeled corpus is an acceptance check we
run at milestone end — not a tuning loop.

Public entry point: ``detector.analyze(video_path, profile) -> Result``.

Layout:
  * ``detector.core`` — orchestration: analyze(), per-frame loop, AxisState,
    pixel kernels, axis-step functions, static-pattern detection, CLI.
  * ``detector.profiles`` — Profile dataclass + PROFILES dict.
  * ``detector.regions`` — HazardRegion + mitigation/counterfactual builders.
  * ``detector.cc_backends`` — CCBackend DI + cv2/tensor implementations.
"""

from .core import (  # noqa: F401
    analyze, Result, PerAxisResult, PerFrame,
    detect_static_pattern_hazard,
)
from .profiles import Profile, PROFILES  # noqa: F401
from .regions import HazardRegion  # noqa: F401
from .cc_backends import CCBackend, CV2_CC_BACKEND, TENSOR_CC_BACKEND  # noqa: F401
