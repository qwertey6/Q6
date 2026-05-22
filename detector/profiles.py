"""detector/profiles.py -- standards profiles.

A ``Profile`` is a frozen dataclass bundling the numeric thresholds for
one PSE standard's reading. Each numeric constant is justified in
``detector/THRESHOLDS.md`` by a clause citation; no value is tuned
against benchmark labels.

The ``PROFILES`` dict maps profile names to ``Profile`` instances. The
detector and harness adapters reference profiles by these string names;
the runner sets ``standard_profile`` in each normalized result so
scoring can route to the correct per-standard label column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Profile:
    name: str
    general_flash_max_per_second: int = 3
    general_flash_luminance_delta: float = 0.1
    general_flash_darker_bound: float = 0.8
    area_fraction_limit: float = 0.25
    area_pixels_limit: int = 10_000_000   # effectively disabled by default
    red_flash_max_per_second: int = 3
    red_sat_delta: int = 20
    red_sat_min: int = 80
    sliding_window_seconds: float = 1.0
    absolute_flashes_per_second_cap: Optional[int] = None
    pattern_hazard_enabled: bool = False
    pattern_min_bars: int = 5
    pattern_min_area_fraction: float = 0.40


# Reference visual-field rectangle (Harding / CRS FCS Implementation
# Guide convention: 341×256 px = 10° of central vision; 25% of that =
# 21,824 px is the WCAG-strict hazard threshold).
REF_RECT_W      = 341
REF_RECT_H      = 256
REF_RECT_AREA   = REF_RECT_W * REF_RECT_H
WCAG_AREA_LIMIT = int(round(0.25 * REF_RECT_AREA))


PROFILES: dict[str, Profile] = {
    "WCAG2.2-SC2.3.1": Profile(
        name="WCAG2.2-SC2.3.1",
        area_fraction_limit=0.25,
        area_pixels_limit=WCAG_AREA_LIMIT,
        pattern_hazard_enabled=True,
    ),
    "WCAG2.2-classic": Profile(
        name="WCAG2.2-classic",
        area_fraction_limit=1.0,
        area_pixels_limit=REF_RECT_AREA,
        pattern_hazard_enabled=True,
    ),
    "ITU-R-BT.1702":    Profile(name="ITU-R-BT.1702",
                                  area_pixels_limit=REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Ofcom-GN2-Annex1": Profile(name="Ofcom-GN2-Annex1",
                                  area_pixels_limit=REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "Trace24":          Profile(name="Trace24",
                                  area_pixels_limit=REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
    "NAB-J":            Profile(name="NAB-J",
                                  area_pixels_limit=REF_RECT_AREA,
                                  absolute_flashes_per_second_cap=5,
                                  pattern_hazard_enabled=True),
    # ISO 9241-391: "Ergonomics of human-system interaction -- Part 391:
    # Requirements, analysis and compliance test methods for the
    # reduction of photosensitive seizures." Harding-classic-style area
    # thresholds + WCAG-shared 0.10 ΔL intensity + 3-flash/sec count.
    "ISO9241-391":      Profile(name="ISO9241-391",
                                  area_pixels_limit=REF_RECT_AREA,
                                  pattern_hazard_enabled=True),
}
