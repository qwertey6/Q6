"""detector/regions.py -- hazard region modeling.

A ``HazardRegion`` is a spatially-coherent area within a single frame
that triggers one or more hazard classes (``luminance``, ``red``,
``pattern``, ``count``). A frame can have multiple regions; a single
region can trip multiple classes.

This module defines the data shape and the builders that wrap a raw
(bbox, area, per-class peak) tuple into a fully populated region with
severity, mitigation hints, standards clauses, and counterfactual.

Builders intentionally live here -- not inside the detector loop --
so the per-frame loop stays focused on numerics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .profiles import Profile


# Severity bands derived from the score (peak / threshold ratio).
# score < 1 means the axis isn't in hazard territory yet.
SEVERITY_BAND_MARGINAL = "marginal"   # 1.00 <= score < 1.10
SEVERITY_BAND_CLEAR    = "clear"      # 1.10 <= score < 1.50
SEVERITY_BAND_SEVERE   = "severe"     # score >= 1.50


# Standards clause references emitted on every HazardRegion. Used by the
# HTML report to link readers to the normative text.
STANDARDS_CLAUSE_REFS: dict[str, dict] = {
    "WCAG2.2-SC2.3.1": {
        "standard": "WCAG 2.2",
        "clause": "Success Criterion 2.3.1 - Three Flashes or Below Threshold",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold",
        "interpretation_note_id": "OQ-4",   # link to detector/THRESHOLDS.md
    },
    "WCAG2.2-classic": {
        "standard": "WCAG 2.2 (Harding-classic area reading)",
        "clause": "Success Criterion 2.3.1 - Three Flashes or Below Threshold",
        "url": "https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold",
        "interpretation_note_id": "OQ-4",
    },
    "Trace24": {
        "standard": "Trace24",
        "clause": "Photosensitive content evaluation guidance",
        "url": "https://trace.umd.edu/peat/",
    },
    "ITU-R-BT.1702": {
        "standard": "ITU-R BT.1702",
        "clause": "Guidance for the reduction of photosensitive epileptic seizures caused by television",
        "url": "https://www.itu.int/rec/R-REC-BT.1702/en",
    },
    "Ofcom-GN2-Annex1": {
        "standard": "Ofcom Guidance Note 2 Annex 1",
        "clause": "Flashing images and regular patterns",
        "url": "https://www.ofcom.org.uk/__data/assets/pdf_file/0024/24296/section2.pdf",
    },
    "NAB-J": {
        "standard": "NAB J 'BJP' (Japan broadcasters)",
        "clause": "Photosensitivity guidelines",
        "url": "https://j-ba.or.jp/",
    },
    "ISO9241-391": {
        "standard": "ISO 9241-391",
        "clause": "Ergonomics: requirements for the reduction of photosensitive seizures",
        "url": "https://www.iso.org/standard/76376.html",
    },
}


def severity_band(score: float) -> str:
    if score < 1.10: return SEVERITY_BAND_MARGINAL
    if score < 1.50: return SEVERITY_BAND_CLEAR
    return SEVERITY_BAND_SEVERE


@dataclass(frozen=True)
class HazardRegion:
    """A spatially-coherent hazardous region within a single frame.

    A frame can contain multiple regions; a region can trigger multiple
    hazard classes simultaneously (e.g. a flashing saturated-red square
    trips both ``luminance`` and ``red``).
    """
    bbox: tuple[int, int, int, int]                  # (x0, y0, x1, y1) inclusive
    area_px: int
    centroid: tuple[float, float]                    # (cx, cy) in pixels
    classes: frozenset[str]                           # which hazard axes triggered
    severity: dict[str, float] = field(default_factory=dict)
    confidence_band: str = SEVERITY_BAND_MARGINAL
    mitigation: list[dict] = field(default_factory=list)
    standards_clauses: list[dict] = field(default_factory=list)
    counterfactual: dict = field(default_factory=dict)
    track_id: Optional[int] = None                    # filled by later track step


# --- Builders --------------------------------------------------------------

def _build_mitigation(cls: str, peak: int, limit: int, region_area_px: int,
                       profile: Profile) -> dict:
    """Per-class actionable mitigation hint."""
    if cls == "luminance":
        return {
            "axis": "luminance",
            "current": int(peak),
            "limit": int(limit),
            "unit": "windowed transitions (per 1-sec window)",
            "suggestion": (
                f"Reduce flash rate so the per-pixel windowed transition "
                f"count is at most {limit} (currently {int(peak)})."
            ),
            "alternatives": [
                f"Shrink the hazardous region from {int(region_area_px)} px to "
                f"< {int(profile.area_pixels_limit)} px (so the area axis no "
                f"longer triggers).",
                "Lower the inter-state ΔL below the intensity threshold "
                f"({profile.general_flash_luminance_delta:.2f}).",
            ],
        }
    if cls == "red":
        return {
            "axis": "red",
            "current": int(peak),
            "limit": int(limit),
            "unit": "windowed saturated-red transitions (per 1-sec window)",
            "suggestion": (
                f"Reduce red oscillation frequency so the per-pixel windowed "
                f"transition count is at most {limit} (currently {int(peak)})."
            ),
            "alternatives": [
                "Desaturate the red component to bring the larger of the two "
                f"endpoints below the Harding minimum ({profile.red_sat_min}).",
                f"Shrink the hazardous region from {int(region_area_px)} px to "
                f"< {int(profile.area_pixels_limit)} px.",
            ],
        }
    if cls == "count":
        return {
            "axis": "count",
            "current": int(peak),
            "limit": int(limit),
            "unit": "absolute transitions (per 1-sec window)",
            "suggestion": (
                "Reduce the absolute count of opposing transitions per "
                "second below the standard's hard cap."
            ),
            "alternatives": [],
        }
    return {"axis": cls, "current": int(peak), "limit": int(limit),
            "suggestion": "Reduce the offending signal below threshold."}


def _build_counterfactual(severity: dict, region_area_px: int,
                           profile: Profile) -> dict:
    """What would change for this region to PASS?"""
    out_flags: dict[str, bool] = {}
    out_edits: list[str] = []
    for cls, score in severity.items():
        if score < 1.0:
            out_flags[f"{cls}_under_threshold"] = True
            continue
        out_flags[f"{cls}_under_threshold"] = False
        if cls == "luminance":
            limit = 2 * profile.general_flash_max_per_second
            out_edits.append(
                f"Reduce {cls} windowed-transition count to <= {limit} "
                f"(currently ~{int(score * limit)})."
            )
        elif cls == "red":
            limit = 2 * profile.red_flash_max_per_second
            out_edits.append(
                f"Reduce {cls} windowed-transition count to <= {limit} "
                f"(currently ~{int(score * limit)})."
            )
    if region_area_px >= profile.area_pixels_limit:
        out_edits.append(
            f"OR shrink hazardous region area to < {profile.area_pixels_limit} "
            f"px (currently {int(region_area_px)} px)."
        )
    out_flags["area_under_threshold"] = region_area_px < profile.area_pixels_limit
    return {"flip_flags": out_flags, "minimal_edits": out_edits}


def make_hazard_region(bbox: tuple[int, int, int, int],
                         area_px: int,
                         centroid: tuple[float, float],
                         per_class_peak: dict[str, int],
                         profile: Profile) -> HazardRegion:
    """Assemble a fully-populated HazardRegion. ``per_class_peak`` maps
    a hazard class (e.g. ``"luminance"``) to the peak windowed-transition
    count observed within the region's pixels for that axis."""
    classes: set[str] = set()
    severity: dict[str, float] = {}
    mitigation_list: list[dict] = []
    for cls, peak in per_class_peak.items():
        if cls == "luminance":
            limit = 2 * profile.general_flash_max_per_second
        elif cls == "red":
            limit = 2 * profile.red_flash_max_per_second
        elif cls == "count" and profile.absolute_flashes_per_second_cap is not None:
            limit = 2 * profile.absolute_flashes_per_second_cap
        else:
            limit = 0
        if limit <= 0:
            continue
        score = float(peak) / float(limit)
        severity[cls] = score
        if score >= 1.0:
            classes.add(cls)
            mitigation_list.append(_build_mitigation(cls, peak, limit,
                                                      area_px, profile))
    overall_score = max(severity.values()) if severity else 0.0
    band = severity_band(overall_score)
    clauses_dict = STANDARDS_CLAUSE_REFS.get(profile.name)
    clauses = [clauses_dict] if clauses_dict is not None else []
    counterfactual = _build_counterfactual(severity, area_px, profile)
    return HazardRegion(
        bbox=bbox, area_px=int(area_px), centroid=centroid,
        classes=frozenset(classes), severity=severity,
        confidence_band=band, mitigation=mitigation_list,
        standards_clauses=clauses, counterfactual=counterfactual,
    )
