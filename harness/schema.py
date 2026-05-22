"""Normalized result schema for every PSE-detector adapter.

This module is the cross-cutting contract for the whole benchmark. Every
adapter — including the adapter for our own detector — must produce results
that validate against ``NORMALIZED_RESULT_SCHEMA``. Scoring (`scoring.py`)
joins these results to the corpus manifest **after** all adapters have
finished; adapters never see ground-truth labels.

The four-verdict vocabulary is deliberate:

* ``PASS``        — the tool judged the fixture harmless under the active standard profile.
* ``FAIL``        — the tool judged the fixture hazardous under the active standard profile.
* ``ERROR``       — the tool crashed, returned malformed output, or timed out. Counts against the tool.
* ``UNSUPPORTED`` — the tool structurally cannot judge this fixture (e.g. FFmpeg
                    vf_photosensitivity has no pass/fail concept; an image-only
                    detector handed a video). Excluded from the metric and reported
                    per-tool with a reason. Distinct from ERROR.

The ``failed_dimensions`` list answers *why* a FAIL was returned: useful for
per-dimension slicing in scoring. Empty for PASS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


# --- Vocabulary -------------------------------------------------------------

VERDICTS = ("PASS", "FAIL", "ERROR", "UNSUPPORTED")

# The dimension axes scoring slices over. Keep stable; the report depends on them.
DIMENSIONS = (
    "luminance",   # WCAG general flash threshold
    "red",         # WCAG saturated-red transition threshold
    "area",        # screen-fraction / reference-rectangle area threshold
    "count",       # number of flashes per time window
    "pattern",     # bold static spatial pattern (separate hazard class)
    "extended",    # Q6-extended axes (high fps, wide gamut, etc.)
)


# --- JSON Schema ------------------------------------------------------------
# A jsonschema-compatible Draft 2020-12 schema. Adapters serialize their result
# as JSON and the runner validates it before writing to disk. Schema violations
# are recorded as ERROR for that fixture-tool pair.

NORMALIZED_RESULT_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://internal/pse-bench/normalized-result.schema.json",
    "title": "Normalized PSE detector result",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "fixture_id",
        "verdict",
        "failed_dimensions",
        "first_fail_timestamp",
        "tool",
        "tool_version",
        "runtime_seconds",
        "raw_output_path",
    ],
    "properties": {
        "fixture_id":           {"type": "string", "minLength": 1},
        "verdict":              {"type": "string", "enum": list(VERDICTS)},
        "failed_dimensions": {
            "type": "array",
            "items": {"type": "string", "enum": list(DIMENSIONS)},
            "uniqueItems": True,
        },
        "first_fail_timestamp": {"type": ["number", "null"], "minimum": 0},
        "per_frame_csv":        {"type": ["string", "null"]},
        "tool":                 {"type": "string", "minLength": 1},
        "tool_version":         {"type": "string", "minLength": 1},
        "runtime_seconds":      {"type": "number", "minimum": 0},
        "raw_output_path":      {"type": "string"},
        "standard_profile":     {"type": "string"},
        "score":                {"type": ["number", "null"]},
        "per_axis_scores":      {"type": "object",
                                   "additionalProperties": {"type": "number"}},
        "unsupported_reason":   {"type": ["string", "null"]},
        "error_message":        {"type": ["string", "null"]},
    },
    "allOf": [
        {
            "if":   {"properties": {"verdict": {"const": "UNSUPPORTED"}}},
            "then": {"required": ["unsupported_reason"],
                     "properties": {"unsupported_reason": {"type": "string", "minLength": 1}}},
        },
        {
            "if":   {"properties": {"verdict": {"const": "ERROR"}}},
            "then": {"required": ["error_message"],
                     "properties": {"error_message": {"type": "string", "minLength": 1}}},
        },
        {
            "if":   {"properties": {"verdict": {"const": "PASS"}}},
            "then": {"properties": {"failed_dimensions": {"maxItems": 0}}},
        },
    ],
}


# --- Per-frame CSV contract -------------------------------------------------
# When an adapter can produce per-frame data, it writes a CSV with exactly this
# header. Per-frame agreement is one of our independent corroboration signals
# (especially against IRIS's shipped *_RELATIVE.csv expected logs).

PER_FRAME_CSV_HEADER = (
    "frame",
    "lum_transitions",   # number of luminance flash transitions completed by this frame
    "red_transitions",   # number of saturated-red transitions completed by this frame
    "flash_area",        # area (in screen-fraction units, 0..1) of the active flash region
    "pattern_risk",      # bold-static-pattern risk score, 0..1
)


# --- Convenience builder ----------------------------------------------------

@dataclass
class NormalizedResult:
    """Adapter-friendly dataclass that serializes to the canonical JSON shape.

    The ``score`` field is the adapter's continuous detection score for
    this fixture. Adapters that produce one (e.g. our detector's
    severity-ratio, q6_mlp's predicted probability, flicker_filter's
    ElasticNet output) should emit it; adapters whose underlying tool
    only emits PASS/FAIL leave it as None. Scoring uses it for AUROC
    and PR-AUC computation where available.
    """

    fixture_id: str
    verdict: str                        # one of VERDICTS
    tool: str
    tool_version: str
    runtime_seconds: float
    raw_output_path: str
    failed_dimensions: list[str] = field(default_factory=list)
    first_fail_timestamp: Optional[float] = None
    per_frame_csv: Optional[str] = None
    standard_profile: str = "WCAG2.2-SC2.3.1"
    unsupported_reason: Optional[str] = None
    error_message: Optional[str] = None
    score: Optional[float] = None       # continuous detection score
    per_axis_scores: dict = field(default_factory=dict)  # per-hazard-class scores

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop keys that are None unless they are required-for-this-verdict.
        if self.verdict != "UNSUPPORTED":
            d.pop("unsupported_reason", None)
        if self.verdict != "ERROR":
            d.pop("error_message", None)
        if self.score is None:
            d.pop("score", None)
        if not self.per_axis_scores:
            d.pop("per_axis_scores", None)
        return d


def validate(payload: dict) -> None:
    """Validate ``payload`` against NORMALIZED_RESULT_SCHEMA. Raises on failure.

    Imported lazily so this module can be inspected without the jsonschema
    dependency installed (useful in detector unit tests).
    """
    import jsonschema  # type: ignore

    jsonschema.validate(instance=payload, schema=NORMALIZED_RESULT_SCHEMA)
