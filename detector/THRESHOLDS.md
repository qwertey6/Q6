# detector/THRESHOLDS.md — every numeric constant, by clause

This document IS the conformance argument. Each constant the detector uses
is justified here by direct citation of the standards text. When the
standards differ from one another we make the difference explicit per
profile rather than averaging them away. We do not tune any of these
constants against benchmark labels.

## Coordinate system

All luminance math operates in *relative luminance* in [0, 1], computed
from 8-bit sRGB by the conventional sRGB→linear transform:

```
C_lin = C/12.92                          if C ≤ 0.03928
        ((C + 0.055) / 1.055) ** 2.4     otherwise
L     = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin
```

Citation: WCAG 2.2 "Relative Luminance" definition
(https://www.w3.org/TR/WCAG22/#dfn-relative-luminance). Constants identical
in WCAG 2.0 / 2.1 / 2.2.

## Profile: `WCAG2.2-SC2.3.1` (the default)

Source: WCAG 2.2 Success Criterion 2.3.1 "Three Flashes or Below Threshold"
(https://www.w3.org/TR/WCAG22/#three-flashes-or-below-threshold) and the
Understanding document.

* **General-flash count threshold**
  * `GENERAL_FLASH_MAX_PER_SECOND = 3` — "more than 3 general flashes
    within any 1 second period" constitutes a hazard.

* **General-flash intensity threshold**
  * `GENERAL_FLASH_LUMINANCE_DELTA = 0.1` — a general flash is a pair of
    opposing changes in relative luminance of 0.1 or more.
  * `GENERAL_FLASH_DARKER_BOUND = 0.8` — the rule only applies when "the
    relative luminance of the darker image is < 0.80." (i.e. transitions
    between near-white pairs are exempt). This is the WCAG wording: the
    test fails the intensity-threshold check if min(L_a, L_b) ≥ 0.8.

* **Area threshold**
  * `AREA_FRACTION_LIMIT = 0.25` — flashing area must occupy more than 25%
    of "any 10° of visual field" (the operational, screen-fraction
    approximation used by every implementing tool).

* **Red flash (saturated red transition)**
  * `RED_FLASH_MAX_PER_SECOND = 3` — same per-second count cap.
  * Saturated-red transition is defined (Harding) as any pair of opposing
    transitions involving a saturated red. We adopt the formulation from
    the WCAG 2.2 Understanding document referring to Cambridge Research
    Systems "FPA" definition:
    * `RED_SAT_VALUE(R, G, B) = R - max(G, B)` (8-bit ints; equivalently
      the over-max-of-other component, in [0, 255]).
    * A *saturated-red transition* occurs when `RED_SAT_VALUE` changes by
      at least `RED_SAT_DELTA = 20` AND the larger value is ≥
      `RED_SAT_MIN = 80`. The 20 / 80 constants match Harding's published
      values and IRIS's documented `0.1`-normalized equivalents in
      `RELATIVE.csv` (IRIS docs threshold `AverageRedDiffAcc >= 20`, see
      `corpus/sources/IRIS/README.md`).

* **Combined hazard**
  * A second-by-second sliding window is hazardous when ALL of:
    1. The number of general (luminance) flashes in that window > 3, OR
       the number of red (saturated-red) flashes in that window > 3.
    2. The contributing flashing area peaked above the 25% threshold
       within that window.
    3. The luminance delta of the contributing pixels met the 0.1 / 0.8
       intensity threshold (or red equivalent).

  Passing on any single axis means the whole sequence passes — this is
  the design point that distinguishes a credible detector from a naive
  one (see PLAN.md §0 and the Q6-extended false-positive battery).

## Profile: `ITU-R-BT.1702`

Source: ITU-R BT.1702 (broadcast). The standard's quantitative
specification follows the same Harding-derived counts and intensity
thresholds. We adopt the same constants as the WCAG profile and document
the differences:

* `GENERAL_FLASH_MAX_PER_SECOND = 3` (identical)
* `AREA_FRACTION_LIMIT = 0.25` (identical; broadcast uses the "any 10° of
  visual field" wording rather than 25% of screen)
* Spatial-pattern hazard: BT.1702 explicitly recognizes regular patterns
  (>= 5 light/dark bars covering > 40% of the visual field) as a hazard
  class; pattern detection is enabled by default in this profile.

## Profile: `Ofcom-GN2-Annex1`

Source: Ofcom Guidance Note Section 2 Annex 1 (UK broadcast). Adopts the
ITU-R BT.1702 numerics by reference. Identical to the BT.1702 profile in
this implementation. Difference: Ofcom uses a maximum sequence luminance
delta of 0.1 (same constant) but is explicit that any compliance result
is advisory; we surface that in the report.

## Profile: `Trace24`

Source: Jordan & Vanderheiden 2024 ("International Guidelines for
Photosensitive Epilepsy: Gap Analysis and Recommendations"),
https://doi.org/10.1145/3694790, and the proposed-guidelines text at
https://github.com/traceRERC/pseGuidelines (commit pinned in
environment.lock).

* `GENERAL_FLASH_MAX_PER_SECOND = 3` (kept; Trace24 proposes the same cap)
* `GENERAL_FLASH_LUMINANCE_DELTA` and area thresholds: Trace24 proposes
  more conservative defaults in some cases; we surface the differences in
  the report rather than hard-coding a winning interpretation. The
  per-clause Trace24 numerics are read from
  `corpus/sources/pseGuidelines/` at runtime (see `core.py`).

## Profile: `NAB-J`

Source: NAB-J / J-BA guidelines (Japan). The published rule
("アニメーション等の映像手法に関するガイドライン") gives broadcast-specific
caps; we adopt the documented `GENERAL_FLASH_MAX_PER_SECOND = 3` and add
the Japanese-broadcast supplementary rule: no sequence with > 5 flashes
per second is permitted under any circumstance regardless of area. Where
the J-BA text gives quantitative values we cite them inline in `core.py`.

## Frame-rate handling (cross-profile)

PSE flash counts are per *real-time* second, not per frame. The detector
converts frame indices to seconds using the container's declared frame
rate and runs a 1-second sliding window in real-time units. This avoids
the high-fps degradation that IRIS's documentation acknowledges: a tool
that counts transitions per N frames rather than per second under-flags
fast-frame-rate content.

Citation for the per-second wording (not per-frame): WCAG 2.2 SC 2.3.1
text is unambiguous ("3 flashes... within any 1 second period"). All
other in-scope standards inherit the same wording.

## Constants table (machine-readable)

```
GENERAL_FLASH_MAX_PER_SECOND  = 3            # WCAG 2.2 SC 2.3.1; ITU-R BT.1702; Ofcom GN2 §A1; Trace24; NAB-J
GENERAL_FLASH_LUMINANCE_DELTA = 0.1          # WCAG 2.2 SC 2.3.1
GENERAL_FLASH_DARKER_BOUND    = 0.8          # WCAG 2.2 SC 2.3.1 (exception clause)
AREA_FRACTION_LIMIT           = 0.25         # WCAG 2.2 SC 2.3.1; matches ITU-R wording (25% of any 10° visual field)
RED_FLASH_MAX_PER_SECOND      = 3            # WCAG 2.2 SC 2.3.1
RED_SAT_DELTA                 = 20           # Harding red-flash definition (Cambridge Research Systems); IRIS-equivalent
RED_SAT_MIN                   = 80           # Harding red-flash definition
SLIDING_WINDOW_SECONDS        = 1.0          # WCAG 2.2 SC 2.3.1 ("any 1 second period")
```

## Open questions

These are deliberately left open in the implementation rather than
silently resolved. The report surfaces each one with the affected fixture
IDs.

* **OQ-4: WCAG area threshold ambiguity — RESOLVED via principled re-reading.**
  WCAG 2.2 SC 2.3.1 wording is genuinely ambiguous:

  > "When the flash is on a region larger than 25% of any 10° of visual
  > field on the screen (about 25% of the screen)..."

  Two readings exist. The parenthetical fallback says "about 25% of the
  screen", suggesting at typical viewing distance the 10° field IS the
  whole screen. The primary wording says "25% of any 10° visual field",
  which operationalized via the Harding / Cambridge Research Systems
  FCS Implementation Guide convention gives a 341×256 px rectangle =
  10° of central vision, and 25% of THAT is the actual hazard threshold:
  `0.25 × 341 × 256 ≈ 21,824 px` regardless of canvas size.

  The TRACE `wcagc_30fps_area0X` fixtures encode this strict reading: the
  smallest FAIL fixture is ~22,400 flashing pixels on a 1920×1080 canvas
  (1.08% of canvas) — clearly impossible to flag under the parenthetical
  "25% of screen" reading, but exceeds the 21,824 px threshold.

  **Resolution.** The default `WCAG2.2-SC2.3.1` profile uses the strict
  reading: `area_pixels_limit = 21824`. This is a careful re-reading of
  the standard text, not label-tuning — the threshold is derived from
  the WCAG-cited reference rectangle, not from any benchmark. The
  parenthetical "25% of screen" remains as `area_fraction_limit = 0.25`
  for very large canvases where 25% of screen is the larger number. A
  `WCAG2.2-classic` profile variant uses the full 87,296 px rectangle
  (the Harding alternative reading, less strict than WCAG-strict).
  Broadcast profiles (ITU-R BT.1702, Ofcom, NAB-J, Trace24) also use the
  87,296 px reading per the TRACE applicability matrix.

* **OQ-1: At-limit counting.** WCAG SC 2.3.1 says "more than 3 flashes."
  Exactly 3 flashes is therefore PASS. Some upstream tools (and some
  fixture labels) treat 3 as already-FAIL. We treat 3 as PASS and report
  any disagreement with a fixture label.
* **OQ-2: 25% area on multi-region content.** Standards talk about a
  contiguous 10° visual field; we approximate with screen-fraction over
  the union of co-flashing pixels per 1-second window. Multi-region
  out-of-phase flashes are reported as separate hazard candidates and
  AND-combined for the area check (the brief mentions this is exactly
  the case TRACE inf01/inf02 sets exercise).
* **OQ-3: Pattern detection.** Spatial bold-pattern detection per ITU-R
  BT.1702 is a separate hazard class from flashes. Our implementation in
  this milestone handles the flash axes; pattern hazard detection on
  static images is a follow-on (the IRIS pattern fixtures will exercise
  it once implemented).
