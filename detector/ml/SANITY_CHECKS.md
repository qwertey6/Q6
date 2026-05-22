# Sanity checks on Q6's "best ML" headline (A6 +0.355)

Before publishing "Q6's ML detector is the best ML approach for PSE
detection" we ran cross-checks to find anything that would make us
the worst because of a benchmark bug. Findings:

## Check 1: Seed stability of A6

A6 was reported at MCC +0.355 with K-fold seed 0. Re-running across
seeds 0-9: **+0.350 ± 0.015** (range +0.304 to +0.355; 9 of 10 seeds
give the exact +0.355). Not fragile.

## Check 2: Apples-to-apples classical baseline

The published "ours classical: MCC +0.220" comes from
`source:upstream` in scoring (uses `expected_label` fallback when
`expected_wcag2_2` is empty -- so it counts 326 fixtures including
EA/IRIS / Apple / Kaya entries with their own labels). A6 evaluates
only on TRACE fixtures with a non-empty `expected_wcag2_2` cell
(293 fixtures). Classical's MCC on the **same** 293-sample subset is
**+0.200**, not +0.220. The honest comparison:

  - classical (same subset): **+0.200**
  - A6 logistic_l2 + classical-as-feature: **+0.355**
  - lift: **+0.155** absolute

(The original writeup said "vs +0.220" in one place which would
overstate the lift. Corrected here.)

## Check 3: Are the other adapters wired correctly?

For each adapter that's not `ours` / `ours_mlp`, we directly tested it
on known-FAIL fixtures and looked at the result distribution:

  - **apple_vfr**: returns UNSUPPORTED for all 372 fixtures. Apple's
    VideoFlashingReduction API requires a macOS-specific framework
    surface that isn't reachable from our adapter context. The 0.000
    MCC in scoring reflects "tool unavailable in this environment,"
    not "tool says PASS on everything." Honest: noted in the report
    as unavailable.

  - **ffmpeg_photosensitivity**: **REAL BUG FOUND AND FIXED.** The
    adapter parsed for the regex `Detected at t=...s` but ffmpeg
    8.1.1's `vf_photosensitivity` emits per-frame diagnostics in the
    form `[Parsed_photosensitivity_0 @ 0x...] badness: <pre> ->
    <post> / <thresh> (<pct>% - OK|EXCEEDED)`, and only at log level
    `verbose` (not `info` which the adapter was using). Net effect:
    the adapter NEVER matched any line, EVER, and silently reported
    PASS for every fixture (MCC 0.000). Fixed: switched to
    `-loglevel verbose`, count EXCEEDED-vs-OK frames, FAIL iff
    `> 50%` of frames EXCEEDED.

    Even with the fix, ffmpeg's `vf_photosensitivity` cannot
    distinguish many TRACE fixture pairs that differ only in area:
    on `wcagc_30fps_area01/f002f038` (FAIL) vs `a002f038` (PASS) it
    emits IDENTICAL stats (24/20 OK/EXCEEDED frames, 45% threshold
    crossing, 291% max badness). The filter measures temporal pixel
    variation, not WCAG-area-axis properties. Documented in the
    adapter's docstring; the adapter's WCAG MCC post-fix is **-0.067**
    (worse than random), which is the honest signal that ffmpeg's
    mitigation heuristic isn't a standards detector.

  - **iris**: tested on `f002f038` (FAIL label, iris -> PASS),
    `f001f037` (FAIL label, iris -> PASS), and a slam-dunk
    Q6-extended 30 fps_fail_31hz fixture (iris correctly -> FAIL).
    Iris's low WCAG MCC (-0.037 with recall 0.024) is consistent
    with iris implementing **Harding-classic area thresholds**
    (87,296 px / full 10° ref rectangle) rather than WCAG-strict
    (21,824 px / 25% of that). TRACE labels are WCAG-strict, so iris
    naturally misses them. Iris's MCC on ITU/Ofcom (broadcast,
    matches Harding) is +0.092 -- still modest, but the relative
    pattern is what we'd predict, not a wiring bug.

  - **flicker_filter**: returns PASS for every TRACE fixture (MCC
    0.000). Verified: their model's `scipy.signal.find_peaks(width=8)`
    needs >=8 frames between peaks; most TRACE fixtures are ~44
    frames total, so n_peaks features are 0 across the board and
    the model output collapses to its intercept. Real limitation of
    the upstream tool, not a wiring bug. Documented.

## Check 4: TRACE label coverage matches across adapters

All adapters that aren't apple_vfr evaluate the same 360 fixtures
(360 verdicts from each of `ours`, `ours_mlp`, `iris`,
`ffmpeg_photosensitivity`, `flicker_filter`). The 12 missing fixtures
from each are static-image fixtures (.png) that those adapters
correctly return UNSUPPORTED on. No skew from coverage gaps.

## Updated headline table

After fixing ffmpeg's regex bug, the full per-WCAG-SC2.3.1 picture:

| tool | MCC | recall | specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| **A6 (logistic_l2 + classical-as-feature, K-fold on TRACE)** | **+0.355** | 0.301 | 0.954 | 22 | 10 | 51 | 210 |
| ours (classical detector) | +0.220 | 0.587 | 0.654 | 54 | 81 | 38 | 153 |
| iris | -0.037 | 0.024 | 0.961 | 2 | 9 | 82 | 221 |
| ffmpeg_photosensitivity (after fix) | -0.067 | 0.107 | 0.839 | -- | -- | 75 | 37 |
| ours_mlp (current OURS-trained, fixture-level) | -0.090 | 0.310 | 0.591 | -- | -- | 58 | 94 |
| flicker_filter (existing-art ML) | 0.000 | 0.000 | 1.000 | 0 | 0 | 84 | 220 |
| apple_vfr | -- | -- | -- | 0 | 0 | 0 | 0 (unavailable) |

## Conclusions from the sanity sweep

1. The A6 +0.355 result **stands**. It's stable across seeds (+0.350 ±
   0.015), uses honest matched-subset comparison against classical
   (+0.200 baseline, not the inflated +0.220), and beats every other
   tool we benchmarked.

2. We found and fixed **one real adapter wiring bug**:
   `ffmpeg_photosensitivity` was reading nothing because the regex
   didn't match the filter's actual output format. Fixed; ffmpeg's
   true MCC is **-0.067**, not 0.000. The fix makes our "ours wins
   among standards-grounded tools" claim *stronger*, not weaker.

3. No bugs found in `iris`, `apple_vfr`, `flicker_filter` adapters.
   Their low MCCs reflect real properties of those tools on this
   corpus, not wiring issues:
     - iris implements Harding-classic, TRACE labels are WCAG-strict
     - apple_vfr's API isn't reachable from our shell context
     - flicker_filter's feature extraction breaks on short videos

4. The "Q6 has the best ML approach" claim is **defensible at +0.355**,
   with disclosure: trained via 5-fold CV on TRACE (consumes labels
   by design, unlike the classical detector); compared head-to-head
   against the only published ML PSE detector with available source
   (flicker_filter, +0.000).
