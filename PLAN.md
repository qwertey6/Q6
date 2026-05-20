# PLAN.md — PSE Detector + Conformance Benchmark Harness

## Architecture summary

Four pillars, each developed behind a hard interface so they can be audited independently:

1. **Corpus** (`/corpus/`). Read-only set of test fixtures with labeled ground truth, every label traceable to a published standard clause. Upstream sources are cloned at pinned commits into `/corpus/sources/`. TRACE's 306-test set is reproducibly *generated* into `/corpus/generated/` from those pinned generators. We extend the corpus with our own labeled cases (frame-rate sweep, boundary precision, false-positive battery, codec round-trip, color space) under `source = "OURS-extended"` so the report can show upstream-peer-reviewed numbers in isolation.

2. **Harness** (`/harness/`). One normalized result schema. One adapter per tool-under-test. Adapters receive only the fixture path — **never** the label. Scoring runs in a separate process after all adapters finish and joins results to `MANIFEST.csv`. This isolation is the architectural property that makes the benchmark non-gameable; it is the first thing an auditor will check, so it is enforced by file layout, not convention.

3. **Detector** (`/detector/`). Our PSE detector, implemented test-driven from the *standard text* (WCAG 2.2 SC 2.3.1 baseline, plus pluggable profiles for ITU-R BT.1702 / Ofcom / Trace24 / NAB-J). Every numeric constant is justified in `detector/THRESHOLDS.md` by clause citation. The upstream-labeled corpus is a *held-out* acceptance check; we never tune constants against test labels. When our reading of the standard differs from a fixture label, that disagreement is recorded as an open question in the report — not silently conformed to.

4. **Report** (`/report/`). HTML + CSV + JSON. Leads with the upstream peer-reviewed subset (the defensible third-party number). Separately shows full extended corpus results. Lists every FN ("missed hazard" — the dangerous error) and every FP ("false alarm" — the credibility error) by tool, with the cited clause. Includes "known but excluded" table so absent competitors are explained, not hidden.

## Milestone order

### M0 — Repo foundation (this commit)
- `git init`, directory layout, `.gitignore`, `PLAN.md`, `README.md`, `THIRD_PARTY_NOTICES.md` skeleton, `environment.lock` skeleton, `Makefile` skeleton, `Dockerfile` skeleton.
- Stub `harness/schema.py` with the normalized result schema and a JSON Schema validator so adapters can self-check.

### M1 — Corpus (the bulk of the credibility work)
1. Clone the 4 cloneable upstream repos at pinned commits into `/corpus/sources/`:
   - `traceRERC/pse-test-media` (BSD-3) — generators, NOT videos.
   - `traceRERC/pseGuidelines` (BSD-3) — the Trace24 spec; our "answer key" when standards disagree.
   - `electronicarts/IRIS` (BSD-3) — 8 test videos with `*_RELATIVE.csv` expected logs, ~12 pattern images with expected results, AND the reference C++ detector (built in M2).
   - `apple/VideoFlashingReduction` (Apple sample) — 1 demo clip (verify the Xcode/MATLAB/Mathematica copies are byte-identical), Swift/MATLAB/Mathematica reference impls.
   - `electronicarts/IRIS-Unreal-Plugin` — clone for provenance only; documented as **excluded** (not headless-runnable).
2. Record every commit hash + retrieval date in `corpus/PROVENANCE.md`. Preserve each LICENSE in place.
3. Build `corpus/MANIFEST.csv` per the schema in §2.2 of the brief. Every label cites a clause. No vote-derived labels.
4. **Fresh search pass** (web search) for newer open-source PSE detectors (Flikcer, Carreira et al., Alzubaidi/Otoom/Al-Tamimi, Chiquet & Ochs, TRACE D2, browser extensions). For each: add adapter if open-source + runnable + headless; else add a row to the "known but excluded" table with reason.
5. Materialize TRACE videos via `corpus/build_trace_videos.sh` using *pinned* deps recorded in `environment.lock`. Codec/library version differences flip near-threshold cases — determinism is non-negotiable here.
6. Extend the corpus (`source = "OURS-extended"`) with:
   - Frame rates {24, 25, 30, 50, 60, 90, 120}.
   - Boundary precision: at threshold, ±1 frame, ±1 unit either side.
   - Area at WCAG 341×256 / 25% rectangle: just under, at, just over, multiple positions.
   - False-positive battery: alarming-looking content that passes because it falls short on exactly one axis.
   - Codec round-trip: H.264 multi-CRF, ProRes, VP9 re-encodes of near-threshold cases.
   - Color space: sRGB + ≥1 wider-gamut, flagged "extended coverage; standards under-specify."

### M2 — Harness
1. Finalize `harness/schema.py` (normalized verdict format + per-frame CSV format).
2. Adapters:
   - `ours` — wraps `/detector/`. Even our own tool goes through the adapter and does **not** read labels.
   - `iris` — build EA IRIS from source at pinned commit (needs `cmake`; install as a Docker layer). Run console example app, parse output. Cross-check our per-frame output against IRIS's shipped `*_RELATIVE.csv` as an independent anchor.
   - `apple_vfr` — wrap MATLAB/Octave reference if runnable in CI; else mark UNSUPPORTED with documented reason (no faking).
   - `ffmpeg_photosensitivity` — run `vf_photosensitivity`, derive a proxy verdict from how much correction it applies, label clearly as **mitigation, non-conformant by design**.
   - Adapters for any runnable detectors discovered in the M1 fresh search.
3. `harness/runner.py` orchestrates: each fixture × each adapter → normalized result file. Strict label isolation enforced in code (adapter API takes only the fixture path).
4. `harness/scoring.py`: confusion matrix, MCC (headline single number, robust to imbalance), recall, specificity, balanced accuracy, F1, sliced by `{standard, dimension, frame_rate, source, codec}`. Dedicated "missed hazards" and "false alarm" lists. Per-frame agreement against IRIS expected logs. Codec-stability metric.
5. `UNSUPPORTED ≠ ERROR` — UNSUPPORTED cases excluded from that metric and counted; reported per tool with reason.

### M3 — Our detector
1. Implement from *standard text*, not from labels:
   - WCAG 2.2 SC 2.3.1 general flash + red flash + classic area/luminance thresholds as baseline.
   - Pluggable profiles: ITU-R BT.1702, Ofcom GN2 Annex 1, Trace24, NAB-J. Each profile cites source.
   - Correct relative-luminance + saturated-red transition definitions per the standards' exact constants. Each constant documented in code with a clause citation.
   - Frame-rate aware: resample/normalize to the standard's reference timing. 24–120 fps must behave correctly (IRIS's high-fps degradation is documented; we should not inherit it).
   - Area analysis over the reference rectangle / screen-fraction, position independent.
   - Extended flash + bold static pattern checks.
   - Deterministic, headless, file-in / verdict-out.
2. `detector/THRESHOLDS.md` — every numeric constant and rule mapped to its standard clause. This document **is** the conformance argument.
3. Acceptance: run the held-out upstream subset *once* at milestone end. No tuning loops. Any standard-vs-label disagreement → open question in the report.

### M4 — Report
- `report/generate_report.py` consumes `scores.json` + raw results + MANIFEST.csv.
- HTML output with: executive summary table (upstream-peer-reviewed subset **first**, full extended corpus second, never blended); methodology & provenance; per-competitor gap analysis (concrete fixture examples + cited clauses); our results under the same scrutiny including our own FNs/FPs; per-axis deep dives (frame-rate curves, boundary precision, FP battery, codec stability); known-but-excluded table; honest limitations section.
- Reproducible from clean clone via Docker. `make corpus && make harness && make report` must be CI-green from a clean checkout.

## Cross-cutting invariants (enforced in code where possible)

- **Adapter label isolation**: adapter functions take `(fixture_path, profile) -> NormalizedResult`. No `label` parameter exists. Scoring is a separate module.
- **Determinism**: pinned Python + pinned pip + pinned ffmpeg + pinned IRIS commit + pinned OS image. Seed any randomness. Same checkout → same scores.
- **Provenance**: every fixture row in MANIFEST cites both upstream provenance (commit hash) and standard clause. No label without a clause.
- **Separation of upstream vs OURS-extended**: never blended in headline numbers. The report's lede number is the upstream-only score.
- **No benchmark-tuning**: when in doubt, the question is "what does the standard say," not "what makes the test pass."
- **Honesty**: known-but-excluded tools listed with reasons. UNSUPPORTED reported, not hidden. Our own FNs/FPs reported with the same prominence as competitors'.

## Open / deferred items

- **`cmake` is not installed locally** — needed to build EA IRIS from source. Resolution: handled in Dockerfile; local IRIS build is best-effort and Docker is the authoritative path.
- **ISO 9241-391** referenced by number only; text not fetched/vendored (non-free). The report's limitations section names this explicitly.
- **Apple VFR MATLAB reference runtime** — if MATLAB is not available in CI, try GNU Octave compatibility; if neither works headless, the Apple tool is marked UNSUPPORTED with a documented reason rather than faked.
- **IRIS-Unreal-Plugin** — runtime requires Unreal Engine; excluded from automated scoring with documented reason in the report's known-but-excluded table.
- **Fresh competitive landscape search** — performed during M1; results table populated then.

## Deviations from the brief

* **Per-standard fixture ground truth discovered late.** Every TRACE fixture
  JSON carries an `expected_result` block with per-standard PASS/FAIL
  (`trace24`, `wcag2_2`, `ofcom2017`, `itu_r1702_4`, `iso`). The brief
  describes ground truth via per-set CSVs, which we use; but the JSONs
  carry strictly richer information. The current manifest collapses per
  fixture using the per-set CSV (most-conservative applicable-standard
  verdict). Wiring the JSON `expected_result` through the manifest and
  scoring is the natural next step and would mechanically resolve OQ-4
  (the WCAG-classic area-threshold question) by letting each fixture be
  scored against each applicable standard's specific label rather than
  one collapsed label.

* **Local-environment limits on external adapters.**
  * **EA IRIS (C++):** building IRIS requires `cmake` + `vcpkg`, which
    are not installed locally. The IRIS adapter is implemented and
    reports `UNSUPPORTED` with a documented reason when the binary is
    absent. The Dockerfile installs the build chain; the Docker path is
    authoritative.
  * **Apple VFR (MATLAB):** MATLAB is non-free. The adapter attempts
    GNU Octave (installed in Docker) as a best-effort substitute but
    reports `UNSUPPORTED` until compatibility is verified. Per the
    brief: no faking.

* **Corpus partially materialized in this session.** TRACE provides 306
  fixtures across 15 sets. In this session we materialized 5 sets (~92
  videos: `30fps_alternating_01`, `broadcast_30fps_01`,
  `wcagc_30fps_area01..03`) to demonstrate the end-to-end pipeline.
  The full 306 are reproducible from `corpus/build_trace_videos.sh`;
  the Makefile + Dockerfile build all 15.

* **OURS-extended labels diverge from detector on `area_exactly_25pct`.**
  This is OQ-2 in `detector/THRESHOLDS.md` (the at-limit question). Our
  detector reads "more than 25%" literally and returns PASS for the
  at-limit case; our extended-corpus generator labels it FAIL. Surface
  in the report as an open question. The generator's label is the one
  that's wrong; left as-is rather than retuning silently to make the
  metric look better.
