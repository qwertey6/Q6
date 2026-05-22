# PLAN.md — Q6 architecture + status

> This document combines the original architecture plan (M0–M4 below)
> with current implementation status. For the day-to-day "what is Q6
> and how do I run it" front door, read [`README.md`](README.md) first.

## Status as of last update

  - **M0 (foundation)** — done. Repo layout, schema, Makefile, Dockerfile
    skeleton all in place; CI runs the detector self-test + pytest panel
    on every push.
  - **M1 (corpus)** — done. 306 TRACE fixtures + 45 Q6-extended fixtures
    materialized. MANIFEST.csv has per-standard label columns for all
    seven supported standards (NAB-J labels derived analytically — see
    `corpus/derive_per_standard_labels.py`).
  - **M2 (harness)** — done. Six adapters wired (Q6 classical + Q6 MLP,
    IRIS, FFmpeg vf_photosensitivity, Apple VFR, flickerfilter), runner
    enforces label isolation in code, scoring emits MCC + F2 + AUROC +
    PR-AUC + precision/recall/specificity per (tool, profile, bucket).
  - **M3 (Q6 detector)** — done for the classical path. Algorithm
    implements WCAG-strict thresholds with seven profile variants,
    each cited clause-by-clause in `detector/THRESHOLDS.md`. Native-
    tensor CC backend exists but is currently gated on upstream
    PyTorch MPS fixes (see `detector/ml/SANITY_CHECKS.md`).
  - **M4 (report)** — done for the comparative LEDE; per-fixture HTML
    report + spatial-temporal heatmap shipped. The full single-page
    comparison report (M4 originally scoped) is the next natural
    deliverable.
  - **ML layer (post-M4 addition)** — A1–A6 experiment sweep complete;
    A6 stacking + K-fold-on-TRACE produces MCC +0.355, the
    best ML result on this corpus. Documented in
    `detector/ml/EXPERIMENTS.md` and `detector/ml/SANITY_CHECKS.md`.

## Architecture summary

Four pillars, each developed behind a hard interface so they can be audited independently:

1. **Corpus** (`/corpus/`). Read-only set of test fixtures with labeled ground truth, every label traceable to a published standard clause. Upstream sources are cloned at pinned commits into `/corpus/sources/`. TRACE's 306-test set is reproducibly *generated* into `/corpus/generated/` from those pinned generators. We extend the corpus with our own labeled cases (frame-rate sweep, boundary precision, false-positive battery, codec round-trip, color space) under `source = "Q6-extended"` so the report can show upstream-peer-reviewed numbers in isolation.

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
4. **Fresh search pass** (web search) for newer open-source PSE detectors (Flikcer, Kaya et al. 2025 / `samfatu/pse-detection-correction`, Alzubaidi/Otoom/Al-Tamimi, Chiquet & Ochs, TRACE D2, browser extensions). For each: add adapter if open-source + runnable + headless; else add a row to the "known but excluded" table with reason.
5. Materialize TRACE videos via `corpus/build_trace_videos.sh` using *pinned* deps recorded in `environment.lock`. Codec/library version differences flip near-threshold cases — determinism is non-negotiable here.
6. Extend the corpus (`source = "Q6-extended"`) with:
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
- **Separation of upstream vs Q6-extended**: never blended in headline numbers. The report's lede number is the upstream-only score.
- **No benchmark-tuning**: when in doubt, the question is "what does the standard say," not "what makes the test pass."
- **Honesty**: known-but-excluded tools listed with reasons. UNSUPPORTED reported, not hidden. Our own FNs/FPs reported with the same prominence as competitors'.

## Open / deferred items

- **External validation.** All current ground-truth comes from TRACE's
  corpus (labels), Q6-extended (analytic labels from generation params),
  or derived from those (NAB-J via `derive_per_standard_labels.py`).
  Real cross-source validation requires running Q6 against an external
  corpus we don't author — e.g., NHK / J-BA / Akatsuki / Imagica EMS
  internal datasets, BBC / EBU samples, fresh academic shares. Outreach
  is in flight; see `CORRESPONDENCE.md` (gitignored locally).
- **MPS tensor CC backend.** Native-tensor CC works correctly on CPU but
  is 100-1000× slower than cv2 on MPS due to three upstream PyTorch
  perf issues (`bincount`, `unique(return_counts)`, `roll`). Pending
  upstream fixes (separate PRs in flight). The cv2 backend is the
  default and is fast enough; the tensor backend is opt-in via
  `Q6_CC_BACKEND=tensor`.
- **Ship the report.** The comparative single-page HTML report from
  M4's original scope (LEDE table + methodology + per-tool gap
  analysis) hasn't been produced as one artifact yet, even though all
  the underlying data + per-fixture reports + heatmaps are in place.
- **Q6-extended labels at the area boundary.** OQ-2 in
  `detector/THRESHOLDS.md` — at `area_exactly_25pct`, the detector
  reads "more than 25%" literally and returns PASS, but the generator
  labels FAIL. Surface as an open question in the comparative report;
  the generator's label is the one that's wrong.

## Resolved items (kept for historical context)

- **Per-standard fixture ground truth.** OQ-5: TRACE's per-fixture
  JSONs carry an `expected_result` block with per-standard PASS/FAIL.
  We now route each tool's verdict to the matching per-standard label
  column via `harness/scoring.py::PROFILE_TO_STANDARD_SLUG`.
- **EA IRIS local build.** Verified via native macOS build (Docker
  blocked on libtool macros). IRIS adapter parses `result.json` from
  IrisApp's per-fixture tempdir.
- **Corpus materialization.** All 306 TRACE fixtures + 45 Q6-extended
  fixtures materialized; `make corpus` rebuilds from sources at pinned
  commits.
