# Corpus Provenance

This file is the audit record for every external resource in this corpus.
Every fixture's ground truth must trace back to a row below (commit hash + path)
*and* to a standards clause cited in `MANIFEST.csv`.

## Cloned sources (in `/corpus/sources/`)

Retrieval date: **2026-05-19** (UTC date of clone). All clones are
`--depth 50`; full history is not needed for provenance because the commit
hash is the pin.

| Source dir | Upstream URL | Pinned commit | Upstream commit date | License | File count |
|---|---|---|---|---|---|
| `pse-test-media` | https://github.com/traceRERC/pse-test-media | `edf799a15cc1a8817a58c0120a7b25b2b28a1932` | 2025-08-11 | BSD-3-Clause | 1043 |
| `pseGuidelines`  | https://github.com/traceRERC/pseGuidelines  | `48d0c20f22a3333f64f444159b52c8c9eb097c71` | 2024-12-17 | BSD-3-Clause | 3 |
| `IRIS`           | https://github.com/electronicarts/IRIS      | `d96978ac1107f3463b77f69a9c1b1ec5d45291a0` | 2025-01-14 | BSD-3-Clause | 113 |
| `VideoFlashingReduction` | https://github.com/apple/VideoFlashingReduction | `7357d2f347c8659cc5ab4804b1338cfb0e95f362` | 2023-05-10 | Apple Sample Code License | 32 |
| `IRIS-Unreal-Plugin` | https://github.com/electronicarts/IRIS-Unreal-Plugin | `85311532a588d951b833a7b942234bcc9b578bd1` | 2024-11-22 | BSD-3-Clause | 840 |

Each clone preserves its upstream `LICENSE` / `LICENSE.txt` / `LICENSE.md`
in place. Nothing is modified in `/corpus/sources/`.

## Apple VideoFlashingReduction triplicate verification

The Apple repo ships ONE demo clip in three sub-projects (Xcode, MATLAB,
Mathematica). The brief flagged a need to verify they are byte-identical;
they are. SHA-256 of all three:

```
896551b3857a8096d0243046ce21655f858a1e3310d5cf8b43156504b071a25b
  VideoFlashingReduction_MATLAB/TestContent/TestVideo.mp4
  VideoFlashingReduction_Mathematica/Resources/movie.mp4
  VideoFlashingReduction_Xcode/VideoFlashingReduction/Resources/movie.mp4
```

Therefore the manifest carries this as a **single** fixture
(`apple_vfr/movie.mp4`), with the canonical path being the MATLAB copy.
The Mathematica/Xcode copies are noted in `MANIFEST.csv` `notes` for
provenance but are not duplicated as fixtures.

## TRACE pse-test-media set inventory

The 15 sets in `corpus/sources/pse-test-media/video_creation/` sum to
**306 tests** (matches the brief). Each set ships:

* a `.csv` ground-truth file with header `filename,pass,dimension`
  (`pass` = `TRUE`/`FALSE`; `dimension` ∈ {`area`, `saturation`, `count`, `FAIL`}),
* a per-test JSON combiner file (referenced by `filename` from the CSV),
* per-set notes in the README.

Per-set applicability across the six standards (`iso`, `itu_r1702`,
`ofcom`, `trace24`, `wcag2`, `nab_j`) is encoded in
`corpus/sources/pse-test-media/video_creation/README.md` and is mirrored
into `MANIFEST.csv` per-row so scoring can slice by `standard`.

Set counts (verified): 16+40+14+10+10+18+30+54+14+16+18+30+12+12+12 = 306. ✓

## IRIS expected per-frame logs

`corpus/sources/IRIS/test/Iris.Tests/data/ExpectedVideoLogFiles/*_RELATIVE.csv`
ships frame-level expected outputs for 8 videos:

```
2Hz_5s.mp4  2Hz_6s.mp4  3Hz_6s.mp4  flashStripes.mp4
extendedFLONG.mp4  intermitentEF.mp4  GradualRedIncrease.mp4  gray.mp4
```

These ship a per-frame schema with 18 columns including
`AverageLuminanceDiffAcc`, `LuminanceTransitions`, `RedTransitions`,
`FlashAreaLuminance`, `FlashAreaRed`, `PatternRisk` and frame-level result
codes. This is our **independent** corroboration signal: harness scoring
diffs our per-frame output against these expected logs (see §3.3 of the
brief).

IRIS pattern test images: `data/TestImages/Patterns/` contains 12
inputs and two sub-directories (`CircularExpectedResults/`,
`LineExpectedResults/`) with the corresponding expected detection outputs.

## "Known but excluded" tools (from fresh search 2026-05-19)

Fully populated in `MANIFEST.csv` (`source = "EXCLUDED"`) and surfaced in
the report's Known-but-Excluded table.

| Tool | Reason for exclusion | Source |
|---|---|---|
| EA IRIS-Unreal-Plugin | Not headless-runnable; requires Unreal Engine 5 runtime. | Brief §2.1; upstream README. |
| TRACE D2 PSE analysis tool | **Not yet publicly released** as of 2026-05-19; no source repo published. Re-evaluate at next benchmark run. | https://trace.umd.edu/open-source-photosensitive-epilepsy-analysis-tool/ |
| Flikcer (FlikcerApp) | Web-app only; no published open-source library entry point; usage flow is interactive upload via flikcerapp.com. | https://flikcerapp.com / Devpost listing. |
| Kaya, Kilic, Genc & Kural 2025 (SIViP) — implementation at samfatu/pse-detection-correction | Implementation repository did not carry a LICENSE file at probe date; included pending license clarification from the authors. | [Paper](https://link.springer.com/article/10.1007/s11760-025-04608-4) — DOI 10.1007/s11760-025-04608-4 ; [implementation repo](https://github.com/samfatu/pse-detection-correction) (probed 2026-05-19) |
| USPTO patent "Detection of photosensitive triggers in video content" | Patent, not software. No public implementation. | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10742923 |

## Determinism notes

* Codec / OpenCV / Pillow version differences shift pixel values near
  thresholds and can flip a PASS↔FAIL verdict. The exact pins are in
  `environment.lock` and `requirements.txt`.
* TRACE videos are GENERATED from the upstream scripts; they are not
  copied. Therefore `corpus/generated/` is git-ignored and rebuilt by
  `corpus/build_trace_videos.sh`.
* Generated extended corpus (`source = "Q6-extended"`) is also rebuilt by
  `corpus/build_extended_corpus.py`. Both scripts are seeded.
