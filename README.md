# PSE Detector + Conformance Benchmark Harness

A standards-grounded photosensitive-epilepsy (PSE) detector for video, plus a
reproducible benchmark harness that runs our detector and every available
open-source PSE detector through the **same** labeled corpus and reports the
results with full provenance.

**Status:** in active development. See `PLAN.md` for architecture and milestone order.

## License

This work is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

Permitted purposes include personal use, academic research, work by
educational institutions, public research organizations, public-safety or
health organizations, environmental-protection organizations, and government
institutions — see `LICENSE` for the full definition.

For commercial use, contact the maintainer for a separate license. The
project is academic-research-first by design; commercial relicensing
discussions are welcome — please open a GitHub Discussion at
<https://github.com/qwertey6/Q6/discussions> or contact the maintainer
directly.

### Contributing

External contributions are very welcome, but **please open an issue or
draft PR before substantial changes** — the project is currently a
single-author work, and a Contributor License Agreement (CLA) will be
required before merging non-trivial contributions to preserve the
ability to relicense in the future. Light fixes (typos, small bugs,
documentation) don't need a CLA.

Third-party components (TRACE pse-test-media, EA IRIS, Apple
VideoFlashingReduction, FFmpeg) retain their own upstream licenses; see
`THIRD_PARTY_NOTICES.md`.

## Layout

```
/corpus/             # labeled test fixtures (with full provenance + standards citations)
/harness/            # adapter-isolated benchmark runner + scoring
/detector/           # our PSE detector, implemented from standard first principles
/report/             # comparative report generator (HTML + CSV + JSON)
```

## Standards covered

- WCAG 2.2 SC 2.3.1 (Three Flashes or Below Threshold)
- ITU-R BT.1702 (TV broadcast baseline)
- Ofcom Guidance Note Section 2 Annex 1 (UK broadcast)
- NAB-J / J-BA (Japan)
- TRACE "Trace24" proposed guideline (Jordan & Vanderheiden, 2024)
- ISO 9241-391 — referenced by number only; non-free standard text not vendored.

## Honesty & legal notes

This repository is engineering tooling, not legal advice. Before any public
publication of benchmark results or commercial release, the licensing and any
redistribution of generated/derived media should be reviewed by counsel. We do
not claim endorsement from any upstream project or standards body; we report
factual methodology only.

See `THIRD_PARTY_NOTICES.md` for every external suite/tool, its license, and
how it is used.

## Reproducibility

`make corpus && make harness && make report` from a clean clone, inside the
provided Docker image, must produce byte-identical scores. Pinned versions live
in `environment.lock`. Near-threshold cases are pixel-sensitive; nondeterminism
would make the benchmark worthless as evidence.
