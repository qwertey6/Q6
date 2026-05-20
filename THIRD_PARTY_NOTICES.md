# Third-Party Notices

This repository's own code is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE) — academic-research-first;
contact the maintainer for commercial relicensing.

The repository additionally **invokes and references** several third-party
projects for the purpose of internal conformance testing. Those projects
retain their own upstream licenses; none of their source is vendored here.
Each upstream project's LICENSE file is preserved in place alongside its
files in `/corpus/sources/` (and, where small fixtures are copied for direct
use, in `/corpus/vendored_fixtures/`).

We make no claim of endorsement by any third party. Factual statements
("tested against EA IRIS's published suite") are methodology; phrases that
imply endorsement ("EA-validated", "Apple-approved") are not used.

| Project | License | Upstream | Used for | Disposition |
|---|---|---|---|---|
| TRACE pse-test-media | BSD-3-Clause | https://github.com/traceRERC/pse-test-media | Test fixture *generators* (PNG patterns + CSV time/color + JSON combiners). 306 tests, 15 sets. Ground-truth labels via per-set CSV. | Cloned to `/corpus/sources/pse-test-media/` at pinned commit; LICENSE retained. Videos materialized into `/corpus/generated/`, not redistributed. |
| TRACE pseGuidelines | BSD-3-Clause | https://github.com/traceRERC/pseGuidelines | The Trace24 spec; reference when standards disagree. | Cloned to `/corpus/sources/pseGuidelines/` at pinned commit; LICENSE retained. Not redistributed. |
| EA IRIS | BSD-3-Clause | https://github.com/electronicarts/IRIS | 8 test videos with frame-level expected logs; ~12 spatial-pattern images with expected results. Also the reference C++ detector — built and invoked as a tool-under-test. | Cloned to `/corpus/sources/IRIS/` at pinned commit; LICENSE retained. Detector binary built locally in Docker; not redistributed. Test data used in-place for internal conformance testing. |
| EA IRIS-Unreal-Plugin | BSD-3-Clause | https://github.com/electronicarts/IRIS-Unreal-Plugin | Real-time UE5 PSE detection variant. **Excluded** from automated scoring (not headless-runnable). | Cloned for provenance/documentation only; LICENSE retained. Listed in the report's "known but excluded" table. |
| Apple VideoFlashingReduction | Apple Sample Code License (see upstream) | https://github.com/apple/VideoFlashingReduction | One demo clip (Xcode/MATLAB/Mathematica copies verified byte-identical); Swift/MATLAB/Mathematica reference implementations. | Cloned to `/corpus/sources/VideoFlashingReduction/` at pinned commit; Apple LICENSE retained. Reference implementation wrapped if runnable headless; otherwise documented exclusion. |
| FFmpeg `vf_photosensitivity` filter | LGPL-2.1+ | https://ffmpeg.org/ | Mitigation/reduction filter (not a standards detector). Included as tool-under-test, clearly labeled "mitigation, non-conformant by design." | Invoked as an external system binary; FFmpeg source is NOT vendored. The pinned binary version is recorded in `environment.lock`. No linkage to FFmpeg code, so LGPL dynamic-linking obligations do not arise here. |

## BSD-3-Clause non-endorsement reminder

The BSD-3-Clause license third clause forbids using the names of the
copyright holders or contributors to endorse or promote products derived
from this software without specific prior written permission. The names
"TRACE", "RERC", "Electronic Arts", and "EA IRIS" are used in this repository
only for factual methodology statements and provenance citation. They are
NOT used to imply endorsement of our detector or report by those parties.

## Apple Sample Code License

Apple's sample-code license applies to the materials in
`/corpus/sources/VideoFlashingReduction/`. We do not redistribute Apple's
code. The demo clip is used in-place for testing only.

## ISO 9241-391

ISO 9241-391 is referenced by standard number only. Its text is non-free
and is not fetched, vendored, or quoted in this repository. The report's
limitations section names this explicitly.

## Legal note

This repository is engineering tooling, not legal advice. Before any public
publication of benchmark results or commercial release, the licensing posture
and any redistribution of generated/derived media should be reviewed by
counsel.
