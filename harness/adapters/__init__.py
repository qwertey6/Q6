"""PSE detector adapters.

Architectural contract (enforced by code review and runner signature):

    def run(fixture_path: pathlib.Path, profile: str) -> dict

is the ONLY interface point. No adapter receives, imports, parses, or
otherwise observes the ground-truth label for any fixture. Labels live in
``corpus/MANIFEST.csv`` and are joined to results in ``harness/scoring.py``,
a separate process that runs AFTER all adapters complete. This separation is
the property that makes the benchmark non-gameable and is the first thing an
auditor will check.
"""
