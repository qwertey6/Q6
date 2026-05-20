"""harness/runner.py — orchestrates fixtures × adapters.

Strict label isolation: this module reads ``corpus/MANIFEST.csv`` only to
get the *list of fixtures and their paths*, never their expected labels.
The expected-label column is dropped before the work loop runs. Scoring
joins back to MANIFEST.csv later in a separate process.

Each adapter is imported lazily so a missing optional dependency in one
adapter doesn't block the rest of the run.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from harness.schema import validate


REPO_ROOT = Path(__file__).resolve().parents[1]


ADAPTERS = ("ours", "ffmpeg_photosensitivity", "iris", "apple_vfr")


def _load_adapter(name: str):
    return importlib.import_module(f"harness.adapters.{name}")


def _fixture_iter(manifest_csv: Path) -> Iterable[tuple[str, Path, str]]:
    """Yield (fixture_id, absolute_path, type) for runnable fixtures.

    Reads ONLY the columns the runner is permitted to see: source, type,
    path. Notably does NOT read expected_label, expected_detail_file, or
    standard_clause. This is enforced by code review: keep this function
    free of those column names.
    """
    with manifest_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ftype = row["type"]
            if ftype == "excluded-tool":
                continue
            rel = row["path"]
            # Skip rows whose path is a URL (excluded tools we already filtered;
            # this is defense-in-depth).
            if rel.startswith("http"):
                continue
            abs_path = (REPO_ROOT / rel).resolve()
            fixture_id = rel  # the corpus-relative path is the canonical fixture id
            yield fixture_id, abs_path, ftype


def _run_one(adapter_name: str, fixture_id: str, fixture_path: Path,
             out_dir: Path) -> dict:
    """Invoke one adapter on one fixture; persist + return the result."""
    adapter = _load_adapter(adapter_name)
    # Per-frame CSV path for adapters that support one (currently "ours").
    per_frame = None
    if adapter_name == "ours":
        per_frame = out_dir / "per_frame" / adapter_name / (fixture_path.stem + ".csv")
    if not fixture_path.exists():
        # Fixture file missing (e.g. TRACE set not yet materialized for this run).
        # ERROR is the right verdict — it counts against the run's coverage.
        from harness.schema import NormalizedResult
        result = NormalizedResult(
            fixture_id=fixture_id, verdict="ERROR",
            tool=adapter_name, tool_version="n/a",
            runtime_seconds=0.0, raw_output_path="",
            error_message=f"fixture file missing: {fixture_path}",
        ).to_dict()
    else:
        try:
            if adapter_name == "ours":
                result = adapter.run(fixture_path, per_frame_out=per_frame)
            else:
                result = adapter.run(fixture_path)
        except Exception as e:
            from harness.schema import NormalizedResult
            result = NormalizedResult(
                fixture_id=fixture_id, verdict="ERROR",
                tool=adapter_name, tool_version="n/a",
                runtime_seconds=0.0, raw_output_path="",
                error_message=f"{type(e).__name__}: {e}",
            ).to_dict()

    # The adapter may have set fixture_id to a different value (e.g.
    # filename only). Standardize to the corpus-relative path.
    result["fixture_id"] = fixture_id
    try:
        validate(result)
    except Exception as e:
        # Schema violation → coerce to ERROR for that fixture-tool pair.
        from harness.schema import NormalizedResult
        result = NormalizedResult(
            fixture_id=fixture_id, verdict="ERROR",
            tool=adapter_name, tool_version=result.get("tool_version", "n/a"),
            runtime_seconds=float(result.get("runtime_seconds", 0.0)),
            raw_output_path=result.get("raw_output_path", ""),
            error_message=f"schema validation failed: {e}",
        ).to_dict()

    # Persist as <out_dir>/<adapter>/<fixture_id_safe>.json.
    safe = fixture_id.replace("/", "__").replace("\\", "__")
    per_adapter = out_dir / "results" / adapter_name
    per_adapter.mkdir(parents=True, exist_ok=True)
    with (per_adapter / f"{safe}.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run PSE adapters across the corpus.")
    ap.add_argument("--corpus",   type=Path, default=REPO_ROOT / "corpus" / "MANIFEST.csv")
    ap.add_argument("--out",      type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--adapters", nargs="*", default=list(ADAPTERS),
                    help=f"Which adapters to run; defaults to all: {ADAPTERS}")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap on fixtures (0 = unlimited). Useful for smoke runs.")
    ap.add_argument("--source-filter", default="",
                    help="If set, only run fixtures whose source contains this substring. "
                         "Used to scope to upstream-only or OURS-extended-only.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Process pool size for per-fixture parallelism. "
                         "Each fixture-adapter pair is one job.")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Build the job list. Read source for filtering ONLY; do NOT read label.
    jobs: list[tuple[str, str, Path]] = []
    with args.corpus.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["type"] == "excluded-tool":
                continue
            if row["path"].startswith("http"):
                continue
            if args.source_filter and args.source_filter not in row["source"]:
                continue
            abs_path = (REPO_ROOT / row["path"]).resolve()
            for adapter in args.adapters:
                jobs.append((adapter, row["path"], abs_path))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"runner: {len(jobs)} (fixture × adapter) jobs across {len(args.adapters)} adapters.")
    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, ad, fid, fp, args.out)
                   for ad, fid, fp in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if n_done % 25 == 0 or n_done == len(jobs):
                el = time.perf_counter() - t0
                print(f"runner: {n_done}/{len(jobs)} done in {el:.1f}s "
                      f"(latest: {res['tool']} → {res['verdict']} on {res['fixture_id']})")

    print(f"runner: complete in {time.perf_counter() - t0:.1f}s; "
          f"results under {args.out}/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
