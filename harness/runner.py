"""harness/runner.py -- orchestrates fixtures x adapters x profiles.

Strict label isolation: this module reads ``corpus/MANIFEST.csv`` only to
get the *list of fixtures and their paths*, never their expected labels.
The expected-label column is dropped before the work loop runs. Scoring
joins back to MANIFEST.csv later in a separate process.

Each adapter is imported lazily so a missing optional dependency in one
adapter doesn't block the rest of the run.

Multi-profile contract: each adapter exposes ``SUPPORTED_PROFILES`` (a
list of profile names) and ``PROFILE_AFFECTS_BEHAVIOR`` (bool). If
behaviour varies, the runner invokes ``adapter.run(...)`` once per
profile per fixture. If it doesn't, the runner runs the adapter ONCE
per fixture and writes the same verdict out under each profile name
(so scoring's per-standard slicing has data for every (tool, profile)
cell without wasting CPU on identical computations).
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


ADAPTERS = ("ours", "ffmpeg_photosensitivity", "iris", "apple_vfr",
            "flicker_filter", "ours_mlp")


def _load_adapter(name: str):
    return importlib.import_module(f"harness.adapters.{name}")


def _adapter_meta(name: str) -> tuple[list[str], bool]:
    mod = _load_adapter(name)
    profiles = list(getattr(mod, "SUPPORTED_PROFILES", ["WCAG2.2-SC2.3.1"]))
    varies = bool(getattr(mod, "PROFILE_AFFECTS_BEHAVIOR", True))
    return profiles, varies


def _write_result(out_dir: Path, adapter_name: str, profile: str,
                  fixture_id: str, result: dict) -> None:
    """Persist a single result at <out>/results/<adapter>/<profile>/<safe>.json."""
    safe = fixture_id.replace("/", "__").replace("\\", "__")
    per_path = out_dir / "results" / adapter_name / profile
    per_path.mkdir(parents=True, exist_ok=True)
    with (per_path / f"{safe}.json").open("w") as fh:
        json.dump(result, fh, indent=2)


def _error_result(adapter_name: str, fixture_id: str, profile: str,
                   message: str) -> dict:
    from harness.schema import NormalizedResult
    return NormalizedResult(
        fixture_id=fixture_id, verdict="ERROR",
        tool=adapter_name, tool_version="n/a",
        runtime_seconds=0.0, raw_output_path="",
        standard_profile=profile,
        error_message=message,
    ).to_dict()


def _coerce_and_validate(result: dict, adapter_name: str, profile: str,
                          fixture_id: str) -> dict:
    """Make sure the result has expected metadata, then schema-validate it.
    On validation failure, coerce to ERROR."""
    result["fixture_id"] = fixture_id
    result.setdefault("standard_profile", profile)
    try:
        validate(result)
        return result
    except Exception as e:
        from harness.schema import NormalizedResult
        return NormalizedResult(
            fixture_id=fixture_id, verdict="ERROR",
            tool=adapter_name, tool_version=result.get("tool_version", "n/a"),
            runtime_seconds=float(result.get("runtime_seconds", 0.0)),
            raw_output_path=result.get("raw_output_path", ""),
            standard_profile=profile,
            error_message=f"schema validation failed: {e}",
        ).to_dict()


def _run_one(adapter_name: str, profiles_to_emit: list[str], fixture_id: str,
             fixture_path: Path, out_dir: Path) -> dict:
    """Invoke the adapter once for ``fixture_path`` and persist results
    under EACH of ``profiles_to_emit``. The caller decides whether
    ``profiles_to_emit`` has one entry (profile-affects-behavior, called
    multiple times) or many entries (profile-doesn't-affect-behavior,
    called once and replicated)."""
    if not fixture_path.exists():
        # Fixture file missing (e.g. TRACE set not yet materialized).
        # ERROR for every requested profile.
        for p in profiles_to_emit:
            r = _error_result(adapter_name, fixture_id, p,
                              f"fixture file missing: {fixture_path}")
            _write_result(out_dir, adapter_name, p, fixture_id, r)
        return _error_result(adapter_name, fixture_id, profiles_to_emit[0],
                              f"fixture file missing: {fixture_path}")

    adapter = _load_adapter(adapter_name)
    # Profile to pass to the adapter call. If multiple are being emitted,
    # we still call the adapter with the first (the adapter's behaviour
    # doesn't depend on it under the "profile-doesn't-affect-behavior"
    # contract).
    invocation_profile = profiles_to_emit[0]

    per_frame = None
    if adapter_name == "ours":
        # Per-frame CSV path embeds the profile so concurrent profile runs
        # don't overwrite each other.
        per_frame = (out_dir / "per_frame" / adapter_name / invocation_profile /
                     (fixture_path.stem + ".csv"))

    try:
        if adapter_name == "ours":
            result = adapter.run(fixture_path, profile=invocation_profile,
                                 per_frame_out=per_frame)
        else:
            result = adapter.run(fixture_path, profile=invocation_profile)
    except Exception as e:
        result = _error_result(adapter_name, fixture_id, invocation_profile,
                                f"{type(e).__name__}: {e}")

    # Persist under each requested profile (overriding standard_profile
    # in the JSON so scoring sees the right key).
    last_validated: dict | None = None
    for p in profiles_to_emit:
        r = dict(result)
        r["standard_profile"] = p
        r = _coerce_and_validate(r, adapter_name, p, fixture_id)
        _write_result(out_dir, adapter_name, p, fixture_id, r)
        last_validated = r
    return last_validated if last_validated is not None else result


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
                    help="Process pool size. Each (adapter, fixture, "
                         "single-or-bulk-profile-group) is one job.")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Resolve per-adapter profile metadata once.
    adapter_profiles: dict[str, tuple[list[str], bool]] = {}
    for ad in args.adapters:
        adapter_profiles[ad] = _adapter_meta(ad)

    # Build the job list. Read source for filtering ONLY; do NOT read label.
    # Each job = (adapter, profiles_to_emit, fixture_id, fixture_path).
    jobs: list[tuple[str, list[str], str, Path]] = []
    fixture_rows = []
    with args.corpus.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["type"] == "excluded-tool":
                continue
            if row["path"].startswith("http"):
                continue
            if args.source_filter and args.source_filter not in row["source"]:
                continue
            fixture_rows.append(row)

    for row in fixture_rows:
        abs_path = (REPO_ROOT / row["path"]).resolve()
        fid = row["path"]
        for ad in args.adapters:
            profiles, varies = adapter_profiles[ad]
            if varies:
                # One job per profile -- adapter actually behaves differently
                # under each.
                for p in profiles:
                    jobs.append((ad, [p], fid, abs_path))
            else:
                # One job, emit all profiles from a single computation.
                jobs.append((ad, list(profiles), fid, abs_path))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    total_emissions = sum(len(p) for _, p, _, _ in jobs)
    print(
        f"runner: {len(jobs)} jobs across {len(args.adapters)} adapters "
        f"(emits {total_emissions} per-(tool,profile,fixture) result files)."
    )
    t0 = time.perf_counter()
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, ad, ps, fid, fp, args.out)
                   for ad, ps, fid, fp in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(jobs):
                el = time.perf_counter() - t0
                print(f"runner: {n_done}/{len(jobs)} done in {el:.1f}s "
                      f"(latest: {res['tool']}@{res['standard_profile']} "
                      f"-> {res['verdict']} on {res['fixture_id']})")

    print(f"runner: complete in {time.perf_counter() - t0:.1f}s; "
          f"results under {args.out}/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
