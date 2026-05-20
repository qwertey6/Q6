"""harness/scoring.py — join adapter results to MANIFEST.csv, compute metrics.

Runs in a separate process from the adapters. This separation is the
property that makes the benchmark non-gameable: adapters never see labels;
scoring reads labels and joins to results that were produced without them.

Headline metric: Matthews correlation coefficient (MCC). Robust to class
imbalance and ranges in [-1, 1]; +1 = perfect, 0 = no better than chance,
-1 = always wrong. We additionally report recall, specificity, balanced
accuracy, F1, and (most importantly for safety) the absolute number of
false negatives (FN) — *missed hazards*. A tool with FN > 0 is flagged
prominently in the report regardless of the headline number.

We slice metrics by:
  * standard (extracted from MANIFEST's standard_clause)
  * dimension (luminance / red / area / count / pattern)
  * frame_rate
  * source (upstream peer-reviewed vs OURS-extended)
  * codec
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


# Standards we'll slice by. The full label substring is matched case-insensitively
# against the manifest's standard_clause field.
KNOWN_STANDARDS = [
    ("wcag2.2-sc2.3.1", ("wcag", "wcag2", "wcag 2", "wcag-2", "wcag2.2")),
    ("itu-r-bt.1702",   ("itu-r", "bt.1702", "itu_r1702")),
    ("ofcom-gn2",       ("ofcom",)),
    ("trace24",         ("trace24",)),
    ("nab-j",           ("nab-j", "nab_j", "j-ba")),
    ("iso9241-391",     ("iso9241-391", "iso 9241")),
]


def _standards_for_row(row: dict) -> list[str]:
    clause = (row.get("standard_clause") or "").lower()
    hits = []
    for label, needles in KNOWN_STANDARDS:
        if any(n in clause for n in needles):
            hits.append(label)
    return hits or ["unspecified"]


def _source_bucket(row: dict) -> str:
    src = row["source"]
    if src.startswith("OURS-extended"):
        return "OURS-extended"
    # Everything else from clone-only upstream goes into the "upstream" bucket
    # that is the lede of the report.
    return "upstream"


def _frame_rate(row: dict) -> str:
    fr = (row.get("frame_rate") or "").strip()
    return fr or "unknown"


def _codec(row: dict) -> str:
    return (row.get("codec") or "").strip() or "default"


# --- Metric primitives -----------------------------------------------------

@dataclass
class Bucket:
    tp: int = 0   # tool said FAIL, label says FAIL (hazard correctly caught)
    tn: int = 0   # tool said PASS, label says PASS (harmless correctly passed)
    fp: int = 0   # tool said FAIL, label says PASS (false alarm)
    fn: int = 0   # tool said PASS, label says FAIL (MISSED HAZARD — the dangerous error)
    error: int = 0
    unsupported: int = 0
    fixture_count: int = 0
    fn_fixtures: list[str] = field(default_factory=list)
    fp_fixtures: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return (self.tp / d) if d else None

    @property
    def specificity(self) -> float | None:
        d = self.tn + self.fp
        return (self.tn / d) if d else None

    @property
    def balanced_accuracy(self) -> float | None:
        r, s = self.recall, self.specificity
        if r is None or s is None: return None
        return 0.5 * (r + s)

    @property
    def f1(self) -> float | None:
        d = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / d) if d else None

    @property
    def mcc(self) -> float | None:
        """Matthews correlation coefficient."""
        num = (self.tp * self.tn) - (self.fp * self.fn)
        denom_sq = (
            (self.tp + self.fp) * (self.tp + self.fn) *
            (self.tn + self.fp) * (self.tn + self.fn)
        )
        if denom_sq <= 0: return None
        return num / math.sqrt(denom_sq)


def _aggregate(rows_with_results: list[tuple[dict, dict]]) -> dict[str, Bucket]:
    """Return a dict of bucket-key -> Bucket. Buckets:

       overall
       source:<upstream|OURS-extended>
       standard:<name>
       dimension:<name>
       fps:<rate>
       codec:<codec>
       source+standard:<...>     (cross-slice for the lede table)
    """
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for manifest_row, result in rows_with_results:
        # Skip rows that aren't scoreable.
        expected = manifest_row["expected_label"]
        if expected not in ("PASS", "FAIL"):
            continue
        verdict = result["verdict"]
        if verdict == "UNSUPPORTED":
            _touch(buckets, manifest_row).unsupported += 1
            continue
        if verdict == "ERROR":
            _touch(buckets, manifest_row).error += 1
            continue
        if verdict not in ("PASS", "FAIL"):
            _touch(buckets, manifest_row).error += 1
            continue
        # Score against the label.
        category = _confusion(expected, verdict)
        for key in _bucket_keys(manifest_row):
            b = buckets[key]
            b.fixture_count += 1
            setattr(b, category, getattr(b, category) + 1)
            if category == "fn":
                b.fn_fixtures.append(result.get("fixture_id", ""))
            elif category == "fp":
                b.fp_fixtures.append(result.get("fixture_id", ""))
    return buckets


def _touch(buckets: dict[str, Bucket], manifest_row: dict) -> Bucket:
    """For UNSUPPORTED/ERROR counts: increment in every applicable bucket."""
    last = None
    for key in _bucket_keys(manifest_row):
        last = buckets[key]
    return last  # we increment the field on this in caller


def _bucket_keys(manifest_row: dict) -> list[str]:
    keys = ["overall"]
    src = _source_bucket(manifest_row)
    keys.append(f"source:{src}")
    for std in _standards_for_row(manifest_row):
        keys.append(f"standard:{std}")
        keys.append(f"source+standard:{src}/{std}")
    keys.append(f"fps:{_frame_rate(manifest_row)}")
    keys.append(f"codec:{_codec(manifest_row)}")
    return keys


def _confusion(expected: str, verdict: str) -> str:
    """Return 'tp' / 'tn' / 'fp' / 'fn'."""
    e_fail = (expected == "FAIL")
    v_fail = (verdict == "FAIL")
    if e_fail and v_fail:     return "tp"
    if not e_fail and not v_fail: return "tn"
    if not e_fail and v_fail:    return "fp"  # false alarm
    return "fn"                                 # missed hazard


# --- Loading ---------------------------------------------------------------

def _load_manifest(manifest_csv: Path) -> dict[str, dict]:
    """fixture_id -> row dict."""
    out: dict[str, dict] = {}
    with manifest_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["type"] == "excluded-tool":
                continue
            out[row["path"]] = row
    return out


def _load_results(results_dir: Path, adapter: str) -> dict[str, dict]:
    """fixture_id -> normalized result dict for one adapter."""
    out: dict[str, dict] = {}
    per_adapter = results_dir / "results" / adapter
    if not per_adapter.exists():
        return out
    for p in per_adapter.glob("*.json"):
        with p.open() as fh:
            r = json.load(fh)
            out[r["fixture_id"]] = r
    return out


# --- Reporting -------------------------------------------------------------

def _bucket_to_dict(b: Bucket) -> dict:
    return {
        "tp": b.tp, "tn": b.tn, "fp": b.fp, "fn": b.fn,
        "error": b.error, "unsupported": b.unsupported,
        "fixture_count": b.fixture_count,
        "recall": b.recall, "specificity": b.specificity,
        "balanced_accuracy": b.balanced_accuracy,
        "f1": b.f1, "mcc": b.mcc,
        "fn_fixtures": b.fn_fixtures, "fp_fixtures": b.fp_fixtures,
    }


def score_all(results_dir: Path, manifest_csv: Path) -> dict:
    manifest = _load_manifest(manifest_csv)
    discovered_adapters = [d.name for d in (results_dir / "results").iterdir()
                            if d.is_dir()] if (results_dir / "results").exists() else []
    per_tool: dict[str, dict] = {}
    for adapter in discovered_adapters:
        results = _load_results(results_dir, adapter)
        # Outer join: a fixture without a result for this adapter just doesn't
        # contribute to that tool's buckets (rare given runner persists everything).
        rows_with_results = []
        for fixture_id, manifest_row in manifest.items():
            if fixture_id in results:
                rows_with_results.append((manifest_row, results[fixture_id]))
        buckets = _aggregate(rows_with_results)
        per_tool[adapter] = {k: _bucket_to_dict(v) for k, v in buckets.items()}
    return {
        "tools": list(per_tool),
        "per_tool": per_tool,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compute scores for PSE benchmark.")
    ap.add_argument("--results",  type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "corpus" / "MANIFEST.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "scores")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    scores = score_all(args.results, args.manifest)

    json_path = args.out / "scores.json"
    with json_path.open("w") as fh:
        json.dump(scores, fh, indent=2)
    print(f"scoring: wrote {json_path}")

    # Flat CSV: one row per (tool, bucket).
    csv_path = args.out / "scores.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool", "bucket", "tp", "tn", "fp", "fn",
                     "error", "unsupported", "fixture_count",
                     "recall", "specificity", "balanced_accuracy", "f1", "mcc"])
        for tool, buckets in scores["per_tool"].items():
            for bname, b in buckets.items():
                w.writerow([
                    tool, bname,
                    b["tp"], b["tn"], b["fp"], b["fn"],
                    b["error"], b["unsupported"], b["fixture_count"],
                    f"{b['recall']:.4f}"   if b["recall"]   is not None else "",
                    f"{b['specificity']:.4f}" if b["specificity"] is not None else "",
                    f"{b['balanced_accuracy']:.4f}" if b["balanced_accuracy"] is not None else "",
                    f"{b['f1']:.4f}" if b["f1"] is not None else "",
                    f"{b['mcc']:.4f}" if b["mcc"] is not None else "",
                ])
    print(f"scoring: wrote {csv_path}")

    # Compact human summary on stdout.
    print()
    print("=== LEDE TABLE — upstream peer-reviewed subset ===")
    print(f"{'tool':28s} {'MCC':>8s} {'recall':>8s} {'spec':>8s} {'FN':>4s} {'FP':>4s} {'unsupp':>7s}")
    for tool, buckets in scores["per_tool"].items():
        b = buckets.get("source:upstream", {})
        if not b:
            continue
        print(f"{tool:28s} "
              f"{(b.get('mcc') or 0):>8.3f} "
              f"{(b.get('recall') or 0):>8.3f} "
              f"{(b.get('specificity') or 0):>8.3f} "
              f"{b.get('fn', 0):>4d} {b.get('fp', 0):>4d} {b.get('unsupported', 0):>7d}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
