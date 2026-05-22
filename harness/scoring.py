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
  * source (upstream peer-reviewed vs Q6-extended)
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


# Standards we slice by. Each entry: (bucket_slug, manifest_column,
# substring_hints). Per-standard labels are read FIRST from the per-fixture
# manifest column (populated for TRACE fixtures from their JSON
# expected_result blocks). For non-TRACE fixtures the column is empty and
# applicability is inferred from `standard_clause` substring hints, with the
# fixture's collapsed `expected_label` as the label.
KNOWN_STANDARDS = [
    ("wcag2.2-sc2.3.1", "expected_wcag2_2",       ("wcag", "wcag2", "wcag 2", "wcag-2", "wcag2.2")),
    ("itu-r-bt.1702",   "expected_itu_r1702_4",   ("itu-r", "bt.1702", "itu_r1702")),
    ("ofcom-gn2",       "expected_ofcom2017",     ("ofcom",)),
    ("trace24",         "expected_trace24",       ("trace24",)),
    ("nab-j",           "",                        ("nab-j", "nab_j", "j-ba")),
    ("iso9241-391",     "expected_iso9241_391",   ("iso9241-391", "iso 9241")),
]


# Map a tool's reported standard_profile string to the canonical bucket
# slug from KNOWN_STANDARDS. WCAG2.2-classic and WCAG2.2-SC2.3.1 are two
# readings of the same standard (see OQ-4) and score against the same
# fixture-level label.
PROFILE_TO_STANDARD_SLUG = {
    "WCAG2.2-SC2.3.1":   "wcag2.2-sc2.3.1",
    "WCAG2.2-classic":   "wcag2.2-sc2.3.1",
    "ITU-R-BT.1702":     "itu-r-bt.1702",
    "Ofcom-GN2-Annex1":  "ofcom-gn2",
    "Trace24":           "trace24",
    "NAB-J":             "nab-j",
}


def _per_standard_labels(row: dict) -> dict[str, str]:
    """Return {standard_slug: 'PASS'|'FAIL'} for this fixture, including only
    standards that have an explicit label. Reads the per-standard columns
    populated from TRACE per-fixture JSONs; for non-TRACE fixtures returns
    the collapsed expected_label keyed under every applicable standard
    inferred from standard_clause."""
    out: dict[str, str] = {}
    for slug, col, _ in KNOWN_STANDARDS:
        if col:
            val = (row.get(col) or "").strip()
            if val in ("PASS", "FAIL"):
                out[slug] = val
    if out:
        return out
    # Fallback for non-TRACE fixtures: use expected_label as the label
    # for every standard the standard_clause names. This preserves the
    # pre-OQ-5 behavior on IRIS / Apple / Q6-extended fixtures.
    fallback = (row.get("expected_label") or "").strip()
    if fallback in ("PASS", "FAIL"):
        clause = (row.get("standard_clause") or "").lower()
        for slug, _, needles in KNOWN_STANDARDS:
            if any(n in clause for n in needles):
                out[slug] = fallback
    return out


def _standards_for_row(row: dict) -> list[str]:
    """Applicable standards for this fixture, derived from per-standard
    columns (TRACE) or from standard_clause substring hints (everything
    else)."""
    standards = list(_per_standard_labels(row).keys())
    return standards or ["unspecified"]


def _label_for_tool(row: dict, result: dict) -> str:
    """Return the per-fixture label to score this tool's result against.

    Priority: per-standard label matching the tool's standard_profile,
    falling back to expected_label if the per-standard column is empty.
    """
    profile = result.get("standard_profile") or ""
    slug = PROFILE_TO_STANDARD_SLUG.get(profile)
    if slug:
        per_std = _per_standard_labels(row)
        if slug in per_std:
            return per_std[slug]
    return (row.get("expected_label") or "").strip()


def _source_bucket(row: dict) -> str:
    src = row["source"]
    if src.startswith("Q6-extended"):
        return "Q6-extended"
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
    # Parallel arrays for AUROC / PR-AUC: (label, score) pairs. Populated
    # only when the adapter emits a continuous `score` field.
    score_pairs: list[tuple[int, float]] = field(default_factory=list)

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
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return (self.tp / d) if d else None

    @property
    def f1(self) -> float | None:
        d = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / d) if d else None

    @property
    def f2(self) -> float | None:
        """F-beta with beta=2. Weights recall 2× over precision -- the
        right framing for safety-critical detection where missing a real
        hazard is much costlier than a false alarm."""
        d = (5 * self.tp) + self.fp + (4 * self.fn)
        return (5 * self.tp / d) if d else None

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

    @property
    def auroc(self) -> float | None:
        """Area under the ROC curve, computed from (label, score) pairs.
        Returns None if scores aren't available or both classes aren't
        represented."""
        if not self.score_pairs:
            return None
        pairs = self.score_pairs
        n_pos = sum(1 for y, _ in pairs if y == 1)
        n_neg = sum(1 for y, _ in pairs if y == 0)
        if n_pos == 0 or n_neg == 0:
            return None
        # Wilcoxon-Mann-Whitney statistic (handles ties correctly).
        sorted_by_score = sorted(pairs, key=lambda x: x[1])
        ranks: dict[int, float] = {}
        i = 0
        while i < len(sorted_by_score):
            j = i
            while j < len(sorted_by_score) and \
                  sorted_by_score[j][1] == sorted_by_score[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-indexed
            for k in range(i, j):
                ranks[id(sorted_by_score[k])] = avg_rank
            i = j
        sum_ranks_pos = sum(ranks[id(p)] for p in pairs if p[0] == 1)
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        return u / (n_pos * n_neg)

    @property
    def pr_auc(self) -> float | None:
        """Average precision (area under the precision-recall curve)."""
        if not self.score_pairs:
            return None
        n_pos = sum(1 for y, _ in self.score_pairs if y == 1)
        if n_pos == 0:
            return None
        sorted_pairs = sorted(self.score_pairs, key=lambda x: -x[1])
        tp = fp = 0
        prev_recall = 0.0
        ap = 0.0
        for y, _ in sorted_pairs:
            if y == 1: tp += 1
            else:      fp += 1
            recall = tp / n_pos
            precision = tp / (tp + fp)
            ap += precision * (recall - prev_recall)
            prev_recall = recall
        return ap


def _aggregate(rows_with_results: list[tuple[dict, dict]]) -> dict[str, Bucket]:
    """Return a dict of bucket-key -> Bucket. Buckets:

       overall
       source:<upstream|Q6-extended>
       standard:<name>
       dimension:<name>
       fps:<rate>
       codec:<codec>
       source+standard:<...>     (cross-slice for the lede table)
    """
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for manifest_row, result in rows_with_results:
        # Select the right label for THIS tool/result based on its
        # standard_profile -- TRACE fixtures get per-standard labels (OQ-5
        # resolution); non-TRACE fixtures get expected_label as before.
        expected = _label_for_tool(manifest_row, result)
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
        score = result.get("score")
        label_int = 1 if expected == "FAIL" else 0
        for key in _bucket_keys(manifest_row):
            b = buckets[key]
            b.fixture_count += 1
            setattr(b, category, getattr(b, category) + 1)
            if category == "fn":
                b.fn_fixtures.append(result.get("fixture_id", ""))
            elif category == "fp":
                b.fp_fixtures.append(result.get("fixture_id", ""))
            if score is not None:
                b.score_pairs.append((label_int, float(score)))
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


def _load_results(results_dir: Path, adapter: str, profile: str) -> dict[str, dict]:
    """fixture_id -> normalized result dict for one (adapter, profile)."""
    out: dict[str, dict] = {}
    per_profile = results_dir / "results" / adapter / profile
    if not per_profile.exists():
        return out
    for p in per_profile.glob("*.json"):
        with p.open() as fh:
            r = json.load(fh)
            out[r["fixture_id"]] = r
    return out


def _discover_adapter_profiles(results_dir: Path) -> list[tuple[str, str]]:
    """Walk results/ directory and return [(adapter, profile), ...] pairs
    that have at least one result file."""
    base = results_dir / "results"
    if not base.exists():
        return []
    out: list[tuple[str, str]] = []
    for adapter_dir in sorted(base.iterdir()):
        if not adapter_dir.is_dir():
            continue
        for profile_dir in sorted(adapter_dir.iterdir()):
            if profile_dir.is_dir() and any(profile_dir.glob("*.json")):
                out.append((adapter_dir.name, profile_dir.name))
    return out


# --- Reporting -------------------------------------------------------------

# Display name mapping. The machine identifier stays as the short
# adapter slug ("q6", "q6_mlp", "flicker_filter"); the display form
# is what's shown in tables and graphs so the brand name "Q6" doesn't
# get lost in screenshots / cropped images.
_DISPLAY_NAMES = {
    "q6":                       "Q6 (classical)",
    "q6_mlp":                   "Q6 (MLP)",
    "q6_cnn":                   "Q6 (CNN)",
    "iris":                     "IRIS",
    "apple_vfr":                "Apple VFR",
    "ffmpeg_photosensitivity":  "FFmpeg vf_photosensitivity",
    "flicker_filter":           "flickerfilter",
}


def _display_name(tool_at_profile: str) -> str:
    """Convert e.g. 'q6@WCAG2.2-SC2.3.1' → 'Q6 (classical) @ WCAG2.2-SC2.3.1'."""
    if "@" in tool_at_profile:
        tool, prof = tool_at_profile.split("@", 1)
        return f"{_DISPLAY_NAMES.get(tool, tool)} @ {prof}"
    return _DISPLAY_NAMES.get(tool_at_profile, tool_at_profile)


def _bucket_to_dict(b: Bucket) -> dict:
    return {
        "tp": b.tp, "tn": b.tn, "fp": b.fp, "fn": b.fn,
        "error": b.error, "unsupported": b.unsupported,
        "fixture_count": b.fixture_count,
        "recall": b.recall, "specificity": b.specificity,
        "precision": b.precision,
        "balanced_accuracy": b.balanced_accuracy,
        "f1": b.f1, "f2": b.f2, "mcc": b.mcc,
        "auroc": b.auroc, "pr_auc": b.pr_auc,
        "n_scored": len(b.score_pairs),
        "fn_fixtures": b.fn_fixtures, "fp_fixtures": b.fp_fixtures,
    }


def score_all(results_dir: Path, manifest_csv: Path) -> dict:
    """Score every (adapter, profile) combo.

    Each (adapter, profile) is treated as its own scorable entity, keyed
    under ``per_tool[f"{adapter}@{profile}"]``. The per-fixture label
    chosen is the one matching the profile's standard slug (see
    ``_label_for_tool``).
    """
    manifest = _load_manifest(manifest_csv)
    pairs = _discover_adapter_profiles(results_dir)
    per_tool: dict[str, dict] = {}
    for adapter, profile in pairs:
        results = _load_results(results_dir, adapter, profile)
        rows_with_results = []
        for fixture_id, manifest_row in manifest.items():
            if fixture_id in results:
                rows_with_results.append((manifest_row, results[fixture_id]))
        buckets = _aggregate(rows_with_results)
        key = f"{adapter}@{profile}"
        per_tool[key] = {k: _bucket_to_dict(v) for k, v in buckets.items()}
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
                     "recall", "specificity", "precision",
                     "balanced_accuracy", "f1", "f2", "mcc",
                     "auroc", "pr_auc", "n_scored"])
        for tool, buckets in scores["per_tool"].items():
            for bname, b in buckets.items():
                w.writerow([
                    tool, bname,
                    b["tp"], b["tn"], b["fp"], b["fn"],
                    b["error"], b["unsupported"], b["fixture_count"],
                    f"{b['recall']:.4f}"      if b["recall"]      is not None else "",
                    f"{b['specificity']:.4f}" if b["specificity"] is not None else "",
                    f"{b['precision']:.4f}"   if b["precision"]   is not None else "",
                    f"{b['balanced_accuracy']:.4f}" if b["balanced_accuracy"] is not None else "",
                    f"{b['f1']:.4f}" if b["f1"] is not None else "",
                    f"{b['f2']:.4f}" if b["f2"] is not None else "",
                    f"{b['mcc']:.4f}" if b["mcc"] is not None else "",
                    f"{b['auroc']:.4f}" if b["auroc"] is not None else "",
                    f"{b['pr_auc']:.4f}" if b["pr_auc"] is not None else "",
                    b["n_scored"],
                ])
    print(f"scoring: wrote {csv_path}")

    # Compact human summary on stdout.
    print()
    print("=== LEDE TABLE — upstream peer-reviewed subset ===")
    print(f"{'tool':32s} {'MCC':>7s} {'F2':>6s} {'AUROC':>6s} "
          f"{'recall':>7s} {'prec':>6s} {'spec':>6s} {'FN':>4s} {'FP':>4s}")
    for tool, buckets in scores["per_tool"].items():
        b = buckets.get("source:upstream", {})
        if not b:
            continue
        auroc = b.get("auroc")
        display = _display_name(tool)
        print(f"{display:32s} "
              f"{(b.get('mcc') or 0):>+7.3f} "
              f"{(b.get('f2') or 0):>6.3f} "
              f"{(auroc if auroc is not None else 0):>6.3f}{'' if auroc is not None else '*'} "
              f"{(b.get('recall') or 0):>7.3f} "
              f"{(b.get('precision') or 0):>6.3f} "
              f"{(b.get('specificity') or 0):>6.3f} "
              f"{b.get('fn', 0):>4d} {b.get('fp', 0):>4d}")
    print("  (* = AUROC not available; tool emits binary verdicts only)")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
