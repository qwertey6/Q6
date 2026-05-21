"""Shared evaluation utilities for A1-A6.

Each experiment records (approach_name, configuration, scores) into a
JSON file under results/ml_experiments/. The writeup task aggregates
across all of them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results" / "ml_experiments"


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num) / den if den > 0 else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n = max(tp + fp + fn + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn,
        "accuracy": (tp + tn) / n,
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0,
        "mcc": mcc(tp, fp, fn, tn),
    }


@dataclass
class ExperimentResult:
    """One row in the comparison table."""
    approach: str                  # A1..A6
    method: str                    # e.g. "logistic_regression"
    config: dict = field(default_factory=dict)
    train_scores: dict = field(default_factory=dict)
    val_scores: dict = field(default_factory=dict)
    test_scores: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def save_result(result: ExperimentResult, suffix: Optional[str] = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name_parts = [result.approach, result.method]
    if suffix:
        name_parts.append(suffix)
    out = RESULTS_DIR / ("_".join(name_parts) + ".json")
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


def print_result(result: ExperimentResult) -> None:
    """Compact human-readable line per result."""
    test = result.test_scores
    val = result.val_scores
    print(f"  [{result.approach}] {result.method:30s}  "
          f"val MCC={val.get('mcc', 0):+.3f}  test MCC={test.get('mcc', 0):+.3f}  "
          f"(test TP={test.get('tp', 0):2d} FP={test.get('fp', 0):3d} "
          f"FN={test.get('fn', 0):3d} TN={test.get('tn', 0):3d})  {result.notes}")
