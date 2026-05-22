"""A6: Stacking / residual learning.

Premise: the classical detector already gets to MCC +0.220 by following
the standards. Instead of learning the verdict from scratch, learn
where it's WRONG and correct it. Lower-variance target than the full
binary classification.

Two formulations tested:

  6a) classical_correction:
        - Add the classical detector's verdict as a feature
        - Train ML to predict the TRACE label
        - If features carry no extra info, ML learns identity-on-classical
        - If features help, ML can selectively flip classical's verdict

  6b) residual_classifier:
        - Compute residual = (TRACE_label XOR classical_verdict)
        - Train ML to predict residual (when to flip)
        - Final verdict = classical_verdict XOR predicted_residual

Both use K-fold CV on TRACE (since we need TRACE labels and the
classical verdict to compute residuals).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..dataset import load_split, MANIFEST, REPO_ROOT, _label_to_int
from .common import (ExperimentResult, confusion, print_result, save_result)


def _load_classical_verdicts(fixture_ids: list[str]) -> dict[str, int]:
    """Load classical detector's verdict for each fixture from the
    saved harness results. Returns id -> {0,1} (0=PASS, 1=FAIL)."""
    import json
    results_dir = REPO_ROOT / "results" / "results" / "q6" / "WCAG2.2-SC2.3.1"
    out = {}
    for fid in fixture_ids:
        safe = fid.replace("/", "__").replace("\\", "__") + ".json"
        path = results_dir / safe
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        verdict = r.get("verdict")
        if verdict in ("PASS", "FAIL"):
            out[fid] = 1 if verdict == "FAIL" else 0
    return out


def _trace_ids_with_labels() -> tuple[list[str], np.ndarray]:
    """Read MANIFEST for TRACE fixtures with non-empty WCAG label;
    return parallel (ids, labels) arrays."""
    import csv
    ids = []
    labels = []
    with MANIFEST.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if "pse-test-media" not in row.get("source", ""):
                continue
            if row.get("path", "").lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            label = _label_to_int(row.get("expected_wcag2_2", ""))
            if label is None:
                continue
            ids.append(row["path"])
            labels.append(label)
    return ids, np.array(labels, dtype=np.int64)


def _aligned_features(fixture_ids: list[str], split_X, split_ids
                        ) -> np.ndarray:
    """Look up features for the given fixture ids from the dataset split.
    Returns (len(fixture_ids), FEATURE_DIM)."""
    id_to_feats = {fid: split_X[i] for i, fid in enumerate(split_ids)}
    return np.stack([id_to_feats[fid] for fid in fixture_ids if fid in id_to_feats])


def run(seed: int = 0, n_folds: int = 5) -> list[ExperimentResult]:
    print("A6: stacking / residual learning on TRACE")
    split = load_split()
    test_ids_all = list(split["ids_test"])
    X_te_all = split["X_test"]
    y_te_all = split["y_test"]

    classical_map = _load_classical_verdicts(test_ids_all)
    # Filter to ids that have BOTH features AND a classical verdict.
    keep_mask = np.array([fid in classical_map for fid in test_ids_all])
    X = X_te_all[keep_mask]
    y = y_te_all[keep_mask]
    ids = [fid for fid in test_ids_all if fid in classical_map]
    classical = np.array([classical_map[fid] for fid in ids], dtype=np.int64)
    print(f"  TRACE fixtures with features+classical: {len(ids)}")

    # 6a: classical verdict as extra feature
    X_aug = np.concatenate([X, classical.reshape(-1, 1).astype(np.float32)], axis=1)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Headline baselines on the same subset (for apples-to-apples)
    classical_baseline = confusion(y, classical)
    print(f"  classical baseline on this subset: MCC={classical_baseline['mcc']:+.3f}  "
          f"TP={classical_baseline['tp']} FP={classical_baseline['fp']} "
          f"FN={classical_baseline['fn']} TN={classical_baseline['tn']}")

    results: list[ExperimentResult] = []
    for name, factory in [
        ("logistic_l2", lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=2000)),
        ("random_forest_200", lambda: RandomForestClassifier(n_estimators=200,
            max_depth=6, min_samples_leaf=2, random_state=0)),
        ("gradient_boost_100", lambda: GradientBoostingClassifier(n_estimators=100,
            max_depth=3, learning_rate=0.05, random_state=0)),
    ]:
        # 6a
        all_preds = np.zeros_like(y)
        for tr_idx, te_idx in skf.split(X_aug, y):
            sc = StandardScaler().fit(X_aug[tr_idx])
            clf = factory()
            clf.fit(sc.transform(X_aug[tr_idx]), y[tr_idx])
            all_preds[te_idx] = clf.predict(sc.transform(X_aug[te_idx]))
        r1 = ExperimentResult(
            approach="A6",
            method=f"{name}_classical_as_feature",
            config={"n_folds": n_folds, "scheme": "6a"},
            test_scores=confusion(y, all_preds),
            notes="classical verdict appended as 11th feature, "
                  "K-fold CV on TRACE.",
        )
        save_result(r1); print_result(r1); results.append(r1)

        # 6b: predict residual = y XOR classical, then apply correction
        residual = (y ^ classical).astype(np.int64)
        all_residual_preds = np.zeros_like(residual)
        for tr_idx, te_idx in skf.split(X, residual):
            sc = StandardScaler().fit(X[tr_idx])
            clf = factory()
            clf.fit(sc.transform(X[tr_idx]), residual[tr_idx])
            all_residual_preds[te_idx] = clf.predict(sc.transform(X[te_idx]))
        final_preds = (classical ^ all_residual_preds).astype(np.int64)
        r2 = ExperimentResult(
            approach="A6",
            method=f"{name}_predict_residual",
            config={"n_folds": n_folds, "scheme": "6b"},
            test_scores=confusion(y, final_preds),
            notes="predict (y XOR classical), apply correction; "
                  "K-fold CV on TRACE.",
        )
        save_result(r2); print_result(r2); results.append(r2)

    # Save the classical baseline too for the writeup table
    bl = ExperimentResult(
        approach="A6",
        method="classical_baseline_on_subset",
        config={"subset_size": len(ids)},
        test_scores=classical_baseline,
        notes="classical detector verdicts on the same subset that A6 "
              "evaluates on; for apples-to-apples comparison.",
    )
    save_result(bl); print_result(bl); results.append(bl)
    return results


if __name__ == "__main__":
    run()
