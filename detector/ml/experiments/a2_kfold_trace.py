"""A2: Stratified K-fold CV on TRACE.

The fundamental question A1 couldn't answer: is the feature space rich
enough to discriminate TRACE labels at all? A2 trains ML ON TRACE
labels (which the classical detector is forbidden from doing, but an
ML detector can do by design as long as eval uses held-out folds) and
measures held-out MCC.

If A2 shows high MCC, the features carry signal but the Q6-extended
training data doesn't. If A2 also fails, the features themselves are
inadequate and we need A3 / A4.

K = 5 folds, stratified by label. Each classifier from A1 sweep is
trained K times; report mean MCC across held-out folds.

The ML detector trained this way would NOT be the production
production tool (since it sees labels at train time) -- this is purely
an upper-bound capability probe.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..dataset import load_split
from .common import (ExperimentResult, confusion, print_result, save_result)


CLASSIFIERS = [
    ("logistic_regression_l2",
        lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=2000)),
    ("random_forest_200",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=6,
                                       min_samples_leaf=2, random_state=0)),
    ("gradient_boost_100",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.05, random_state=0)),
    ("svm_rbf",
        lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=0)),
]


def run(seed: int = 0, n_folds: int = 5) -> list[ExperimentResult]:
    print(f"A2: stratified {n_folds}-fold CV on TRACE")
    split = load_split()
    X_te, y_te = split["X_test"], split["y_test"]
    print(f"  TRACE samples: {len(X_te)} (positive: {int(y_te.sum())}, "
          f"negative: {int((1 - y_te).sum())})")

    results: list[ExperimentResult] = []
    for name, factory in CLASSIFIERS:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_mccs = []
        all_preds = np.zeros_like(y_te)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_te, y_te)):
            X_tr = X_te[tr_idx]; y_tr = y_te[tr_idx]
            X_val = X_te[val_idx]; y_val = y_te[val_idx]
            scaler = StandardScaler().fit(X_tr)
            clf = factory()
            clf.fit(scaler.transform(X_tr), y_tr)
            pred = clf.predict(scaler.transform(X_val))
            all_preds[val_idx] = pred
            fold_mccs.append(confusion(y_val, pred)["mcc"])
        # Aggregate over all folds (since each sample is predicted exactly once)
        overall = confusion(y_te, all_preds)
        result = ExperimentResult(
            approach="A2",
            method=name,
            config={"seed": seed, "n_folds": n_folds,
                    "scaler": "StandardScaler"},
            val_scores={"mean_mcc_per_fold": float(np.mean(fold_mccs)),
                         "std_mcc_per_fold": float(np.std(fold_mccs)),
                         "per_fold_mcc": [float(x) for x in fold_mccs]},
            test_scores=overall,  # held-out-aggregated, the headline result
            notes="MCC reported via out-of-fold predictions (each TRACE "
                  "sample is held out exactly once).",
        )
        save_result(result)
        print_result(result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
