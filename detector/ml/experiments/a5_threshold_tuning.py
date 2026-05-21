"""A5: Threshold tuning via ROC analysis.

For each candidate classifier from A1/A2, replace the default 0.5
classification cutoff with one chosen on the validation set to
maximise MCC (or Youden's J = recall + specificity - 1). The model
outputs may be shifted so the natural cutoff isn't 0.5 -- this is a
no-cost optimisation.

Two regimes (matching A1/A2):
  - OURS->TRACE: val = held-out OURS fixtures
  - K-fold on TRACE: val = the inner fold (we use stacked out-of-fold
    predictions; threshold tuned on the same held-out predictions, so
    this slightly inflates the apparent gain; reported as such)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..dataset import load_split
from .common import (ExperimentResult, confusion, print_result, save_result, mcc)


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


def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Sweep candidate thresholds, pick one that maximises MCC.
    Returns (threshold, mcc_at_threshold)."""
    cands = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), probs]))
    best_t, best_mcc = 0.5, -2.0
    for t in cands:
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        m = mcc(tp, fp, fn, tn)
        if m > best_mcc:
            best_mcc = m
            best_t = float(t)
    return best_t, best_mcc


def _proba(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # Fallback: decision function -> sigmoid-ish
    d = clf.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def run(seed: int = 0, val_frac: float = 0.2, n_folds: int = 5
        ) -> list[ExperimentResult]:
    print("A5: threshold tuning")
    split = load_split()
    X_tr_all, y_tr_all = split["X_train"], split["y_train"]
    X_te, y_te = split["X_test"], split["y_test"]
    results: list[ExperimentResult] = []

    # Mode 1: OURS->TRACE. Tune threshold on val (held-out OURS), apply to test.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr_all))
    n_val = max(1, int(round(val_frac * len(X_tr_all))))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    scaler = StandardScaler().fit(X_tr_all[tr_idx])
    Xtr_s = scaler.transform(X_tr_all[tr_idx])
    Xval_s = scaler.transform(X_tr_all[val_idx])
    Xte_s = scaler.transform(X_te)

    print("  mode 1: OURS->TRACE, threshold tuned on OURS val")
    for name, factory in CLASSIFIERS:
        clf = factory()
        clf.fit(Xtr_s, y_tr_all[tr_idx])
        val_probs = _proba(clf, Xval_s)
        te_probs = _proba(clf, Xte_s)
        t_best, val_mcc = _best_threshold(val_probs, y_tr_all[val_idx])
        te_pred = (te_probs >= t_best).astype(int)
        result = ExperimentResult(
            approach="A5",
            method=f"{name}_OURS_to_TRACE",
            config={"threshold_tuned_on_val": t_best, "regime": "OURS->TRACE"},
            val_scores={"mcc_at_tuned_threshold": float(val_mcc)},
            test_scores=confusion(y_tr_all[val_idx], (val_probs >= t_best).astype(int)),
        )
        # Use test_scores for TEST not val. Re-do.
        result.test_scores = confusion(y_te, te_pred)
        result.val_scores = {"mcc_at_tuned_threshold": float(val_mcc),
                             "tuned_threshold": t_best}
        save_result(result); print_result(result); results.append(result)

    # Mode 2: K-fold on TRACE with per-fold threshold tuning.
    # Tune threshold on each fold's val, apply to that fold's test predictions.
    # We use an inner split: each fold has train/inner-val/test; tune on inner-val,
    # final eval on out-of-fold predictions with their tuned thresholds.
    print("  mode 2: K-fold on TRACE, threshold tuned per-fold")
    for name, factory in CLASSIFIERS:
        all_preds = np.zeros_like(y_te)
        fold_thresholds = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(X_te, y_te):
            X_in, y_in = X_te[tr_idx], y_te[tr_idx]
            X_out, y_out = X_te[te_idx], y_te[te_idx]
            # Inner split for threshold tuning (50/50 of training fold)
            inner_perm = np.random.default_rng(seed).permutation(len(X_in))
            n_inner_val = max(1, len(inner_perm) // 4)
            inner_val_idx = inner_perm[:n_inner_val]
            inner_tr_idx = inner_perm[n_inner_val:]
            sc = StandardScaler().fit(X_in[inner_tr_idx])
            clf = factory()
            clf.fit(sc.transform(X_in[inner_tr_idx]), y_in[inner_tr_idx])
            inner_val_probs = _proba(clf, sc.transform(X_in[inner_val_idx]))
            t_best, _ = _best_threshold(inner_val_probs, y_in[inner_val_idx])
            out_probs = _proba(clf, sc.transform(X_out))
            all_preds[te_idx] = (out_probs >= t_best).astype(int)
            fold_thresholds.append(t_best)
        result = ExperimentResult(
            approach="A5",
            method=f"{name}_kfold_TRACE_tuned",
            config={"regime": "kfold_TRACE", "fold_thresholds": fold_thresholds},
            val_scores={"fold_thresholds": fold_thresholds,
                         "mean_threshold": float(np.mean(fold_thresholds))},
            test_scores=confusion(y_te, all_preds),
        )
        save_result(result); print_result(result); results.append(result)
    return results


if __name__ == "__main__":
    run()
