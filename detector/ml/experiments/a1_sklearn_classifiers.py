"""A1: Different classifiers on the same 10 hand-engineered features.

Sweep through sklearn classifiers (logistic regression, random forest,
gradient boosting, SVM with RBF, k-NN, naive Bayes). With 45 training
samples and 313 MLP parameters, the MLP overfit. Simpler classifiers
with stronger inductive biases may generalize better to TRACE.

Same train/val/test split as q6_mlp:
  - train: Q6-extended minus 20% val
  - val:   20% of Q6-extended (random seed 0)
  - test:  TRACE pse-test-media fixtures

Eval reports MCC + confusion on each split, persists to JSON.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..dataset import load_split
from .common import (ExperimentResult, confusion, print_result, save_result)


# Each entry: (name, factory) -- factory returns a fresh sklearn-style
# classifier that has fit() and predict_proba() / decision_function().
CLASSIFIERS = [
    ("logistic_regression_l1",
        lambda: LogisticRegression(penalty="l1", solver="liblinear",
                                   C=1.0, max_iter=2000)),
    ("logistic_regression_l2",
        lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=2000)),
    ("logistic_regression_l2_strong",
        lambda: LogisticRegression(penalty="l2", C=0.1, max_iter=2000)),
    ("random_forest_50",
        lambda: RandomForestClassifier(n_estimators=50, max_depth=4,
                                       min_samples_leaf=3, random_state=0)),
    ("random_forest_200",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=6,
                                       min_samples_leaf=2, random_state=0)),
    ("gradient_boost_100",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                            learning_rate=0.05, random_state=0)),
    ("svm_rbf",
        lambda: SVC(kernel="rbf", C=1.0, probability=True, random_state=0)),
    ("svm_linear",
        lambda: SVC(kernel="linear", C=1.0, probability=True, random_state=0)),
    ("knn_5",
        lambda: KNeighborsClassifier(n_neighbors=5)),
    ("gaussian_nb",
        lambda: GaussianNB()),
]


def run(seed: int = 0, val_frac: float = 0.2) -> list[ExperimentResult]:
    print("A1: classifier sweep on hand-engineered features")
    split = load_split()
    X_tr_all, y_tr_all = split["X_train"], split["y_train"]
    X_te, y_te = split["X_test"], split["y_test"]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr_all))
    n_val = max(1, int(round(val_frac * len(X_tr_all))))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
    X_val, y_val = X_tr_all[val_idx], y_tr_all[val_idx]

    # Standardize features (sklearn-wise good practice; matches MLP normalisation)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    results: list[ExperimentResult] = []
    for name, factory in CLASSIFIERS:
        clf = factory()
        clf.fit(X_tr_s, y_tr)
        tr_pred = clf.predict(X_tr_s)
        val_pred = clf.predict(X_val_s)
        te_pred = clf.predict(X_te_s)
        result = ExperimentResult(
            approach="A1",
            method=name,
            config={"seed": seed, "val_frac": val_frac, "scaler": "StandardScaler"},
            train_scores=confusion(y_tr, tr_pred),
            val_scores=confusion(y_val, val_pred),
            test_scores=confusion(y_te, te_pred),
        )
        save_result(result)
        print_result(result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
