"""A4: tsfresh-extracted features.

flickerfilter's approach: extract a large automatic feature set from
per-frame time series, feed to a regularized linear classifier.

We use the same minimal-feature-set as tsfresh's MinimalFCParameters
(~10-20 features per channel), applied to three channels:
lum_transitions, red_transitions, flash_area. Each fixture gets
~30-60 features total. Larger feature space than A1's 10 but still
tractable for 45 training samples.

Two evaluation modes:
  - train on Q6-extended, test on TRACE (apples-to-apples vs A1)
  - K-fold CV on TRACE (apples-to-apples vs A2)
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# tsfresh issues many warnings on tiny inputs; the per-frame traces are
# short so silencing them keeps the bench output readable.
warnings.filterwarnings("ignore", category=Warning)

from tsfresh import extract_features as tsfresh_extract
from tsfresh.feature_extraction.settings import MinimalFCParameters

from ..dataset import _iter_fixtures
from .common import (ExperimentResult, confusion, print_result, save_result)


def _per_fixture_dataframe(video_paths_and_labels):
    """Return a long-format DataFrame [id, time, channel_lum, channel_red,
    channel_area] suitable for tsfresh, plus a parallel label vector
    indexed by id."""
    from detector import analyze, CV2_CC_BACKEND  # type: ignore
    rows = []
    labels = {}
    ids = []
    for video_path, fid, label in video_paths_and_labels:
        try:
            res = analyze(video_path, profile="WCAG2.2-SC2.3.1",
                          cc_backend=CV2_CC_BACKEND)
        except Exception as e:
            print(f"  skip {fid}: {e}")
            continue
        if not res.per_frame:
            continue
        for f in res.per_frame:
            rows.append({"id": fid, "time": f.frame,
                         "lum": float(f.lum_transitions),
                         "red": float(f.red_transitions),
                         "area": float(f.flash_area)})
        labels[fid] = label
        ids.append(fid)
    df = pd.DataFrame(rows)
    return df, labels, ids


def _extract_tsfresh_features(df):
    """Run tsfresh on a long-format dataframe. Returns wide-format
    (n_ids, n_features) DataFrame indexed by id."""
    feats = tsfresh_extract(
        df, column_id="id", column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=0, disable_progressbar=True,
    )
    # tsfresh returns NaN for some features (single-value series). Fill 0.
    feats = feats.fillna(0.0)
    return feats


def _materialise():
    """Extract tsfresh features for all Q6-extended + TRACE fixtures."""
    print("  extracting per-frame traces and tsfresh features...")
    train_df, train_labels, train_ids = _per_fixture_dataframe(
        _iter_fixtures("expected_wcag2_2", sources_in=("Q6-extended",))
    )
    test_df, test_labels, test_ids = _per_fixture_dataframe(
        _iter_fixtures("expected_wcag2_2", source_prefix="pse-test-media")
    )

    combined = pd.concat([train_df, test_df], ignore_index=True)
    feats = _extract_tsfresh_features(combined)

    train_feats = feats.loc[train_ids].values.astype(np.float32)
    test_feats  = feats.loc[test_ids].values.astype(np.float32)
    train_y = np.array([train_labels[i] for i in train_ids], dtype=np.int64)
    test_y  = np.array([test_labels[i]  for i in test_ids],  dtype=np.int64)
    print(f"  features extracted: {feats.shape[1]} per fixture")
    print(f"  train: {train_feats.shape}, test: {test_feats.shape}")
    return train_feats, train_y, test_feats, test_y


def run(seed: int = 0) -> list[ExperimentResult]:
    print("A4: tsfresh features")
    X_tr, y_tr, X_te, y_te = _materialise()
    results: list[ExperimentResult] = []

    # Mode 1: train on OURS, test on TRACE (vs A1)
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(penalty="l2", C=0.1, max_iter=5000)
    clf.fit(scaler.transform(X_tr), y_tr)
    te_pred = clf.predict(scaler.transform(X_te))
    tr_pred = clf.predict(scaler.transform(X_tr))
    r1 = ExperimentResult(
        approach="A4",
        method="tsfresh_logistic_l2_OURS_to_TRACE",
        config={"feature_count": X_tr.shape[1], "C": 0.1},
        train_scores=confusion(y_tr, tr_pred),
        test_scores=confusion(y_te, te_pred),
        notes="train Q6-extended, test TRACE -- vs A1 baselines",
    )
    save_result(r1); print_result(r1); results.append(r1)

    # Mode 2: K-fold CV on TRACE (vs A2)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_preds = np.zeros_like(y_te)
    fold_mccs = []
    for tr_idx, val_idx in skf.split(X_te, y_te):
        scaler_k = StandardScaler().fit(X_te[tr_idx])
        clf_k = LogisticRegression(penalty="l2", C=0.1, max_iter=5000)
        clf_k.fit(scaler_k.transform(X_te[tr_idx]), y_te[tr_idx])
        pred = clf_k.predict(scaler_k.transform(X_te[val_idx]))
        all_preds[val_idx] = pred
        fold_mccs.append(confusion(y_te[val_idx], pred)["mcc"])
    r2 = ExperimentResult(
        approach="A4",
        method="tsfresh_logistic_l2_kfold_TRACE",
        config={"feature_count": X_te.shape[1], "C": 0.1, "n_folds": 5},
        val_scores={"per_fold_mcc": [float(x) for x in fold_mccs],
                     "mean_mcc_per_fold": float(np.mean(fold_mccs))},
        test_scores=confusion(y_te, all_preds),
        notes="K-fold CV on TRACE -- vs A2 baselines",
    )
    save_result(r2); print_result(r2); results.append(r2)
    return results


if __name__ == "__main__":
    run()
