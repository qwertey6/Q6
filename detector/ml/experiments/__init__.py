"""ML experiment registry. Each module here implements one approach
from the A1-A6 sweep; run_all.py executes them all and writes results
to results/ml_experiments/.

Approaches:
  A1 = sklearn_classifiers   different classifiers on same features
  A2 = kfold_trace           stratified K-fold CV on TRACE
  A3 = frame_level           per-frame labels from generation params
  A4 = tsfresh_features      automatic time-series feature extraction
  A5 = threshold_tuning      ROC-based threshold selection
  A6 = stacking              ML predicts classical's errors
"""
