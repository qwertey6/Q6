"""detector/ml -- learned PSE detector(s).

The MLP detector (Option A) consumes summary features from the classical
detector's per-frame trace; the CNN detector (Option C, future) consumes
raw frames.

Public API for harness adapters:
    from detector.ml.infer import predict_mlp_verdict
"""
