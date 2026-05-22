"""Train the q6_mlp detector.

Trains on the Q6-extended subset (45 fixtures) -- the synthetic
fixtures with ground-truth labels derived from generation parameters.
Holds out 20% as validation; reports BCE loss + accuracy + MCC; saves
checkpoint to detector/ml/checkpoints/q6_mlp.pt.

Run from repo root:
    PYTHONPATH=. python3 -m detector.ml.train
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import load_split
from .model import OursMlp, FeatureNormaliser


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "detector" / "ml" / "checkpoints"
CKPT_PATH = CKPT_DIR / "q6_mlp.pt"


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num) / den if den > 0 else 0.0


def report(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    print(f"  [{name:5s}] n={len(y_true):3d}  TP={tp:3d} FP={fp:3d} "
          f"FN={fn:3d} TN={tn:3d}  acc={acc:.3f}  mcc={mcc(tp, fp, fn, tn):+.3f}")
    return {"name": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "mcc": mcc(tp, fp, fn, tn)}


def main(seed: int = 0,
         val_frac: float = 0.2,
         epochs: int = 200,
         lr: float = 3e-3,
         weight_decay: float = 1e-3,
         patience: int = 30) -> dict:
    print("Loading dataset...")
    split = load_split()
    X_tr_all = torch.from_numpy(split["X_train"])
    y_tr_all = torch.from_numpy(split["y_train"]).float()
    X_te = torch.from_numpy(split["X_test"])
    y_te = torch.from_numpy(split["y_test"]).float()
    print(f"  train: {len(X_tr_all)}  test: {len(X_te)}")
    if len(X_tr_all) < 10 or len(X_te) < 10:
        raise RuntimeError("not enough data to train / evaluate.")

    # Reproducible val split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr_all))
    n_val = max(1, int(round(val_frac * len(X_tr_all))))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr = X_tr_all[tr_idx]
    y_tr = y_tr_all[tr_idx]
    X_val = X_tr_all[val_idx]
    y_val = y_tr_all[val_idx]

    # Normaliser computed from training set only (no test leakage)
    normaliser = FeatureNormaliser.from_features(X_tr)

    torch.manual_seed(seed)
    model = OursMlp()
    model.set_normaliser(normaliser)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val).item()
            val_pred = (torch.sigmoid(val_logits) > 0.5).long().numpy()
            val_acc = float((val_pred == y_val.long().numpy()).mean())

        history.append({"epoch": epoch, "train_loss": float(loss.item()),
                        "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"  early stop @ epoch {epoch}; best val_loss = {best_val_loss:.4f}")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print(f"Trained in {train_time:.1f}s, {len(history)} epochs.")
    print("Eval on splits:")
    with torch.no_grad():
        tr_pred = (torch.sigmoid(model(X_tr_all)) > 0.5).long().numpy()
        val_pred = (torch.sigmoid(model(X_val)) > 0.5).long().numpy()
        te_pred = (torch.sigmoid(model(X_te)) > 0.5).long().numpy()
    train_report = report(y_tr_all.long().numpy(), tr_pred, "train")
    val_report   = report(y_val.long().numpy(),    val_pred, "val")
    test_report  = report(y_te.long().numpy(),     te_pred,  "test")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "normaliser_mean": normaliser.mean.tolist(),
        "normaliser_std":  normaliser.std.tolist(),
        "config": {"seed": seed, "val_frac": val_frac, "epochs": epochs,
                   "lr": lr, "weight_decay": weight_decay, "patience": patience},
        "reports": {"train": train_report, "val": val_report, "test": test_report},
    }, CKPT_PATH)
    print(f"Saved checkpoint -> {CKPT_PATH}")
    return {"train": train_report, "val": val_report, "test": test_report}


if __name__ == "__main__":
    main()
