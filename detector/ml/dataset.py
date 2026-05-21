"""Dataset for the ours_mlp detector.

Reads MANIFEST.csv, picks fixtures with the requested per-standard label,
extracts features (with on-disk cache), returns (features, label) pairs.

Train/test split philosophy (preserves the adapter-label-isolation
invariant for the *classical* detector that this MLP wraps -- the MLP
itself is explicitly a learned tool that consumes labels by design):

  - TRAIN set: source == "OURS-extended" (synthetic, ground-truth from
    generation params; the MLP NEVER sees TRACE labels during training)
  - TEST set: source startswith "TRACE/pse-test-media" (held out for
    evaluation by the harness scoring path)

The dataset is small (~45 train) so we materialise it fully in memory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .features import extract_features, FEATURE_DIM


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "corpus" / "MANIFEST.csv"
CACHE_DIR = REPO_ROOT / "detector" / "ml" / "feature_cache"


def _cache_key(video_path: Path, profile: str) -> Path:
    h = hashlib.sha256(
        f"{video_path.resolve()}::{profile}::{video_path.stat().st_mtime_ns}"
        .encode()
    ).hexdigest()[:32]
    return CACHE_DIR / f"{h}.npy"


def _cached_features(video_path: Path, profile: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(video_path, profile)
    if cache_path.exists():
        return np.load(cache_path)
    feats = extract_features(video_path, profile=profile)
    np.save(cache_path, feats)
    return feats


def _label_to_int(label_str: str) -> int | None:
    s = label_str.strip().upper()
    if s == "PASS":
        return 0
    if s == "FAIL":
        return 1
    return None


def _iter_fixtures(
    label_column: str,
    source_prefix: str | None = None,
    sources_in: tuple[str, ...] | None = None,
) -> Iterator[tuple[Path, str, int]]:
    """Yield (video_path, fixture_id, label_int) tuples for fixtures
    matching the source filter and with a non-empty label in
    ``label_column``."""
    with MANIFEST.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("type") in ("excluded-tool",):
                continue
            if row.get("path", "").startswith("http"):
                continue
            src = row.get("source", "")
            if sources_in is not None and src not in sources_in:
                continue
            if source_prefix is not None and source_prefix not in src:
                continue
            label = _label_to_int(row.get(label_column, ""))
            if label is None:
                continue
            video_path = (REPO_ROOT / row["path"]).resolve()
            if not video_path.exists():
                continue
            # Skip image fixtures -- the MLP's features assume video.
            if video_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                continue
            yield video_path, row["path"], label


def load_split(label_column: str = "expected_wcag2_2",
               profile: str = "WCAG2.2-SC2.3.1") -> dict[str, np.ndarray]:
    """Return train/test dict with:
        X_train: (N_train, FEATURE_DIM) float32
        y_train: (N_train,) int64
        ids_train: (N_train,) object array of fixture_id strings
        X_test, y_test, ids_test: ... over TRACE fixtures
    """
    def build(filt_kwargs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats_list: list[np.ndarray] = []
        labels: list[int] = []
        ids: list[str] = []
        for video_path, fid, label in _iter_fixtures(label_column, **filt_kwargs):
            try:
                feats_list.append(_cached_features(video_path, profile))
                labels.append(label)
                ids.append(fid)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {fid}: {e}")
        if not feats_list:
            return (np.zeros((0, FEATURE_DIM), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.empty((0,), dtype=object))
        return (np.stack(feats_list).astype(np.float32),
                np.array(labels, dtype=np.int64),
                np.array(ids, dtype=object))

    X_tr, y_tr, ids_tr = build({"sources_in": ("OURS-extended",)})
    X_te, y_te, ids_te = build({"source_prefix": "pse-test-media"})
    return {
        "X_train": X_tr, "y_train": y_tr, "ids_train": ids_tr,
        "X_test":  X_te, "y_test":  y_te, "ids_test":  ids_te,
    }
