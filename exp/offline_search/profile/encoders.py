"""Example encoders for `repr --encode` (templates for later rounds; none of them is a method).

An encoder is `fn(view) -> np.ndarray [len(view), D]`, see repr.RowsView. Library-side statistics must be
computed from `view.lib` (the library view) so that query rows are encoded with library-only information.

  python -m exp.offline_search.profile.repr --reprs "" --encode exp.offline_search.profile.encoders:task_centered_v1
"""
from __future__ import annotations

import numpy as np

_CACHE: dict = {}


def _task_means(lib_view, field: str) -> dict:
    key = (str(lib_view.store.dir), field)
    if key not in _CACHE:
        X = lib_view.get(field).astype(np.float64)
        t = lib_view.task_id
        _CACHE[key] = {int(k): X[t == k].mean(0) for k in np.unique(t)}
    return _CACHE[key]


def _centered(view, field: str) -> np.ndarray:
    mu = _task_means(view.lib, field)
    X = view.get(field).astype(np.float32)
    for k in np.unique(view.task_id):
        m = view.task_id == k
        X[m] -= mu[int(k)].astype(np.float32)
    return X


def task_centered_v0(view) -> np.ndarray:
    """key_v0 minus its per-task library mean (then cosine): removes the shared direction that saturates cos."""
    return _centered(view, "key_v0")


def task_centered_v1(view) -> np.ndarray:
    return _centered(view, "key_v1")


def concat_centered(view) -> np.ndarray:
    """Task-centered v0 | v1, each scaled to unit RMS over the library, one cosine for both views."""
    out = []
    for f in ("key_v0", "key_v1"):
        X = _centered(view, f)
        key = (str(view.lib.store.dir), f, "rms")
        if key not in _CACHE:
            _CACHE[key] = float(np.sqrt(np.mean(_centered(view.lib, f) ** 2)))
        out.append(X / max(_CACHE[key], 1e-12))
    return np.concatenate(out, 1)
