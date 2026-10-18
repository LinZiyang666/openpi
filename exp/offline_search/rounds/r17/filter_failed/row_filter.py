"""Select candidates from frozen Cache fits. Never call a fitter or estimator.

Array slices are stored in the existing serving format; no runtime hook, new
pickle class, or worker change is needed. Full payload row IDs stay unchanged.
"""
from __future__ import annotations

import copy
from pathlib import Path
import pickle

import numpy as np


R17_CANDIDATES = ('rows', 'Z', 'z2', 'HD', 'h2')
R10_CANDIDATES = ('rows', 'Z', 'z2', 'HD', 'h2', 'RS', 'rs2', 'V0', 'V1', 'Z0', 'n20')


def _slice(table, mask, fields):
    result = copy.copy(table)
    for name in fields:
        value = getattr(table, name, None)
        if value is not None:
            value = value[mask]
            value.flags.writeable = False
            setattr(result, name, value)
    return result


def r17_candidates(original, kept_episodes):
    result = copy.copy(original)
    result.tasks = {}
    selected = np.isin(original.episode, list(kept_episodes))
    for task, table in original.tasks.items():
        mask = selected[table.rows]
        if not mask.any():
            continue
        if mask.all():
            result.tasks[task] = table
            continue
        out = _slice(table, mask, R17_CANDIDATES)
        if table.early is not None:
            out.early = _slice(table.early, mask, ('n20',))
        result.tasks[task] = out
    return result


def r10_candidates(blob, kept_episodes):
    result = dict(blob)
    method = copy.copy(blob['method'])
    result['method'] = method
    assert method.variant == 'A', 'only the original pure Cache arm is in scope'
    base = copy.copy(method.inner)
    method.inner = base
    base.tasks = {}
    selected = np.isin(base.lib_ep, list(kept_episodes))
    for task, table in blob['method'].inner.tasks.items():
        mask = selected[table.rows]
        if not mask.any():
            continue
        base.tasks[task] = table if mask.all() else _slice(table, mask, R10_CANDIDATES)
    # n_cand/fixed_bytes and every confidence/normalization parameter remain
    # original. n_cand is a reporting field, never used to choose candidates.
    return result


def load_r10(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def save_r10(blob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        pickle.dump(blob, stream, protocol=4)
