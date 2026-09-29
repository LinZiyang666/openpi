"""Library-only Q2 allocation; no query outcomes or policy calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

DOSES = np.array([0., .125, .25, .5, 1.])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve(rows, rho, c1, allocation='risk'):
    if allocation not in ('risk', 'uniform'):
        raise ValueError('allocation must be risk or uniform')
    if not np.isfinite(rho) or rho < 0 or not 0 < c1 < 1:
        raise ValueError('require finite nonnegative rho and 0<c1<1')
    tasks = sorted(int(k) for k in rows)
    if not tasks:
        raise ValueError('empty library calibration')
    q, h, a, pi = np.array([[rows[str(t)][k] for k in ('risk', 'h', 'a', 'traffic')] for t in tasks], float).T
    if (not np.isfinite([q, h, a, pi]).all() or np.any(q < 0) or np.any(h <= 0)
            or np.any(a <= 0) or np.any(a > h) or np.any(pi <= 0)):
        raise ValueError('invalid library risks, lengths, or traffic')
    pi /= pi.sum()
    q = np.ones_like(q) if allocation == 'uniform' or not q.max() else np.maximum(q, np.finfo(float).eps*q.max())
    w = q/np.dot(pi, q)
    cost = lambda p: float(np.dot(pi, a*(c1+(1-c1)*p))/np.dot(pi, h))
    floor, ceiling = cost(np.zeros_like(w)), cost(np.ones_like(w))
    target = np.clip(rho, floor, ceiling)
    lo, hi = 0., 1/w.min()
    for _ in range(80):
        mid = (lo+hi)/2
        if cost(np.minimum(1., mid*w)) < target:
            lo = mid
        else:
            hi = mid
    p = np.minimum(1., (lo+hi)/2*w)
    if rho <= floor:
        p[:] = 0.
    if rho >= ceiling:
        p[:] = 1.
    out = {}
    for t, v in zip(tasks, p):
        mix = np.zeros(5)
        if v >= 1:
            mix[-1] = 1.
        else:
            j = max(0, int(np.searchsorted(DOSES, v, side='right'))-1)
            u = (v-DOSES[j])/(DOSES[j+1]-DOSES[j])
            mix[j:j+2] = [1-u, u]
        out[int(t)] = dict(p=float(v), weights=tuple(float(z) for z in mix))
    return dict(tasks=out, rho=float(rho), predicted_IR=cost(p), floor=floor, ceiling=ceiling,
                clamped=bool(rho < floor or rho > ceiling), allocation=allocation, c1=float(c1))


def load_calibration(path, expected_sha256, deployed, ctx):
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError('calibration file SHA mismatch')
    all_rows = json.loads(path.read_text())
    key = f'{ctx.lib_key}/{deployed.name}'
    row = all_rows['libraries'][key]
    if row['commitment_blocks'] != 2:
        raise ValueError('this deployment adapter requires 10-control commitment')
    for filename, expected in row['library_sha256'].items():
        if digest(deployed.dir/filename) != expected:
            raise ValueError(f'calibration belongs to different library: {key}/{filename}')
    # Recompute lengths rather than trusting a stale bank label.
    for task, r in row['tasks'].items():
        _, counts = np.unique(deployed.episode[deployed.task_id == int(task)], return_counts=True)
        if not len(counts) or not np.isclose(counts.mean(), r['h']) or not np.isclose(np.ceil(counts/2).mean(), r['a']):
            raise ValueError('library length calibration mismatch')
    if set(map(int, row['tasks'])) != set(map(int, deployed.tasks())):
        raise ValueError('calibration task support mismatch')
    return row
