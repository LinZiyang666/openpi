"""R11 library-only periodic replay with per-task camera prices."""
from __future__ import annotations
import numpy as np
from exp.offline_search.rounds.r10.recipe.fitting import fit_path, library_boundary
from exp.offline_search.rounds.r10.data import STORE
from exp.offline_search.rounds.r11.knob.calibration import tables
from exp.offline_search.rounds.r11.knob.controller import COSTS
from .build import R12


def per_episode_calls(S, x):
    R, E = S.reps, len(S.eps)
    run = np.zeros((R, E), int)
    cap = int(np.floor(x))+(S.gap[0] < x-np.floor(x))
    M = np.zeros((R, E), int)
    for i in range(len(S.alive)):
        g, alive = S.guard[i][None], S.alive[i][None]
        call = g | ((~g) & (run >= cap))
        run = np.where(call, 0, run+1)
        cap = np.where(call, int(np.floor(x))+(S.gap[i+1] < x-np.floor(x)), cap)
        M += call & alive
    assert M.sum() == S.counts('periodic', x)
    return M


def ir(S, x, camera):
    if all(camera[str(t)] == 'both' for t in range(10)):
        return S.ir('periodic', x)  # exact R11 summation order
    a = np.array([COSTS[S.model][0]*(.5 if camera[str(e.task)] == 'wrist' else 1.) for e in S.eps])
    V = np.array([len(e.steps) for e in S.eps])
    M = per_episode_calls(S, x)
    return float((a[None]*V[None]+(1-a[None])*M).sum() / S.N)


def solve(S, target, camera):
    if all(camera[str(t)] == 'both' for t in range(10)):
        return S.solve('periodic', target)
    lo, hi = 0., 40.
    a, b = ir(S, lo, camera), ir(S, hi, camera)
    if b < a:
        lo, hi, a, b = hi, lo, b, a
    if target <= a:
        return lo, a, 'floor'
    if target >= b:
        return hi, b, 'ceiling'
    for _ in range(26):
        mid = (lo+hi)/2
        if ir(S, mid, camera) < target:
            lo = mid
        else:
            hi = mid
    x = (lo+hi)/2
    return x, ir(S, x, camera), 'ok'


def calibrate(model, suite, size, camera, library, target=.32):
    # tables() verifies all SHA-bound R11 LOEO intermediates. The boundary also
    # refuses queries, derived data and closed-loop outcomes for this new fit.
    sources = [R12.parent/'r11/knob/input_audit.json']
    for source in sources:
        fit_path(source, STORE)
    from pathlib import Path
    with library_boundary(STORE, Path(STORE)/'library'/f'{model}_{suite}'/library) as reads:
        _, _, _, S = tables(model, suite, size)
        x, got, status = solve(S, target, camera)
    return dict(library_only=True, target=target, setting=round(x,4), unrounded_setting=x,
        solve_status=status, pred_IR_lib=got, pred_IR_rounded=ir(S,round(x,4),camera),
        prices={t: COSTS[model][0]*(.5 if mode == 'wrist' else 1.) for t,mode in camera.items()},
        fit_reads=sorted(reads), source='SHA-bound frozen R11 LOEO tables/guard flags; unchanged keyed coins')
