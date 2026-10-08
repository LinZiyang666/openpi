"""Layer-4 recalibration sketch for the half-everywhere base (library-only, R11 periodic ScheduleReplay).
Same frozen R11 LOEO anchor tables, guard flags and keyed coins; only the look price changes per task mode:
non-call look a_t (.152/.148 both; .076/.074 wrist), any call look 1.0 in total. Solves the gap cap per target."""
import json, sys
import numpy as np
from exp.offline_search.rounds.r11.knob.calibration import tables

OUT = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design/knob_sketch.json"
FULL = {"pi05": .152, "groot": .148}


def per_episode_calls(S, x):
    """Vectorized periodic replay (R11 ScheduleReplay.counts, method 'periodic') returning calls per (rep, ep)."""
    R, E = S.reps, len(S.eps)
    run = np.zeros((R, E), int)
    cap = np.floor(x).astype(int) + (S.gap[0] < x - np.floor(x))
    M = np.zeros((R, E), int)
    for i in range(len(S.alive)):
        g = S.guard[i][None]; alive = S.alive[i][None]
        k = (~g) & (run >= cap)
        call = g | k
        run = np.where(call, 0, run + 1)
        newcap = int(np.floor(x)) + (S.gap[i + 1] < x - np.floor(x))
        cap = np.where(call, newcap, cap)
        M += (call & alive)
    return M


def ir(S, x, a_by_task, model):
    M = per_episode_calls(S, x) if x is not None else np.broadcast_to(S.guard.sum(0)[None], (S.reps, len(S.eps)))
    V = np.array([len(e.steps) for e in S.eps]); N = np.array([e.n_slots for e in S.eps])
    a = np.array([a_by_task[e.task] for e in S.eps])
    work = (a[None] * V[None] + (1 - a[None]) * M).sum()
    return work / (N.sum() * S.reps), M.sum() / (V.sum() * S.reps)


def solve(S, target, a_by_task, model):
    lo, hi = 0., 40.
    f = lambda x: ir(S, x, a_by_task, model)[0]
    a, b = f(lo), f(hi)
    if b < a: lo, hi, a, b = hi, lo, b, a
    if target <= a: return lo, a
    if target >= b: return hi, b
    for _ in range(30):
        mid = (lo + hi) / 2
        if f(mid) < target: lo = mid
        else: hi = mid
    x = (lo + hi) / 2
    return x, f(x)


if __name__ == "__main__":
    plans = json.loads(sys.argv[1])   # {"pi05_l10_50": {"wrist": [tasks...]}, ...}
    targets = [.25, .32, .40]
    res = {}
    for cell, spec in plans.items():
        model, suite, size = cell.split("_"); size = int(size)
        a_, pack, eps, S = tables(model, suite, size)
        for label, wrist in spec.items():
            a_by_task = {t: (FULL[model] / 2 if t in wrist else FULL[model]) for t in range(10)}
            base_ir, grate = ir(S, None, a_by_task, model)
            row = dict(base_IR_lib=base_ir, guard_per_look=grate)
            for tg in targets:
                x, got = solve(S, tg, a_by_task, model)
                row[f"gap_{tg}"] = round(x, 4); row[f"IR_{tg}"] = got
                row[f"callrate_{tg}"] = float(per_episode_calls(S, x).sum() / (sum(len(e.steps) for e in S.eps) * S.reps))
            res[f"{cell}|{label}"] = row
            print(cell, label, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
