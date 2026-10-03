"""Task-agnostic score quantiles and a minimal guard/cadence accounting model."""
from __future__ import annotations
import numpy as np

VISION = {'pi05': .152, 'groot': .148}


def threshold(score, fraction, weights=None):
    """Miss high scores, randomize exact ties; no task-dependent state."""
    score = np.asarray(score, float)
    if not np.isfinite(score).all() or not 0 <= fraction <= 1:
        raise ValueError('finite scores and a fraction in [0,1] required')
    w = np.ones(len(score)) if weights is None else np.asarray(weights, float)
    order = np.argsort(-score, kind='stable')
    val, ww = score[order], w[order]
    if fraction == 0:
        return float(val[0]), 0.
    if fraction == 1:
        return float(val[-1]), 1.
    target = fraction * w.sum()
    j = min(int(np.searchsorted(np.cumsum(ww), target, side='left')), len(val)-1)
    t = val[j]
    tie = np.clip((target - w[score > t].sum()) / w[score == t].sum(), 0, 1)
    return float(t), float(tie)


def probability(score, fraction, beta=1., weights=None):
    """beta=1 threshold; beta=.5 half uniform floor, half targeted lottery."""
    t, tie = threshold(score, fraction, weights)
    p = (1-beta) * fraction + beta * ((score > t) + (score == t) * tie)
    return p, t, tie


def rank_bounds(score):
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    hi = np.cumsum(counts) / len(score)
    lo = hi - counts / len(score)
    return lo[inverse], hi[inverse]


def packed(a, lib):
    """Pack pure-policy B episodes; progress comes from held-out retrieval donors."""
    eps, counts = np.unique(a['ep'], return_counts=True)
    H = int(counts.max())
    idx = np.full((H, len(eps)), -1, int)
    length = np.empty(len(eps), int)
    for j, e in enumerate(eps):
        r = np.flatnonzero(a['ep'] == e)
        r = r[np.argsort(a['step'][r])]
        np.testing.assert_array_equal(a['step'][r], np.arange(len(r)))
        idx[:len(r), j] = r
        length[j] = len(r)
    top = a['top'][np.maximum(idx, 0)]
    return dict(idx=idx, length=length, episodes=eps,
                prog=np.asarray(lib.progress)[top],
                den=np.maximum(np.asarray(lib.ep_len)[top] - 1, 1),
                risk=a['risk'][np.maximum(idx, 0)])


def dag(pack, probabilities, model, guard=True):
    """Exact expectation on exogenous B states, NOT closed-loop predictions.

    State: previous look lag (1/2), previous no-progress span capped at 2.
    No-progress delta*current-donor-length <= .5, guard span>=2, hit veto span>0.
    Policy call always commits one tail. Guard remains mandatory and unmodified.
    """
    idx, length, prog, den = [pack[k] for k in ('idx', 'length', 'prog', 'den')]
    H, E = idx.shape
    P = probabilities[np.maximum(idx, 0)]
    reach = np.zeros((H+2, 2, 3, E))
    reach[0, 0, 0] = 1
    looks, misses, guards, knobs, overlaps, captured = [np.zeros(E) for _ in range(6)]
    occupancy = np.zeros_like(P)
    guard_occupancy = np.zeros_like(P)
    for i in range(H):
        alive = i < length
        for lag in (1, 2):
            advancement = np.ones(E, bool) if i == 0 else ((prog[i] - prog[max(i-lag, 0)]) * den[i] > .5)
            for previous_span in range(3):
                mass = reach[i, lag-1, previous_span] * alive
                span = np.where(advancement, 0, min(2, previous_span + lag)) if guard else np.zeros(E, int)
                g = span >= 2
                p = np.where(g, 1., P[i])
                occupancy[i] += mass
                guard_occupancy[i] += mass * g
                looks += mass; misses += mass * p; guards += mass * g
                knobs += mass * (~g) * P[i]; overlaps += mass * g * P[i]
                captured += mass * p * pack['risk'][i]
                for s in range(3):
                    selected = span == s
                    # A policy tail is mandatory after each miss; no progress veto.
                    reach[i+2, 1, s] += mass * p * selected
                    # A cache hit with nonzero no-progress span vetoes its tail.
                    h = 2 if s == 0 else 1
                    reach[i+h, h-1, s] += mass * (1-p) * selected
    a = VISION[model]; b = 1-a
    N = length.sum()
    return dict(owner_ir=float((a*looks.sum()+b*misses.sum())/N),
        v=float(looks.sum()/N), m=float(misses.sum()/N),
        miss_per_look=float(misses.sum()/looks.sum()), guard_per_look=float(guards.sum()/looks.sum()),
        knob_per_look=float(knobs.sum()/looks.sum()), overlap_per_look=float(overlaps.sum()/looks.sum()),
        looks=float(looks.sum()), misses=float(misses.sum()), decisions=int(N),
        captured=float(captured.sum()), occupancy=occupancy, guard_occupancy=guard_occupancy,
        per_episode_ir=(a*looks+b*misses)/length)


def calibrate(pack, score, target, model, beta=1., uniform=False):
    lo, hi = 0., 1.
    floor = dag(pack, np.zeros(len(score)), model)['owner_ir']
    ceiling = dag(pack, np.ones(len(score)), model)['owner_ir']
    if target <= floor:
        hi = 0.
    elif target >= ceiling:
        lo = 1.
    else:
        for _ in range(30):
            dose = (lo+hi)/2
            p = np.full(len(score), dose) if uniform else probability(score, dose, beta)[0]
            if dag(pack, p, model)['owner_ir'] < target:
                lo = dose
            else:
                hi = dose
    dose = (lo+hi)/2
    p, t, tie = probability(score, dose, beta)
    if uniform:
        p[:] = dose
    result = dag(pack, p, model)
    return dict(dose=dose, threshold=t, tie_probability=tie, beta=beta,
                random_floor=(1-beta)*dose if not uniform else dose,
                target=target, feasible=floor <= target <= ceiling, floor=floor, ceiling=ceiling,
                **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)})


def simulate(pack, score, dose, target, model, *, adaptive=False, beta=.5,
             seeds=32, eta=.2, guard=True, shift=0., stress=False, seed=20261002):
    """Sequential B-state replay, common seeded draws; no environment or policy."""
    idx, length = pack['idx'], pack['length']
    H, E = idx.shape
    low, high = rank_bounds(score)
    low = low[np.maximum(idx, 0)]
    high = high[np.maximum(idx, 0)]
    p_lo, p_hi = np.clip(low+shift, 0, 1), np.clip(high+shift, 0, 1)
    rng = np.random.default_rng(seed)
    q = np.full((seeds, E), dose)
    last_look = np.zeros((seeds, E), int)
    last_cost = np.zeros((seeds, E))
    last_prog = np.zeros((seeds, E))
    span = np.zeros((seeds, E), int)
    next_look = np.zeros((seeds, E), int)
    V, M, G, capture = [np.zeros((seeds, E)) for _ in range(4)]
    sat = np.zeros((seeds, E))
    a, b = VISION[model], 1-VISION[model]
    for i in range(H):
        active = (next_look == i) & (i < length[None])
        if adaptive and i:
            q = np.where(active, np.clip(q - eta*(last_cost-target*(i-last_look))/b, 0, 1), q)
        advance = (pack['prog'][i][None] - last_prog) * pack['den'][i][None] > .5
        if stress:
            advance[:, :] = (i % 8) < 4  # explicit synthetic persistent-stall scenario
        newspan = np.where(advance | (i == 0), 0, np.minimum(2, span+i-last_look)) if guard else np.zeros_like(span)
        g = active & (newspan >= 2)
        # Randomized rank inside a score's empirical CDF tie interval.
        ranks = p_lo[i][None] + rng.random((seeds, E)) * (p_hi[i]-p_lo[i])[None]
        prob = (1-beta)*q + beta*(ranks > 1-q)
        hit_knob = rng.random((seeds, E)) < prob
        miss = active & (g | hit_knob)
        V += active; M += miss; G += g
        capture += miss * pack['risk'][i][None]
        sat += active & ((q == 0) | (q == 1))
        h = np.where(miss | (newspan == 0), 2, 1)
        next_look = np.where(active, i+h, next_look)
        last_look = np.where(active, i, last_look)
        last_cost = np.where(active, a+b*miss, last_cost)
        last_prog = np.where(active, pack['prog'][i][None], last_prog)
        span = np.where(active, newspan, span)
    ir = (a*V.sum(1)+b*M.sum(1))/length.sum()
    return dict(owner_ir_mean=float(ir.mean()), seed_ir_min=float(ir.min()), seed_ir_max=float(ir.max()),
        seed_ir_std=float(ir.std()), miss_per_look=float(M.sum()/V.sum()),
        guard_per_look=float(G.sum()/V.sum()), saturation=float(sat.sum()/V.sum()),
        captured_mean=float(capture.sum(1).mean()), seeds=seeds, eta=eta,
        shift=shift, synthetic_stall=stress)
