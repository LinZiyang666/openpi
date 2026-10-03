"""Generate evidence, static knob curves, calibration and adaptive stress tests."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import time
import numpy as np

from exp.offline_search.rounds.r11.astra.boundary import HERE, READS, dump, install, sha
from exp.offline_search.rounds.r11.astra.experiment import CELLS, episode_weights
from exp.offline_search.rounds.r11.astra.signals import VISION, probability, packed, dag, calibrate, simulate

SCORES = ('distance', 'predicted_error', 'disagreement', 'gripper_transition',
          'gripper_disagreement', 'chunk_transition')
METHODS = [('distance', 1.), ('predicted_error', 1.), ('disagreement', 1.),
           ('predicted_error', .5), ('disagreement', .5)]


def scalar(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def grouped_mean(a, ep):
    eps, inv, counts = np.unique(ep, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=a) / counts


def value_evidence(a):
    rows, pairs = [], []
    # A causal ten-control cache looks at even steps absent guard vetoes.
    for stride in (1, 2):
        take = a['step'] % stride == 0
        ep, risk = a['ep'][take], a['risk'][take]
        w = episode_weights(ep)
        denom = grouped_mean(risk, ep)
        grip_denom = grouped_mean(a['grip_error'][take], ep)
        rng = np.random.default_rng(20261002)
        boot = rng.integers(0, len(denom), size=(2000, len(denom)))
        captures = {}
        for name in SCORES:
            for dose in (.2, .4, .6):
                p, threshold, tie = probability(a[name][take], dose, weights=w)
                mass = grouped_mean(risk * p, ep)
                fraction = grouped_mean(p, ep)
                cap = mass.mean() / denom.mean()
                gain = mass[boot].mean(1)/denom[boot].mean(1)-fraction[boot].mean(1)
                grip_cap = grouped_mean(a['grip_error'][take]*p, ep).mean()/grip_denom.mean()
                row = dict(method=name, stride=stride, dose=dose, capture=float(cap),
                    random_capture=float(fraction.mean()), lift=float(cap/fraction.mean()),
                    capture_advantage=float(cap-fraction.mean()),
                    advantage_bootstrap_low=float(np.quantile(gain,.025)),
                    advantage_bootstrap_high=float(np.quantile(gain,.975)),
                    grip_capture=float(grip_cap), threshold=threshold, tie_probability=tie)
                rows.append(row)
                captures[name,dose] = mass
        for dose in (.2,.4,.6):
            delta = captures['predicted_error',dose]-captures['disagreement',dose]
            boot_delta = delta[boot].mean(1)/denom[boot].mean(1)
            pairs.append(dict(stride=stride, dose=dose, delta_capture=float(delta.mean()/denom.mean()),
                low=float(np.quantile(boot_delta,.025)), high=float(np.quantile(boot_delta,.975))))
    return rows, pairs


def adaptive_calibrate(pack, score, target, model):
    trials = []
    lo, hi = 0., 1.
    for dose in (lo, hi):
        r = simulate(pack, score, dose, target, model, adaptive=True, seeds=32)
        trials.append((abs(r['owner_ir_mean']-target),dose,r))
    feasible = trials[0][2]['owner_ir_mean'] <= target <= trials[1][2]['owner_ir_mean']
    if feasible:
        for _ in range(16):
            dose = (lo+hi)/2
            r = simulate(pack, score, dose, target, model, adaptive=True, seeds=32)
            trials.append((abs(r['owner_ir_mean']-target),dose,r))
            if r['owner_ir_mean'] < target:
                lo = dose
            else:
                hi = dose
    _, dose, fit = min(trials, key=lambda x: x[0])
    validation = simulate(pack, score, dose, target, model, adaptive=True, seeds=128, seed=20261003)
    return dict(dose=dose, target=target, feasible=feasible, fit=fit, validation=validation,
                calibration_seed=20261002, validation_seed=20261003)


def run_cell(model, suite, size):
    from exp.offline_search.rounds.r10.data import SubsetLibrary, STORE
    started = time.monotonic()
    tag = f'{model}_{suite}_{size}'
    with np.load(HERE / 'data' / tag / 'signals.npz') as z:
        a = {k: np.asarray(z[k]) for k in z.files}
    lib = SubsetLibrary(STORE, f'{model}_{suite}', size)
    np.testing.assert_array_equal(a['row'], lib.rows)
    pack = packed(a, lib)
    np.savez_compressed(HERE / 'data' / tag / 'pack.npz', **pack)
    evidence, pairs = value_evidence(a)
    zeros, ones = np.zeros(len(a['row'])), np.ones(len(a['row']))
    base = dag(pack, zeros, model)
    curve = []
    for name in SCORES:
        for dose in (0., .2, .4, .6, .8, 1.):
            p,t,tie = probability(a[name],dose)
            curve.append(dict(method=name, beta=1., dose=dose, threshold=t,tie_probability=tie,
                              **scalar(dag(pack,p,model))))
    for name in ('predicted_error','disagreement'):
        for dose in (.2,.4,.6,.8):
            p,t,tie=probability(a[name],dose,.5)
            curve.append(dict(method=name, beta=.5,dose=dose,threshold=t,tie_probability=tie,
                              **scalar(dag(pack,p,model))))
    print(tag, 'base', base['owner_ir'], 'curves done', flush=True)
    # Rates/thresholds all cell-wide. Stronger libraries get only one spend level.
    targets = (.25,.32,.40) if size == 50 else (.25,)
    calibrations = []
    for target in targets:
        r=calibrate(pack,a['distance'],target,model,uniform=True)
        calibrations.append(dict(method='random_baseline',**r))
        for name,beta in METHODS:
            r=calibrate(pack,a[name],target,model,beta)
            calibrations.append(dict(method=name,**r))
    print(tag, 'static calibrated', flush=True)
    adaptive = []
    # Adaptive variant shares the predicted-error hybrid's score, beta and eta.
    for target in ((.32,.40) if size==50 else (.25,)):
        r=adaptive_calibrate(pack,a['predicted_error'],target,model)
        static=next(c for c in calibrations if c['method']=='predicted_error' and c['beta']==.5 and c['target']==target)
        scenarios=[]
        for shift,stress in ((0.,False),(.2,False),(-.2,False),(0.,True)):
            shared=dict(seeds=64,shift=shift,stress=stress,seed=20261004)
            s=simulate(pack,a['predicted_error'],static['dose'],target,model,adaptive=False,**shared)
            d=simulate(pack,a['predicted_error'],r['dose'],target,model,adaptive=True,**shared)
            scenarios.append(dict(shift=shift,stress=stress,static=s,adaptive=d))
        adaptive.append(dict(method='adaptive_error_hybrid',**r,scenarios=scenarios))
    # No-progress overlap: compare raw signal precision before/after guard occupancy.
    occupancy = base['occupancy']; guard_occ=base['guard_occupancy']
    flat=np.maximum(pack['idx'],0)
    overlaps=[]
    for name in SCORES[:3]:
        p=probability(a[name],.4)[0][flat]
        denom=(occupancy*p).sum()
        nonguard=occupancy-guard_occ
        overlaps.append(dict(method=name,base_signal_overlap=float((guard_occ*p).sum()/denom),
            nonguard_capture=float((nonguard*p*pack['risk']).sum()/(nonguard*pack['risk']).sum()),
            nonguard_miss_fraction=float((nonguard*p).sum()/nonguard.sum())))
    dump(HERE/'data'/tag/'analysis.json',dict(cell=tag,model=model,suite=suite,size=size,
        timestamp=datetime.now(timezone.utc).isoformat(),rows=len(a['row']),episodes=len(lib.episodes),
        failed_library_episodes=sum(not e['success'] for e in lib.episodes),
        base=scalar(base),no_guard=scalar(dag(pack,zeros,model,False)),ceiling=scalar(dag(pack,ones,model)),
        evidence=evidence,predicted_vs_disagreement=pairs,curve=curve,calibrations=calibrations,
        adaptive=adaptive,overlaps=overlaps,seconds=time.monotonic()-started,
        data_sha256=sha(HERE/'data'/tag/'signals.npz')))
    print(f'DONE analysis {tag} {time.monotonic()-started:.1f}s',flush=True)


def main():
    install()
    p=argparse.ArgumentParser()
    p.add_argument('--cell');p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    jobs=[c for c in CELLS if not args.cell or '_'.join(map(str,c))==args.cell]
    with ThreadPoolExecutor(args.concurrency) as pool:
        futures=[pool.submit(run_cell,*c) for c in jobs]
        for f in futures:
            f.result()
    dump(HERE/'analysis_reads.json',sorted(READS))


if __name__=='__main__':
    main()
