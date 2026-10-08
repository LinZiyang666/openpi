"""Reproduce frozen Opus B-dev replay and audit the serving feature extractor.

guard_final.json is the risk-only precursor; H2's reference is guard_m2.json.
The published tables normalize by each dev arm median. Serving uses the library
median required by SPEC; this discrepancy is quantified, never hidden or refit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np

from exp.offline_search.closed_loop.guard import FEATURES, feature_matrix, risk_score, h2_flags
from exp.offline_search.closed_loop.look_recording import read_looks, validate_root
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, metrics
from exp.offline_search.rounds.r12.opus.tools.guard_final import risk_fn
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load, score
from .build import HERE, R12

OUT = R12/'opus/out'
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
TOL = 1e-12


def _first(z, flags):
    first = []
    for uid in np.unique(z['uid']):
        js = np.flatnonzero(z['uid'] == uid)
        js = js[np.argsort(z['step'][js])]
        if z['fail'][js[0]] and z['pure_ok'][js[0]]:
            called = js[flags[js]]
            first.append(int(z['step'][called[0]]) if len(called) else 999)
    return float(np.median(first)) if first else np.nan


def _schedule(z, d, sig, calls, model):
    idx = {(u, int(s)): i for i, (u, s) in enumerate(zip(d['uid'], d['step']))}
    by = defaultdict(list)
    for j, uid in enumerate(z['uid']):
        by[uid].append(j)
    half, delta = [], []
    knob = z['called'] & (z['os_reason'] == 62)
    for js in by.values():
        js.sort(key=lambda j: z['step'][j])
        lag = 0
        for k, j in enumerate(js):
            lag = 0 if not k else lag+2
            i = idx[(z['uid'][j], int(z['step'][j]))]
            if not k or calls[j] or knob[j] or lag > 8:
                lag = 0; half.append(False); delta.append(0.); continue
            L = min(8, max(2, lag))
            accepted = score(d['S_W'][i], d[f'S_WL{L}'][i], sig) <= .10
            half.append(accepted)
            if accepted:
                sh = d['shadow'][i]
                delta.append(float(np.sqrt(np.mean(((sh-d[f'S_WL{L}'][i])[:, :6]/sig[:6])**2))
                    - np.sqrt(np.mean(((sh-d['S_F'][i])[:, :6]/sig[:6])**2))) if np.isfinite(sh).all() else np.nan)
            else:
                lag = 0; delta.append(0.)
    half = float(np.mean(half))
    return dict(half=half, dIR_vision=-.505*(.076 if model == 'pi05' else .074)*half,
                dq_retrieval=float(np.mean(delta)) if np.isfinite(delta).all() else None)


def _check_metrics(actual, expected, allowance=0.):
    maximum = 0.
    for key in expected:
        if key not in actual:
            continue
        a, b = actual[key], expected[key]
        if a is None or b is None:
            if a != b:
                raise AssertionError((key, a, b))
            continue
        maximum = max(maximum, abs(a-b))
        np.testing.assert_allclose(a, b, atol=TOL+allowance, rtol=TOL, err_msg=key)
    return maximum


def _raw_features(z, d, meta, limit):
    model, suite, size = meta['model'], meta['suite'], meta['size']
    if size != 50 or 'devknob' in meta['arm']:
        return None
    with (HERE/'artifacts'/f'{model}_{suite}_{size}'/'h2.pkl').open('rb') as stream:
        m = pickle.load(stream)['method']
    # Only the normalization is replaced to audit the published replay. The
    # production artifact retains its library-only median and frozen constants.
    m.risk_fit = dict(m.risk_fit, distance_median=float(np.median(d['W_d1'])))
    zi = {(u, int(s)): i for i, (u, s) in enumerate(zip(z['uid'], z['step']))}
    di = {(u, int(s)): i for i, (u, s) in enumerate(zip(d['uid'], d['step']))}
    inverse = {int(row): j for j, row in enumerate(m.row_subset)}
    Xref = feature_matrix(z)
    histories = defaultdict(list)
    looked, eligible, maximum, risk_delta = 0, 0, 0., 0.
    distance_delta, real_risk_delta, top_agree = 0., 0., 0
    root = RUNS/meta['root']
    validate_root(root)
    best = {}
    directories = sorted((root/'runs'/meta['arm']).glob('server_*/looks'))
    for directory in directories:
        for manifest in directory.glob('manifest_*.json'):
            record = json.loads(manifest.read_text())
            if record['init_pool'] != 'B' or not record['dev']:
                raise ValueError('raw replay refuses non-B looks')
            for shard in record['shards']:
                best[shard['uid']] = max(best.get(shard['uid'], 0), shard['attempt'])
    for directory in directories:
        for lk in read_looks(directory):
            uid, step = lk['meta']['uid'], int(lk['step'])
            if lk['meta']['attempt'] != best[uid] or (uid, step) not in zi:
                continue
            hist = histories[uid]
            if hist and step <= hist[-1]['step']:
                continue
            i, j = zi[(uid, step)], di[(uid, step)]
            actions = np.zeros((step, 10, 7), np.float64)
            if hist and step-hist[-1]['step'] == 2:
                prev = hist[-1]
                actions[prev['step'], :5] = prev['action'][:5]
                actions[prev['step']+1, :5] = prev['action'][5:10]
            m._guard_history = [dict(step=r['step'], rs=r['rs'], served_prog=r['served_prog']) for r in hist[-3:]]
            m._pending_guard = None
            q = NS(step=step, task_id=int(lk['task_id']), rs=lk['rs'], key_v1=lk['key_v1'],
                   hist_a_exec=actions, prev_hit=True)
            actual = m._prepare_guard(q)
            m._pending_guard = None
            W = dict(rows=np.array([inverse[int(d['W_top1'][j])]]), distance=float(d['W_d1'][j]))
            with patch.object(m, '_query_metric', return_value=W):
                pending = m._prepare_guard(q)
            if step >= 4 and len(hist) >= 2:
                X = np.asarray(pending['features'])
                maximum = max(maximum, float(np.max(np.abs(X-Xref[i]))))
                np.testing.assert_allclose(X, Xref[i], rtol=TOL, atol=TOL)
                risk_delta = max(risk_delta, abs(pending['scalars']['risk']-float(risk_score(Xref[i], m.risk_fit['coef'])[0])))
                distance_delta = max(distance_delta, abs(actual['W']['distance']-W['distance']))
                real_risk_delta = max(real_risk_delta, abs(actual['scalars']['risk']-pending['scalars']['risk']))
                top_agree += int(actual['W']['rows'][0] == W['rows'][0])
                eligible += 1
            hist.append(dict(step=step, rs=np.asarray(lk['rs'][:8], np.float64),
                             action=np.asarray(lk['action'][:10, :7], np.float64), served_prog=float(d['F_top1_prog'][j])))
            looked += 1
            if looked >= limit:
                break
        if looked >= limit:
            break
    if not eligible:
        raise AssertionError('no eligible raw feature audits')
    assert top_agree/eligible >= .99, 'wrist retrieval top-1 drift'
    assert real_risk_delta <= 1e-6, 'serving wrist risk exceeds numerical reproduction tolerance'
    return dict(raw_B_looks=looked, eligible_feature_checks=eligible, max_feature_abs=maximum,
                max_risk_abs=risk_delta, real_wrist_max_distance_abs=distance_delta,
                real_wrist_max_risk_abs=real_risk_delta, real_wrist_top1_agreement=top_agree/eligible,
                real_wrist_tolerance=1e-6, wrist_retrieval='real serving regime-2 W checked; injected reference isolates history')


def main():
    """Compare published replay exactly and record serving-normalization drift."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-looks', type=int, default=128)
    args = ap.parse_args()
    fit = json.loads((HERE/'risk_model.json').read_text())
    coefref = json.loads((OUT/'guard_lib.json').read_text())
    final = json.loads((OUT/'guard_final.json').read_text())
    frozen = json.loads((OUT/'guard_m2.json').read_text())
    devfiles = {p.stem.split('__')[-1]: p for p in (OUT/'dev').glob('*.npz')}
    reports = {}
    for path in sorted((OUT/'guard').glob('*.npz')):
        arm = path.stem
        model = 'pi05' if '_pi05_' in arm else 'groot'
        with np.load(path, allow_pickle=False) as data:
            z = dict(data)
        d, meta = load(devfiles[arm])
        validate_root(RUNS/meta['root'])
        coef = fit['models'][model]['coef']
        X = feature_matrix(z)
        np.testing.assert_array_equal(X, X_of(z, FEATURES))
        risk = risk_score(X, coef)
        expected_risk = risk_fn(coefref[f'{model}|coef'])(X)
        np.testing.assert_allclose(risk, expected_risk, rtol=TOL, atol=TOL)
        base = arm.replace('devknob', 'dev').replace('_periodic_ir32', '')
        theta = frozen['thetas'][base]
        calls = h2_flags(z['step'], z['W_adv'], risk, theta['theta_a'], theta['theta_b'])
        expected_calls = h2_flags(z['step'], z['W_adv'], expected_risk, theta['theta_a'], theta['theta_b'])
        diff = calls != expected_calls
        if diff.any():
            distance = np.minimum(abs(risk-theta['theta_a']), abs(risk-theta['theta_b']))
            assert np.all(distance[diff] <= TOL), 'H2 decisions differ beyond numerical threshold tolerance'
        # At exact calibration ties use the reference's inclusivity for aggregate
        # reproduction, and expose any sub-tolerance tie disagreements separately.
        reproduced = dict(metrics(z, expected_calls), first=_first(z, expected_calls))
        error = _check_metrics(reproduced, frozen[arm]['H2'])
        sigma = np.asarray(meta['sig_r'])
        schedule = _schedule(z, d, sigma, expected_calls, model)
        error = max(error, _check_metrics(schedule, frozen[arm]['M2']))
        thr = final[f'{model}|theta_per_cell'][base]
        glib = (expected_risk >= thr) & (z['step'] >= 4)
        error = max(error, _check_metrics(metrics(z, glib), final[arm]['Glib']))
        error = max(error, _check_metrics(_schedule(z, d, sigma, glib, model), final[arm]['M2']))
        cell = fit['cells'].get(f"{model}_{meta['suite']}_{meta['size']}")
        drift = None
        if cell is not None:
            index = {(u, int(s)): i for i, (u, s) in enumerate(zip(d['uid'], d['step']))}
            table = dict(z, W_d1rel=d['W_d1'][np.array([index[(u, int(s))]
                for u, s in zip(z['uid'], z['step'])])]/cell['distance_median'])
            prod_risk = risk_score(feature_matrix(table), coef)
            prod_calls = h2_flags(z['step'], z['W_adv'], prod_risk, theta['theta_a'], theta['theta_b'])
            with (HERE/'artifacts'/f"{model}_{meta['suite']}_{meta['size']}"/'h2.pkl').open('rb') as stream:
                cache_sigma = pickle.load(stream)['method'].inner.base.sig
            drift = dict(library_median=cell['distance_median'], dev_arm_median=float(np.median(d['W_d1'])),
                published_h2_rate=float(expected_calls.mean()), library_normalized_h2_rate=float(prod_calls.mean()),
                decision_disagreements=int(np.count_nonzero(prod_calls != expected_calls)),
                runtime_M2_half=_schedule(z, d, np.asarray(cache_sigma), prod_calls, model)['half'],
                runtime_consistency_sigma='selected-cache sigma (SPEC); published replay used parent-library sig_r')
        raw = _raw_features(z, d, meta, args.raw_looks) if args.raw_looks else None
        reports[arm] = dict(looks=len(risk), max_risk_abs=float(np.max(abs(risk-expected_risk))),
            threshold_tie_disagreements=int(diff.sum()), max_reference_metric_abs=error,
            H2=reproduced, M2=schedule, raw_feature_audit=raw, normalization_drift=drift)
        print(json.dumps(dict(arm=arm, looks=len(risk), max_metric_abs=error, raw=raw, normalization=drift)), flush=True)
    (HERE/'REPLAY.json').write_text(json.dumps(dict(tolerance=TOL, arms=reports, passed=True,
        reference='guard_m2.json H2/M2 and guard_final.json risk-only precursor',
        serving_normalization='library median; published replay uses dev-arm median (drift reported)', launches=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
