"""Fail-closed CPU comparison of numerical stages and cache decisions."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from artifacts import load_tree, sha, tensor_leaves, write_json


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or not a.size:
        raise ValueError('tensor shape/dtype/empty mismatch')
    if a.dtype.kind not in 'biuf':
        raise ValueError('unsupported comparison dtype')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite tensors cannot pass')
    x, y = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(x - y)
    return dict(max_abs=float(d.max()), p99_abs=float(np.quantile(d, .99)),
        mean_abs=float(d.mean()), relative_max=float((d / np.maximum(np.abs(x), 1e-6)).max()),
        relative_l2=float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(x.ravel()), 1e-6)),
        exact=a.tobytes() == b.tobytes())


def read_replay(root):
    root = Path(root)
    meta = json.loads((root / 'REPLAY.json').read_text())
    if meta['status'] != 'COMPLETE' or not meta['rows']:
        raise ValueError('incomplete replay')
    rows = {}
    for entry in meta['rows']:
        key = entry['input'], entry['repeat']
        if key in rows or Path(entry['file']).name != entry['file']:
            raise ValueError('duplicate/unsafe replay row')
        path = root / entry['file']
        if sha(path.with_suffix('.json')) != entry['sha256']:
            raise ValueError('replay row changed')
        rows[key] = load_tree(path)
        expected = entry.get('actual_nfe') if meta['model'] == 'cosmos' else meta['sampler']['num_steps'] - 1
        if expected is None or len(rows[key]['stages'].get('denoise', {})) != expected:
            raise ValueError('incomplete denoising-step coverage')
    return meta, rows


def budgets(model, name):
    # Proposed screening caps, NOT SR certificates. Only the two FW values and
    # DP wire-action value have existing RUN_CONFIGS measurement support.
    if name.startswith('/noise/') or 'mask' in name:
        return 0., 0.
    if model == 'fastwam' and name == '/committed_normalized':
        return .0078125, .01
    if model == 'fastwam' and (name.startswith('/denoise/') or name == '/final_normalized'):
        return .015625, .01
    if model == 'dp' and name in ('/final_wire', '/committed_wire'):
        return .000345707, .001
    if name.startswith(('/stage1/', '/stage2/', '/capture/')) and model != 'dp':
        return .03125, .002
    return .001, .001


def leaves(row, commit):
    result = dict(tensor_leaves(row['stages']))
    for source, target in [('final_normalized', 'committed_normalized'), ('final_wire', 'committed_wire')]:
        if source not in row['stages']:
            raise ValueError('missing final action tensor')
        result['/' + target] = row['stages'][source][:commit]
    for category in ('stage1', 'stage2', 'noise', 'denoise'):
        # K1 may have no intermediate denoise state; final action is its one update.
        if category not in row['stages'] or (category != 'denoise' and not row['stages'][category]):
            raise ValueError('missing stage category ' + category)
    return result


def cache_report(left, right):
    keys = ('hit', 'top1', 'corrector', 'stall')
    present = [r['cache'] is not None for r in list(left.values()) + list(right.values())]
    if not all(present):
        return dict(status='NOT_MEASURED', reason='missing fitted-cache execution', eligible=False)
    primary = [key for key in left if key[1] == 0]
    # Empirical boundary bands: comparison errors and same-GPU scalar drift.
    boundary = {}
    for scalar, margin in [('score', 'score_margin'), ('neighbour_margin', 'neighbour_margin'),
                           ('corrector_ratio', 'corrector_margin'), ('stall_score', 'stall_margin')]:
        observed = []
        for key in left:
            l, r = left[key]['cache'], right[key]['cache']
            lv, rv = l.get(scalar), r.get(scalar)
            if lv is not None and rv is not None:
                observed.append(abs(float(lv) - float(rv)))
        repeat_drift = []
        for rows in (left, right):
            for key in rows:
                v, v0 = rows[key]['cache'].get(scalar), rows[(key[0], 0)]['cache'].get(scalar)
                if v is not None and v0 is not None:
                    repeat_drift.append(abs(float(v) - float(v0)))
        boundary[margin] = dict(epsilon=max(1e-6, 2*max(observed, default=0.), 5*max(repeat_drift, default=0.)),
            scalar_coverage=len(observed), cross_max=max(observed, default=None),
            same_gpu_max=max(repeat_drift, default=None))
    metrics, mismatches = {}, []
    for field in keys:
        unknown = [k for k in left if left[k]['cache'].get(field) is None or right[k]['cache'].get(field) is None]
        bad = [k for k in primary if left[k]['cache'].get(field) != right[k]['cache'].get(field)]
        floor = 0
        total_repeat = 0
        for rows in (left, right):
            for key in rows:
                if key[1] != 0:
                    floor += rows[key]['cache'].get(field) != rows[(key[0], 0)]['cache'].get(field)
                    total_repeat += 1
        metrics[field] = dict(n=len(primary), missing=len(unknown), mismatches=len(bad),
            agreement=1 - len(bad) / len(primary), same_gpu_repeat_disagreement=floor / max(total_repeat, 1),
            # Exact one-sided 95% bound for zero failures; Wilson for nonzero.
            upper95_disagreement=(1 - .05 ** (1 / len(primary))) if not bad else wilson_upper(len(bad), len(primary)))
        for key in bad:
            l, r = left[key]['cache'], right[key]['cache']
            margin_name = {'top1':'neighbour_margin','hit':'score_margin','corrector':'corrector_margin','stall':'stall_margin'}[field]
            margin = l.get(margin_name)
            band = boundary[margin_name]
            mismatches.append(dict(input=key[0], field=field, left=l.get(field), right=r.get(field),
                margin=margin, epsilon=band['epsilon'], near_threshold='unknown' if margin is None or not band['scalar_coverage'] else abs(margin) <= band['epsilon'],
                corrected_action=difference(l['corrected_action'], r['corrected_action'])
                    if 'corrected_action' in l and 'corrected_action' in r else None))
    missing = any(v['missing'] for v in metrics.values())
    passes = all(v['agreement'] >= .999 and v['same_gpu_repeat_disagreement'] <= .001 for v in metrics.values())
    # Conservative smoke gate: no unexplained decision changes, even at boundaries.
    eligible = not missing and passes and not mismatches and all(v['same_gpu_repeat_disagreement'] == 0 for v in metrics.values())
    return dict(status='SMOKE_AGREEMENT' if eligible else 'INCOMPLETE' if missing else 'MISMATCH',
        eligible=eligible, statistical_certificate=False,
        independent_query_count_bound_met=eligible and all(v['upper95_disagreement'] <= .001 for v in metrics.values()),
        metrics=metrics, mismatches=mismatches,
        boundary_bands=boundary,
        margin_coverage=sum(left[k]['cache'].get('score_margin') is not None for k in primary),
        limitation='query independence/history coverage unattested; initial-only cohorts do not cover stalls; missing margins never count as far from threshold')


def wilson_upper(k, n):
    z = 1.6448536269514722
    p = k / n
    return (p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)


def compare(left_dir, right_dir):
    lm, left = read_replay(left_dir)
    rm, right = read_replay(right_dir)
    for key in ('input_identity', 'runtime_identity', 'binding_identity', 'sampler', 'cache_identity', 'model', 'commit_horizon'):
        if key not in lm or key not in rm or lm[key] != rm[key]:
            raise ValueError('incomparable identity: ' + key)
    if set(left) != set(right):
        raise ValueError('different input/repeat coverage')
    indices = sorted({k[0] for k in left})
    commit = lm['commit_horizon']
    if type(commit) is not int or commit <= 0:
        raise ValueError('invalid execution commit horizon')
    reports = []
    for i in indices:
        repetitions = sorted(k[1] for k in left if k[0] == i)
        if len(repetitions) < 3 or repetitions != list(range(len(repetitions))):
            raise ValueError('at least three contiguous repeats per GPU required')
        ls = [leaves(left[i, r], commit) for r in repetitions]
        rs = [leaves(right[i, r], commit) for r in repetitions]
        for item in ls + rs:
            if set(item) != set(ls[0]):
                raise ValueError('tensor coverage mismatch')
        for rows in (left, right):
            if any(rows[i,r]['request'] != left[i,0]['request'] for r in repetitions):
                raise ValueError('request mismatch')
        for name in sorted(ls[0]):
            cross = [difference(l[name], r[name]) for l in ls for r in rs]
            drift = [difference(a[name], b[name]) for group in (ls, rs)
                     for a, b in itertools.combinations(group, 2)]
            worst = {k: max(d[k] for d in cross) for k in ('max_abs', 'relative_max', 'relative_l2', 'p99_abs')}
            floor = {k: max(d[k] for d in drift) for k in ('max_abs', 'relative_l2')}
            cap, relcap = budgets(lm['model'], name)
            strict = name.startswith('/noise/') or 'mask' in name or ls[0][name].dtype.kind in 'biu'
            within_budget = all(d['exact'] for d in cross) if strict else worst['max_abs'] <= cap and worst['relative_l2'] <= relcap
            within_floor = (all(d['exact'] for d in cross) if strict else
                worst['max_abs'] <= max(1e-6, 5 * floor['max_abs']) and
                worst['relative_l2'] <= max(1e-6, 5 * floor['relative_l2']))
            reports.append(dict(input=i, tensor=name, **worst, same_gpu_floor=floor,
                max_abs_cap=cap, relative_l2_cap=relcap, within_budget=within_budget,
                within_noise_floor=within_floor, exact=all(d['exact'] for d in cross)))
    noise_ok = all(r['exact'] for r in reports if r['tensor'].startswith('/noise/'))
    budget_ok = all(r['within_budget'] for r in reports)
    gripper = lm['model_spec']['layout'].get('gripper_channel')
    gripper_ok = gripper is None or all(np.array_equal(
        np.sign(left[k[0],0]['stages']['final_wire'][:commit, gripper]),
        np.sign(rows[k]['stages']['final_wire'][:commit, gripper]))
        for rows in (left, right) for k in left)
    budget_ok = budget_ok and gripper_ok
    cache = cache_report(left, right)
    return dict(schema='xgpu.comparison.v1', status='SCREEN_PASS' if noise_ok and budget_ok and cache['eligible'] else 'NOT_EQUIVALENT_OR_INCOMPLETE',
        deployment_equivalence_established=False, layer_c='NOT_RUN',
        noise_identical=noise_ok, layer_a_within_budget=budget_ok,
        committed_gripper_sign_agreement=gripper_ok,
        layer_a_within_noise_floor=all(r['within_noise_floor'] for r in reports),
        left_environment=lm.get('environment'), right_environment=rm.get('environment'),
        layer_a=reports, layer_b=cache,
        interpretation='SCREEN_PASS permits paired pilot only; neither SR nor dynamic-batch equivalence is established')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    try:
        result = compare(a.left, a.right)
    except (ValueError, KeyError, OSError) as e:
        result = dict(status='INVALID', error=str(e), deployment_equivalence_established=False)
    write_json(a.output, result)
    print(json.dumps({k:v for k,v in result.items() if k in ('status', 'noise_identical', 'layer_a_within_budget', 'error')}))
    raise SystemExit(0 if result['status'] == 'SCREEN_PASS' else 2)


if __name__ == '__main__':
    main()
