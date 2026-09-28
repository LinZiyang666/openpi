"""Read-only K5 audit, unchanged B fitter, held-init/task OPE, and sign sensitivity.

Run via run_analysis.sh. Writes only to the specified Q3 directory or /tmp/q3_*.
All data-derived tables are borrowed big-library information.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm, t

from exp.offline_search.rounds.r04.k5_rand import estimate as k5
from exp.offline_search.rounds.r05.q3_callvalue import cost_solver_reference as solver

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r04_k5')
REFERENCE = HERE.parent / 'ideation_B/cost_solver.py'
METRICS = ['SR_change', 'N_change', 'M_change', 'cost_saving']
DISCLOSURE = 'borrowed big-library information'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': h.hexdigest()}


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def key(row):
    return row['landmark_class'], json.dumps(row['context'], sort_keys=True)


def cluster_keys(rows):
    return sorted({(r['task_id'], r['init']) for r in rows})


def outcome(row, rho):
    return np.array([row['Y'], row['N'], row['M'], .848 * row['M'] + (.152 - rho) * row['N']])


def intervals(vals, comparisons=1):
    """Input: one contribution per init cluster, including both replicate records."""
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1) / np.sqrt(len(vals))
    z = float(norm.ppf(1 - .05 / (2 * comparisons)))
    return {
        'n_clusters': len(vals), 'comparisons_per_outcome': comparisons,
        'mean': dict(zip(METRICS, mean.tolist())), 'SE': dict(zip(METRICS, se.tolist())),
        'ci95_pointwise': dict(zip(METRICS, np.stack([mean - norm.ppf(.975) * se, mean + norm.ppf(.975) * se], 1).tolist())),
        'ci95_bonferroni': dict(zip(METRICS, np.stack([mean - z * se, mean + z * se], 1).tolist())),
        'approximate': True,
    }


def known_keys():
    return [(lm, json.dumps(dict(zip(k5.CONTEXTS, bins)), sort_keys=True))
            for lm, bins in itertools.product((1, 3), itertools.product(*k5.CONTEXTS.values()))]


def crossfit(rows, rho, by, keys):
    clusters = cluster_keys(rows)
    index = {c: i for i, c in enumerate(clusters)}
    bycell = {k: np.zeros((len(clusters), 4)) for k in keys}
    get_fold = (lambda r: r['init'] % 5) if by == 'init_mod_5' else (lambda r: r['task_id'])
    folds = []
    heldout_seen = []
    for fold in sorted({get_fold(r) for r in rows}):
        train = [r for r in rows if get_fold(r) != fold]
        test = [r for r in rows if get_fold(r) == fold]
        assert not set(cluster_keys(train)) & set(cluster_keys(test))
        fit = solver.causal_fit(train, rho, max_loss=.01)
        original_ope = solver.ope(test, fit, rho)
        probabilities = {tuple(c['key']): c['suppress_probability'] for c in fit['cells']}
        vals = np.zeros((len(clusters), 4))
        for row in test:
            x = probabilities.get(key(row), 0.) if row['exposed'] else 0.
            sign = 1 if row['assigned_treatment'] == 'CACHE' else -1
            delta = x * sign * outcome(row, rho)
            delta[3] *= -1  # saving, instead of policy-minus-CALL cost
            idx = index[(row['task_id'], row['init'])]
            vals[idx] += delta
            if row['exposed']:
                bycell[key(row)][idx] += delta
        test_indices = [index[c] for c in cluster_keys(test)]
        heldout_seen.extend(test_indices)
        stats = intervals(vals[test_indices])
        reference_mean = np.array(original_ope['policy_minus_CALL_Y_N_M_C']) * [1, 1, 1, -1]
        np.testing.assert_allclose(list(stats['mean'].values()), reference_mean, atol=1e-13)
        np.testing.assert_allclose(list(stats['SE'].values()), original_ope['SE'], atol=1e-13)
        folds.append({'fold': fold, 'train_clusters': len(cluster_keys(train)), 'test_clusters': len(test_indices),
                      'fit': fit, 'reference_ope': original_ope, 'heldout': stats})
    assert sorted(heldout_seen) == list(range(len(clusters)))
    total = sum(bycell.values(), np.zeros((len(clusters), 4)))
    return {
        'split': by, 'folds': folds, 'pooled': intervals(total),
        'by_context': [{'key': list(k), 'heldout': intervals(v, len(keys))} for k, v in bycell.items()],
        'cluster_contributions': [{'task_id': c[0], 'init': c[1], **dict(zip(METRICS, v.tolist()))}
                                  for c, v in zip(clusters, total)],
    }


def family_analysis(rows, rho, family):
    if family == 'parent':
        rows = [{**r, 'context': {} if r['exposed'] else None} for r in rows]
        keys = [(1, '{}'), (3, '{}')]
    else:
        keys = sorted(set(known_keys()) | {key(r) for r in rows if r['exposed']})
    full = solver.causal_fit(rows, rho, max_loss=.01)
    init_cv = crossfit(rows, rho, 'init_mod_5', keys)
    task_cv = crossfit(rows, rho, 'task_id', keys)
    cells = {tuple(c['key']): c for c in full['cells']}
    init_cells = {tuple(c['key']): c['heldout'] for c in init_cv['by_context']}
    task_cells = {tuple(c['key']): c['heldout'] for c in task_cv['by_context']}
    table = []
    for k in keys:
        c = cells.get(k)
        rec = dict(c) if c else {'key': list(k), 'n': 0, 'clusters': 0, 'n_call': 0, 'n_cache': 0,
                                'supported': False, 'suppress_probability': 0.}
        rec['present_in_fit'] = c is not None
        rec['context'] = json.loads(k[1])
        rec['landmark_class'] = k[0]
        rec['eligible_for_nonzero_probability'] = bool(c and c['supported'] and c['saving_lcb'] > 0)
        if c:
            delta, se, z = np.array(c['delta_Y_N_M_C_population']), np.array(c['SE']), full['z_simultaneous_normal']
            rec['CALL_minus_CACHE_population_ci95_bonferroni'] = np.stack([delta-z*se, delta+z*se], 1).tolist()
        rec['init_fold_probabilities'] = [next((cc['suppress_probability'] for cc in f['fit']['cells'] if tuple(cc['key']) == k), 0.)
                                          for f in init_cv['folds']]
        rec['task_fold_probabilities'] = [next((cc['suppress_probability'] for cc in f['fit']['cells'] if tuple(cc['key']) == k), 0.)
                                          for f in task_cv['folds']]
        rec['init_heldout'] = init_cells[k]
        rec['task_heldout'] = task_cells[k]
        table.append(rec)
    has_override = any(c['suppress_probability'] > 0 for c in full['cells'])
    bounds = init_cv['pooled']['ci95_pointwise']
    deployable = has_override and bounds['cost_saving'][0] > 0 and bounds['SR_change'][0] >= -.01
    return {'family': family, 'fit': full, 'table': table, 'init_crossfit': init_cv, 'task_crossfit': task_cv,
            'deployment': {'deployable': bool(deployable), 'has_full_data_override': has_override,
                           'positive_heldout_cost_lcb': bounds['cost_saving'][0] > 0,
                           'heldout_SR_within_tolerance': bounds['SR_change'][0] >= -.01,
                           'decision': 'deployment gate passed' if deployable else 'baseline CALL everywhere',
                           'sample_supported_cells': sum(c['supported'] for c in full['cells']),
                           'positive_cost_lcb_supported_cells': sum(c['supported'] and c['saving_lcb'] > 0 for c in full['cells'])}}


def itt_array(rows):
    clusters = cluster_keys(rows)
    index = {c: i for i, c in enumerate(clusters)}
    vals = np.zeros((len(clusters), len(k5.METRICS)))
    for r, v in zip(rows, k5.values(rows)):
        vals[index[(r['task_id'], r['init'])]] += (1 if r['assigned_treatment'] == 'CALL' else -1) * v
    return clusters, vals


def effect_summary(vals, weights, tasks):
    draws = weights @ vals / len(vals)
    task_means = np.array([vals[tasks == task].mean(0) for task in sorted(set(tasks))])
    se_task = task_means.std(0, ddof=1) / np.sqrt(len(task_means))
    mean = vals.mean(0)
    return {'mean': dict(zip(k5.METRICS, mean.tolist())),
            'ci95_init_bootstrap': dict(zip(k5.METRICS, np.quantile(draws, [.025, .975], axis=0).T.tolist())),
            'ci95_task_t9': dict(zip(k5.METRICS, np.stack([mean-t.ppf(.975,9)*se_task, mean+t.ppf(.975,9)*se_task], 1).tolist()))}


def task_sensitivity(rows_by_scale):
    clusters50, arr50 = itt_array(rows_by_scale[50])
    clusters500, arr500 = itt_array(rows_by_scale[500])
    assert clusters50 == clusters500
    tasks = np.array([c[0] for c in clusters50])
    _, weights = k5.cluster_draws(rows_by_scale[50], 2000, 20260927)
    out = {'estimand': 'ITT CALL minus CACHE; scale contrast is g500 minus g50, matched original init',
           'bootstrap': {'draws': 2000, 'seed': 20260927, 'cluster': '(task, original init)', 'paired_across_scales': True},
           'scale_effects': {}, 'paired_scale_contrast': effect_summary(arr500-arr50, weights, tasks), 'by_task': [], 'leave_one_task_out': []}
    for scale, arr in [(50, arr50), (500, arr500)]:
        out['scale_effects'][str(scale)] = effect_summary(arr, weights, tasks)
    for task in sorted(set(tasks)):
        for group, mask in [('by_task', tasks == task), ('leave_one_task_out', tasks != task)]:
            out[group].append({'task': int(task), 'clusters_per_scale': int(mask.sum()),
                               'g50': dict(zip(k5.METRICS, arr50[mask].mean(0).tolist())),
                               'g500': dict(zip(k5.METRICS, arr500[mask].mean(0).tolist())),
                               'g500_minus_g50': dict(zip(k5.METRICS, (arr500-arr50)[mask].mean(0).tolist()))})
    out['loto_signs'] = {}
    for scale in (50, 500):
        out['loto_signs'][str(scale)] = {}
        for m in k5.METRICS:
            vv = [r[f'g{scale}'][m] for r in out['leave_one_task_out']]
            out['loto_signs'][str(scale)][m] = {'min': min(vv), 'max': max(vv),
                                              'positive': sum(x>0 for x in vv), 'negative': sum(x<0 for x in vv), 'zero': sum(x==0 for x in vv)}
    return out


def report_tables(results, out):
    lines = ['# Complete context and parent results', '',
             'Borrowed big-library information. C saving is CALL cost minus policy cost in C_rho units per episode; '
             'SR change is policy minus CALL. All effects below are population contributions (denominator 500 init clusters), '
             'not conditional cell means and not IR changes. S = sample support only (>=30 clusters, >=10 observations per treatment).', '',
             'Full-suppression C intervals use the unchanged fitter\'s Bonferroni multiplier over observed cells '
             '(26/27 leaves or 2 parents). OOF intervals use five init-mod-5 folds and Bonferroni over all 48 leaves '
             '(2 parents), separately by outcome. Zero policy effects and [0,0] intervals are exact consequences of '
             'zero suppression on those folds, not evidence that an omitted call is harmless. Unobserved causal effects are —.', '']
    def fmt(x):
        return f'{x:+.6f}'
    def interval(x):
        return '[' + ', '.join(fmt(v) for v in x) + ']'
    for scale in (500, 50):
        for family in ('leaf', 'parent'):
            a = results[scale][family]
            lines += [f'## g{scale} {family}', '',
                      f"rho={results[scale]['rho']:.17g}; full-data z={a['fit']['z_simultaneous_normal']:.12g}. "
                      f"Decision: **{a['deployment']['decision']}**.", '',
                      '| Landmark | Progress | Grip | Confidence | Stall age | Clusters | CALL/CACHE | S | p(full) | p(folds 0–4) | C saving if fully suppressed [Bonf.] | OOF saving [Bonf.] | OOF SR change [Bonf.] |',
                      '|---:|---|---|---|---|---:|---:|---|---:|---|---|---|---|']
            for c in a['table']:
                ctx = c['context']; held = c['init_heldout']
                raw = (fmt(c['delta_Y_N_M_C_population'][3]) + ' ' + interval(c['CALL_minus_CACHE_population_ci95_bonferroni'][3])) if c['present_in_fit'] else '—'
                lines.append('| ' + ' | '.join([str(c['landmark_class']), *[str(ctx.get(k, '*')) for k in ('progress','gripper','confidence','stall_age')],
                              str(c['clusters']), f"{c['n_call']}/{c['n_cache']}", 'yes' if c['supported'] else 'no',
                              f"{c['suppress_probability']:.6g}", ','.join(f'{p:.4g}' for p in c['init_fold_probabilities']), raw,
                              fmt(held['mean']['cost_saving'])+' '+interval(held['ci95_bonferroni']['cost_saving']),
                              fmt(held['mean']['SR_change'])+' '+interval(held['ci95_bonferroni']['SR_change'])]) + ' |')
            lines += ['']
    (out / 'CONTEXTS.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE/'results')
    ap.add_argument('--from-audited', type=Path, help='reuse Q3 raw-audited episodes, checking raw-source hashes')
    args = ap.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(HERE) or str(out).startswith('/tmp/q3_')
    out.mkdir(parents=True, exist_ok=True)
    assert digest(REFERENCE)['sha256'] == digest(Path(solver.__file__))['sha256']
    assert set(os.sched_getaffinity(0)) <= {26,27,28,29,70,71,72,73}
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        assert os.environ.get(variable) == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    rows_by_scale, results = {}, {}
    for scale in (500, 50):
        print(f'g{scale}: loading and auditing accepted episodes', flush=True)
        arms = [f'r4k5_p_l10_g{scale}_r{i}' for i in (1,2)]
        sources = [ROOT/f'k5_g{scale}_episodes.json', ROOT/f'k5_g{scale}_estimate.json']
        sources += [ROOT/'runs'/arm/'client/journal.jsonl' for arm in arms]
        sources += [p for arm in arms for p in sorted((ROOT/'runs'/arm).glob('server_*/decisions_*.jsonl'))]
        source_hashes = [digest(p) for p in sources]
        if args.from_audited:
            saved = json.loads((args.from_audited/f'audited_g{scale}_episodes.json').read_text())
            assert saved['source_hashes'] == source_hashes
            rows, audits = saved['episodes'], saved['input_audit']
        else:
            rows, audits = [], []
            for arm in arms:
                rr, audit = k5.load_arm(ROOT, arm)
                rows.extend(rr); audits.append(audit)
        groups = k5.validate_pairs(rows)
        assert len(rows) == 1000 and len(groups) == 500
        assert set(groups) == set(itertools.product(range(10),range(50)))
        assert rows[0]['controller'] == {
            'method_spec': 'exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge',
            'kwargs': {'base': 'exp/offline_search/rounds/r02/g1_awm/awm.py:AWM',
                       'base_kwargs': {} if scale == 500 else {'lib':'current','kref':5},
                       'guards': True, 'events': 'none'},
            'cell': 'pi05_l10_cache',
            'judge': {'mode':'guard_only','cap':0,'burst':1,'step0':'judge','text':'guard_only'},
            'stage1_mode': 'full', 'miss_steps': 10,
        }
        exported = json.loads((ROOT/f'k5_g{scale}_episodes.json').read_text())['episodes']
        assert sorted(rows, key=lambda r: (r['arm'],r['task_id'],r['init'])) == sorted(exported,key=lambda r: (r['arm'],r['task_id'],r['init']))
        coord = json.loads((ROOT/f'k5_g{scale}_estimate.json').read_text())
        reproduced = k5.estimate(rows, boot=coord['uncertainty']['draws'], seed=coord['uncertainty']['seed'])
        for field in reproduced:
            assert reproduced[field] == coord[field], f'coordinator mismatch: {field}'
        write(out/f'audited_g{scale}_episodes.json', {'schema':'causal_rescue_credit.episodes.v1', 'episodes':rows,
                                                    'input_audit':audits, 'source_hashes':source_hashes})
        write(out/f'k5_g{scale}_reproduced.json', reproduced)
        rho = k5.RHOS[f'g{scale}']
        unknown = sum(r['exposed'] and (r['context'] is None or any(v is None for v in r['context'].values())) for r in rows)
        assert unknown == 0, 'Unknown leaves need an explicit exclusion adaptation before fitting.'
        print(f'g{scale}: 500 pairs audited; fitting leaves, parents, init folds, task folds', flush=True)
        result = {'scale':scale, 'rho':rho, 'provenance':DISCLOSURE,
                  'audit':{'episodes':len(rows), 'init_clusters':len(groups), 'tasks':10, 'inits_per_task':50,
                           'raw_equals_coordinator_export':True, 'recomputed_estimate_equals_coordinator':True,
                           'unknown_context_exposures':unknown,
                           'discordant_pair_exposure':reproduced['discordant_pair_exposure'],
                           'input_audit':audits,'controller':rows[0]['controller'],
                           'exposures':sum(r['exposed'] for r in rows),
                           'total_decisions':sum(r['N'] for r in rows), 'total_misses':sum(r['M'] for r in rows)},
                  'leaf':family_analysis(rows,rho,'leaf'), 'parent':family_analysis(rows,rho,'parent')}
        rows_by_scale[scale], results[scale] = rows, result
        write(out/f'g{scale}.json',result)
        print(json.dumps({'scale':scale, **result['leaf']['deployment'], 'parent':result['parent']['deployment']}),flush=True)
    sensitivity = task_sensitivity(rows_by_scale)
    write(out/'task_sensitivity.json', sensitivity)
    write(out/'source_code.json', {'sources':[digest(p) for p in (REFERENCE,Path(solver.__file__),Path(k5.__file__),Path(k5.__file__).with_name('overlay.py'),Path(__file__))],
                                 'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
                                 'cpu_affinity':sorted(os.sched_getaffinity(0))})
    report_tables(results,out)
    deployable = [s for s in results if any(results[s][f]['deployment']['deployable'] for f in ('leaf','parent'))]
    write(out/'deployment_decisions.json', {'provenance':DISCLOSURE, 'deployable_scales':deployable,
                                           'scales':{str(s):{f:results[s][f]['deployment'] for f in ('leaf','parent')} for s in results}})
    if deployable:
        print('Deployment gate passed: implementation and arm specifications now required.',flush=True)
    else:
        print('No deployable scale. Baseline CALL everywhere; no plugin path or pilot arms.',flush=True)


if __name__ == '__main__':
    main()
