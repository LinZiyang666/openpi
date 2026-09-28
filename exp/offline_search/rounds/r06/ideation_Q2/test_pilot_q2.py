"""Statistical/accounting regression checks using explicitly synthetic outcomes.

Writes only under ideation_Q2. No servers, simulator, real pilot, or network.
"""
import contextlib
import argparse
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import pilot_q2 as q


def fixtures(out):
    out.mkdir(parents=True, exist_ok=False)
    tables = {k: [] for k in ['episodes', 'decisions', 'anchors', 'attempts']}
    rng = np.random.default_rng(82817)
    for cohort in q.COHORTS:
        for task in range(10):
            for init in range(6):
                for block in range(3):
                    arm = f'r6p3v2_pi05_l10_50_{cohort}_r{block}'
                    uid = f'{arm}:{task}:{init}'
                    dose = list(q.DOSES.values())[(task+init+block) % 5] if cohort == 'dose_mix' else q.DOSES.get(cohort, .25)
                    calls = rng.random(4) < dose
                    # Synthetic outcomes are not policy measurements or power evidence.
                    y = int(rng.random() < .45+.4*dose)
                    ident = dict(arm=arm, uid=uid, attempt=1)
                    ep = dict(**ident, model='pi05', suite='l10', lib='current', task_id=task, init=init,
                        Y=y, decisions=8, anchors=4, misses=int(calls.sum()), active_controls=40,
                        controls_verified=True, environment_seed_verified=True, physical_transitions_available=True,
                        collection_mode='client_verified', client_environment_seed=603+block,
                        full_policy_forwards=8, extra_stage3=3)
                    ep.update({'provenance.run_block': block, 'provenance.split': 'calibration' if init%5 == 0 else 'validation',
                        'client.orig_init_state_idx': init, 'client.task_id': task,
                        'episode_assignment.dose': dose, 'episode_assignment.probability': .2 if cohort == 'dose_mix' else 1.,
                        'episode_assignment.random': cohort == 'dose_mix', 'provenance.catalog_sha256': 'synthetic-not-real-fit',
                        'deployment_IR_per_actual_5_controls': (.152*4+.848*calls.sum())/8})
                    tables['episodes'].append(ep)
                    tables['attempts'].append(dict(arm=arm, task_uid=uid, attempt=1, accepted=True, status='done', success=y))
                    for step in range(8):
                        call, vision = bool(calls[step//2]), step%2 == 0
                        source = ('policy' if call else 'cache') if vision else ('policy_tail' if call else 'cache')
                        d = dict(**ident, step=step, source=source, vision=vision, actual_controls=5)
                        d.update({'stage_invocations.stage1': 1, 'stage_invocations.stage2': 1,
                            'stage_invocations.stage3': 4 if step == 0 else 1})
                        tables['decisions'].append(d)
                        if vision:
                            a = dict(**ident, step=step)
                            a.update({f'assignment.{k}': v for k, v in dict(executed_policy=call,
                                actual_propensity=dose, propensity=dose, treatment_probability=dose if call else 1-dose,
                                cohort_probability=1, cohort=cohort, commit_controls=10, override='coin').items()})
                            tables['anchors'].append(a)
    for name, rows in tables.items():
        q.write_csv(out/(name+'.csv'), rows)
    q.dump(out/'audit.json', {k: len(v) for k, v in tables.items()})
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=q.HERE/'pilot_synthetic_tests')
    out = q.owned(ap.parse_args().out)
    out.mkdir(exist_ok=False)
    checks = []
    # Equal init weighting, irrespective of unequal repeat counts.
    z1, z0 = np.array([1, 4, 2, 8, 8.]), np.array([0, 4, 2, 8, 8.])
    maps = {'A': {(0, 0, b): z1 for b in range(3)} | {(0, 1, 0): z0}}
    eng = q.Engine(maps, .152, 4000, 'unequal-init-test')
    assert eng.estimate('A')['SR'] == .5
    assert len(eng.clusters) == 2 and len(eng.units) == 4
    checks.append('equal init weights, preserve seed repeats in clusters')

    # Paired subtraction must be inside each draw, not independent arm resampling.
    both = q.Engine({'A': maps['A'], 'P10': maps['A']}, .152, 4000, 'paired-test')
    assert both.contrast('A', 'P10')['SR_lo'] == both.contrast('A', 'P10')['SR_hi'] == 0
    assert np.isnan(both.contrast('A', 'P10')['SR_simul_lower'])
    checks.append('paired draws cancel identical arms; no equivalence from zero variance')

    one = q.Engine({'A': {(0, 1, b): z1 for b in range(3)}}, .152, 4000, 'one-init')
    assert not one.valid_ci and one.df == 0 and np.isnan(one.estimate('A')['SR_lo'])
    checks.append('three repeats of one init cannot estimate within-task init variance')
    ratio = q.Engine({'A': {u: np.array([0, 4, 0, 8, 8.]) for u in maps['A']},
                      'P10': {u: np.array([1, 4, 4, 8, 8.]) for u in maps['A']}}, .152, 4000, 'ratio')
    ratio.boot[0, ratio.names.index('P10'), q.METRICS.index('m')] = 0
    assert not ratio.slope('A', 'P10')['m_slope_resolved']
    checks.append('undefined ratio bootstrap draws suppress the interval rather than disappear by filtering')

    quality, _ = q.quality_rows()
    for uniform in [True, False]:
        for rho in q.RHOS:
            a = q.allocation(quality, 'pi05_l10_50', rho, uniform)
            assert abs(a['predicted_IR']-rho) < 1e-12
            for row in a['tasks']:
                weights = a['weights'][row['task']]
                assert abs(sum(weights.values())-1) < 1e-12
                assert abs(sum(q.DOSES[k]*v for k, v in weights.items())-row['p']) < 1e-12
    assert q.allocation(quality, 'pi05_l10_50', 0)['clamped']
    assert q.allocation(quality, 'pi05_l10_50', 1)['clamped']
    checks.append('all eight library allocations hit feasible rho and interpolate whole-policy doses')

    # Expected HT score enumerating every possible episode treatment, p=.2.
    potential = np.array([0, 0, 1, 0, 1])
    for j in range(5):
        assert np.isclose(sum(.2*(int(a == j)*potential[a]/.2) for a in range(5)), potential[j])
    checks.append('episode HT scores average to each fixed potential outcome under assigned-dose randomization')

    tabledir = out/'tables'
    fixtures(tabledir)
    eps, audit, support = q.load_tables(tabledir)
    assert len(eps) == 1620 and audit['decisions'] == 12960 and audit['anchors'] == 6480
    p10 = [e for e in eps if e['cohort'] == 'P10']
    assert all(q.metric_values(e['z'], .152)[1] == .5 for e in p10)
    assert all(e['z'][2] == 4 and e['forwards'] == 8 for e in p10)
    checks.append('synthetic CSV join/accounting: policy tails and eight shadows do not become eight MISS calls')
    attempts_path = tabledir/'attempts.csv'
    original_attempts = attempts_path.read_text()
    attempts = list(csv.DictReader(io.StringIO(original_attempts)))
    attempts.append(dict(attempts[0], accepted='', status='running'))
    q.write_csv(attempts_path, attempts)
    assert len(q.load_tables(tabledir)[0]) == 1620
    attempts_path.write_text(original_attempts)
    checks.append('nonterminal journal entries with absent acceptance do not enter the episode denominator')

    # The CLI, JSON serialization, HT branches, risk policies, and plots-free run.
    args = ['pilot_q2.py', '--tables', str(tabledir), '--out', str(out/'analysis'), '--smoke',
            '--bootstrap', '200', '--no-plots']
    with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()) as log:
        q.main()
    result = json.loads((out/'analysis/estimates.json').read_text())
    assert len(result['allocations']) == 80 and result['dose_mix'] and result['dose_mix_vs_fixed']
    assert result['marginal_slopes'] and result['diminishing_returns']
    assert all(r['recommendation'] == 'SMOKE_ONLY_NO_SCIENTIFIC_RECOMMENDATION' for r in result['recommendations'])
    checks.append('synthetic end-to-end CLI exercises fixed doses, B, HT, allocation, marginal slopes, and explicit smoke gating')

    # Reject corrupted counting and treatment propensities, not silent coercion.
    epspath = tabledir/'episodes.csv'
    original = epspath.read_text()
    rows = list(csv.DictReader(io.StringIO(original)))
    rows[0]['misses'] = '1'
    q.write_csv(epspath, rows)
    try:
        q.load_tables(tabledir)
    except ValueError:
        checks.append('corrupt MISS total rejected')
    else:
        raise AssertionError('bad totals accepted')
    epspath.write_text(original)
    apath = tabledir/'anchors.csv'
    original = apath.read_text()
    rows = list(csv.DictReader(io.StringIO(original)))
    rows[0]['assignment.cohort_probability'] = str(1/9)
    q.write_csv(apath, rows)
    try:
        q.load_tables(tabledir)
    except ValueError:
        checks.append('invented 1/9 cohort propensity rejected')
    else:
        raise AssertionError('wrong propensity accepted')
    apath.write_text(original)

    # Adapter checks are simulated; this does NOT claim a raw-log reader run.
    run = out/'fake_run'
    arm = 'r6p3v2_pi05_l10_50_A_r0'
    armroot = run/'runs'/arm
    armroot.mkdir(parents=True)
    (run/'state').mkdir()
    manifest = armroot/'manifest.json'
    manifest.write_text(json.dumps({'selected': [{'task': 0, 'init': 0}]}))
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
    h = load_manifest(manifest)['sha256']
    args = SimpleNamespace(tables=None, run_root=run, arms=[arm], arms_file=None, cell=None)
    adapterout = out/'adapter'
    adapterout.mkdir()
    try:
        q.reader_input(args, adapterout)
    except ValueError:
        checks.append('unfinished arm refused without current selection-hash DONE')
    else:
        raise AssertionError('unfinished arm admitted')
    (run/'state'/f'{arm}.manifest_{h}.DONE').touch()
    with patch.object(q.subprocess, 'run') as call:
        q.reader_input(args, adapterout)
        cmd = call.call_args.args[0]
        assert cmd[:3] == ['taskset', '-c', '18-21,62-65']
        assert cmd[cmd.index('--client-root')+1] == str(armroot)
        assert '--require-stage-counts' in cmd and '--require-snapshots' in cmd
    checks.append('raw reader adapter uses correct manifest hash, one arm, confined client root, CPU prefix, required audits (mock only)')
    report = dict(scope='SYNTHETIC SOFTWARE CHECKS ONLY', checks=checks, passed=len(checks),
                  synthetic_cli_output=log.getvalue().strip(), raw_log_adapter='mocked; not executed')
    q.dump(out/'test_results.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
