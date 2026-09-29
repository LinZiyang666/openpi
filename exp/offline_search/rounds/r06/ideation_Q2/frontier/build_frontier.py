"""Frozen, read-only census of completed LIBERO closed-loop arms and owner frontier.

Run from repo root under the CPU mask in frontier.md. No collection is invoked.
"""
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import json
import datetime
import math
import re

import numpy as np
from scipy.stats import beta, binomtest

OUT = Path(__file__).resolve().parent
Q2 = OUT.parent
REPO = Q2.parents[4]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store/library')
EXPECTED = {(t, i) for t in range(10) for i in range(50)}
C1 = {'pi05': .152, 'groot': .148}
MARGIN = .02


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def csv_out(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def journal(path):
    rows = {}
    for line in path.open():
        r = json.loads(line)
        if r.get('accepted') and r.get('status') in ('done', 'failed') and not r.get('error'):
            key = tuple(map(int, r['task_uid'].split(':')[-2:]))
            if key in rows and rows[key] != r:
                raise ValueError(f'multiple accepted attempts: {path} {key}')
            rows[key] = r
    return rows


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    center = (p+z*z/(2*n))/(1+z*z/n)
    return center-half, center+half


def family(s, spec, run, arm, pure, L):
    method = s.get('method', '')
    kw = s.get('kwargs', {})
    base = kw.get('base_kwargs', kw)
    judge = (s.get('mixed') or {}).get('judge', {}).get('mode', spec.get('judge', '')) or ''
    if pure:
        return f'pure_inference_L{int(L)}' + ('_K2' if '_k2_' in arm else '')
    if 'GrootCommitJudge' in method or (method.endswith(':CommitJudge') and kw.get('monitor', 'off') == 'off'):
        return 'B_guard_committed_rescue'
    if 'CommitJudge' in method:
        return 'monitored_committed_rescue'
    if 'GraspCheckJudge' in method:
        return 'D1_grasp_guard'
    if 'q4_growth' in method:
        return 'library_growth_'+str(kw.get('variant', ''))
    if run == 'r06_abl':
        return 'ablation_'+method.rsplit(':', 1)[-1]
    if run == 'r04_k5':
        return 'randomized_guard_landmark'
    if 'CycleTail' in method:
        return 'periodic_anchor_policy_tail_G10'
    if 'wrist' in method.lower():
        return 'wrist_guard_or_tail'
    if run == 'r05_b1':
        variant = arm.rsplit('_', 1)[-1]
        controller = 'K7_guard' if 'VisionConfirmed' in method else method.rsplit(':', 1)[-1]
        return f'B1_{variant}_{controller}'
    if 'AWM3' in method:
        return 'AWM3_metric_variant'
    if judge.startswith('quantile'):
        return 'confidence_quantile'
    if judge.startswith('periodic'):
        return 'periodic_MISS'
    if base.get('serving') == 'anchor_tail' and not s.get('mixed', {}).get('misses', 0):
        return 'A_commit_cache' if base.get('budget') == 1 else 'cache_longer_tail'
    if 'PolicyTailJudge' in method:
        return 'guard_policy_tail'
    if 'VisionConfirmed' in method:
        return 'K7_vision_guard'
    if 'MixedJudge' in method:
        return 'guard_only' if not base.get('serving') else 'blind_guard_'+base['serving']
    if 'BlindAWM' in method:
        return 'blind_cache_'+base.get('serving', 'unspecified')
    if 'ControlStepLibrary' in method:
        return 'control_step_library'
    if 'shadow' in arm:
        return 'retrieval_shadow'
    return 'five_step_cache_'+method.rsplit(':', 1)[-1]


@lru_cache(None)
def lib_metadata(model, suite, lib):
    d = STORE/f'{model}_{suite}'/lib
    f = d/'manifest.json'
    if not f.exists():
        return dict(path=str(d), episodes=None, manifest_sha=None)
    ep = np.load(d/'episode.npy', mmap_mode='r')
    return dict(path=str(d), episodes=int(len(np.unique(ep))), manifest_sha=sha(f))


def ledger_counts(s, spec, cached, jp):
    ledger = s.get('cost_ledger') or {}
    if ledger:
        N, V, M = (int(ledger[k]) for k in ['decisions', 'vision_decisions', 'misses'])
        if N != s['client_decisions'] or not 0 <= M <= V <= N:
            raise ValueError('ledger/client mismatch')
        if not np.isclose(ledger['v'], V/N) or not np.isclose(ledger['m'], M/N):
            raise ValueError('ledger v/m inconsistent with counts')
        L = float(ledger['l_per_request']['mean'])
        return N, V, M, L, 'summary.cost_ledger', True
    if cached and cached.get('cost_valid'):
        if sha(jp) != cached['hashes'][str(jp)]:
            raise ValueError('legacy journal differs from cached accepted raw audit')
        for f in cached.get('sources', []):
            stat = Path(f['path']).stat()
            if stat.st_size != f['bytes'] or stat.st_mtime_ns != f['mtime_ns']:
                raise ValueError('legacy raw file differs from cached audit')
        return cached['N'], cached['V'], cached['M'], cached['L'], 'prior accepted raw audit; journal SHA and raw size/mtime rechecked', True
    # Preserve uncertain historical SR/cost points in the census, outside dominance.
    mixed = s.get('mixed') or {}
    N = int(mixed.get('decisions', s['client_decisions']))
    return N, N, int(mixed.get('misses', 0)), 5., 'conflicting legacy summary; cost UNVERIFIED', False


def collect():
    cache = {(r['run'], r['arm']): r for r in json.loads((Q2/'arms.json').read_text())}
    paths_file = OUT/'summary_paths.json'
    if paths_file.exists():
        paths = [Path(p) for p in json.loads(paths_file.read_text())]
    else:
        paths = sorted(RUNS.glob('*/runs/*/summary.json'))
        dump('summary_paths.json', [str(p) for p in paths])
    points, skipped, evidence, outcomes = [], [], {}, {}
    specs_cache = {}
    for path in paths:
        run, arm = path.parents[2].name, path.parent.name
        if 'smoke' in run or 'pilot' in run:
            skipped.append(dict(run=run, arm=arm, reason='smoke/subset/pilot; excluded before outcome collection'))
            continue
        s = json.loads(path.read_text())
        if s.get('model') not in C1:
            skipped.append(dict(run=run, arm=arm, reason='outside requested models'))
            continue
        suite = {'libero_10': 'l10', 'libero_spatial': 'spatial', 'l10': 'l10', 'spatial': 'spatial'}.get(s['suite'])
        if suite is None:
            skipped.append(dict(run=run, arm=arm, reason='outside requested suites'))
            continue
        runroot = path.parents[2]
        done = sorted((runroot/'state').glob(arm+'.*DONE'))
        if not done or (runroot/'state'/(arm+'.SKIPPED')).exists():
            skipped.append(dict(run=run, arm=arm, reason='no completion marker or explicitly skipped'))
            continue
        jp = path.parent/'client/journal.jsonl'
        if not jp.exists():
            skipped.append(dict(run=run, arm=arm, reason='missing accepted journal'))
            continue
        acc = journal(jp)
        if len(acc) != s['complete'] or sum(bool(v['success']) for v in acc.values()) != s['success']:
            raise ValueError('summary/journal mismatch: '+str(path))
        if run not in specs_cache:
            af = runroot/'arms.json'
            specs_cache[run] = {x.get('arm', x.get('name')): x for x in json.loads(af.read_text())} if af.exists() else {}
        spec = specs_cache[run].get(arm, {})
        N, V, M, L, cost_source, cost_valid = ledger_counts(s, spec, cache.get((run, arm)), jp)
        pure = 'SeededInference' in s.get('method', '') or 'inferL' in arm or 'policy_L' in arm
        actual = list(s.get('server', {}).get('lib_mix', {}))
        if len(actual) != 1 and not pure:
            raise ValueError('unknown/multiple served libraries: '+str(path))
        lib = 'none' if pure else actual[0]
        nominal = {'current': 50, 'bpool_cs': 500, 'bpool_all': 500}.get(lib)
        lm = dict(path=None, episodes=0, manifest_sha=None) if pure else lib_metadata(s['model'], suite, lib)
        kw = s.get('kwargs', {})
        notes = []
        comparable = set(acc) == EXPECTED
        if not comparable:
            notes.append('different episode set: '+str(len(acc))+' task/init pairs')
        eligible = comparable and cost_valid and (nominal in (50, 500) or pure)
        if not cost_valid:
            notes.append('legacy conflicting accepted decision records; excluded from dominance/NI selection')
        if not pure and nominal is None:
            notes.append('separate library variant, excluded from 50/500 frontiers')
        if kw.get('prior_alpha', 0) > 0:
            eligible = False
            notes.append('current retrieval uses additional 500-episode fitted prior; separate library-information condition')
        if 'wrist' in s.get('method', '').lower() or '_k2' in arm:
            notes.append('requested full owner coefficients; no wrist/K2 discount applied')
        if pure:
            notes.append('policy-only, no cache-library requirement; shared across both library sizes')
        if not s.get('cost_ledger'):
            notes.append('no summary cost_ledger; '+cost_source)
        if L != 5:
            notes.append(f'owner normalization 5/L={5/L:g}; no blind-vision charge')
        if lm['episodes'] and nominal and lm['episodes'] != nominal:
            notes.append(f'nominal {nominal}-episode bank has {lm["episodes"]} actual source episodes')
        v, m = V/N, M/N
        owner = (C1[s['model']]*v+(1-C1[s['model']])*m)*5/L
        if pure and not np.isclose(owner, 5/L):
            raise ValueError('policy-only owner IR mismatch')
        if pure:
            owner = 5/L  # exact accounting endpoint, not a floating-roundoff tie break
        sr = s['success']/len(acc)
        lo, hi = wilson(s['success'], len(acc))
        ident = f'{run}/{arm}'
        point = dict(id=ident, cell=f'{s["model"]}_{suite}_{nominal}' if nominal else f'{s["model"]}_{suite}_'+('reference' if pure else lib),
            model=s['model'], suite=suite, arm=arm, run=run,
            method_family=family(s, spec, run, arm, pure, L), method_class=s.get('method'),
            library=lib, library_nominal=nominal, library_actual_episodes=lm['episodes'], library_path=lm['path'],
            library_manifest_sha256=lm['manifest_sha'], n=len(acc), success=s['success'], SR=sr,
            SR_lo=lo, SR_hi=hi, SR_interval='Wilson 95%; single 500-init run', owner_IR=owner,
            v=V/N, m=M/N, v_per_five=V/N*5/L, m_per_five=M/N*5/L, L=L, N=N, V=V, M=M,
            pure=pure, comparable=comparable, eligible=eligible, cost_valid=cost_valid,
            cost_source=cost_source, notes='; '.join(notes), summary=str(path), journal=str(jp),
            episode_set_sha256=hashlib.sha256(json.dumps(sorted(acc)).encode()).hexdigest())
        points.append(point)
        outcomes[ident] = {f'{t}:{i}': int(r['success']) for (t, i), r in sorted(acc.items())}
        evidence[ident] = dict(summary_sha256=sha(path), journal_sha256=sha(jp), markers=[str(p) for p in done], spec=spec, kwargs=kw)
        if not eligible:
            skipped.append(dict(run=run, arm=arm, reason=point['notes'], retained_in_points=True))
    # The L=5 GR00T reference is historical and outside the requested root; explicit provenance.
    for a in cache.values():
        if a['run'] != 'DUAL':
            continue
        jp = Path(a['summary'])
        acc = journal(jp)
        if set(acc) != EXPECTED:
            raise ValueError('historical reference pair set differs')
        model, suite = a['cell'].split('_', 1)
        sr = sum(r['success'] for r in acc.values())/500
        lo, hi = wilson(int(round(sr*500)), 500)
        ident = 'DUAL/'+a['arm']
        points.append(dict(id=ident, cell=f'{model}_{suite}_reference', model=model, suite=suite,
            arm=a['arm'], run='DUAL', method_family='pure_inference_L5', method_class='historical native policy',
            library='none', library_nominal=None, library_actual_episodes=0, library_path=None,
            library_manifest_sha256=None, n=500, success=int(round(sr*500)), SR=sr, SR_lo=lo, SR_hi=hi,
            SR_interval='Wilson 95%; historical 500-init run', owner_IR=1., v=1., m=1., v_per_five=1., m_per_five=1.,
            L=5., N=a['N'], V=a['N'], M=a['N'], pure=True, comparable=True, eligible=True, cost_valid=True,
            cost_source='native L=5 policy definition; requested owner price', notes='historical DUAL harness; same task/init identities; not same environment/policy seed experiment',
            summary=None, journal=str(jp), episode_set_sha256=hashlib.sha256(json.dumps(sorted(acc)).encode()).hexdigest()))
        outcomes[ident] = {f'{t}:{i}': int(r['success']) for (t, i), r in sorted(acc.items())}
        evidence[ident] = dict(journal_sha256=sha(jp), source='historical L=5 reference required by PAPER_AB')
    dump('outcomes.json', outcomes)
    dump('source_evidence.json', evidence)
    csv_out('skipped.csv', skipped)
    return points, outcomes, evidence, skipped


def pareto(rows):
    return [r for r in rows if not any(s['owner_IR'] <= r['owner_IR'] and s['SR'] >= r['SR'] and
            (s['owner_IR'] < r['owner_IR'] or s['SR'] > r['SR']) for s in rows if s is not r)]


def reference_ids(model, suite, length):
    short = 'sp' if suite == 'spatial' else 'l10'
    if length == 10:
        return [f'r04_cost/r4f_p_{short}_inf_k10_L10'] if model == 'pi05' else [f'r05_q2/r5q2_g_{suite}_policy_L10']
    dual = f'DUAL/tr_{model}_{short}_inf'
    return [dual, f'r04_cost/r4f_p_{short}_inf_s1001', f'r04_cost/r4f_p_{short}_inf_s2001'] if model == 'pi05' else [dual]


def paired_counts(a, b, outcomes):
    x, y = outcomes[a], outcomes[b]
    if set(x) != set(y):
        raise ValueError('attempt to pair different evaluation sets')
    wins = sum(x[k] == 1 and y[k] == 0 for k in x)
    losses = sum(x[k] == 0 and y[k] == 1 for k in x)
    return wins, losses, len(x)


def exact_lower(w, l, n, alpha):
    # Two one-sided Clopper–Pearson bounds, Bonferroni within paired multinomial.
    win_lower = float(beta.ppf(alpha/2, w, n-w+1)) if w else 0.
    loss_upper = float(beta.ppf(1-alpha/2, l+1, n-l)) if l < n else 1.
    return win_lower-loss_upper


def aggregate_lower(candidate_ids, refids, outcomes, alpha):
    # Repeated references/candidates share init clusters; do NOT stack independent pairs.
    k = len(candidate_ids)*len(refids)
    return float(np.mean([0. if a == b else exact_lower(*paired_counts(a, b, outcomes), alpha/k)
        for a in candidate_ids for b in refids]))


def paired_boot(candidate_ids, refs, outcomes):
    keys = sorted(outcomes[candidate_ids[0]])
    diff = np.mean([[outcomes[a][k] for k in keys] for a in candidate_ids], axis=0)-np.mean([[outcomes[b][k] for k in keys] for b in refs], axis=0)
    # Resample within task, keeping all repeated run outcomes of an init together.
    rng = np.random.default_rng(26092902)
    means = np.zeros(4000)
    for task in range(10):
        d = np.array([diff[i] for i, k in enumerate(keys) if int(k.split(':')[0]) == task])
        means += d[rng.integers(len(d), size=(4000, len(d)))].mean(1)/10
    return float(diff.mean()), *map(float, np.quantile(means, [.025, .975]))


def pool_ab(points, outcomes):
    by = {r['id']: r for r in points}
    groups = []
    for model in C1:
        for suite in ['l10', 'spatial']:
            sh = 'sp' if suite == 'spatial' else 'l10'
            for size in [50, 500]:
                a = f'r5x_g_{sh}_{size}_tail1u' if model == 'groot' else f'r5t_p_{sh}_{size}_tail1uc'
                run = 'r05_x' if model == 'groot' else 'r05_ptail'
                if model == 'pi05' and suite == 'spatial' and size == 500:
                    a, run = 'r4b3_p_sp_500_tail1uc', 'r04_blind'
                b = f'r6p1_c10_g_{sh}_{size}' if model == 'groot' else f'r5q1_c10_p_{sh}_{size}'
                brun = 'r06_paper' if model == 'groot' else 'r05_q1'
                for label, base, rr in [('A', a, run), ('B', b, brun)]:
                    ids = [f'{rr}/{base}', f'r06_paper/{base}_rep2', f'r06_paper/{base}_rep3']
                    selected = [by[i] for i in ids]
                    keys = sorted(outcomes[ids[0]])
                    y = np.mean([[outcomes[i][k] for k in keys] for i in ids], axis=0)
                    rng = np.random.default_rng(26092901)
                    boot = np.zeros(4000)
                    for t in range(10):
                        vals = np.array([y[k] for k, key in enumerate(keys) if int(key.split(':')[0]) == t])
                        boot += vals[rng.integers(50, size=(4000, 50))].mean(1)/10
                    N, V, M = (sum(r[k] for r in selected) for k in ['N', 'V', 'M'])
                    groups.append(dict(cell=f'{model}_{suite}_{size}', label=label, n=1500, init_clusters=500,
                        SR=float(y.mean()), SR_lo=float(np.quantile(boot, .025)), SR_hi=float(np.quantile(boot, .975)),
                        owner_IR=C1[model]*V/N+(1-C1[model])*M/N, runs=ids,
                        replicate_SR=[r['SR'] for r in selected], interval='task-stratified init cluster bootstrap; repeats kept together'))
    return groups


def analyze(points, outcomes, evidence, skipped):
    by = {r['id']: r for r in points}
    ni, mc, front, gaps, refs = [], [], {}, [], []
    pooled = pool_ab(points, outcomes)
    total_tests = 2*sum(2 if r['pure'] else 1 for r in points if r['eligible'])
    for model in C1:
        for suite in ['l10', 'spatial']:
            for length in [5, 10]:
                ids = reference_ids(model, suite, length)
                refs.append(dict(model=model, suite=suite, L=length, SR=float(np.mean([by[x]['SR'] for x in ids])),
                                 owner_IR=5/length, n=sum(by[x]['n'] for x in ids), init_clusters=500,
                                 constituent_SR=[by[x]['SR'] for x in ids], runs=ids))
            for size in [50, 500]:
                cell = f'{model}_{suite}_{size}'
                rows = [r for r in points if r['eligible'] and (r['cell'] == cell or (r['pure'] and r['model'] == model and r['suite'] == suite))]
                eff = sorted(pareto(rows), key=lambda r: r['owner_IR'])
                front[cell] = [r['id'] for r in eff]
                for r in rows:
                    if not r['pure']:
                        r['pareto'] = r in eff
                    for L in [5, 10]:
                        refids = reference_ids(model, suite, L)
                        lower = aggregate_lower([r['id']], refids, outcomes, .05)
                        simultaneous = aggregate_lower([r['id']], refids, outcomes, .05/total_tests)
                        reference_sr = float(np.mean([by[x]['SR'] for x in refids]))
                        d, blo, bhi = paired_boot([r['id']], refids, outcomes)
                        ni.append(dict(cell=cell, id=r['id'], reference_L=L, reference_SR=reference_sr,
                            delta_SR=d, margin=MARGIN, paired_exact_lower95=lower, NI_nominal=lower > -MARGIN,
                            paired_exact_lower_simultaneous=simultaneous, NI_simultaneous=simultaneous > -MARGIN,
                            bootstrap_delta_lo=blo, bootstrap_delta_hi=bhi, bootstrap_scope='sensitivity; 500 init clusters, never pooled independent replicates',
                            owner_IR=r['owner_IR'], pure=r['pure']))
                        for ref in refids:
                            w, l, n = paired_counts(r['id'], ref, outcomes)
                            p = float(binomtest(w, w+l, .5).pvalue) if w+l else 1.
                            mc.append(dict(cell=cell, candidate=r['id'], reference=ref, reference_L=L,
                                n=n, wins=w, losses=l, delta_SR=(w-l)/n, mcnemar_two_sided_exact_p=p,
                                NI_paired_exact_lower95=0. if r['id'] == ref else exact_lower(w, l, n, .05),
                                note='McNemar tests zero difference; NI uses paired exact risk-difference bound'))
                nonpure = [r for r in rows if not r['pure']]
                best = max(nonpure, key=lambda r: (r['SR'], -r['owner_IR']))
                b = next(r for r in pooled if r['cell'] == cell and r['label'] == 'B')
                for L in [5, 10]:
                    refids = reference_ids(model, suite, L)
                    reference_sr = np.mean([by[x]['SR'] for x in refids])
                    comparisons = [r for r in ni if r['cell'] == cell and r['reference_L'] == L]
                    qualifying = sorted([r for r in comparisons if r['NI_nominal']], key=lambda r: (r['owner_IR'], -by[r['id']]['SR']))
                    simultaneous = sorted([r for r in comparisons if r['NI_simultaneous']], key=lambda r: (r['owner_IR'], -by[r['id']]['SR']))
                    # The reference controller/declared reference mixture matches itself by identity.
                    # This is a baseline option, not another observed 500-episode arm.
                    identity_ref = dict(id=f'REFERENCE/{model}_{suite}_L{L}', owner_IR=5/L, paired_exact_lower95=0.)
                    if not qualifying or qualifying[0]['owner_IR'] >= 5/L:
                        qualifying.insert(0, identity_ref)
                    if not simultaneous or simultaneous[0]['owner_IR'] >= 5/L:
                        simultaneous.insert(0, identity_ref)
                    point = sorted([r for r in rows if r['SR'] >= reference_sr-1e-12], key=lambda r: (r['owner_IR'], -r['SR']))
                    within = sorted([r for r in rows if r['SR'] >= reference_sr-MARGIN], key=lambda r: (r['owner_IR'], -r['SR']))
                    gaps.append(dict(cell=cell, reference_L=L, reference_SR=float(reference_sr),
                        B_mean_SR=b['SR'], B_owner_IR=b['owner_IR'], B_loss_pp=100*(reference_sr-b['SR']),
                        best_cache_id=best['id'], best_cache_SR=best['SR'], best_cache_IR=best['owner_IR'], best_cache_loss_pp=100*(reference_sr-best['SR']),
                        minimum_observed_crossing=point[0]['id'] if point else None, observed_crossing_IR=point[0]['owner_IR'] if point else None,
                        minimum_point_within_2pp=within[0]['id'] if within else None, point_within_2pp_IR=within[0]['owner_IR'] if within else None,
                        minimum_NI=qualifying[0]['id'] if qualifying else None, minimum_NI_IR=qualifying[0]['owner_IR'] if qualifying else None,
                        minimum_NI_lower=qualifying[0]['paired_exact_lower95'] if qualifying else None,
                        minimum_simultaneous_NI=simultaneous[0]['id'] if simultaneous else None,
                        minimum_simultaneous_NI_IR=simultaneous[0]['owner_IR'] if simultaneous else None))
    for r in pooled:
        model, suite, size = r['cell'].split('_')
        for L in [5, 10]:
            ids = reference_ids(model, suite, L)
            d, lo, hi = paired_boot(r['runs'], ids, outcomes)
            r[f'delta_L{L}'] = d
            r[f'delta_L{L}_lo'], r[f'delta_L{L}_hi'] = lo, hi
            r[f'exact_lower_L{L}'] = aggregate_lower(r['runs'], ids, outcomes, .05)
    csv_out('frontier_points.csv', points)
    csv_out('noninferiority.csv', ni)
    csv_out('paired_mcnemar.csv', mc)
    csv_out('cell_gaps.csv', gaps)
    csv_out('paper_AB_recomputed.csv', pooled)
    csv_out('pure_references.csv', refs)
    dump('frontier_data.json', dict(points=points, frontiers=front, gaps=gaps, references=refs, pooled_AB=pooled,
        ni_tests=total_tests, margin=MARGIN, snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    report(points, front, gaps, refs, pooled, skipped, total_tests)


def report(points, front, gaps, refs, pooled, skipped, tests):
    by = {r['id']: r for r in points}
    text = ['# R6 owner IR–SR frontier (completed full evaluations)', '',
        'This is a retrospective observed frontier, not a guarantee about untested controllers or the global minimum IR. '
        'Pilot outcomes are excluded from these estimates. See `completion_plan.md` for placement-only pilot use.', '',
        '## Accounting and inclusion', '',
        f'Census: {len(points)} completed arm/reference records; {sum(r["eligible"] for r in points)} are eligible on the common 500-pair evaluation set. '
        f'Every excluded summary is listed in `skipped.csv` ({len(skipped)} records). `frontier_points.csv` also retains completed library variants and uncertain-cost rows with `eligible=false`.', '',
        'This frozen census contains 165 summary-ledger rows, 54 verified legacy raw-audit rows, four historical native-policy references, and two conflicting-cost rows. '
        'The exclusions are 147 smoke/pilot summaries, eight demo-bank variants, four grow250 holdouts, four AWM3 extra-bank-prior variants, and the two cost conflicts. '
        'Four DUAL references are explicitly added from outside os_closed_loop because PAPER_AB uses them for L=5.', '',
        'Owner IR = `(c1*v + (1-c1)*m)*5/L`, c1=.152 (pi05), .148 (GR00T), with v=V/N and m=M/N. '
        'L=5 for normal requests; pure L=10 has v=m=1 but costs .5 per five controls. Blind and policy-tail requests contribute neither V nor M. '
        'Summary eager/measured IR is ignored. Per the current request, wrist and K2 arms receive the same owner coefficients; no historical hardware discount is applied.', '',
        'For summaries without cost_ledger, reuse the earlier accepted raw-decision N/V/M audit in `../arms.json`, after verifying the current journal SHA and every recorded raw-log size/mtime. '
        'The two conflicting legacy R3 records remain visible but are excluded from the frontier. The actual served bank comes from `server.lib_mix`, verified against its manifest and episode.npy. '
        'Nominal 50/500 labels describe named banks, with actual source-episode counts in a separate column. demo100/200/300 and grow250 are separate banks; grow250 additionally evaluates only inits25–49. '
        'AWM3 prior_alpha=.5 uses 500-bank information with current-bank retrieval and is excluded from a strict 50-bank comparison.', '',
        'All comparable individual runs have exactly tasks0–9 × inits0–49 and validated accepted outcomes. Single-run SR intervals are Wilson 95%. '
        'A/B three-run aggregates retain the same 500 init clusters in their task-stratified bootstrap. Individual run frontiers are shown as requested; selecting their best replicate is optimistic. '
        'The A/B aggregate table below prevents confusing a winning single replicate with the reproducible headline. Historical DUAL L=5 references have matching task/init identities but different harness/seed provenance.', '',
        '## Noninferiority definition', '',
        'Margin epsilon=.02 absolute SR; nominal one-sided alpha=.05. Exact McNemar p-values test equality, **not** noninferiority. '
        '`paired_mcnemar.csv` gives wins/losses and exact two-sided binomial McNemar p for every matched run/reference pair. '
        'For NI use the conservative exact paired risk-difference lower bound '
        '`BetaQuantile(alpha/2; wins,n-wins+1) - BetaQuantile(1-alpha/2; losses+1,n-losses)`, with the usual 0/1 endpoint conventions. '
        'The two component Clopper–Pearson bounds cover jointly by Bonferroni; matching makes wins/losses the paired discordant events. '
        'Declare NI only if this lower bound is greater than −.02. This can be less powerful than optimized paired score procedures; failure is inconclusive, not inferiority.', '',
        'For the owner’s pi05 L=5 mean, combine the DUAL and two full-step r04_cost references. Bounds are averaged after dividing alpha among the constituent paired comparisons; never treat repeated init outcomes as independent. '
        'L=10 uses the owner’s r04_cost K10 reference (.904/.986), with the additional .900 l10 run still visible as another point. GR00T L=5 is DUAL, L=10 is r05_q2. '
        'A candidate identical to its reference has difference zero by identity. `REFERENCE/...` denotes the reference controller or declared reference mixture as an available baseline option at its known cost; it is not an extra observed run. '
        'A separate paired cluster-bootstrap interval is reported as a sensitivity, never mislabeled exact McNemar. '
        'These intervals use an independent-init/exchangeable-pair sampling model; the fixed, stratified benchmark is not a random sample of future robots or task families.', '',
        f'The nominal minimum is an exploratory selection over many arms. Also report simultaneous bounds with alpha=.05/{tests} across the {tests} cell/arm × reference tests (shared policy references counted conservatively in both library cells). '
        'A nominally passing single run therefore does not establish a portable or repeat-robust optimum.', '',
        '## Current gap and minimum observed costs', '',
        '| Cell | Reference | Pure SR | B SR @ IR (3 runs) | B loss pp | Cheapest point reaching reference SR: IR | Cheapest nominal 2 pp NI: IR | Simultaneous NI: IR |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for g in gaps:
        fmt = lambda x: 'unresolved' if x is None else f'{x:.3f}'
        text.append(f'| {g["cell"]} | L={g["reference_L"]} | {g["reference_SR"]:.3f} | {g["B_mean_SR"]:.3f} @ {g["B_owner_IR"]:.3f} | {g["B_loss_pp"]:+.2f} | {fmt(g["observed_crossing_IR"])} | {fmt(g["minimum_NI_IR"])} | {fmt(g["minimum_simultaneous_NI_IR"])} |')
    text += ['', 'Negative loss means SR is above the reference. Point crossing, being within 2 pp, and statistical NI are separate columns in `cell_gaps.csv`; none is an interpolation or an unobserved crossing.', '',
             '## Pareto points by cell', '', 'Dominance uses exact point estimates: no other eligible point has at least this SR at no greater IR, with one strict improvement. Ties are retained. Pure policy points require no cache bank and are shared across the two library columns.']
    for cell, ids in front.items():
        text += ['', f'### {cell}', '', '| Run / arm | Family | Library | SR [95%] | Owner IR |', '|---|---|---|---|---:|']
        for ident in ids:
            r = by[ident]
            text.append(f'| `{ident}` | {r["method_family"]} | {r["library"]} | {r["SR"]:.3f} [{r["SR_lo"]:.3f},{r["SR_hi"]:.3f}] | {r["owner_IR"]:.5f} |')
        for g in [g for g in gaps if g['cell'] == cell]:
            text += ['', f'L={g["reference_L"]} reference {g["reference_SR"]:.3f}: cheapest nominal NI point is `{g["minimum_NI"]}` '
                f'at IR {g["minimum_NI_IR"]}, lower paired SR difference {g["minimum_NI_lower"]}. '
                f'Best observed cache/hybrid SR is {g["best_cache_SR"]:.3f} at IR {g["best_cache_IR"]:.3f} (`{g["best_cache_id"]}`), loss {g["best_cache_loss_pp"]:+.2f} pp.']
    text += ['', '## Three-replicate A/B check', '', '| Cell | Arm | Replicate SRs | Mean SR [cluster 95%] | Owner IR |', '|---|---|---|---|---:|']
    for r in pooled:
        text.append(f'| {r["cell"]} | {r["label"]} | '+', '.join(f'{v:.3f}' for v in r['replicate_SR'])+f' | {r["SR"]:.3f} [{r["SR_lo"]:.3f},{r["SR_hi"]:.3f}] | {r["owner_IR"]:.5f} |')
    text += ['', '## Coverage by method family', '', '| Family | Eligible runs |', '|---|---:|']
    for fam, count in sorted(Counter(r['method_family'] for r in points if r['eligible']).items()):
        text.append(f'| {fam} | {count} |')
    text += ['', '## Reproduce / provenance', '',
        'Sources: `logs/offline_search_exploration.log.md` §10; rounds r04/r05 ANALYSIS; `r06/PAPER_AB.md`; each summary, emitted arms.json, accepted journal, served-bank manifest; the earlier Q2 raw audit. '
        '`source_evidence.json` records hashes/configurations; `summary_paths.json` freezes the census; `outcomes.json` contains accepted per-init outcomes. '
        'No simulator, GPU inference, server, worker, chain, or external host was invoked.', '', '```bash',
        'taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/build_frontier.py', '```', '',
        'Figure script and PNG/PDF are outside the repository: `/home/weiland/projects/openpi_ext/artifacts/frontier_r6/plot_frontier.py`. '
        'Pilot-only placement data and proposed configurations are separate artifacts, never frontier evidence.', '',
        '```bash',
        'taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /home/weiland/projects/openpi_ext/artifacts/frontier_r6/plot_frontier.py',
        'taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/validate_frontier.py', '```', '',
        'Validation output is `validation.json`; the figure was visually inspected. '
        '[Provisional completion plan](completion_plan.md), [existing-controller emit specs](emit_arms_existing.json), '
        '[PNG](/home/weiland/projects/openpi_ext/artifacts/frontier_r6/frontier_r6.png), '
        '[PDF](/home/weiland/projects/openpi_ext/artifacts/frontier_r6/frontier_r6.pdf).']
    (OUT/'frontier.md').write_text('\n'.join(text)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    points, outcomes, evidence, skipped = collect()
    analyze(points, outcomes, evidence, skipped)
    print(json.dumps(dict(points=len(points), eligible=sum(r['eligible'] for r in points), skipped=len(skipped), output=str(OUT))))


if __name__ == '__main__':
    main()
