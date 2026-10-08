"""Read-only confirmation of frozen B tables; writes only beside this script.

Run with the CPU command in HANDBACK.md. No fitting, selection, launch, GPU,
pickle execution, or mutation of B files. A is measurement only.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from analyze import (Reader, Episodes, require, sha, verify_protocol, paired,
                     uid_pair, markdown_table, cell_name, write_outputs)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
EMITTER = HERE.parent / 'descent/test_a_selected'
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50')
ARMS = ('pure_c10', 'uniform', 'task_descent')
SEED, DRAWS = 20261005, 5000
EXPECTED = [(t, i) for t in range(10) for i in range(50)]


def load_a(cell, levels, uniform, p, caps, reader, template):
    root = TRACE / ('r14_descent_test_' + cell)
    seal = reader.json(root / 'ROOT.json')
    require(seal['cell'] == cell and seal['init_pool'] == 'A' and
            seal['arm_count'] == 3 and seal['episodes_per_arm'] == 500, 'Wrong A design')
    for name, digest in seal['definitions_sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'Definition outside root')
        require(sha(reader.read(path)) == digest, f'Changed A definition: {path}')
    for path, digest in seal['fit_sha256'].items():
        require(sha(reader.read(path)) == digest, f'Changed fit: {path}')
    require(reader.read(root / 'eval500.json') == template, 'Changed R11 eval500 template')
    require(reader.json(root / 'eval500.json') == [list(x) for x in EXPECTED], 'Wrong A pairs')
    table = reader.json(root / 'descent_table.json')
    require(table['levels'] == levels and table['uniform'] == uniform, 'Changed frozen selection')
    require(table['template_sha256'] == sha(template), 'Wrong template hash')
    require(table['off_tasks'] == [t for t in range(10) if levels[str(t)] == 'off'], 'Wrong off mask')
    capvec = [0. if levels[str(t)] == 'off' else caps['cells'][cell]['periodic_cap_by_level'][levels[str(t)]]
              for t in range(10)]
    require(table['caps'] == capvec, 'Changed frozen caps')
    for level, digest in table['source_artifacts'].items():
        require(digest == caps['cells'][cell]['fit_sha256_by_level'][level], 'Wrong cap source')
    rows = reader.json(root / 'arms.json')
    names = {a: f'r14dt_{cell}_' + ('uniform_' + uniform if a == 'uniform' else a) for a in ARMS}
    require(len(rows) == 3 and {r['arm'] for r in rows} == set(names.values()) == set(seal['arms']), 'Wrong A arms')
    arms = {}
    positions = {pair: j for j, pair in enumerate(EXPECTED)}
    for arm in ARMS:
        name = names[arm]
        row = next(r for r in rows if r['arm'] == name)
        run = root / 'runs' / name
        level = 'c10' if arm == 'pure_c10' else uniform if arm == 'uniform' else 'task_descent'
        require(row['init_pool'] == 'A' and row['dev'] is False and row['episodes'] == 500 and
                row['r14_descent_level'] == level and row['model'] == cell.split('_')[0], 'Wrong arm metadata')
        require(row['replan_steps'] == 5 and row['full_model'] and not row['recording'], 'Wrong execution settings')
        require(Path(row['manifest']) == root / 'eval500.json', 'Wrong arm manifest pointer')
        require(reader.read(run / 'manifest.json') == template, 'Changed per-arm manifest')
        selected = reader.json(run / 'selection.json')
        # Legacy A selection.json only records the manifest pointer. A identity
        # is independently explicit in the sealed arm, every journal and summary.
        require(selected.get('init_pool', 'A') == 'A' and Path(selected['manifest']) == run / 'manifest.json', 'Wrong selection binding')
        fit = Path(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1])
        fit_digest = seal['fit_sha256'][str(fit)]
        if arm == 'task_descent':
            require(row['kwargs']['task_levels'] == levels and row['kwargs']['task_caps'] == capvec,
                    'Changed adapter settings')
            require(table['fit']['sha256'] == fit_digest and Path(table['fit']['path']) == fit, 'Wrong adapter fit')
        else:
            require(fit_digest == caps['cells'][cell]['fit_sha256_by_level'][level], 'Wrong frozen reference fit')

        accepted, errors, journal_count, duplicate_accepted = {}, 0, 0, 0
        for r in reader.jsonl(run / 'client/journal.jsonl'):
            journal_count += 1
            require(r.get('init_pool') == 'A', 'Non-A journal row')
            uid = r.get('task_uid')
            if uid:
                require(uid.startswith(name + ':') and uid_pair(uid) in positions, 'Unexpected journal UID')
            errors += bool(r.get('error'))
            if r.get('accepted') and r.get('status') in ('done', 'failed') and not r.get('error'):
                require(type(r.get('success')) is bool, 'Nonbinary journal outcome')
                duplicate_accepted += uid in accepted
                accepted[uid] = r
        by_pair = {uid_pair(uid): r for uid, r in accepted.items()}
        require(len(accepted) == len(by_pair) == 500 and set(by_pair) == set(EXPECTED), 'Missing/extra A episodes')
        y = np.array([by_pair[x]['success'] for x in EXPECTED], dtype=np.int64).reshape(10, 50)
        summary = reader.json(run / 'summary.json')
        require(summary['init_pool'] == 'A' and summary['arm'] == name and summary['model'] == row['model'], 'Wrong summary')
        require(summary['complete'] == 500 and summary['success'] == int(y.sum()) and
                math.isclose(summary['sr'], float(y.mean()), abs_tol=1e-12), 'Summary SR mismatch')
        require(summary['journal_rows'] == journal_count and summary['error_rows'] == errors, 'Summary journal mismatch')
        retried = sum(int(r.get('attempt', 1) or 1) > 1 for r in accepted.values())
        require(summary['retried'] == retried, 'Summary retry mismatch')
        require(all(summary['per_task'][str(t)] == {'n': 50, 'success': int(y[t].sum())} for t in range(10)),
                'Task summary mismatch')

        decs, duplicates, rejected, hosts, audit_rows = {}, 0, 0, set(), 0
        shards = sorted(run.glob('server_*/decisions_*.jsonl'))
        require(shards, 'Missing decision shards')
        reader.shards[str(run)] = list(map(str, shards))
        for path in shards:
            for d in reader.jsonl(path):
                if d.get('ev') == 'startup':
                    hosts.add(d['host'])
                    require(d['kwargs'] == row['kwargs'] and d['method_spec'] == row['method'], 'Wrong served method')
                    args = d['argv']
                    require(Path(args[args.index('--os-fit-artifact') + 1]) == fit, 'Wrong served artifact')
                    continue
                if d.get('ev') != 'dec':
                    continue
                uid = d['uid']
                require(uid.startswith(name + ':') and uid_pair(uid) in positions, 'Unexpected decision UID')
                require(d.get('init_pool', 'A') == 'A', 'Wrong decision pool')
                if uid not in accepted or int(d.get('attempt', 1) or 1) != int(accepted[uid].get('attempt', 1) or 1):
                    rejected += 1
                    continue
                step = d['step']
                require(type(step) is int and step >= 0, 'Invalid decision step')
                require(type(d.get('hit')) is bool and type(d.get('vision')) is bool, 'Nonboolean decision cost')
                require(d.get('ok', True) and (d['hit'] or d['vision']), 'Invalid full-policy decision')
                require(d.get('stage1_mode', 'full') == 'full' and d.get('replan_steps', 5) == 5, 'Unsupported cost mode')
                require(d['hit'] or float(d['miss_k']) == float(summary['cost_ledger']['full_stage_costs']['k']), 'Partial miss')
                require(int(d['task_id']) == uid_pair(uid)[0] and int(d['init']) == uid_pair(uid)[1], 'Wrong decision identity')
                diag = d.get('r14')
                if diag:
                    audit_rows += 1
                    require(diag['fit_sha256'] == fit_digest, 'Served fit hash mismatch')
                    if arm == 'task_descent' and 'descent_level' in diag:
                        require(diag['descent_level'] == levels[str(uid_pair(uid)[0])], 'Served task level mismatch')
                key = (uid, step)
                duplicates += key in decs
                # Keep only the last accepted-attempt decision in sorted shard/line order.
                decs[key] = (d['vision'], not d['hit'])
        n, v, m = (np.zeros(500, dtype=np.int64) for _ in range(3))
        steps = {uid: [] for uid in accepted}
        for (uid, step), (vision, miss) in decs.items():
            j = positions[uid_pair(uid)]
            n[j] += 1; v[j] += vision; m[j] += miss
            steps[uid].append(step)
        require(all(s and sorted(s) == list(range(len(s))) for s in steps.values()), 'Noncontiguous decisions')
        ledger = summary['cost_ledger']; full = ledger['full_stage_costs']
        require([int(x.sum()) for x in (n, v, m)] == [ledger[k] for k in ('decisions', 'vision_decisions', 'misses')],
                'N/V/M ledger mismatch')
        # k is the model's full denoising step count (10 for pi05, 8 for GR00T),
        # not the action chunk length. Each miss was checked against it above.
        require(ledger['controls'] == int(n.sum()) * 5 and float(full['k']) > 0., 'Wrong controls or policy length')
        require(set(ledger['l_per_request']['counts']) == {'5.0'} and
                set(ledger['stage1_modes']) == {'full'}, 'Ledger cost-mode mismatch')
        stage_ir = (full['s1'] * v.sum() + (full['s2'] + full['s3']) * m.sum()) / (
            sum(full[k] for k in ('s1', 's2', 's3')) * n.sum())
        require(math.isclose(stage_ir, ledger['ir_per_five_controls'], abs_tol=1e-8), 'Stage ledger IR mismatch')
        cv, cm = p['cost']['coefficients'][row['model']]
        arms[arm] = Episodes(y, n.reshape(10, 50), v.reshape(10, 50), m.reshape(10, 50),
                            (cv * v + cm * m).reshape(10, 50), dict(
            arm=name, root=str(root), accepted_episodes=500, journal_rows=journal_count, error_rows=errors,
            duplicate_accepted_rows=duplicate_accepted, duplicate_decisions=duplicates,
            rejected_attempt_decisions=rejected, retried_episodes=retried,
            selection_pool_field_present='init_pool' in selected,
            earliest_accepted_terminal_ts=min(r['ts'] for r in accepted.values()),
            hosts=sorted(hosts), decision_shards=len(shards), r14_audit_rows=audit_rows,
            summary_stage_ir=stage_ir, full_stage_costs=full, frozen_fit_sha256=fit_digest))
    return arms


def sum_sample(a, idx):
    return a[np.arange(10)[None, :, None], idx].sum(axis=(1, 2))


def ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def paired_boot(y, ref, sample_delta):
    gains = int(((y == 1) & (ref == 0)).sum())
    losses = int(((y == 0) & (ref == 1)).sum())
    return dict(n=int(y.size), gains=gains, losses=losses, delta=(gains-losses)/y.size,
                bootstrap_ci95=ci(sample_delta), bootstrap_lower95=float(np.quantile(sample_delta, .05)))


def ir_savings(task_stats, ref_stats, task_boot, ref_boot):
    delta = ref_stats['ir'] - task_stats['ir']
    bs = ref_boot['ir'] - task_boot['ir']
    return dict(ir_saved=delta, relative_ir_saved=delta/ref_stats['ir'], bootstrap_ci95=ci(bs),
                cost_saved_per_episode=ref_stats['cost_per_episode']-task_stats['cost_per_episode'],
                relative_cost_saved_per_episode=1-task_stats['cost_per_episode']/ref_stats['cost_per_episode'],
                relative_bootstrap_ci95=ci(bs/ref_boot['ir']))


def pool_stats(stats, coefficients):
    n = sum(s['decisions'] for s in stats)
    cost = sum(cv * s['vision'] + cm * s['misses'] for s, (cv, cm) in zip(stats, coefficients, strict=True))
    episodes = sum(s['n_episodes'] for s in stats)
    successes = sum(s['successes'] for s in stats)
    return dict(n_episodes=episodes, successes=successes, sr=successes/episodes, decisions=n,
                vision=sum(s['vision'] for s in stats), misses=sum(s['misses'] for s in stats),
                normalized_cost=cost, ir=cost/n, cost_per_episode=cost/episodes)


def optimism(bstats, astats):
    out = {'arms': {}}
    for arm in ARMS:
        b, a = bstats[arm], astats[arm]
        out['arms'][arm] = dict(b_predicted=b, test_a_measured=a,
                               sr_b_minus_a=b['sr']-a['sr'], ir_a_minus_b=a['ir']-b['ir'])
    out['contrasts'] = {}
    for arm, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
        bd = bstats[arm]['sr'] - bstats[ref]['sr']; ad = astats[arm]['sr'] - astats[ref]['sr']
        bs = bstats[ref]['ir'] - bstats[arm]['ir']; ass = astats[ref]['ir'] - astats[arm]['ir']
        out['contrasts'][arm + '_vs_' + ref] = dict(b_sr_delta=bd, a_sr_delta=ad,
            sr_advantage_optimism=bd-ad, b_ir_saved=bs, a_ir_saved=ass, ir_saving_optimism=bs-ass)
    return out


def calculate(all_arms, b, p, levels, uniforms):
    # Same sampled original suite/task/init for every arm AND both models.
    rng = np.random.default_rng(SEED)
    idxs = {suite: rng.integers(0, 50, size=(DRAWS, 10, 50)) for suite in ('l10', 'spatial')}
    result = dict(measurement_only=True, primary_tolerance_pp=5, levels=levels, uniform=uniforms,
        methods=dict(seed=SEED, replicates=DRAWS, resampling='whole paired episode within fixed suite/task; shared across arms and models',
            bootstrap_interval='percentile 2.5/97.5%; one-sided lower 5%; numpy linear quantiles',
            task_exact_lower='CP_low(g,n,.025)-CP_high(l,n,.025); no identical-policy shortcut for reruns',
            task_simultaneous_lower='CP tails .05/(2*40), forty model-task comparisons per contrast family',
            pooled_sr='equal model x suite x task mixture, 2000 episode outcomes per arm; 1000 original suite/task/init clusters',
            pooled_ir='sum_cell(cV_cell*V_cell+cM_cell*M_cell)/sum_cell(N_cell), not average cell IR',
            coefficients=p['cost']['coefficients'], inference_scope='fixed tasks and observed single runs; no between-run uncertainty'),
        cells={}, pooled={})
    boots, bstats = {}, {}
    for cell in CELLS:
        arms = all_arms[cell]; idx = idxs[cell.split('_')[1]]
        stats = {a: arms[a].stats() for a in ARMS}
        boot = {}
        for a in ARMS:
            ep = arms[a]
            bs = {k: sum_sample(getattr(ep, k), idx) for k in ('y', 'n', 'cost')}
            bs['sr'] = bs['y']/500; bs['ir'] = bs['cost']/bs['n']; boot[a] = bs
            stats[a]['normalized_cost'] = float(ep.cost.sum())
        c = dict(arms=stats, comparisons={}, ir_savings={}, tasks={}, audit={a: arms[a].audit for a in ARMS})
        for a, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
            c['comparisons'][a + '_vs_' + ref] = paired_boot(arms[a].y, arms[ref].y, boot[a]['sr']-boot[ref]['sr'])
        for ref in ('pure_c10', 'uniform'):
            c['ir_savings']['vs_' + ref] = ir_savings(stats['task_descent'], stats[ref], boot['task_descent'], boot[ref])
        for t in range(10):
            tr = dict(level=levels[cell][str(t)], pure_successes=int(arms['pure_c10'].y[t].sum()),
                      task_successes=int(arms['task_descent'].y[t].sum()), uniform_successes=int(arms['uniform'].y[t].sum()), n=50)
            for a in ('task_descent', 'uniform'):
                exact = paired(arms[a].y[t], arms['pure_c10'].y[t], family=40)
                delta = (arms[a].y[t] - arms['pure_c10'].y[t])[idx[:, t, :]].mean(axis=1)
                exact.update(bootstrap_ci95=ci(delta), bootstrap_lower95=float(np.quantile(delta, .05)),
                             below_minus_5pp=100*(exact['successes']-exact['pure_successes']) < -5*50,
                             exact_lower_within_5pp=exact['lower95'] >= -.05,
                             simultaneous_lower_within_5pp=exact['simultaneous_lower95'] >= -.05)
                tr[a + '_vs_pure'] = exact
            c['tasks'][str(t)] = tr
        c['task_tolerance'] = {a: dict(
            below_minus_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['below_minus_5pp']],
            exact_lower_within_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['exact_lower_within_5pp']],
            simultaneous_lower_within_5pp=[t for t, tr in c['tasks'].items() if tr[a+'_vs_pure']['simultaneous_lower_within_5pp']])
            for a in ('task_descent', 'uniform')}
        bcell = b['cells'][cell]; tol = bcell['tolerances']['5']
        require(all(arms[a].audit['full_stage_costs'] == bcell['arm_stats']['c10']['full_stage_costs'] for a in ARMS),
                'A full-policy cost specification differs from frozen B')
        bstats[cell] = dict(pure_c10=bcell['arm_stats']['c10'], uniform=tol['savings']['uniform'], task_descent=tol['savings']['task'])
        c['selection_optimism'] = optimism(bstats[cell], stats)
        c['frozen_b_library_agreement'] = tol['library']
        c['frozen_b_matched_ir'] = bcell['matched_ir']
        result['cells'][cell] = c; boots[cell] = boot
    coeffs = [p['cost']['coefficients'][c.split('_')[0]] for c in CELLS]
    pool = result['pooled']; pool['arms'] = {}; pool['comparisons'] = {}; pool['ir_savings'] = {}
    pb, bpool = {}, {}
    for arm in ARMS:
        pool['arms'][arm] = pool_stats([result['cells'][c]['arms'][arm] for c in CELLS], coeffs)
        bpool[arm] = pool_stats([bstats[c][arm] for c in CELLS], coeffs)
        samples = {k: sum(boots[c][arm][k] for c in CELLS) for k in ('y', 'n', 'cost')}
        samples['sr'] = samples['y']/2000; samples['ir'] = samples['cost']/samples['n']; pb[arm] = samples
    for a, ref in [('task_descent', 'pure_c10'), ('uniform', 'pure_c10'), ('task_descent', 'uniform')]:
        pool['comparisons'][a + '_vs_' + ref] = paired_boot(np.stack([all_arms[c][a].y for c in CELLS]),
            np.stack([all_arms[c][ref].y for c in CELLS]), pb[a]['sr'] - pb[ref]['sr'])
    for ref in ('pure_c10', 'uniform'):
        pool['ir_savings']['vs_' + ref] = ir_savings(pool['arms']['task_descent'], pool['arms'][ref], pb['task_descent'], pb[ref])
    pool['task_tolerance'] = {a: {key: [f'{c}:{t}' for c in CELLS for t in result['cells'][c]['task_tolerance'][a][key]]
        for key in ('below_minus_5pp', 'exact_lower_within_5pp', 'simultaneous_lower_within_5pp')}
        for a in ('task_descent', 'uniform')}
    pool['selection_optimism'] = optimism(bpool, pool['arms'])
    return result


def f(value, scale=1, digits=2):
    return f'{value*scale:.{digits}f}'


def interval_text(value, scale=100):
    return '[' + ', '.join(f(x, scale) for x in value) + ']'


def scopes(result):
    return [*result['cells'].items(), ('pooled', result['pooled'])]


def stats_table(result):
    return markdown_table(['Cell', 'Arm', 'Success / episodes', 'SR %', 'N', 'V', 'M', 'Owner IR', 'Cost / episode'], [
        [cell, arm, f"{s['successes']}/{s['n_episodes']}", f(s['sr'],100), s['decisions'], s['vision'], s['misses'], f(s['ir'],1,6), f(s['cost_per_episode'],1,6)]
        for cell, c in scopes(result) for arm, s in c['arms'].items()])


def render_test(result):
    lines = ['# R14 test-A confirmation: frozen per-task descent', '',
        'Measurement only. Both tables were frozen from B before these runs. No A outcome changes any level, cap, stopping rule, tolerance or arm. All tables below are generated by `analyze_test_a.py` from accepted terminal journals and their accepted-attempt decision logs.', '',
        'The four cells each have three arms × 500 episodes, 50 inits per task. All use the byte-identical R11 eval500 A template. Runs were on the local 4090 per the execution handoff; startup host names are audited below. The original H100 host was unavailable. π0.5 host insensitivity is inherited from R11 as supplied by the owner; this analysis does not newly test it. Host metadata still says intended H100 in some sealed definitions.', '',
        '## Estimands and uncertainty', '',
        'SR uses all accepted task/init outcomes. IR = (cV·V + cM·M)/N with frozen π0.5 coefficients 0.152/0.848 and GR00T 0.148/0.852. N counts nominal five-control requests, V full-vision calls and M full-policy calls. Pooled IR sums each model’s priced numerator and all N, rather than averaging cell IR. A terminal chunk may execute fewer controls; this is the frozen nominal-control estimand, not wall time. Pure C10 reuses the second half of ten-action chunks, so its IR is approximately 0.5. Off disables extra knob calls; vision and mandatory inference remain.', '',
        f'Paired bootstrap: {DRAWS:,} whole-episode resamples, seed {SEED}, independently within each fixed suite/task; the same sampled init is used for all arms and both models. Pooled SR has 2,000 outcomes/arm but 1,000 original suite/task/init clusters. Tasks and model/suite weights are fixed. Two-sided intervals use 2.5/97.5 percentiles; one-sided lower bounds use the fifth percentile. These are descriptive intervals conditional on the frozen table and observed single runs; there is no estimate of between-run variance or new-task generalization.', '',
        'Per-task conservative paired lower95 = CP_low(g,n,.025) − CP_high(l,n,.025); the simultaneous lower95 uses tail .05/(2×40) over forty model-task comparisons within each contrast family. Exact two-sided intervals in JSON use four .0125 tails. Even if the selected level is C10, its A run is a separate measurement: no zero-width identity shortcut. Bootstrap intervals can degenerate with zero discordances; use the exact lower bounds for noninferiority interpretation. A point loss strictly greater than 5 pp is counted as a violation. Marginal or pooled success does not establish the every-task requirement.', '',
        '## Three-arm SR and frozen-price IR', '', stats_table(result),
        '## Paired SR contrasts', '', markdown_table(['Cell','Candidate − reference','Gains / losses','ΔSR pp','Bootstrap 95% interval pp','Bootstrap lower95 pp'], [
            [cell, name, f"{s['gains']}/{s['losses']}", f(s['delta'],100), interval_text(s['bootstrap_ci95']), f(s['bootstrap_lower95'],100)]
            for cell,c in scopes(result) for name,s in c['comparisons'].items()]),
        '## Task-descent IR savings (negative means more cost)', '',
        markdown_table(['Cell','Reference','IR saved pp','Bootstrap 95% interval pp','Relative saved %','Relative 95% interval %','Cost saved / episode','Relative cost saved / episode %'], [
            [cell, ref, f(s['ir_saved'],100), interval_text(s['bootstrap_ci95']), f(s['relative_ir_saved'],100), interval_text(s['relative_bootstrap_ci95']),f(s['cost_saved_per_episode'],1,6),f(s['relative_cost_saved_per_episode'],100)]
            for cell,c in scopes(result) for ref,s in c['ir_savings'].items()]),
        '## Every-task tolerance check', '',
        markdown_table(['Cell','Policy','Tasks below pure − 5 pp','Task IDs','Marginal exact lower ≥ −5 pp','Simultaneous lower ≥ −5 pp'], [
            [cell,a,len(s['below_minus_5pp']), ', '.join(s['below_minus_5pp']) or 'none',len(s['exact_lower_within_5pp']),len(s['simultaneous_lower_within_5pp'])]
            for cell,c in scopes(result) for a,s in c['task_tolerance'].items()]),
        markdown_table(['Cell','Task','Frozen level','Pure /50','Descent /50','Pure SR %','Descent SR %','Gains/losses','ΔSR pp','Bootstrap CI95 pp','Bootstrap lower95 pp','Exact lower95 pp','Simultaneous lower95 pp','Below −5 pp'], [
            [cell,t,tr['level'],tr['pure_successes'],tr['task_successes'],f(s['pure_sr'],100),f(s['sr'],100),f"{s['gains']}/{s['losses']}",f(s['delta'],100),interval_text(s['bootstrap_ci95']),f(s['bootstrap_lower95'],100),f(s['lower95'],100),f(s['simultaneous_lower95'],100),'YES' if s['below_minus_5pp'] else 'no']
            for cell,c in result['cells'].items() for t,tr in c['tasks'].items() for s in [tr['task_descent_vs_pure']]]),
        'Uniform per-task successes, gains/losses, bootstrap and exact bounds are also retained in `test_a.json`.', '',
        '## B stitched prediction versus independent execution on A', '',
        'B was selected and measured on the same development episodes; its task-descent result stitches whole selected task trajectories from uniform-level runs. A actually executes the fixed mixed controller. The signed B−A SR gap and A−B IR gap are observed transfer discrepancies, not an unbiased decomposition of selection bias. They also include pool, rollout, host and stitched-versus-executed effects. Pure-relative contrast optimism subtracts the pure-reference shift; it is the more informative selection diagnostic. Negative optimism is retained. No B/A paired interval is implied.', '',
        markdown_table(['Cell','Arm','B SR %','A SR %','B−A SR pp','B IR','A IR','A−B IR pp'], [
            [cell,a,f(s['b_predicted']['sr'],100),f(s['test_a_measured']['sr'],100),f(s['sr_b_minus_a'],100),f(s['b_predicted']['ir'],1,6),f(s['test_a_measured']['ir'],1,6),f(s['ir_a_minus_b'],100)]
            for cell,c in scopes(result) for a,s in c['selection_optimism']['arms'].items()]),
        markdown_table(['Cell','Contrast','B ΔSR pp','A ΔSR pp','SR advantage optimism pp','B IR saved pp','A IR saved pp','IR-saving optimism pp'], [
            [cell,name,f(s['b_sr_delta'],100),f(s['a_sr_delta'],100),f(s['sr_advantage_optimism'],100),f(s['b_ir_saved'],100),f(s['a_ir_saved'],100),f(s['ir_saving_optimism'],100)]
            for cell,c in scopes(result) for name,s in c['selection_optimism']['contrasts'].items()]),
        '## Input and accounting audit', '',
        markdown_table(['Cell','Arm','Accepted','Error rows','Retries','Duplicate accepted','Duplicate decisions','Rejected attempt decisions','Shards','Startup hosts','Stage-ledger IR'], [
            [cell,a,s['accepted_episodes'],s['error_rows'],s['retried_episodes'],s['duplicate_accepted_rows'],s['duplicate_decisions'],s['rejected_attempt_decisions'],s['decision_shards'],','.join(s['hosts']),f(s['summary_stage_ir'],1,6)]
            for cell,c in result['cells'].items() for a,s in c['audit'].items()]),
        'The original stage-ledger IR is shown only for reconciliation; all comparisons use the frozen owner coefficients. Every root definition, manifest, fitted artifact, served method/table, accepted identity, per-task success total, contiguous decision sequence and N/V/M ledger is checked. Last error-free accepted terminal journal record per UID wins; normal task failures count. Only that attempt’s decisions count, with last UID/step in sorted shard/line order. All consumed files are SHA-256 snapshotted and rehashed before writing; the shard sets are rechecked. Frozen protocol seals and all B output seals must pass.', '',
        'A loader schema adaptations: legacy A selection.json contains the manifest pointer without an init_pool field; the exact R11 manifest bytes and A labels in root/arm/journal/summary independently bind the pool, and an explicitly wrong pool is rejected. The full-policy k field is the model-specific denoising count (π0.5: 10; GR00T: 8), not action chunk length; every miss matches its model’s full k and the full cost specification matches frozen B. Initial runs aborted on these schema assumptions before writing results; only this adapter was corrected. No frozen B analysis or selection was edited.', '',
        'Exact episode-level y/N/V/M/cost arrays, all full-precision metrics, software versions, source hashes and input hashes are in `test_a.json`. Rerun with the command in `HANDBACK.md`.', '']
    return '\n'.join(lines)


def render_report(r):
    pool = r['pooled']; tol = pool['task_tolerance']; arms = pool['arms']
    opt = pool['selection_optimism']; adv = opt['contrasts']['task_descent_vs_pure_c10']
    saving = pool['ir_savings']['vs_pure_c10']; usaving = pool['ir_savings']['vs_uniform']
    lib = [s for c in r['cells'].values() for s in c['frozen_b_library_agreement'].values()]
    raw = opt['arms']['task_descent']
    fail_count = len(tol['task_descent']['below_minus_5pp'])
    level_rows = [[cell_name(c), *[r['levels'][c][str(t)] for t in range(10)], r['uniform'][c]] for c in CELLS]
    summary_rows = []
    for cell, c in scopes(r):
        summary_rows.append(['汇总' if cell == 'pooled' else cell_name(cell),
            *[f(c['arms'][a]['sr'],100) + ' / ' + f(c['arms'][a]['ir'],1,3) for a in ARMS],
            len(c['task_tolerance']['task_descent']['below_minus_5pp']),
            f(c['ir_savings']['vs_pure_c10']['ir_saved'],100),f(c['ir_savings']['vs_uniform']['ir_saved'],100)])
    lines = ['# 给负责人的结论', '',
        '先在开发集按“每任务成功率最多下降 5 个百分点、首次不达标即停”逐级选档，并冻结逐任务表和统一档位；随后在测试集独立执行三种策略，每组每种策略 500 回合。测试结果只作确认，没有重新选档。', '',
        markdown_table(['模型 / 任务组', *[f'任务 {t}' for t in range(10)], '统一档位'], level_rows),
        'c10 是纯推理；off 关闭额外旋钮调用，仍保留视觉和必要推理；ir 数字是档位名，不是实测算力。下表每格为“成功率 % / IR”，IR 越低越省；节省为负表示更费。', '',
        markdown_table(['模型 / 任务组','纯推理','统一档位','逐任务表','逐任务超容差数','比纯推理省 IR 百分点','比统一省 IR 百分点'],summary_rows),
        f'（1）没有守住“每个任务都在容差内”。逐任务表有 {fail_count}/40 个模型×任务的实测下降超过 5 个百分点，统一档位也有 {len(tol["uniform"]["below_minus_5pp"])}/40 个。逐任务表汇总比纯推理少成功 {f(-pool["comparisons"]["task_descent_vs_pure_c10"]["delta"],100)} 个百分点，配对 95% 区间为 {interval_text(pool["comparisons"]["task_descent_vs_pure_c10"]["bootstrap_ci95"])}，相对节省 IR {f(saving["relative_ir_saved"],100)}%；比统一档位多成功 {f(pool["comparisons"]["task_descent_vs_uniform"]["delta"],100)} 个百分点，但 IR 增加 {f(-usaving["relative_ir_saved"],100)}%。平均达标不能代替逐任务达标。', '',
        f'（2）库估计只能提供有限的排序线索，尚不能可靠决定档位。四组排序相关依次为 {"、".join(f(r["cells"][c]["frozen_b_library_agreement"]["ir32"]["spearman"],1,3) for c in CELLS)}；两种预算映射的档位完全吻合率为 {f(min(s["exact_fraction"] for s in lib),100,0)}–{f(max(s["exact_fraction"] for s in lib),100,0)}%。有限预算上限本来就不能预测 off；估计还用过较早的开发集校准。本次测试只运行冻结表，不能反推新的任务边界。', '',
        f'（3）开发集对成功率收益偏乐观，算力预测较准。逐任务表汇总成功率预测 {f(raw["b_predicted"]["sr"],100)}%，测试为 {f(raw["test_a_measured"]["sr"],100)}%，高估 {f(raw["sr_b_minus_a"],100)} 个百分点；扣除纯推理基线变化后，相对优势高估 {f(adv["sr_advantage_optimism"],100)} 个百分点。IR 预测 {f(raw["b_predicted"]["ir"],1,4)}，实测 {f(raw["test_a_measured"]["ir"],1,4)}。这些差额也包含初始状态、单次运行及轨迹拼合与实际执行的差异，不能全算作选档偏差。', '',
        '第 5 层建议：把逐任务降档保留为开发工具，暂不作为“逐任务守住成功率”的部署保证。下一轮只在开发集增加独立确认与重复运行、采用更保守的逐任务规则，再用新的未用评估数据确认；库估计只辅助安排试验。本次表保持冻结，不能按测试中的坏任务补档。', '',
        '## Failure detail and limits', '',
        markdown_table(['Cell','Tasks below −5 pp','Task-descent ΔSR pp / exact paired lower95 pp'],[
            [cell, ', '.join(c['task_tolerance']['task_descent']['below_minus_5pp']), '; '.join(
                t + ': ' + f(c['tasks'][t]['task_descent_vs_pure']['delta'],100) + ' / ' + f(c['tasks'][t]['task_descent_vs_pure']['lower95'],100)
                for t in c['task_tolerance']['task_descent']['below_minus_5pp'])] for cell,c in r['cells'].items()]),
        f'All four cell-average task-descent point losses are within the registered tolerance, while each cell has task violations. Only {len(tol["task_descent"]["exact_lower_within_5pp"])}/40 task comparisons has a marginal conservative paired lower bound at or above −5 pp; {len(tol["task_descent"]["simultaneous_lower_within_5pp"])}/40 meet the simultaneous lower-bound criterion. The point violations are observed losses, not forty individually proven population effects. Neither the cell means nor the bootstrap interval for the pooled effect certify every task.', '',
        f'GR00T spatial task 5 retains C10, yet its success changes from {f(r["cells"]["groot_spatial_50"]["tasks"]["5"]["task_descent_vs_pure"]["pure_sr"],100)}% to {f(r["cells"]["groot_spatial_50"]["tasks"]["5"]["task_descent_vs_pure"]["sr"],100)}%. B stitching reused the same reference trajectory for a C10-selected task; A reruns it inside the mixed arm. This shows why not every observed loss can be attributed to a reduced knob. One run per arm cannot separate stochastic/runtime variation from controller effects. Do not delete this failure or substitute its pure outcome.', '',
        f'π0.5 long-task descent is the most promising cell-level tradeoff versus uniform: ΔSR {f(r["cells"]["pi05_l10_50"]["comparisons"]["task_descent_vs_uniform"]["delta"],100)} pp with interval {interval_text(r["cells"]["pi05_l10_50"]["comparisons"]["task_descent_vs_uniform"]["bootstrap_ci95"])} and IR saving {f(r["cells"]["pi05_l10_50"]["ir_savings"]["vs_uniform"]["ir_saved"],100)} pp, but its every-task requirement still fails. Both spatial cells pay substantially more than uniform off; GR00T spatial gains only {f(r["cells"]["groot_spatial_50"]["comparisons"]["task_descent_vs_uniform"]["delta"],100)} pp SR.', '',
        f'GR00T long-task descent lowers the IR ratio versus uniform by {f(r["cells"]["groot_l10_50"]["ir_savings"]["vs_uniform"]["ir_saved"],100)} pp, but has lower SR and increases normalized cost per episode by {f(-r["cells"]["groot_l10_50"]["ir_savings"]["vs_uniform"]["relative_cost_saved_per_episode"],100)}%. Its two arms have the same total policy-call count ({r["cells"]["groot_l10_50"]["arms"]["uniform"]["misses"]}); descent uses more requests. A lower ratio alone is not a reduction in total episode work. No same-realized-IR superiority claim follows from these three-arm A results.', '',
        '## How optimistic was B?', '',
        markdown_table(['Cell','Descent B−A SR pp','B descent−pure pp','A descent−pure pp','Pure-relative optimism pp','B descent IR','A descent IR','A−B IR pp'],[
            [cell,f(o['arms']['task_descent']['sr_b_minus_a'],100),f(s['b_sr_delta'],100),f(s['a_sr_delta'],100),f(s['sr_advantage_optimism'],100),f(o['arms']['task_descent']['b_predicted']['ir'],1,6),f(o['arms']['task_descent']['test_a_measured']['ir'],1,6),f(o['arms']['task_descent']['ir_a_minus_b'],100)]
            for cell,c in scopes(r) for o in [c['selection_optimism']] for s in [o['contrasts']['task_descent_vs_pure_c10']]]),
        f'The pooled advantage over uniform shrinks from {f(opt["contrasts"]["task_descent_vs_uniform"]["b_sr_delta"],100)} pp on B to {f(opt["contrasts"]["task_descent_vs_uniform"]["a_sr_delta"],100)} pp on A. Negative raw B−A SR gaps and negative IR-saving optimism are retained in [TEST_A.md](TEST_A.md). The pooled IR may move in a different direction from every cell IR because each pool uses its own decision-count weights. B and A outcomes are not paired across pools. These are observed discrepancies, with no claim to isolate a causal winner’s-curse term.', '',
        '## Library agreement with the frozen B descent', '',
        markdown_table(['Cell','Frozen allocation target','Spearman ρ','Kendall τ-b','Exact level %','Within one level %','Mean absolute ordinal error'],[
            [cell,dose,f(s['spearman'],1,3),f(s['kendall_tau_b'],1,3),f(s['exact_fraction'],100,0),f(s['within_one_fraction'],100,0),f(s['mean_absolute_ordinal_error'])]
            for cell,c in r['cells'].items() for dose,s in c['frozen_b_library_agreement'].items()]),
        'This reproduces the sealed B result, including both budgets and all cells. Priorities are library-derived; cap bias also used earlier B state-path calibration. Caps are budget allocations rather than calibrated SR tolerance boundaries, and the registered finite-cap mapping cannot output off. With ten tasks per cell, ties and discrete levels limit the rank evidence. A validates only the selected controller, not the unmeasured full A descent curve.', '',
        'For context, the original frozen library allocation was also compared with uniform periodic on B:', '',
        markdown_table(['Cell','Target','Allocation−uniform IR pp','Paired ΔSR pp','Exact-IR expected-mixture ΔSR pp'],[
            [cell,dose,f(s['actual_ir_gap'],100),f(s['paired_sr']['delta'],100),f(s['exact_ir_expected_mixture']['allocation_minus_expected_uniform_sr'],100)]
            for cell,c in r['cells'].items() for dose,s in c['frozen_b_matched_ir'].items()]),
        'These small B effects have mixed signs. All nominal pairs meet the previously registered approximate IR-gap criterion; the exact-IR comparison is an expected mixture of adjacent uniform trajectories, not an observed deterministic controller. Neither it nor the rank table demonstrates a general deployment benefit.', '',
        '## Reproducibility and scope', '',
        'The confirmation covers the four model×suite cells at library size 50. It does not answer the broader layer-5 component-ablation or larger-library questions. All A runs were on the local 4090 (the intended H100 host became unavailable); the R11 π0.5 host-insensitivity finding is inherited from the supplied handoff. No fresh host-equivalence claim or wall-time saving is inferred.', '',
        f'[TEST_A.md](TEST_A.md) contains every arm’s N/V/M, frozen-price SR/IR, {DRAWS:,}-draw paired cluster-bootstrap intervals (seed {SEED}), all task lower bounds, signed savings and full B/A comparisons. [test_a.json](test_a.json) contains episode arrays and hashed provenance. [PROTOCOL.md](PROTOCOL.md), [DESCENT.md](DESCENT.md), [levels.json](levels.json) and [uniform.json](uniform.json) remain unchanged. [analyze_test_a.py](analyze_test_a.py) regenerates this report and the handback as well as the measurement outputs.', '']
    return '\n'.join(lines)


def render_handback(r):
    p = r['pooled']
    return '\n'.join(['# R14 final descent handback', '',
        'Complete: [REPORT.md](REPORT.md) leads with the short Chinese owner conclusion and frozen task table; [TEST_A.md](TEST_A.md) has full confirmation statistics; [test_a.json](test_a.json) includes episode arrays and input hashes. No test-A selection or tuning; frozen protocol, B outputs, levels and uniforms are unchanged.', '',
        f'Confirmation audited all {sum(c["audit"][a]["accepted_episodes"] for c in r["cells"].values() for a in ARMS):,} accepted episodes. Task-descent SR {f(p["arms"]["task_descent"]["sr"],100)}% vs pure {f(p["arms"]["pure_c10"]["sr"],100)}%; paired Δ {f(p["comparisons"]["task_descent_vs_pure_c10"]["delta"],100)} pp, CI {interval_text(p["comparisons"]["task_descent_vs_pure_c10"]["bootstrap_ci95"])}. There are {len(p["task_tolerance"]["task_descent"]["below_minus_5pp"])}/40 task point violations. Pooled IR {f(p["arms"]["task_descent"]["ir"],1,6)} saves {f(p["ir_savings"]["vs_pure_c10"]["relative_ir_saved"],100)}% vs pure but costs {f(-p["ir_savings"]["vs_uniform"]["relative_ir_saved"],100)}% more than uniform. B pure-relative SR advantage optimism: {f(p["selection_optimism"]["contrasts"]["task_descent_vs_pure_c10"]["sr_advantage_optimism"],100)} pp. Do not advertise an every-task guarantee or matched-IR superiority.', '',
        'Rerun from the repository root (only analysis and report writes):', '', '```bash',
        'taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r14/descent_astra/analyze_test_a.py', '```', '',
        'A schema adaptations are documented in TEST_A.md. The script verifies frozen B seals, A manifests/definitions/fits, paired accepted attempts, all decision ledgers and input stability before writing. All analysis writes stay in this directory. No git, experiment launch, remote command or GPU use. Recommend a separately versioned B-only development confirmation before any deployment claim; never repair the table from A.', ''])


def main():
    require(set(os.sched_getaffinity(0)) <= set(range(10,22)) | set(range(54,66)), 'Use required CPU affinity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide GPUs')
    reader = Reader(); reader.read(__file__); reader.read(HERE / 'analyze.py')
    p = verify_protocol(reader)
    prov = reader.json(HERE / 'provenance.json')
    for name, digest in prov['selected_output_sha256'].items():
        require(sha(reader.read(HERE / name)) == digest, f'Frozen B output changed: {name}')
    b = reader.json(HERE / 'analysis.json')
    levels = reader.json(HERE / 'levels.json'); uniforms = reader.json(HERE / 'uniform.json')
    require(set(levels) == set(uniforms) == set(CELLS), 'Incomplete frozen tables')
    require(reader.json(EMITTER / 'input.json') == dict(levels=levels, uniform=uniforms), 'Emitter input differs from B')
    for cell in CELLS:
        require(b['cells'][cell]['tolerances']['5']['levels'] == levels[cell] and
                b['cells'][cell]['tolerances']['5']['uniform'] == uniforms[cell], 'B table mismatch')
    caps = reader.json(HERE.parent / 'descent/cap_tables.json')
    templates = [reader.read(TRACE / f'r11_knob_{i}/eval500.json') for i in range(1,5)]
    require(len(set(templates)) == 1, 'R11 A templates differ')
    all_arms = {}
    for cell in CELLS:
        print('Read and audit A:', cell, flush=True)
        all_arms[cell] = load_a(cell, levels[cell], uniforms[cell], p, caps, reader, templates[0])
    btime = datetime.fromisoformat(prov['analyzed_at_utc']).timestamp()
    require(all(btime < ep.audit['earliest_accepted_terminal_ts'] for a in all_arms.values() for ep in a.values()),
            'B analysis not earlier than A completion records')
    result = calculate(all_arms, b, p, levels, uniforms)
    result['episode_data'] = {c: {a: {k:getattr(ep,k).tolist() for k in ('y','n','v','m','cost')} for a,ep in arms.items()} for c,arms in all_arms.items()}
    result['provenance'] = dict(protocol_frozen_at_utc=p['frozen_at_utc'], inputs=reader.inputs,
        decision_shards=reader.shards, cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
        numpy=np.__version__, bootstrap_draws=DRAWS, bootstrap_seed=SEED, pool='A', reselection=False,
        frozen_b_output_sha256=prov['selected_output_sha256'])
    print('Rehash inputs and verify stable shard sets', flush=True)
    reader.stable()
    write_outputs({'test_a.json': result, 'TEST_A.md': render_test(result),
                   'REPORT.md': render_report(result), 'HANDBACK.md': render_handback(result)}, HERE)
    print(json.dumps({c: dict(sr={a:s['sr'] for a,s in x['arms'].items()},
        ir={a:s['ir'] for a,s in x['arms'].items()}, violations=x['task_tolerance']['task_descent']['below_minus_5pp'])
        for c,x in scopes(result)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
