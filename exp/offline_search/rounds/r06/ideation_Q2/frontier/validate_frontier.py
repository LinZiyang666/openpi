"""Independent accounting, support, frontier, statistics, and plan checks.

Reads the frozen outputs and their completed-run sources; never invokes a run.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.stats import beta, binomtest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
ART = Path('/home/weiland/projects/openpi_ext/artifacts/frontier_r6')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = json.loads((HERE/'frontier_data.json').read_text())
    evidence = json.loads((HERE/'source_evidence.json').read_text())
    outcomes = json.loads((HERE/'outcomes.json').read_text())
    points = data['points']
    by = {p['id']: p for p in points}
    assert len(by) == len(points)
    expected = {f'{t}:{i}' for t in range(10) for i in range(50)}
    source_checks, paired_checks, dominance_checks = 0, 0, 0
    for p in points:
        y = outcomes[p['id']]
        assert sum(y.values()) == p['success'] and len(y) == p['n']
        assert np.isclose(p['SR'], sum(y.values())/len(y))
        assert p['SR_lo'] <= p['SR'] <= p['SR_hi']
        if p['eligible']:
            assert set(y) == expected and p['cost_valid']
            assert p['pure'] or p['library'] in ('current', 'bpool_cs', 'bpool_all')
        if p['cost_valid']:
            assert 0 <= p['M'] <= p['V'] <= p['N']
            assert np.isclose(p['v'], p['V']/p['N']) and np.isclose(p['m'], p['M']/p['N'])
            c = .152 if p['model'] == 'pi05' else .148
            assert np.isclose(p['owner_IR'], (c*p['V']+(1-c)*p['M'])/p['N']*5/p['L'])
        if p['pure']:
            assert p['owner_IR'] == 5/p['L']
        ev = evidence[p['id']]
        assert sha(p['journal']) == ev['journal_sha256']
        source_checks += 1
        if p['summary']:
            assert sha(p['summary']) == ev['summary_sha256']
            source_checks += 1
    for cell, ids in data['frontiers'].items():
        model, suite, _ = cell.split('_')
        rows = [p for p in points if p['eligible'] and (p['cell'] == cell or
                (p['pure'] and p['model'] == model and p['suite'] == suite))]
        # Independent cost-sorted scan; retain exact ties.
        running_best, frontier = -math.inf, set()
        for cost in sorted({p['owner_IR'] for p in rows}):
            at_cost = [p for p in rows if p['owner_IR'] == cost]
            best = max(p['SR'] for p in at_cost)
            if best > running_best:
                frontier.update(p['id'] for p in at_cost if p['SR'] == best)
                running_best = best
        assert frontier == set(ids), cell
        dominance_checks += len(rows)
    import csv
    for r in csv.DictReader((HERE/'paired_mcnemar.csv').open()):
        a, b = outcomes[r['candidate']], outcomes[r['reference']]
        wins = sum(a[k] > b[k] for k in a)
        losses = sum(a[k] < b[k] for k in a)
        assert (wins, losses, len(a)) == (int(r['wins']), int(r['losses']), int(r['n']))
        assert np.isclose((wins-losses)/len(a), float(r['delta_SR']))
        p = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
        assert np.isclose(p, float(r['mcnemar_two_sided_exact_p']))
        paired_checks += 1
    ni = list(csv.DictReader((HERE/'noninferiority.csv').open()))
    assert len(ni) == data['ni_tests']
    for r in ni:
        assert (float(r['paired_exact_lower95']) > -.02) == (r['NI_nominal'] == 'True')
        assert float(r['paired_exact_lower_simultaneous']) <= float(r['paired_exact_lower95'])+1e-12
    # No-discordance and McNemar-nonsignificance are distinct from equality/NI.
    from build_frontier import exact_lower
    assert np.isclose(exact_lower(0, 0, 500, .05), -(1-.025**(1/500)))
    assert binomtest(50, 100, .5).pvalue == 1. and exact_lower(50, 50, 500, .05) < -.02
    assert exact_lower(50, 0, 500, .05) > -.02
    # PAPER_AB source values, in cell order explicitly keyed below.
    headline = {
        'pi05_l10_50': ([.706,.710,.726], [.830,.794,.826]),
        'pi05_l10_500': ([.828,.820,.834], [.866,.864,.890]),
        'pi05_spatial_50': ([.838,.844,.828], [.910,.912,.908]),
        'pi05_spatial_500': ([.982,.974,.972], [.982,.978,.986]),
        'groot_l10_50': ([.608,.618,.602], [.726,.702,.728]),
        'groot_l10_500': ([.830,.832,.828], [.864,.868,.868]),
        'groot_spatial_50': ([.868,.866,.868], [.874,.874,.878]),
        'groot_spatial_500': ([.964,.964,.964], [.958,.960,.958]),
    }
    for p in data['pooled_AB']:
        assert p['replicate_SR'] == headline[p['cell']][p['label'] == 'B']
        assert np.isclose(p['SR'], np.mean(p['replicate_SR'])) and p['init_clusters'] == 500
    plan = json.loads((HERE/'completion_plan.json').read_text())
    specs = json.loads((HERE/'emit_arms_existing.json').read_text())
    needs = json.loads((HERE/'needs_code_configs.json').read_text())
    assert plan['total_arms'] == len(specs)+len(needs) <= 48
    assert max(Counter(p['cell'] for p in plan['arms']).values()) <= 6
    assert len({p['name'] for p in plan['arms']}) == plan['total_arms']
    emitter = REPO/'exp/offline_search/closed_loop/ops/emit_arms.py'
    module_spec = importlib.util.spec_from_file_location('q2_emit_readonly', emitter)
    mod = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(mod)  # definitions only; main is never called
    expanded = mod.expand_replicates(specs)
    assert expanded == specs
    for s in specs:
        assert re.fullmatch(r'[A-Za-z0-9_]+', s['name'])
        assert s['model'] in ('pi05','groot') and s['suite'] in mod.SUITES
        assert s['mode'] == 'plugin' and s['full_model'] and s['cost_ledger']
        assert s['manifest'] == '<RUN>/manifests/eval500.json'
        assert s['client_overrides']['replan_steps'] == 5
        assert '--os-no-shadow-native' in s['plugin_args']
        assert s['plugin_args'].count('--os-fit-artifact') == 1
    for s in needs:
        assert s['shadow'] is False and s['resamples'] == 0 and s['episodes'] == 500
        if 'task_mixture_weights' in s:
            assert len(s['task_mixture_weights']) == 10
            for w in s['task_mixture_weights'].values():
                assert min(w.values()) >= 0 and np.isclose(sum(w.values()), 1.)
    figure = json.loads((ART/'figure_provenance.json').read_text())
    assert figure['input_sha256'] == sha(HERE/'frontier_data.json')
    assert figure['script_sha256'] == sha(ART/'plot_frontier.py')
    assert (ART/'frontier_r6.png').stat().st_size > 0 and (ART/'frontier_r6.pdf').stat().st_size > 0
    result = dict(status='PASS', points=len(points), eligible=sum(p['eligible'] for p in points),
        source_hash_checks=source_checks, paired_exact_comparisons=paired_checks, NI_rows=len(ni),
        dominance_memberships=dominance_checks, headline_replicates=48,
        existing_emit_specs=len(specs), proposed_adapter_specs=len(needs),
        checks=['accounting and accepted journal counts', 'unchanged source hashes', 'independent Pareto scan',
                'paired discordance and exact McNemar', 'NI/multiplicity flags', 'exact-bound endpoint and nonsignificance counterexample',
                'PAPER_AB replicate identity', 'read-only emitter expansion and schema', 'plan caps and lottery weights', 'figure hashes'],
        limitations='No fitting, emit main, serving, or rollout validation; fixed-init intervals require the stated sampling model',
        artifacts={f: sha(HERE/f) for f in ['frontier_points.csv','frontier_data.json','noninferiority.csv','paired_mcnemar.csv','completion_plan.json','emit_arms_existing.json']})
    (HERE/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
