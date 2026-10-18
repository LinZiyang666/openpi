"""Audit collected formal-run evidence. No remote calls; no new GPU trial."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .compare import rows, wait_io
from .node import atomic, read
from .policy import GIB


def audit(root, reference_per_min, *, case):
    root = Path(root)
    c = read(root / 'campaign.json')
    expected_nodes = set(c['autoscale_node_maxima'])
    tasks = read(root / 'EPISODE_PLAN.json')['tasks']
    planned = Counter(t['task_uid'] for t in tasks)
    accepted = Counter()
    errors = 0
    for row in rows(root / 'driver/journal.jsonl'):
        if row.get('accepted') is True:
            accepted[row['task_uid']] += 1
            errors += bool(row.get('error'))
    decisions = list(rows(root / 'AUTOSCALE_DECISIONS.jsonl'))
    windows = [r for r in decisions if 'rate' in r['signals']]
    rates = []
    stable_streak = maximum_streak = 0
    previous_end = None
    for row in windows:
        s = row['signals']
        # No cherry-picking by achieved rate. Exclude count transitions,
        # workload-cohort changes and explicit upward probes by recorded state.
        stable = bool(s.get('settled') and s.get('comparable') and s.get('calibrated') and not s.get('probing') and
                      row['reason'] in ('hold', 'marginal_plateau') and s.get('window_s') == 180)
        end = s.get('window_end', row['time'])
        stable_streak = (stable_streak + 1 if previous_end is not None and abs(end - previous_end - 180) <= 5 else 1) if stable else 0
        maximum_streak = max(maximum_streak, stable_streak)
        previous_end = end
        rates.append(dict(end=end, accepted_per_min=s['rate'] * 60, stable=stable,
                          reason=row['reason'], active=row['counts']['active']))
    nodes, curve, memory_exercised = {}, [], False
    for node in sorted(expected_nodes):
        folder = root / 'workers' / node
        samples = list(rows(folder / 'AUTOSCALE_SAMPLES.jsonl')) if (folder / 'AUTOSCALE_SAMPLES.jsonl').exists() else []
        events = list(rows(folder / 'AUTOSCALE_DECISIONS.jsonl')) if (folder / 'AUTOSCALE_DECISIONS.jsonl').exists() else []
        evidence = read(folder / 'AUTOSCALE_KERNEL.json', {})
        minima = [r['signals'].get('minimum_available', r['signals']['available']) for r in samples]
        final = read(folder / 'AUTOSCALE_NODE.json', {})
        if final.get('minimum_available') is not None:minima.append(final['minimum_available'])
        nodes[node] = dict(min_available_gib=min(minima) / GIB if minima else None,
                          kernel_status=evidence.get('status', 'UNKNOWN'), oom_lines=evidence.get('oom_lines', []),
                          memory_known=bool(samples) and not any(r['reason'] == 'memory_unknown' for r in events),
                          kernel_evidence=evidence)
        for row in events:
            count = row['counts']
            curve.append(dict(time=row['time'], node=node, delta=row['delta'], reason=row['reason'],
                              active=count.get('active'), target=count.get('target'), draining=count.get('draining', 0)))
            sig = row['signals']
            if node == 'timan107' and row['reason'] in ('memory_growth', 'memory_hold') and sig.get('available', 0) >= 10 * GIB:
                memory_exercised = True
    stable_rates = [r['accepted_per_min'] for r in rates if r['stable']]
    critical = c.get('autoscale', {}).get('critical', 10 * GIB) / GIB
    checks = dict(
        throughput=len(stable_rates) >= 5 and min(stable_rates) >= .95 * reference_per_min,
        zero_oom=all(n['kernel_status'] == 'PASS' and n['kernel_evidence'].get('coverage') is True and not n['oom_lines'] for n in nodes.values()),
        memory_signals_known=all(n['memory_known'] for n in nodes.values()),
        above_critical=all(n['min_available_gib'] is not None and n['min_available_gib'] >= critical for n in nodes.values()),
        accepted_errors_zero=errors == 0,
        every_uid_once=bool(planned) and all(v == 1 for v in planned.values()) and accepted == planned,
        scaling_exercised=any(r['reason'] == 'server_queue' for r in decisions) if case == 'server' else memory_exercised,
        curve_present=all(any(r['node'] == n and r['reason'] for r in curve) for n in expected_nodes))
    curve.sort(key=lambda r: (r['time'], r['node']))
    with (root / 'AUTOSCALE_ACTIVE_CURVE.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['time', 'node', 'delta', 'reason', 'active', 'target', 'draining'])
        writer.writeheader();writer.writerows(curve)
    result = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                  reference_per_min=reference_per_min, threshold_per_min=.95 * reference_per_min,
                  accepted=sum(accepted.values()), unique_accepted=len(accepted), accepted_errors=errors,
                  duplicate_uids=sum(v > 1 for v in accepted.values()), maximum_stable_streak=maximum_streak,
                  nodes=nodes, windows_180s=rates, curve='AUTOSCALE_ACTIVE_CURVE.csv')
    atomic(root / 'AUTOSCALE_ACCEPTANCE.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--case', choices=('server', 'memory'), required=True)
    p.add_argument('--reference-per-min', required=True, type=float)
    a = p.parse_args(argv)
    if a.reference_per_min <= 0:p.error('a measured fixed-mode reference is required')
    wait_io()
    result = audit(a.run, a.reference_per_min, case=a.case)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':raise SystemExit(main())
