"""Stream already-collected formal-run evidence; never query remote machines."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def wait_io():
    while True:
        full = next(line for line in Path('/proc/pressure/io').read_text().splitlines() if line.startswith('full '))
        avg10 = float(dict(s.split('=') for s in full.split()[1:])['avg10'])
        if avg10 <= 20:return
        print('IO full avg10 >20; waiting before reading evidence', flush=True)
        time.sleep(5)


def rows(path):
    wait_io()
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if index and index % 4096 == 0:wait_io()
            if line.strip():yield json.loads(line)


def same_plan(a, b):
    left = json.loads((Path(a) / 'EPISODE_PLAN.json').read_text())['tasks']
    right = json.loads((Path(b) / 'EPISODE_PLAN.json').read_text())['tasks']
    if left != right:raise ValueError('paired episode tasks differ (UID/init/seed/endpoint/config); do not compare or launch the second lane')
    return len(left)


def stamp(row):
    for key in ('ts', 'timestamp', 'time'):
        value = row.get(key)
        if isinstance(value, (float, int)):return float(value)
        if isinstance(value, str):return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    raise ValueError('journal row lacks a timestamp')


def summarize(root):
    root = Path(root)
    accepted, errors, times = set(), 0, []
    for row in rows(root / 'driver' / 'journal.jsonl'):
        if row.get('accepted') is True:
            accepted.add(row['task_uid'])
            errors += bool(row.get('error'))
            times.append(stamp(row))
    nodes = {}
    for path in sorted((root / 'workers').glob('*/AUTOSCALE_SAMPLES.jsonl')):
        samples = list(rows(path))
        if not samples:continue
        nodes[path.parent.name] = dict(samples=len(samples),
            min_available_gib=min(r['signals']['available'] for r in samples) / 1024**3,
            max_rss_gib=max(r['signals']['rss'] for r in samples) / 1024**3,
            max_memory_full_avg10=max(r['signals']['pressure'].get('full', {}).get('avg10', 0) for r in samples),
            active_min=min(r['counts']['active'] for r in samples), active_max=max(r['counts']['active'] for r in samples))
    windows = [row for row in rows(root / 'AUTOSCALE_DECISIONS.jsonl') if 'rate' in row['signals']]
    return dict(accepted=len(accepted), accepted_errors=errors,
                accepted_per_s_between_first_last=(len(times) - 1) / (max(times) - min(times)) if len(times) > 1 and max(times) > min(times) else None,
                nodes=nodes, window_rates=[r['signals']['rate'] for r in windows],
                window_wait_fraction=[r['signals']['wait_s'] / max(1e-9, r['signals']['sim_s'] + r['signals']['wait_s']) for r in windows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixed', type=Path);p.add_argument('dynamic', type=Path, nargs='?')
    p.add_argument('--closed', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    wait_io()
    if a.closed:
        receipt = json.loads((a.fixed / 'RECEIPT.json').read_text())
        if receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True:
            raise ValueError('prior trial is not PASS with all_owned_closed=true; do not start another lane')
        print(json.dumps(dict(run=str(a.fixed), closed=True)))
        return
    if a.dynamic is None:p.error('dynamic run required unless --closed')
    n = same_plan(a.fixed, a.dynamic)
    result = dict(identical_episode_tasks=n)
    if not a.plan_only:
        result.update(fixed=summarize(a.fixed), dynamic=summarize(a.dynamic))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
