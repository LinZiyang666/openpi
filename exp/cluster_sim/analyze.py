"""Per-N server throughput and latency from a sweep directory (sweep.py layout).

<run>/server/requests_<pid>.csv  conn,kind,t_recv,t_sent  (serve_entry.py)
<run>/replay/n<NNN>_s<seed>_summary.json  replay window and episode total (replay.py)

For each replay: the infer requests the server received inside its window, requests per second over the window,
latency (t_sent - t_recv) mean and p10 / p25 / p50 / p75 / p90 / p99 in ms, and the time to finish its 500 episodes. Diagnostics:
steady_requests_per_s counts only the stretch where all N robots run (from the last robot's first episode to the
last episode's start, after which robots drop out); from <run>/server/batches_<pid>.csv, the share of the window
with any stage batch on the GPU (gpu_busy) and per stage the batches, mean size, mean batch ms and mean queue wait
of the batch's oldest request.

  python -m exp.cluster_sim.analyze --run DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def requests(server):
    rows = []
    for path in Path(server).glob('requests_*.csv'):
        for line in path.read_text().splitlines():
            conn, kind, t_recv, t_sent = line.split(',')
            if kind == 'infer':
                rows.append((float(t_recv), float(t_sent)))
    return np.array(sorted(rows)) if rows else np.zeros((0, 2))


def batches(server):
    rows = [tuple(map(float, line.split(','))) for path in Path(server).glob('batches_*.csv')
            for line in path.read_text().splitlines()]
    return np.array(sorted(rows, key=lambda r: r[1])) if rows else np.zeros((0, 5))


def busy_share(spans, start, end):
    covered, edge = 0.0, start
    for a, b in spans:
        a, b = max(a, edge), min(b, end)
        if b > a:
            covered += b - a
            edge = b
    return covered / (end - start)


def diagnostics(bat, start, end):
    inside = bat[(bat[:, 1] >= start) & (bat[:, 2] <= end)] if len(bat) else bat
    stages = {}
    for stage in (1, 2, 3):
        b = inside[inside[:, 0] == stage]
        if len(b):
            stages[stage] = dict(batches=int(len(b)), mean_size=round(float(b[:, 3].mean()), 2),
                                 batch_ms=round(float((b[:, 2] - b[:, 1]).mean() * 1000), 2),
                                 oldest_wait_ms=round(float(b[:, 4].mean() * 1000), 2))
    return dict(gpu_busy=round(busy_share(inside[:, 1:3], start, end), 3) if len(inside) else None, stages=stages)


def analyze(run):
    run = Path(run)
    reqs = requests(run / 'server')
    bat = batches(run / 'server')
    out = []
    for path in sorted((run / 'replay').glob('n*_summary.json')):
        s = json.loads(path.read_text())
        inside = reqs[(reqs[:, 0] >= s['start']) & (reqs[:, 1] <= s['end'])] if len(reqs) else reqs
        lat = (inside[:, 1] - inside[:, 0]) * 1000.0
        pct = (10, 25, 50, 75, 90, 99)
        q = np.percentile(lat, pct) if len(lat) else [None] * len(pct)
        eps = [json.loads(line) for line in path.with_name(path.name.replace('_summary.json', '_episodes.jsonl')).read_text().splitlines()]
        first = {}
        for e in eps:
            first[e['slot']] = min(first.get(e['slot'], e['start']), e['start'])
        a, b = max(first.values()), max(e['start'] for e in eps)
        steady = inside[(inside[:, 0] >= a) & (inside[:, 0] < b)] if b > a + 1 else None
        out.append(dict(robots=s['robots'], seed=s['seed'], episodes=s['episodes'], total_s=round(s['total_s'], 2),
                        infer_requests=int(len(inside)), requests_per_s=round(len(inside) / s['total_s'], 2),
                        steady_requests_per_s=round(len(steady) / (b - a), 2) if steady is not None else None,
                        latency_ms=dict(mean=round(float(lat.mean()), 2) if len(lat) else None,
                                        **{f'p{p}': round(float(v), 2) if len(lat) else None for p, v in zip(pct, q)}),
                        **diagnostics(bat, s['start'], s['end'])))
    out.sort(key=lambda r: r['robots'])
    (run / 'RESULTS.json').write_text(json.dumps(out, indent=1) + '\n')
    for r in out:
        print(json.dumps(r), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    analyze(parser.parse_args().run)


if __name__ == '__main__':
    main()
