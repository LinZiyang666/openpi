"""Per-N server throughput and latency from a sweep directory (sweep.py layout).

<run>/server/requests_<pid>.csv  conn,kind,t_recv,t_sent  (serve_entry.py)
<run>/replay/n<NNN>_s<seed>_summary.json  replay window and episode total (replay.py)

For each replay: the infer requests the server received inside its window, requests per second over the window,
latency (t_sent - t_recv) mean / p50 / p90 / p99 in ms, and the time to finish its 500 episodes.

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


def analyze(run):
    run = Path(run)
    reqs = requests(run / 'server')
    out = []
    for path in sorted((run / 'replay').glob('n*_summary.json')):
        s = json.loads(path.read_text())
        inside = reqs[(reqs[:, 0] >= s['start']) & (reqs[:, 1] <= s['end'])] if len(reqs) else reqs
        lat = (inside[:, 1] - inside[:, 0]) * 1000.0
        q = np.percentile(lat, [50, 90, 99]) if len(lat) else [None] * 3
        out.append(dict(robots=s['robots'], seed=s['seed'], episodes=s['episodes'], total_s=round(s['total_s'], 2),
                        infer_requests=int(len(inside)), requests_per_s=round(len(inside) / s['total_s'], 2),
                        latency_ms=dict(mean=round(float(lat.mean()), 2) if len(lat) else None,
                                        p50=round(float(q[0]), 2) if len(lat) else None,
                                        p90=round(float(q[1]), 2) if len(lat) else None,
                                        p99=round(float(q[2]), 2) if len(lat) else None)))
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
