"""Mean server latency (receive -> send) of hits and misses per N: runs/<run>/HITSPLIT.json.

The request log (server/requests_<pid>.csv) has no hit flag; the plugin's decision log has one per infer request
(`ev=dec`, `hit`, with the episode's task_id / init). Episode ep_<task>_<init> ran on one robot slot (one websocket
connection) between its start and end (replay/n<NNN>_s0_episodes.jsonl), so its decisions in time order are that
connection's infer requests in that span, in order. Slots map to connections in order of first contact. Every episode
must match exactly (same count of decisions and requests); the all-request mean equals RESULTS.json's latency mean.

  python -m exp.cluster_sim.hitsplit --runs /nvme/weiland/openpi/cluster_sim/runs cache_full cache cache_seg
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
from pathlib import Path

import numpy as np

DECISIONS = '/data/mw_finetune/eval/cs_rec_pi05/runs/*/server_8000/decisions_*.jsonl'
CHUNK = 64 << 20


def scan(job):
    path, start, end, windows = job
    out = defaultdict(list)
    f = open(path, 'rb')
    f.seek(start)
    if start:
        f.readline()
    while f.tell() <= end:
        line = f.readline()
        if not line:
            break
        if b'"dec"' not in line:
            continue
        r = json.loads(line)
        if r.get('ev') != 'dec':
            continue
        for key, (a, b) in windows.items():
            if a <= r['ts'] <= b + 5:
                out[key].append((r['ts'], f"ep_{r['task_id']}_{r['init']:02d}", bool(r['hit'])))
                break
    return out


def requests(server):
    rows, first = defaultdict(list), {}
    for path in Path(server).glob('requests_*.csv'):
        for line in path.read_text().splitlines():
            conn, kind, a, b = line.split(',')
            conn, a, b = int(conn), float(a), float(b)
            first[conn] = min(first.get(conn, a), a)
            if kind == 'infer':
                rows[conn].append((a, b))
    return {c: np.array(sorted(v)) for c, v in rows.items()}, first


def split(run, summary, decisions, reqs, first):
    a, b = summary['start'], summary['end']
    eps = [json.loads(line) for line in (run / 'replay' / f"n{summary['robots']:03d}_s{summary['seed']}_episodes.jsonl").read_text().splitlines()]
    slot_start = {}
    for e in eps:
        slot_start[e['slot']] = min(slot_start.get(e['slot'], e['start']), e['start'])
    conns = sorted((c for c in reqs if a - 5 <= first[c] <= b), key=first.get)
    assert len(conns) == len(slot_start), (len(conns), len(slot_start))
    conn_of = dict(zip(sorted(slot_start, key=slot_start.get), conns))
    by_ep = defaultdict(list)
    for ts, ep, hit in decisions:
        by_ep[ep].append((ts, hit))
    lat, hits = [], []
    for e in eps:
        r = reqs[conn_of[e['slot']]]
        r = r[(r[:, 0] >= e['start']) & (r[:, 0] < e['end'])]
        d = sorted(by_ep[e['episode']])
        assert len(d) == len(r) == e['infers'], (e, len(d), len(r))
        lat += list((r[:, 1] - r[:, 0]) * 1000.0)
        hits += [h for _, h in d]
    lat, hits = np.array(lat), np.array(hits)
    return dict(robots=summary['robots'], hits=int(hits.sum()), misses=int((~hits).sum()),
                all_ms=round(float(lat.mean()), 2), hit_ms=round(float(lat[hits].mean()), 2), miss_ms=round(float(lat[~hits].mean()), 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default='/nvme/weiland/openpi/cluster_sim/runs')
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    summaries = {(name, s['robots']): s for name in args.names
                 for s in map(lambda p: json.loads(p.read_text()), (Path(args.runs) / name / 'replay').glob('n*_summary.json'))}
    windows = {k: (s['start'], s['end']) for k, s in summaries.items()}
    jobs = [(p, s, min(s + CHUNK, os.path.getsize(p)) - 1, windows) for p in glob.glob(DECISIONS) for s in range(0, os.path.getsize(p), CHUNK)]
    decisions = defaultdict(list)
    with ProcessPoolExecutor(min(40, len(jobs))) as ex:
        for out in ex.map(scan, jobs):
            for k, v in out.items():
                decisions[k] += v
    for name in args.names:
        run = Path(args.runs) / name
        reqs, first = requests(run / 'server')
        rows = [split(run, summaries[k], decisions[k], reqs, first) for k in sorted(summaries) if k[0] == name]
        (run / 'HITSPLIT.json').write_text(json.dumps(rows, indent=1) + '\n')
        for r in rows:
            print(name, json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
