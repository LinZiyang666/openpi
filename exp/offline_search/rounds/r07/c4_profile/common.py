from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.profile import common as PC
from exp.offline_search.profile.timeline import moves, FLAG
from exp.offline_search.profile.runprof import pct
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources, load_base
from exp.offline_search.rounds.r06.p3_profiling.read_v2 import identical

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/r7_C4')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
SEED = 20260930
csv.field_size_limit(32 << 20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def write(path, obj):
    path = Path(path).resolve()
    if not any(path == r or r in path.parents for r in (HERE, SCRATCH)):
        raise ValueError('C4 writes only its owned directory or /tmp/r7_C4')
    path.parent.mkdir(parents=True, exist_ok=True)
    PC.write_json(path, clean(obj))


def csv_write(path, rows):
    path = Path(path).resolve()
    if not any(r in path.parents for r in (HERE, SCRATCH)):
        raise ValueError('C4 output outside owned paths')
    path.parent.mkdir(parents=True, exist_ok=True)
    PC.write_csv(path, [clean(r) for r in rows])


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def decode(value, default=None):
    return default if value in ('', None) else json.loads(value)


def quant(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), p50=pct(x, 50), p90=pct(x, 90), p95=pct(x, 95),
                mean=float(x.mean()) if len(x) else None)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--cells', nargs='+', default=['all'])
    p.add_argument('--out', type=Path, required=True)
    return p


def cells(values):
    return sorted(sources()) if values == ['all'] else [v.replace('_sp_', '_spatial_') for v in values]


def bank(cell):
    base, blob = load_base(sources()[cell])
    model, suite, size = cell.split('_')
    path = STORE / 'library' / f'{model}_{suite}' / base.cand_name
    manifest = json.loads((path / 'manifest.json').read_text())
    lib = PC.RowStore(path)
    if not np.array_equal(lib['episode'], base.lib_ep) or not np.array_equal(lib['action'], base.act):
        raise ValueError('A fit/library identity mismatch')
    return base, lib, manifest


def stage_table(lib, manifest, base):
    # No local segmentation copy: import the C1-owned frozen table.
    from exp.offline_search.rounds.r07.stages.stages import StageTable
    arrays = {k: lib[k] for k in ('action','rs','task_id','episode','step','success','next')}
    path = SCRATCH/'stages'/(lib.dir.parent.name+'_'+lib.dir.name+'.pkl')
    if path.exists():
        table = StageTable.load(path, library=arrays, manifest=manifest)
        from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
        if table.retrieval_fingerprint != fingerprint(base):
            raise ValueError('cached stages belong to another A fit')
        return table
    table = StageTable.fit(arrays, manifest={**manifest,'_retrieval':base})
    path.parent.mkdir(parents=True,exist_ok=True)
    table.save(path)
    return table


def stage_label(table, rows, weights):
    result = table.online(np.asarray(rows, int), np.asarray(weights, float))
    if float(result['unknown_mass']) > 0:
        return 'unknown'
    if not bool(result['unanimous']):
        return 'mixed'
    mode = int(table.mode[int(rows[0])])
    runs=table.stage_run[np.asarray(rows,int)]
    run=str(int(runs[0])) if np.all(runs==runs[0]) else 'mixed'
    return f'run{run}_' + ('event' if float(result['event_mass']) > 0 else 'interior') + f'_m{mode}'


def successors(lib, count):
    nxt = np.asarray(lib['next'], int).copy()
    ok = (nxt >= 0) & (nxt < len(nxt))
    i = np.flatnonzero(ok)
    j = nxt[i]
    ok[i] &= (lib['episode'][i] == lib['episode'][j]) & (lib['task_id'][i] == lib['task_id'][j]) & (lib['step'][j] == lib['step'][i] + 1)
    nxt[~ok] = -1
    chain = [np.arange(len(nxt))]
    for _ in range(count):
        old = chain[-1]
        chain.append(np.where(old >= 0, nxt[np.maximum(old, 0)], -1))
    return np.asarray(chain)


def tables(cell, campaign='bval', table='anchors', arm=None):
    name = cell.replace('_spatial_', '_sp_') if campaign == 'p3' else cell
    root = RUNS / ('r06_p3_pilot' if campaign == 'p3' else 'r06_c_cal') / 'tables' / name
    audit = json.loads((root / 'audit.json').read_text())
    if not audit.get('episodes') or not audit.get('decisions'):
        raise ValueError('strict-reader product has no accepted episodes/decisions')
    with (root / (table + '.csv')).open() as f:
        for r in csv.DictReader(f):
            keep = r.get('arm') == arm if arm else campaign != 'p3' or r.get('arm', '').endswith('_A_r0')
            if keep:
                yield r


def stream(cell, campaign, state_dims, max_age=4):
    ds = list(tables(cell, campaign, 'decisions'))
    states = {}
    for r in ds:
        with np.load(r['absolute_input_archive'], allow_pickle=False) as z:
            states[r['arm'], r['uid'], int(r['step'])] = np.asarray(z['robot_state'][state_dims], float)
    anchors = list(tables(cell, campaign))
    result = []
    for a in anchors:
        key = a['arm'], a['uid']
        s = int(a['step'])
        result.append(dict(anchor=a, rows=np.asarray(decode(a['retrieval.rows']), int),
                           weights=np.asarray(decode(a['retrieval.weights']), float),
                           states=np.asarray([states.get((*key, s+h), np.full(len(state_dims), np.nan)) for h in range(max_age+1)])))
    return result


def accepted_arm(root):
    """Join accepted attempts; resolve legacy reused-attempt prefixes before acceptance.

    Exact duplicates collapse with the P3 reader's identical comparator. Conflicts,
    gaps, missing episodes, and ambiguous incarnation boundaries fail closed.
    """
    root = Path(root)
    accepted = {}
    for j in jsonl(root / 'client/journal.jsonl'):
        if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error'):
            uid = j['task_uid']
            if uid in accepted and accepted[uid] != j:
                raise ValueError(f'multiple accepted outcomes: {uid}')
            accepted[uid] = j
    groups = defaultdict(list)
    for f in sorted(root.glob('server_*/decisions_*.jsonl')):
        for d in jsonl(f):
            j = accepted.get(d.get('uid'))
            if d.get('ev') == 'dec' and j and int(d.get('attempt', 1) or 1) == int(j.get('attempt', 1) or 1):
                if d.get('ts', 0) <= j.get('ts', math.inf):
                    groups[d['uid']].append(d)
    episodes, discarded = [], 0
    for uid, j in sorted(accepted.items()):
        ds = groups[uid]
        starts = [d.get('ts', 0) for d in ds if d['step'] == 0]
        if not starts:
            raise ValueError(f'missing accepted stream: {uid}')
        start = max(starts)
        kept = [d for d in ds if d.get('ts', 0) >= start]
        discarded += len(ds) - len(kept)
        unique = {}
        for d in kept:
            s = int(d['step'])
            if s in unique and not identical(unique[s], d):
                raise ValueError(f'conflicting duplicate: {uid}:{s}')
            unique[s] = d
        if sorted(unique) != list(range(len(unique))):
            raise ValueError(f'gapped accepted stream: {uid}')
        ordered = [unique[s] for s in sorted(unique)]
        first = ordered[0]
        episodes.append(dict(uid=uid, task=int(first['task_id']), init=int(first['init']),
                             Y=int(j['success']), attempt=int(j.get('attempt', 1) or 1), decisions=ordered))
    if not episodes:
        raise ValueError('no accepted episodes')
    return episodes, dict(episodes=len(episodes), discarded_prior_prefix=discarded)


def task_bootstrap(records, value, reps=2000, alpha=.05):
    """Fixed-task, task/init cluster bootstrap; repeats remain in the same cluster."""
    clusters = defaultdict(list)
    for r, v in zip(records, value, strict=True):
        clusters[r['task'], r['init']].append(v)
    by_task = defaultdict(list)
    for (t, _), v in clusters.items():
        by_task[t].append(float(np.mean(v)))
    rng = np.random.default_rng(SEED)
    means = [np.asarray(v) for _, v in sorted(by_task.items())]
    draw = np.mean([v[rng.integers(len(v), size=(reps, len(v)))].mean(1) for v in means], axis=0)
    return dict(estimate=float(np.mean([v.mean() for v in means])),
                lo=float(np.quantile(draw, alpha/2)), hi=float(np.quantile(draw, 1-alpha/2)),
                clusters=len(clusters), tasks=len(means), draws=reps, seed=SEED)
