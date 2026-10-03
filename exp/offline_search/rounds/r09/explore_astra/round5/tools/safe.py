"""Identity-first readers. Outcome payloads are parsed only after admission.

No arbitrary run-root parameter, no summaries, no raw server stdout. JSON array
objects are framed without decoding their fields, then admitted by identity.
NPZ members are lazy: identity first, reject mixed archives before payload access.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[5]
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_r5')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
ROOTS = {n: RUN.parent/n for n in ('r09_astra_round2b', 'r09_astra_round2b_g', 'r09_fable_r3c')}
EVAL = {(t, i) for t in range(10) for i in range(20, 30)}


def reject_root(path):
    p = Path(path).resolve()
    if any(s.startswith(('r09_holdout', 'r09_astra_holdout')) for s in p.parts):
        raise ValueError('forbidden root')
    return p


def identity(raw, lo=0, hi=30):
    """Lexical identity parser runs BEFORE json.loads (including rejected rows)."""
    mi = re.search(r'"init"\s*:\s*(-?\d+)(?=\s*[,}])', raw)
    mt = re.search(r'"task_id"\s*:\s*(-?\d+)(?=\s*[,}])', raw)
    mu = re.search(r'"(?:task_uid|uid)"\s*:\s*"([^"\\]+)"', raw)
    if mi and mt:
        t, i = int(mt[1]), int(mi[1])
    elif mu:
        try:
            t, i = map(int, mu[1].rsplit(':', 2)[-2:])
        except ValueError:
            return None
    else:
        return None
    if not (0 <= t < 10 and lo <= i < hi <= 30):
        return None
    if mi and int(mi[1]) != i:
        raise ValueError('identity disagreement')
    return t, i


def records(path, lo=20, hi=30):
    path = reject_root(path)
    if not any(path.is_relative_to(p) for p in [*ROOTS.values(), RUN, HERE]):
        raise ValueError('journal root not allowlisted')
    with path.open() as f:
        for raw in f:
            pair = identity(raw, lo, hi)
            if pair is not None:
                yield pair, json.loads(raw)


def framed_objects(path):
    """Stream a JSON object array without deserializing unadmitted outcomes."""
    with reject_root(path).open() as f:
        depth = 0
        quote = escape = False
        buf = []
        while chunk := f.read(65536):
            for c in chunk:
                if depth:
                    buf.append(c)
                if quote:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        quote = False
                elif c == '"':
                    quote = True
                elif c == '{':
                    if depth == 0:
                        buf = [c]
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        yield ''.join(buf)
                        buf = []
        if depth or quote:
            raise ValueError('truncated JSON array')


def episode_metadata(path, pairs=EVAL):
    for raw in framed_objects(path):
        pair = identity(raw, 0, 30)
        if pair is not None and pair in pairs:
            yield json.loads(raw)


def safe_npz(path, keys, lo=20):
    with np.load(reject_root(path), allow_pickle=False) as z:
        task, init = z['task'], z['init']
        if not np.all((task >= 0) & (task < 10) & (init >= 0) & (init < 30)):
            raise ValueError('mixed archive rejected BEFORE payload load')
        take = (init >= lo) & (init < 30)
        return {k: z[k][take] for k in keys}


def dump(path, value):
    path = Path(path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(RUN)):
        raise ValueError('write outside owned roots')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def sha(path):
    h = hashlib.sha256()
    with reject_root(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_arm(root, arm):
    root = ROOTS[root]
    specs = {a['arm']: a for a in json.loads((root/'arms.json').read_text())}
    if arm not in specs:
        raise ValueError('unknown arm')
    d = root/'runs'/arm
    outcomes = {}
    for pair, row in records(d/'client/journal.jsonl'):
        if row.get('accepted') and row.get('success') is not None and not row.get('error'):
            if pair in outcomes and outcomes[pair] != row:
                raise ValueError('duplicate accepted journal row')
            outcomes[pair] = row
    if set(outcomes) != EVAL:
        raise ValueError(f'incomplete screen {arm}: {len(outcomes)}')
    traces = {p: {} for p in EVAL}
    for path in sorted(d.glob('server_*/decisions_*.jsonl')):
        for pair, row in records(path):
            if row.get('ev') != 'dec' or row.get('attempt') != outcomes[pair]['attempt']:
                continue
            if row.get('uid') != outcomes[pair]['task_uid']:
                raise ValueError('journal/decision UID disagreement')
            step = int(row['step'])
            if step in traces[pair]:
                raise ValueError('duplicate accepted decision')
            traces[pair][step] = row
    for pair, rows in traces.items():
        if not rows or sorted(rows) != list(range(len(rows))):
            raise ValueError('incomplete decision trace')
        traces[pair] = [rows[k] for k in sorted(rows)]
    return specs[arm], outcomes, traces
