"""Identity-first data admission. No outcomes are needed for this experiment."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[5]
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_r7')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
COMPACT = STORE / 'derived/r09_astra/compact'
CELLS = ('groot_l10_50', 'groot_spatial_50')
VARIANTS = ('A', 'CU', 'IP')


def admitted(path):
    p = Path(path)
    if any(s.startswith(('r09_holdout', 'r09_astra_holdout')) for s in p.parts):
        raise ValueError('forbidden root')
    p = p.resolve()
    if any(s.startswith(('r09_holdout', 'r09_astra_holdout')) for s in p.parts):
        raise ValueError('forbidden root')
    return p


def owned(path):
    p = admitted(path)
    if not any(p.is_relative_to(r) for r in (HERE, RUN)):
        raise ValueError('write outside round7 roots')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def dump(path, value):
    owned(path).write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()
        if isinstance(x, np.generic) else str(x)) + '\n')


def sha(path):
    h = hashlib.sha256()
    with admitted(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity_arrays(z):
    # NPZ members are lazy. Reject an entire mixed archive before a payload is accessed.
    task, init = z['task'], z['init']
    if (task.shape != init.shape or task.ndim != 1 or
            task.dtype.kind not in 'iu' or init.dtype.kind not in 'iu' or
            not np.all((task >= 0) & (task < 10) & (init >= 0) & (init < 30))):
        raise ValueError('archive is not exclusive to admitted inits 0..29')
    return task, init


def dataset(cell, variant, split):
    if cell not in CELLS or variant not in VARIANTS or split not in ('train', 'eval'):
        raise ValueError('population not allowlisted')
    path = admitted(COMPACT / f'r8_{cell}_{variant}.npz')
    with np.load(path, allow_pickle=False) as z:
        task, init = identity_arrays(z)
        take = (init < 20) if split == 'train' else (init >= 20)
        # All members are known to contain only 0..29; split before constructing a dataset.
        fields = ('seq', 'vision', 'state_norm', 'shadow_look_rows', 'shadow_look_weights',
            'shadow_look_cache_chunk', 'policy_shadow_chunk',
            'shadow_look_keys_pca_third', 'shadow_look_keys_pca_wrist')
        a = {k: z[k][take] for k in fields}
        a.update(task=task[take], init=init[take])
    valid = (a['vision'] & np.isfinite(a['shadow_look_cache_chunk']).all((1, 2)) &
             np.isfinite(a['policy_shadow_chunk']).all((1, 2)) &
             np.isfinite(a['state_norm']).all(1))
    a = {k: v[valid] for k, v in a.items()}
    rows = a['shadow_look_rows'].astype(np.int64)
    w = a['shadow_look_weights'].astype(np.float64)
    if not np.all(rows >= 0) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('invalid retrieval')
    w /= w.sum(1, keepdims=True)
    base = a['shadow_look_cache_chunk'].astype(np.float32)
    return dict(task=a['task'], init=a['init'], seq=a['seq'], rows=rows, weights=w,
        episode=a['task'].astype(int) * 30 + a['init'] + VARIANTS.index(variant) * 300,
        base=base, teacher=a['policy_shadow_chunk'].astype(np.float32),
        x=np.c_[a['shadow_look_keys_pca_third'], a['shadow_look_keys_pca_wrist'],
                a['state_norm'][:, :8], base.reshape(len(base), -1),
                np.minimum(a['seq'], 120) / 120].astype(np.float32))


def subset(d, take):
    return {k: v[take] for k, v in d.items()}


def combine(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def balanced(episode):
    _, inv, counts = np.unique(episode, return_inverse=True, return_counts=True)
    w = 1 / counts[inv]
    return w / w.mean()


def episode_errors(pred, d):
    value = ((pred - d['teacher'][:, :, :6]) ** 2).mean((1, 2))
    ep, inv, counts = np.unique(d['episode'], return_inverse=True, return_counts=True)
    return ep, np.bincount(inv, weights=value) / counts


def control_row(cell):
    rows = json.loads((RUN.parent / 'r09_astra_r6/arms.json').read_text())
    return next(a for a in rows if a['arm'] == f'r9a6_{cell}_control')


def artifact(row):
    a = row['plugin_args']
    return Path(a[a.index('--os-fit-artifact') + 1])


def framed_objects(path):
    """Frame JSON objects without decoding rejected records or their outcomes."""
    with admitted(path).open() as f:
        depth = 0
        quote = escape = False
        buf = []
        while chunk := f.read(65536):
            for c in chunk:
                if depth:
                    buf.append(c)
                if quote:
                    if escape: escape = False
                    elif c == '\\': escape = True
                    elif c == '"': quote = False
                elif c == '"': quote = True
                elif c == '{':
                    if depth == 0: buf = [c]
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        yield ''.join(buf)
                        buf = []
        if depth or quote:
            raise ValueError('truncated metadata')


def episode_metadata(path, pairs):
    if any(not (0 <= t < 10 and 0 <= i < 30) for t, i in pairs):
        raise ValueError('forbidden requested identity')
    for raw in framed_objects(path):
        mi = re.search(r'"init"\s*:\s*(\d+)(?=\s*[,}])', raw)
        mt = re.search(r'"task_id"\s*:\s*(\d+)(?=\s*[,}])', raw)
        if mi and mt and (int(mt[1]), int(mi[1])) in pairs:
            yield json.loads(raw)


def json_identity(raw):
    """Lex only scalar identity values, skipping all other values before decoding payload.

    JSON strings are scanned with escapes. Nested payloads are skipped without
    deserialization. Duplicate identity fields fail closed.
    """
    dec = json.JSONDecoder()
    i = 0
    while i < len(raw) and raw[i].isspace(): i += 1
    if i == len(raw) or raw[i] != '{': return None
    i += 1
    fields = {}
    wanted = {'init', 'task_id', 'task_uid', 'uid'}
    while i < len(raw):
        while i < len(raw) and (raw[i].isspace() or raw[i] == ','): i += 1
        if i == len(raw) or raw[i] == '}': break
        key, i = dec.raw_decode(raw, i)
        while raw[i].isspace(): i += 1
        if raw[i] != ':': raise ValueError('bad JSON framing')
        i += 1
        while raw[i].isspace(): i += 1
        start = i
        depth, quote, escape = 0, False, False
        while i < len(raw):
            c = raw[i]
            if quote:
                if escape: escape = False
                elif c == chr(92): escape = True
                elif c == '"': quote = False
            elif c == '"': quote = True
            elif c in '[{': depth += 1
            elif c in ']}':
                if depth == 0: break
                depth -= 1
            elif c == ',' and depth == 0: break
            i += 1
        if key in wanted:
            if key in fields: raise ValueError('duplicate identity field')
            fields[key] = json.loads(raw[start:i])
    identities = []
    if fields.get('init') is not None and fields.get('task_id') is not None:
        identities.append((int(fields['task_id']), int(fields['init'])))
    for k in ('uid', 'task_uid'):
        if fields.get(k):
            parts = fields[k].rsplit(':', 2)
            if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
                identities.append(tuple(map(int, parts[-2:])))
    if not identities: return None
    if len(set(identities)) != 1: raise ValueError('conflicting identities')
    return identities[0]


def admitted_jsonl(path, inits=range(20,30)):
    allowed = set(inits)
    if not allowed <= set(range(30)): raise ValueError('forbidden requested inits')
    with admitted(path).open() as f:
        for raw in f:
            identity = json_identity(raw)
            if identity is None or not (0 <= identity[0] < 10 and identity[1] in allowed):
                continue
            yield json.loads(raw)


def phases(closed, episode, seq):
    """Descriptive commanded-gripper phase; no object/contact truth."""
    out = np.full(len(closed), 'post', dtype='<U8')
    for ep in np.unique(episode):
        ix = np.flatnonzero(episode == ep)
        ix = ix[np.argsort(seq[ix])]
        c = closed[ix]
        first = np.flatnonzero(c)
        out[ix] = np.where(c, 'carry', 'post')
        out[ix[:first[0] if len(first) else len(ix)]] = 'approach'
        for j in range(1,len(ix)):
            if not c[j-1] and c[j]: out[ix[max(0,j-1):j+1]] = 'grasp'
            elif c[j-1] and not c[j]: out[ix[max(0,j-1):min(len(ix),j+2)]] = 'release'
    return out


def annotated(cell, variant, split):
    d = dataset(cell, variant, split)
    with np.load(admitted(COMPACT / f'r8_{cell}_{variant}.npz'), allow_pickle=False) as z:
        task, init = identity_arrays(z)
        take = (init < 20) if split == 'train' else (init >= 20)
        t, ii, seq = task[take], init[take], z['seq'][take]
        served = z['served_chunk'][take]
        ep = t.astype(int)*30+ii
        phase = phases((served[:,:5,6]<0).mean(1)>.5, ep, seq)
        ix = {(int(t0),int(i0),int(s0)):j for j,(t0,i0,s0) in enumerate(zip(t,ii,seq))}
        sel = [ix[int(t0),int(i0),int(s0)] for t0,i0,s0 in zip(d['task'],d['init'],d['seq'])]
        d['phase'] = phase[sel]
        d['distance'] = z['d1'][take][sel]
        d['state_wire'] = z['state_wire'][take][sel]
    d['grip_event'] = np.any(np.diff(d['base'][:,:,6]<0,axis=1),axis=1)
    return d
