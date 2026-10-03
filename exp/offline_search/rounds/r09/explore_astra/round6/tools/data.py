"""Identity-first data admission. No outcomes are needed for this experiment."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[5]
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_r6')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
COMPACT = STORE / 'derived/r09_astra/compact'
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50')
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
        raise ValueError('write outside round6 roots')
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
    model, suite, _ = cell.split('_')
    root = 'r09_fable_r3c' if suite == 'l10' else 'r09_fable_r4'
    prefix = 'r9f3c' if suite == 'l10' else 'r9f4'
    # Match the established leaders: escalation only in pi0.5 long; spatial is guard+corrector.
    name = f'{prefix}_{cell}_np_corr05' + ('_esc' if cell == 'pi05_l10_50' else '')
    rows = json.loads((RUN.parent / root / 'arms.json').read_text())  # configuration only
    return next(a for a in rows if a['arm'] == name)


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
