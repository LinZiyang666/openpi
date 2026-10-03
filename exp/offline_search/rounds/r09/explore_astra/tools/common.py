"""Small, strict data and statistical primitives shared by the R9 analyses."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
DERIVED = STORE / 'derived/r09_astra'
SEED = 20261001


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if not line.endswith('\n'):
                raise ValueError(f'unpublished line in {path}')
            if line.strip():
                yield json.loads(line)


def identity(row):
    task, init = map(int, row['task_uid'].rsplit(':', 2)[-2:])
    return task, init


def discovery(task, init):
    return 0 <= task < 10 and 0 <= init < 30


def assert_discovery(task, init):
    if not np.all((np.asarray(task) >= 0) & (np.asarray(task) < 10) &
                  (np.asarray(init) >= 0) & (np.asarray(init) < 30)):
        raise ValueError('Only tasks 0..9 and discovery inits 0..29 are permitted')


def accepted(path):
    """Filter split BEFORE admitting terminal outcomes; reject conflicting accepts."""
    out = {}
    for row in jsonl(path):
        if 'task_uid' not in row or not discovery(*identity(row)):
            continue
        if row.get('accepted') and row.get('status') in ('done', 'failed') and not row.get('error'):
            key = identity(row)
            if key in out and (out[key].get('attempt'), out[key].get('success')) != (row.get('attempt'), row.get('success')):
                raise ValueError(f'conflicting accepted attempt {key}')
            out[key] = row
    if set(out) != {(t, i) for t in range(10) for i in range(30)}:
        raise ValueError(f'incomplete discovery population: {len(out)}')
    return out


def paired_bootstrap(x, draws=10000, seed=SEED):
    """Fixed-task, paired init bootstrap. x is [10,30,...]."""
    x = np.asarray(x)
    if x.shape[:2] != (10, 30):
        raise ValueError('Expected all 300 discovery pairs in [task,init] order')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 30, (draws, 10, 30))
    values = x[np.arange(10)[None, :, None], idx].mean(axis=(1, 2))
    return dict(mean=np.mean(x, axis=(0, 1)).tolist(),
                lo=np.quantile(values, .025, axis=0).tolist(),
                hi=np.quantile(values, .975, axis=0).tolist())


def mixture_ir(cost, decisions, probabilities):
    """Ratio of expected SUMS; an episode lottery does not average arm IRs."""
    c, n, p = map(np.asarray, (cost, decisions, probabilities))
    if c.shape != n.shape or c.shape != p.shape or np.any(p < 0) or not np.allclose(p.sum(-1), 1):
        raise ValueError('invalid mixture')
    return float(np.sum(c * p) / np.sum(n * p))


def load_compact(arm):
    with np.load(DERIVED / 'compact' / f'{arm}.npz', allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    assert_discovery(out['task'], out['init'])
    return out
