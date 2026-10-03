"""Read ONLY the already discovery-filtered round-1 compact store.

Identity is inspected before loading any outcome/action member. Files containing
any init outside 0..29 are rejected, not analyzed. Raw mixed-init captures and
holdout roots are deliberately unsupported. Every learning script must further
split 0..19 (fit) from 20..29 (evaluation) BEFORE computing fitted statistics.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parents[1]
COMPACT = Path('/home/weiland/trace_runs/offline_search_store/derived/r09_astra/compact')
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
SEED = 20261002
CELLS = [f'{m}_{s}_{n}' for m in ['pi05', 'groot'] for s in ['l10', 'spatial'] for n in [50, 500]]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def validate_identity(task, init):
    if not np.all((task >= 0) & (task < 10) & (init >= 0) & (init < 30)):
        raise ValueError('Forbidden identity: ONLY discovery inits 0..29 supported')


def masks(init):
    if not np.all((init >= 0) & (init < 30)): raise ValueError('forbidden init')
    return init < 20, init >= 20


def load(cell, variant='A'):
    if cell not in CELLS or variant not in ['A', 'CU', 'IP', 'A5', 'W10']:
        raise ValueError('Only explicitly allowlisted compact populations supported')
    path = COMPACT / f'r8_{cell}_{variant}.npz'
    if not path.exists(): return None
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['episodes'] != 300 or 'inits 0..29' not in meta['split']:
        raise ValueError('Missing discovery-only provenance')
    with np.load(path, allow_pickle=False) as z:
        task, init = z['task'], z['init']
        validate_identity(task, init)  # BEFORE any outcome, trajectory, or action load.
        # Explicit admission filter, even though upstream compact files are exclusive.
        admission = (init >= 0) & (init < 30)
        names = ['seq', 'vision', 'state_norm', 'shadow_look_cache_chunk',
            'policy_shadow_chunk', 'shadow_look_keys_pca_third', 'shadow_look_keys_pca_wrist',
            'shadow_look_weights', 'd1_rel', 'dst', 'disp5', 'eligible', 'treatment', 'p', 'coin']
        out = {k: z[k][admission] for k in names}
        if variant == 'W10': out['cache_chunk'] = z['cache_chunk'][admission]
        if variant == 'A' and cell.startswith('pi05'):
            out['camera_shadow_wrist_cache_chunk'] = z['camera_shadow_wrist_cache_chunk'][admission]
        out.update(task=task[admission], init=init[admission])
        for k in ['success', 'cost', 'decisions']:
            if z[k].shape != (10, 30): raise ValueError('Ledger must be discovery-only')
            out[k] = z[k][:, :30]
    return out


def temporal(state, base, seq, episode):
    """Previous fresh observation only; reset every episode, never future state."""
    delta = np.zeros_like(state)
    prev = np.zeros((len(state), 7))
    gap = np.zeros((len(state), 1))
    same = episode[1:] == episode[:-1]
    j = np.flatnonzero(same) + 1
    delta[j] = state[j] - state[j-1]
    prev[j] = base[j-1].mean(1)
    gap[j, 0] = np.minimum(seq[j] - seq[j-1], 12) / 12
    return np.c_[delta, prev, gap]


def dataset(cell, variant='A'):
    z = load(cell, variant)
    if z is None: return None
    take = z['vision'] & np.isfinite(z['shadow_look_cache_chunk']).all((1,2)) & np.isfinite(z['state_norm']).all(1)
    d = {k: z[k][take] for k in ['task', 'init', 'seq', 'state_norm', 'eligible', 'p', 'treatment', 'coin']}
    d['base'] = z['cache_chunk' if variant == 'W10' else 'shadow_look_cache_chunk'][take]
    d['teacher'] = z['policy_shadow_chunk'][take]
    if not np.isfinite(d['teacher']).all(): raise ValueError('Missing teacher')
    d['episode'] = d['task'].astype(int)*30 + d['init']
    # task and init are grouping metadata ONLY, absent from all learned inputs.
    d['x'] = np.c_[z['shadow_look_keys_pca_third'][take], z['shadow_look_keys_pca_wrist'][take],
        d['state_norm'], d['base'].reshape(len(d['task']), -1), np.minimum(d['seq'], 120)/120]
    d['history'] = temporal(d['state_norm'], d['base'], d['seq'], d['episode'])
    w = z['shadow_look_weights'][take]
    d['diagnostics'] = np.c_[z['d1_rel'][take], z['dst'][take], z['disp5'][take],
        -(w*np.log(np.maximum(w, 1e-12))).sum(1), w.max(1)]
    d['success'] = z['success'][d['task'], d['init']]
    d['variant'] = variant
    return d


def subset(d, mask):
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v for k,v in d.items()}


def features(d, history=False):
    return np.c_[d['x'], d['history']] if history else d['x']


def balanced_weights(episode):
    _, inverse, counts = np.unique(episode, return_inverse=True, return_counts=True)
    w = 1. / counts[inverse]
    return w / w.mean()


def epmeans(value, d):
    ep, inverse = np.unique(d['episode'], return_inverse=True)
    totals = np.bincount(inverse, weights=value)
    counts = np.bincount(inverse)
    return ep, totals / counts


def interval(values, episodes, draws=3000):
    """Paired bootstrap of init IDs, jointly across fixed tasks (10 eval inits)."""
    inits = np.unique(episodes % 30)
    grid = np.array([values[episodes % 30 == i].mean() for i in inits])
    rng = np.random.default_rng(SEED)
    b = grid[rng.integers(0, len(grid), (draws, len(grid)))].mean(1)
    return dict(mean=float(grid.mean()), lo=float(np.quantile(b,.025)), hi=float(np.quantile(b,.975)),
                init_clusters=len(grid))
