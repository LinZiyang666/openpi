"""Shared paths and deterministic, CPU-only growth utilities."""
from pathlib import Path
import hashlib
import pickle
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class

OUT = Path(__file__).resolve().parent
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r05_growth')
COLD = Path('/home/weiland/trace_runs/offline_search_store')
SHM = Path('/dev/shm/offline_search_store')
SPEC = 'exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM'
BASE_SPEC = 'exp/offline_search/rounds/r02/g1_awm/awm.py:AWM'
FIELDS = ('key_v0', 'key_v1', 'rs', 'action', 'task_id', 'episode', 'step',
          'ep_len', 'progress', 'success', 'prev', 'next')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def base_artifact(suite, scale=50):
    short = 'sp' if suite == 'spatial' else suite
    return RUN.parent / f'r02_g{scale}' / 'fits' / f'oscl{scale}_p_{short}_cl2.pkl'

def load_base(suite, scale=50):
    load_method_class(BASE_SPEC)  # Original artifacts use the file-loader's module name.
    p = base_artifact(suite, scale)
    with p.open('rb') as f:
        blob = pickle.load(f)
    expected = {'lib': 'current', 'kref': 5} if scale == 50 else {}
    assert blob['spec'] == BASE_SPEC and blob['kwargs'] == expected
    assert blob['cell'] == f'pi05_{suite}_cache'
    blob['method'].prof = api.NULL_PROFILER
    return blob['method']

def features(method, obj, rows=None):
    """Frozen PCA projections; no fitting and no query outcome access."""
    rows = np.arange(len(obj.rs)) if rows is None else np.asarray(rows)
    x = np.empty((len(rows), 136), np.float32)
    for lo in range(0, len(rows), 128):
        rr = rows[lo:lo + 128]
        x[lo:lo + len(rr), :64] = obj.key_v0[rr] @ method.B0T.T - method.muB0
        x[lo:lo + len(rr), 64:128] = obj.key_v1[rr] @ method.B1T.T - method.muB1
        x[lo:lo + len(rr), 128:] = obj.rs[rr, :8]
    assert np.isfinite(x).all()
    return x

def fingerprint(task, x, action):
    # All full-horizon valid action dimensions, never head-only deduplication.
    a = np.concatenate((np.asarray(x, np.float32).ravel(),
                        np.asarray(action[:, :7], np.float32).ravel()))
    assert np.isfinite(a).all()
    a[a == 0] = 0  # Treat signed zero identically, as array_equal does.
    return hashlib.sha256(int(task).to_bytes(4, 'little') + a.tobytes()).digest()

def episode_view(e, index):
    return api.EpisodeView(e['uid'], e['task'], e['task_id'], e['init'], index, 0)

def readonly(method):
    for a in list(vars(method).values()) + [v for t in method.tasks.values() for v in vars(t).values()]:
        if isinstance(a, np.ndarray):
            a.flags.writeable = False
