"""Per-look CPU latency breakdown of the deployed cache, with and without the frozen corrector.

Warm single-thread replay (one pinned core), K8 method. Queries are B-pool library frames
(never offline_search_store/queries/*): for sizes < 500 the episodes are outside the library.
Arms: A = R10 pure cache fit; AC = same cache + frozen GC_dist corrector (deployed arithmetic,
DistanceCorrectedBase); AC_log = the R15 run class (adds one telemetry distance call).
"""
import argparse, collections, contextlib, functools, hashlib, json, os, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path('/home/weiland/projects/openpi')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
LIB = Path('/home/weiland/trace_runs/offline_search_store/library')
OUT = Path(__file__).resolve().parent
from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.recipe import recipe as r9
from exp.offline_search.rounds.r10.astra.method import DistanceCorrectedBase


class Loader(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith('osm_') and mod not in sys.modules:
            for p in (ROOT / 'exp/offline_search').rglob('*.py'):
                if 'osm_' + hashlib.md5(str(p).encode()).hexdigest()[:12] == mod:
                    plugin.load_method_class(str(p) + ':' + name)
                    break
        return super().find_class(mod, name)


def load(p):
    with open(p, 'rb') as f:
        return Loader(f).load()['method']


def base(m):
    while hasattr(m, 'inner') or hasattr(m, 'base'):
        m = getattr(m, 'inner', None) or m.base
    return m


def queries(cell, size, per_task=64):
    d = LIB / cell / 'bpool_cs'
    a = {n: np.load(d / f'{n}.npy', mmap_mode='r') for n in ('key_v0', 'key_v1', 'rs', 'action', 'episode', 'task_id', 'step')}
    inlib = set(np.unique(a['episode'][np.load(ROOT / f'exp/offline_search/rounds/r10/subsets/{cell}_{size}.npy')]).tolist())
    eps = np.asarray(a['episode']); tasks = np.asarray(a['task_id'])
    seq = []
    for task in range(10):
        cand = [e for e in np.unique(eps[tasks == task]) if (size == 500 or e not in inlib)]
        e = int(cand[0])
        rows = np.flatnonzero(eps == e)[:per_task]
        uid = f'latency:{cell}:{task}:{e}'
        ep = api.EpisodeView(uid, f'task{task}', task, e, task, plugin.ep_seed(0, uid))
        s = SimpleNamespace(rt=SimpleNamespace(model='pi05', api=api, opts=SimpleNamespace(os_tokens='off')),
                            hits=[True] * len(rows), has_vision=[True] * len(rows), last_vision_step=0, _age_before=0)
        for name, arr in (('b_v0', a['key_v0'][rows]), ('b_v1', a['key_v1'][rows]), ('b_rs', a['rs'][rows]),
                          ('b_raw', a['rs'][rows][:, :8]), ('b_aex', a['action'][rows])):
            arr = np.array(arr, np.float32, copy=True); arr.flags.writeable = False
            buf = plugin._Buf(arr.shape[1:], arr.dtype); buf.a = arr; buf.n = len(arr); setattr(s, name, buf)
        seq.append((ep, [plugin.OnlineQueryView(s, j, task, ep) for j in range(len(rows))]))
    return seq, len(inlib)


class Recorder:
    """Exclusive section times; sections nested under the corrector are prefixed 'corr:'."""
    def __init__(self):
        self.stack = []; self.acc = collections.defaultdict(float); self.rows = []

    @contextlib.contextmanager
    def section(self, name):
        if name != 'corrector' and any(f[0] == 'corrector' or f[0].startswith('corr:') for f in self.stack):
            name = 'corr:' + name
        t = time.perf_counter_ns(); frame = [name, t, 0]; self.stack.append(frame)
        try:
            yield
        finally:
            el = time.perf_counter_ns() - t; self.stack.pop(); self.acc[name] += (el - frame[2]) / 1e6
            if self.stack:
                self.stack[-1][2] += el

    def finish(self):
        self.rows.append(dict(self.acc)); self.acc.clear()

    def wrap(self, obj, name, label):
        old = getattr(obj, name)
        @functools.wraps(old)
        def timed(*x, **kw):
            with self.section(label):
                return old(*x, **kw)
        setattr(obj, name, timed)
        return lambda: setattr(obj, name, old)


def run(m, seq, n, rec=None):
    times, dig = [], []
    while len(times) < n:
        for ep, qs in seq:
            m.reset(ep)
            for q in qs:
                t = time.perf_counter_ns(); r = m.query(q); times.append((time.perf_counter_ns() - t) / 1e6)
                if rec:
                    rec.finish()
                if len(dig) < 64:
                    dig.append(hashlib.sha256(np.asarray(r.action).tobytes()).hexdigest())
                if len(times) >= n:
                    return times, dig
    return times, dig


def stats(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.percentile(x, 50)), p90=float(np.percentile(x, 90)), p99=float(np.percentile(x, 99)))


def profile(m, seq, n):
    rec = Recorder(); b = base(m); undo = []
    b.prof = rec
    for name in ('argpartition', 'partition', 'argsort', 'lexsort'):
        undo.append(rec.wrap(np, name, 'topk'))
    undo.append(rec.wrap(b, '_mix', 'synth'))
    if hasattr(b, '_correction'):
        undo.append(rec.wrap(b, '_correction', 'corrector'))
        undo.append(rec.wrap(b, '_features', 'features'))
        undo.append(rec.wrap(r9, '_predict', 'rff_ridge'))
        undo.append(rec.wrap(b, '_apply', 'corr:apply'))
    try:
        t, dig = run(m, seq, n, rec)
    finally:
        for f in reversed(undo):
            f()
        b.prof = api.NULL_PROFILER
    keys = sorted(set().union(*rec.rows))
    return stats(t), {k: stats([r.get(k, 0.) for r in rec.rows]) for k in keys}, dig


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cell', required=True); p.add_argument('--size', type=int, required=True)
    p.add_argument('--n', type=int, default=2048); a = p.parse_args()
    seq, nlib = queries(a.cell, a.size)
    A = load(RUNS / f'r10_size_pi05/fits/r10_{a.cell}_{a.size}_A.pkl')
    AC = load(RUNS / f'r15_purecorr_{a.cell.replace("pi05_", "pi05_")}/fits/r15_purecorr_{a.cell}_{a.size}_AC_dist.pkl')
    ACL = load(RUNS / f'r15_purecorr_{a.cell}/fits/r15_purecorr_{a.cell}_{a.size}_AC_dist.pkl')
    base(AC).__class__ = DistanceCorrectedBase          # deployed GC_dist arithmetic, no R15 telemetry call
    out = dict(cell=a.cell, size=a.size, library_episodes=nlib, queries=sum(len(q) for _, q in seq),
               affinity=sorted(os.sched_getaffinity(0)), load_start=os.getloadavg())
    for label, m in (('A', A), ('AC', AC), ('AC_log', ACL)):
        for mm in (m, base(m)):
            mm.prof = api.NULL_PROFILER
        run(m, seq, a.n)                                  # warm
        t, dig = run(m, seq, a.n)
        pt, comp, pdig = profile(m, seq, a.n)
        assert dig == pdig, 'profiling changed outputs'
        out[label] = dict(total_ms=stats(t), profiled_total_ms=pt, components_ms=comp)
        print(a.cell, a.size, label, 'p50 %.3f mean %.3f' % (out[label]['total_ms']['p50'], out[label]['total_ms']['mean']),
              {k: round(v['mean'], 3) for k, v in comp.items()}, flush=True)
    out['load_end'] = os.getloadavg()
    (OUT / f'bench_{a.cell}_{a.size}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
