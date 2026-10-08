"""CPU plugin replay on B-library inputs, in isolated processes and /tmp scratch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import MethodType, SimpleNamespace

import numpy as np

from .build import HERE, REPO, SPEC, load, fit_path, write_json
from .runtime import PureCorrector
from exp.offline_search.rounds.r10.data import STORE, PARENT


def trace(row, output, guarded):
    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    plugin._git_head = lambda: None
    args = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']),
            '--os-cell', row['cell'], '--os-log-dir', str(output), '--os-tag', 'purecorr_cpu', *row['plugin_args']]
    opts, rest = plugin.parse_cli(args)
    assert not rest
    rt = plugin.install(opts, row['model'])
    cfg = cc.load_cache_config(row['yaml'])
    shared = cc.build_shared_storage(cfg)
    lib = store.LibraryView(STORE, f'{row["model"]}_{row["suite_short"]}', PARENT[row['model']])
    qc = SimpleNamespace(key_v0=lib.key_v0, key_v1=lib.key_v1, rs=lib.rs)

    class CPUFake(FakePolicy):
        def stage1(self, obs):
            pass

        def _osp_prepare_blind(self, obs):
            return np.array(lib.rs[obs['_row']], copy=True), None

        def _osp_blind_output(self, action, state):
            return {'actions': action.copy()}

    def factory(_base, bundle_id='default'):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = FakeKB(qc)
        for session in plugin._TLS.new_sessions:
            session.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
            judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
            write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
            library_stats=comps.get('library_stats'))
        return CPUFake(orch, kb, lib.action)

    conns = [plugin._wrap_factory(factory)(None, str(i)) for i in range(2)]
    if guarded:
        for conn in conns:
            inner = conn._osp_sessions[0].method.inner
            original = inner._progress
            def forced(self, q, top1, original=original):
                span = original(q, top1)
                if int(q.step) == 2:
                    self._noprog_span = self.noprog_n - 1
                    return self._noprog_span
                return span
            inner._progress = MethodType(forced, inner)
    actions = []
    candidates = {t: np.flatnonzero(lib.task_id == t) for t in range(10)}
    # Two interleaved connections, every task, two episode resets, static and
    # moving states. Only B-library keys/state/chunks, never test A trajectories.
    for pair in range(5):
        for scenario in range(2):
            for i, conn in enumerate(conns):
                t = 2 * pair + i
                conn.on_episode_start(task=lib.meta['tasks'][str(t)], episode_id=scenario,
                    extra_metadata={'task_uid': f'purecorr-{t}-{scenario}', 'task_id': t,
                                    'orig_init_state_idx': scenario})
            for step in range(8):
                for i, conn in enumerate(conns):
                    t = 2 * pair + i
                    ids = candidates[t]
                    # Choose actual selected-library rows to guarantee nonzero engagement.
                    subset = conn._osp_sessions[0].method.row_subset
                    selected = np.intersect1d(ids, subset)
                    r = int(selected[0] if scenario == 0 else selected[min(step, len(selected)-1)])
                    obs = {'observation/state': np.asarray(lib.rs[r, :8], np.float64),
                           'prompt': lib.meta['tasks'][str(t)], '_row': r,
                           'observation/image': np.zeros((2, 2, 3), np.uint8),
                           'observation/wrist_image': np.zeros((2, 2, 3), np.uint8),
                           '__extra__': {'decision_id': step, 'executed_steps': 5}}
                    result = conn.infer(obs)
                    sess = conn._osp_sessions[0]
                    assert sess.step == step + 1 == sess.b_aex.n
                    assert np.array_equal(sess.b_aex.a[step], result['actions'])
                    assert np.isfinite(result['actions'][:, :7]).all()
                    actions.append(np.asarray(result['actions']).tobytes())
            for conn in conns:
                conn.on_episode_end(success=False)
    rt.flush_all()
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines()
            if json.loads(line).get('ev') == 'dec']
    assert len(decs) == 160
    if guarded:
        forced = [d for d in decs if d['step'] == 2]
        assert len(forced) == 20
        assert all(d['vision'] and not d['hit'] and int(d['extras']['os_flags']) == 8 for d in forced)
        assert any(d['src'] == 'policy_tail' for d in decs)
    else:
        assert all(d['hit'] for d in decs)
    assert any(d['src'] == 'cache_blind' for d in decs)
    ignored = {'ts', 'conn', 'fit_s', 'method', 'tag', 'wall_ms', 'timing', 'timings', 'ms'}
    semantic = []
    for d in decs:
        clean = {k: v for k, v in d.items() if k not in ignored and not k.endswith(('_ms', '_us'))}
        clean['extras'] = {k: v for k, v in clean['extras'].items() if not k.startswith('os_pc_')}
        semantic.append(clean)
    applied = [d for d in decs if d['extras'].get('os_pc_applied') == 1.]
    inherited = [d for d in decs if d['extras'].get('os_pc_inherited') == 1.]
    if row['kwargs'].get('corrector_enabled') and not guarded:
        assert applied and inherited
        assert all(d['extras'].get('os_pc_count') == 0 for d in decs if d['step'] == 0)
    report = dict(PASS=True, decisions=160, tasks=10, connections=2, episodes=20,
                  forced_guard_misses=20 if guarded else 0, correction_applied=len(applied),
                  correction_inherited=len(inherited), miss=sum(not d['hit'] for d in decs),
                  telemetry_example=applied[0]['extras'] if applied else None)
    write_json(output / 'report.json', report)
    with gzip.open(output / 'trace.pkl.gz', 'wb') as f:
        pickle.dump(dict(actions=actions, decisions=semantic), f, protocol=4)


def one(record):
    name = record['arm']
    with tempfile.TemporaryDirectory(prefix=f'r15_purecorr_{name}_') as td:
        scratch = Path(td)
        prod, = [r for r in json.loads((Path(record['root']) / 'arms.json').read_text()) if r['arm'] == name]
        refs = record['references']
        rows = dict(A=refs['A']['row'], GC_dist=refs['GC_dist']['row'], production=prod)
        for label, guarded in (('off', False), ('guard_on', True)):
            a, gc = (load(refs[v]['files']['fit']['path']) for v in ('A', 'GC_dist'))
            method = PureCorrector.assemble(a['method'], gc['method'], corrector_enabled=guarded,
                                            guard_enabled=guarded)
            row = copy.deepcopy(refs['GC_dist' if guarded else 'A']['row'])
            kwargs = dict(size=record['size'], corrector_enabled=guarded, guard_enabled=guarded)
            row.update(method=SPEC, kwargs=kwargs)
            path = scratch / f'{label}.pkl'
            path.write_bytes(pickle.dumps(dict(method=method, registered={}, spec=SPEC, kwargs=kwargs,
                                               cell=row['cell'], fit_s=0.), protocol=4))
            row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1] = str(path)
            rows[label] = row
        traces, reports = {}, {}
        for label, row in rows.items():
            out = scratch / label
            write_json(out / 'row.json', row)
            with (out / 'trace.log').open('w') as log:
                proc = subprocess.run([sys.executable, '-m', 'exp.offline_search.rounds.r15.pure_corr.proof',
                    'trace', '--row', str(out / 'row.json'),
                    '--out', str(out), *(['--guard'] if label in ('GC_dist', 'guard_on') else [])],
                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f'{name}/{label}: {proc.returncode}\n{(out / "trace.log").read_text()[-6000:]}')
            with gzip.open(out / 'trace.pkl.gz', 'rb') as f:
                traces[label] = pickle.load(f)
            reports[label] = json.loads((out / 'report.json').read_text())
        for new, old in (('off', 'A'), ('guard_on', 'GC_dist')):
            for key in ('actions', 'decisions'):
                if traces[new][key] != traces[old][key]:
                    first = next(i for i, (x, y) in enumerate(zip(traces[new][key], traces[old][key])) if x != y)
                    raise AssertionError(f'{name}: {new} vs {old} {key} at {first}: '
                                         f'{traces[new][key][first]!r} vs {traces[old][key][first]!r}')
        changed = [i for i, (new, old) in enumerate(zip(traces['production']['actions'], traces['A']['actions'])) if new != old]
        assert changed, 'production corrector must change served actions'
        p = dict(PASS=True, arm=name, corrector_off_vs_A_differences=0, guard_on_vs_GC_dist_differences=0,
                 production_changed_decisions=len(changed), proofs=reports,
                 fixture='B selected-library states; synthetic resets; no A outcomes/recordings read')
        write_json(HERE / 'proofs' / f'{name}.json', p)
        print(f'PROOF PASS {name}: A/off and GC_dist/guard_on, engagement visible', flush=True)
        return p


def all_proofs(workers):
    records = json.loads((HERE / 'audit.json').read_text())['records']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = list(pool.map(one, records))
    write_json(HERE / 'proofs.json', dict(PASS=True, cell_sizes=24,
        compared_decisions=24 * 160 * 2, production_decisions=24 * 160,
        corrector_off_vs_A_differences=0, guard_on_vs_GC_dist_differences=0,
        forced_guard_misses=24 * 20, records=result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['all', 'one', 'trace'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--arm')
    p.add_argument('--row', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--guard', action='store_true')
    a = p.parse_args()
    if a.action == 'trace':
        trace(json.loads(a.row.read_text()), a.out, a.guard)
    elif a.action == 'one':
        record, = [r for r in json.loads((HERE / 'audit.json').read_text())['records'] if r['arm'] == a.arm]
        one(record)
    else:
        all_proofs(a.workers)


if __name__ == '__main__':
    main()
