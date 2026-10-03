"""CPU integration test using recorded B library queries, real plugin/orchestrator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np

from .data import STORE, PARENT, write_json


def run_test(run, arm):
    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    row, = [r for r in json.loads((run / 'arms.json').read_text()) if r['arm'] == arm]
    plugin._git_head = lambda: None
    out = run / 'selftest' / arm
    # This directory contains only this test's generated logs.
    for p in out.glob('decisions_*.jsonl'):
        p.unlink()
    args = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']), '--os-cell', row['cell'],
            '--os-log-dir', str(out), '--os-tag', 'r10test', *row['plugin_args']]
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
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
            judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
            write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
            library_stats=comps.get('library_stats'))
        return CPUFake(orch, kb, lib.action)

    conns = [plugin._wrap_factory(factory)(None, str(i)) for i in range(2)]
    corrector_checks = []
    if row['kwargs']['variant'].startswith('GC_'):
        from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
        from exp.offline_search.rounds.r09.recipe.recipe import RecipeCorrectedBase
        for conn in conns:
            base = conn._osp_sessions[0].method.inner.base
            assert isinstance(base, RecipeCorrectedBase) and base.blend == .5 and base.chans == 6
            assert set(base.heads) == set(map(str, range(10)))
            for h in base.heads.values():
                assert h['coef'].shape == (60, 601) and h['w'].shape == (217, 384)
            original_synth = base.os_synth
            def checked(self, q, rows, weights, original_synth=original_synth):
                raw = BlindAWM.os_synth(self, q, rows, weights)
                expected = self._apply(raw, self._correction(q, raw))
                action = original_synth(q, rows, weights)
                assert action.tobytes() == expected.tobytes()
                assert np.array_equal(action[:, 6:], raw[:, 6:])
                assert np.array_equal(action[10:], raw[10:])
                corrector_checks.append(1)
                return action
            base.os_synth = MethodType(checked, base)
    if row['kwargs']['variant'] != 'A':
        for conn in conns:
            inner = conn._osp_sessions[0].method.inner
            original = inner._progress
            def forced(self, q, top1, original=original):
                span = original(q, top1)
                if int(q.step) == 2:
                    self._noprog_span = self.noprog_n - 1
                    return self._noprog_span
                return span
            inner._progress = MethodType(forced, inner)  # test-only, never serialized
    served = []
    for restart in range(2):
        for i, conn in enumerate(conns):
            t = i
            conn.on_episode_start(task=lib.meta['tasks'][str(t)], episode_id=restart,
                extra_metadata={'task_uid': f'r10test-{i}-{restart}', 'task_id': t, 'orig_init_state_idx': restart})
        for step in range(12):
            for i, conn in enumerate(conns):
                r = int(np.flatnonzero(lib.task_id == i)[0])
                obs = {'observation/state': np.asarray(lib.rs[r, :8], np.float64),
                    'prompt': lib.meta['tasks'][str(i)], '_row': r,
                    'observation/image': np.zeros((2, 2, 3), np.uint8),
                    'observation/wrist_image': np.zeros((2, 2, 3), np.uint8),
                    '__extra__': {'decision_id': step, 'executed_steps': 5}}
                result = conn.infer(obs)
                sess = conn._osp_sessions[0]
                assert sess.step == step + 1 == sess.b_aex.n
                assert np.array_equal(sess.b_aex.a[step], result['actions'])
                assert np.isfinite(result['actions'][:, :7]).all()
                served.append(np.asarray(result['actions'])[:5, :7].tolist())
        for conn in conns:
            conn.on_episode_end(success=False)
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decs) == 48
    assert [d['served_head'] for d in decs] == served
    assert all(d['lib'] == PARENT[row['model']] for d in decs if d['src'] in ('cache', 'cache_blind'))
    forced_decs = [d for d in decs if d['step'] == 2 and d['vision']]
    if row['kwargs']['variant'] != 'A':
        assert len(forced_decs) == 4
        assert all(not d['hit'] and d['extras']['os_force_miss'] == 1 and int(d['extras']['os_flags']) == 8 for d in forced_decs)
        assert any(d['src'] == 'policy_tail' for d in decs)
    else:
        assert all(d['hit'] for d in decs)
        assert any(d['src'] == 'cache_blind' for d in decs)
    if row['kwargs']['variant'].startswith('GC_'):
        assert len(corrector_checks) >= 4
    report = dict(PASS=True, arm=arm, fit_inputs='B libraries only', query_inputs='recorded B-library keys/state/chunks',
        decisions=48, connections=2, episodes=4, miss=sum(not d['hit'] for d in decs),
        blind=sum(d['src'] == 'cache_blind' for d in decs), policy_tail=sum(d['src'] == 'policy_tail' for d in decs),
        forced_guard_triggers=len(forced_decs) if row['kwargs']['variant'] != 'A' else 0,
        serving_corrector_checks=len(corrector_checks))
    write_json(out / 'selftest_report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--arm', required=True)
    a = p.parse_args()
    run_test(a.run, a.arm)
