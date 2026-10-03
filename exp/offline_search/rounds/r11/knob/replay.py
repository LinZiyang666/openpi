"""Real CPU plugin integration fixture, copied from R10 and scoped to knob/."""
from pathlib import Path
import json
from types import MethodType, SimpleNamespace
import numpy as np
from exp.offline_search.rounds.r10.data import STORE, PARENT, write_json
from .build import HERE
from .verification import audit_trace

def replay(row, label, size=None):
    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
    from exp.offline_search.harness import store
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    plugin._git_head = lambda: None
    output = HERE / 'replays' / label
    # Remove only this replay's own synthetic decision files, never real logs.
    for p in output.glob('decisions_*.jsonl'):
        p.unlink()
    args = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']), '--os-cell', row['cell'],
            '--os-log-dir', str(output), '--os-tag', 'recipe_replay', *row['plugin_args']]
    opts, rest = plugin.parse_cli(args)
    assert not rest
    rt = plugin.install(opts, row['model'])
    cfg = cc.load_cache_config(row['yaml'])
    shared = cc.build_shared_storage(cfg)
    library = 'current' if size is None else PARENT[row['model']]
    lib = store.LibraryView(STORE, f'{row["model"]}_{row["suite_short"]}', library)
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
    synth_checks, fingerprints, gate_factors = [], [], []
    for conn in conns:
        inner = conn._osp_sessions[0].method.inner
        progress = inner._progress
        def forced(self, q, top1, original=progress):
            span = original(q, top1)
            if int(q.step) in (2, 8):
                self._noprog_span = self.noprog_n - 1
                return self._noprog_span
            return span
        inner._progress = MethodType(forced, inner)  # never serialized in artifacts
        base = inner.base
        synth = base.os_synth
        def checked(self, q, rows, weights, original=synth):
            raw = BlindAWM.os_synth(self, q, rows, weights)
            correction = self._correction(q, raw)
            expected = self._apply(raw, correction)
            action = original(q, rows, weights)
            assert action.tobytes() == expected.tobytes()
            assert action[:, 6:].tobytes() == raw[:, 6:].tobytes()
            assert action[10:].tobytes() == raw[10:].tobytes()
            if int(q.step) == 0:
                assert action.tobytes() == raw.tobytes()
            T, _, _, _, _, _, _, d, _, _, _ = self._dist(q)
            positions = np.searchsorted(T.rows, rows)
            ratio = float(np.min(d[positions])) / self.distance_scale
            gate_factors.append(0. if q.step == 0 else float(np.clip((2-ratio)/1.25, 0, 1)))
            fingerprints.append(dict(step=int(q.step), task=int(q.task_id), rows=np.asarray(rows).tobytes().hex(),
                weights=np.asarray(weights).tobytes().hex(), raw=raw.tobytes().hex(), action=action.tobytes().hex()))
            synth_checks.append(1)
            return action
        base.os_synth = MethodType(checked, base)

    actions = []
    # Ten tasks, two simultaneous sessions, each reset three times: static
    # no-progress states, sequential library states, then distant library keys.
    for pair in range(5):
        for scenario in range(3):
            for i, conn in enumerate(conns):
                t = 2 * pair + i
                conn.on_episode_start(task=lib.meta['tasks'][str(t)], episode_id=scenario,
                    extra_metadata={'task_uid': f'replay-{t}-{scenario}', 'task_id': t, 'orig_init_state_idx': scenario})
            for step in range(12):
                for i, conn in enumerate(conns):
                    t = 2 * pair + i
                    candidates = np.flatnonzero(lib.task_id == t)
                    if scenario == 0:
                        r = int(candidates[0])
                    elif scenario == 1:
                        r = int(candidates[min(step, len(candidates)-1)])
                    else:
                        r = int(candidates[(step * 97 + len(candidates)//2) % len(candidates)])
                    obs = {'observation/state': np.asarray(lib.rs[r, :8], np.float64),
                           'prompt': lib.meta['tasks'][str(t)], '_row': r,
                           'observation/image': np.zeros((2, 2, 3), np.uint8),
                           'observation/wrist_image': np.zeros((2, 2, 3), np.uint8),
                           '__extra__': {'decision_id': step, 'executed_steps':
                               4 if row.get('fault_lifecycle') and scenario == 2 and step == 3 else 5}}
                    result = conn.infer(obs)
                    sess = conn._osp_sessions[0]
                    assert sess.step == step + 1 == sess.b_aex.n
                    assert np.array_equal(sess.b_aex.a[step], result['actions'])
                    if step > 0 and not sess.has_vision[step] and sess.hits[step-1] == 0:
                        np.testing.assert_array_equal(result['actions'][:5, :7], sess.b_aex.a[step-1, 5:10, :7])
                    assert np.isfinite(result['actions'][:, :7]).all()
                    actions.append(np.asarray(result['actions']).tobytes().hex())
            for conn in conns:
                conn.on_episode_end(success=False)
    rt.flush_all()
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decs) == 360
    assert all(d['lib'] == library for d in decs if d['src'] in ('cache', 'cache_blind'))
    forced_decs = [d for d in decs if d['step'] in (2, 8)]
    assert len(forced_decs) == 60
    if not row.get('fault_lifecycle'):
        assert all(d['vision'] and not d['hit'] and d['extras']['os_force_miss'] == 1
                   and int(d['extras']['os_flags']) == 8 for d in forced_decs)
    assert any(d['src'] == 'policy_tail' for d in decs)
    if row.get('r11_method','off') == 'off':
        assert any(d['src'] == 'cache_blind' for d in decs)
    # Compare all decision semantics. Machine/time/clone identifiers do not
    # describe a decision; preserve scores, confidence, rows, flags and extras.
    ignored = {'ts', 'conn', 'fit_s', 'method', 'tag', 'wall_ms', 'timing', 'timings', 'ms'}
    semantic = [{k: v for k, v in d.items() if k not in ignored and not k.endswith(('_ms', '_us'))} for d in decs]
    report = dict(PASS=True, decisions=len(decs), tasks=10, connections=2, episodes=30,
                  forced_guard_triggers=len(forced_decs), serving_corrector_checks=len(synth_checks),
                  miss=sum(not d['hit'] for d in decs), blind=sum(d['src'] == 'cache_blind' for d in decs),
                  policy_tail=sum(d['src'] == 'policy_tail' for d in decs),
                  gate_zero=sum(f == 0 for f in gate_factors),
                  gate_partial=sum(0 < f < 1 for f in gate_factors), gate_full=sum(f == 1 for f in gate_factors))
    report.update(audit_trace(decs, row))
    write_json(output / 'selftest_report.json', report)
    return dict(decisions=semantic, actions=actions, synthesis=fingerprints), report
