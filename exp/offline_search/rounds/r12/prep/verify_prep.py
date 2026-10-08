"""Default dependency/spec byte proof and real B-library CPU plugin replay."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'


def default_proof():
    from exp.offline_search.closed_loop.ops.h100 import assets, control
    root = Path('/home/weiland/trace_runs/os_closed_loop/r10_recipe_current')
    rows = json.loads((root / 'arms.json').read_text())
    with (EVIDENCE / 'default_plan_after.log').open('w') as f, contextlib.redirect_stdout(f):
        # Baseline used a relative --work-dir; match that spelling exactly.
        plan = assets.build_plan(root, [r['arm'] for r in rows], EVIDENCE.relative_to(Path.cwd()) / 'default_plan')
    after = (EVIDENCE / 'default_plan/plan.json').read_bytes()
    before = (EVIDENCE / 'default_plan_before.json').read_bytes()
    specs = json.dumps({r['arm']: dict(selection=control.selection(root, r), server=control.server_spec(root, r, 23220))
                       for r in rows}, sort_keys=True)
    result = dict(plan_byte_identical=after == before, plan_sha256=hashlib.sha256(after).hexdigest(),
                  files=len(plan['files']), bytes=plan['bytes'],
                  selection_server_bytes_identical=specs == (EVIDENCE / 'default_specs_before.json').read_text(),
                  worker_script_bytes_identical=(control.HERE / 'run_arm.sh').read_bytes() == (EVIDENCE / 'run_arm_before.sh').read_bytes(),
                  remote_calls=0, gpu=False)
    assert all(result[k] for k in ('plan_byte_identical', 'selection_server_bytes_identical', 'worker_script_bytes_identical'))
    (HERE / 'default_unchanged.json').write_text(json.dumps(result, indent=1) + '\n')
    print('DEFAULT_UNCHANGED', json.dumps(result), flush=True)


def replay():
    from exp.offline_search.closed_loop import plugin, look_recording
    from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
    from exp.offline_search.closed_loop.ops.h100 import assets
    from exp.offline_search.harness.store import LibraryView
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest

    source = Path('/home/weiland/trace_runs/os_closed_loop/r11_dev_current')
    row = next(r for r in json.loads((source / 'arms.json').read_text()) if r['model'] == 'pi05' and r['suite'] == 'libero_10')
    # Fresh evidence directories per invocation; never overwrite real run roots.
    output = EVIDENCE / ('cpu_replay_' + str(time.time_ns()))
    output.mkdir()
    root = output / 'dev_root'
    rows = look_recording.prepare(source, root, [row['arm']], tokens=True, images=56, shadow=True)
    with (output / 'plan.log').open('w') as f, contextlib.redirect_stdout(f):
        plan = assets.build_plan(root, [row['arm']], output / 'plan', local=True)
    ready = plan['arms'][0]
    opts, rest = plugin.parse_cli(['--os-method', ready['method'], '--os-kwargs', json.dumps(ready['kwargs']),
                                 '--os-cell', ready['cell'], '--os-log-dir', str(root / 'runs' / row['arm'] / 'server_replay'),
                                 '--os-tag', 'cpu_replay', *ready['plugin_args']])
    assert not rest
    plugin._git_head = lambda: None  # no git is invoked by this proof
    rt = plugin.install(opts, 'pi05')
    assert not rt.ep_index  # recording runtime never opens test-A query metadata
    cfg = cc.load_cache_config(ready['yaml'])
    shared = cc.build_shared_storage(cfg)
    lib = LibraryView('/home/weiland/trace_runs/offline_search_store', 'pi05_l10', 'bpool_cs')

    class CPUFake(FakePolicy):
        def stage1(self, obs):
            return None

        def _osp_prepare_blind(self, obs):
            return np.array(lib.rs[obs['_row']], copy=True), None

        def _osp_blind_output(self, action, state):
            return {'actions': action.copy()}

        def _osp_shadow_chunk(self, obs):
            return np.array(lib.action[obs['_row']], copy=True)

        def infer(self, obs):
            self.stage1(obs)
            return super().infer(obs)

    def factory(base, bundle_id='default'):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = FakeKB(lib)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
            judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
            write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
            library_stats=comps.get('library_stats'))
        return CPUFake(orch, kb, lib.action)

    conn = plugin._wrap_factory(factory)(None, row['arm'])
    manifest = load_manifest(row['manifest'])
    executed, blind, misses = [], 0, 0
    for task, init in sorted(manifest['selected'])[:3]:
        ep = next(e for e in lib.episodes if e['task_id'] == task and e['orig_init_state_idx'] == init)
        uid = f'{row["arm"]}:eval:{task}:{init}'
        conn.on_episode_start(task=ep['task'], episode_id=50*task+init,
            extra_metadata=dict(task_uid=uid, task_id=task, orig_init_state_idx=init, attempt=1,
                                run_id=look_recording.token(rt.looks.contract)))
        token_rows = lib.tok_rows[(lib.tok_rows >= ep['start']) & (lib.tok_rows < ep['end'])]
        assert len(token_rows)
        for step in range(10):
            r = int(token_rows[min(step, len(token_rows)-1)])
            obs = dict(_row=r, **{'observation/state': np.zeros(8,np.float64),
                       'observation/image': np.array(lib.tok_of('img0',r),np.uint8),
                       'observation/wrist_image': np.array(lib.tok_of('img1',r),np.uint8)})
            out = conn.infer(obs)
            session = conn._osp_sessions[0]
            executed.append((uid, step, out['actions'].copy(), r))
            blind += int(not session.has_vision[-1])
            misses += int(session.hits[-1] == 0)
        conn.on_episode_end(success=False)
    recs = list(look_recording.read_looks(rt.looks.directory))
    assert len(recs) == len(executed) - blind
    for rec in recs:
        _,_,wire,r = next(x for x in executed if x[:2] == (rec['meta']['uid'], int(rec['step'])))
        assert np.array_equal(wire, rec['wire_actions'])
        assert rec['key_v0'].tobytes() == np.asarray(lib.key_v0[r], np.float32).tobytes()
        assert rec['key_v1'].tobytes() == np.asarray(lib.key_v1[r], np.float32).tobytes()
        assert np.isclose(rec['donor_weights'].sum(), 1)
        assert len(rec['topk']) == 16 and np.isfinite(rec['distances']).all()
        assert rec['tok_v0'].shape == (256,2048) and rec['img0'].shape == (56,56,3)
        # All actual donors must be mapped to the same parent row space as topk.
        assert set(rec['donor_rows']) <= set(rec['topk'])
    result = dict(passed=True, model='pi05', suite='l10', root=str(root), output=str(rt.looks.directory),
                  init_pool='B', episodes=3, decisions=len(executed), looks=len(recs), blind_decisions=blind,
                  policy_decisions=misses, wire_action_bytes_equal=True, key_bytes_equal=True,
                  full_neighbors_and_parent_donors_verified=True, tokens_images_verified=True,
                  shadow='CPU fake policy only; no GPU model or policy launch', gpu=False, remote_calls=0)
    (HERE / 'cpu_replay.json').write_text(json.dumps(result, indent=1) + '\n')
    print('CPU_REPLAY', json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=('default', 'replay'))
    args = ap.parse_args()
    default_proof() if args.action == 'default' else replay()
