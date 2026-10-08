"""Real plugin/orchestrator CPU harness using recorded B-library observations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
import re
from types import MethodType, SimpleNamespace as NS
import numpy as np

from exp.offline_search.closed_loop import plugin, devset
from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
from exp.offline_search.harness import store
from .common import HERE, source, key

EVIDENCE = {}


def load(cell, method, target=None):
    catalog = json.loads((HERE / 'fits.json').read_text())
    rec = next(r for r in catalog if r['config_key'] == key(cell, method, target))
    return pickle.loads(Path(rec['path']).read_bytes())['method']


def equal_result(a, b):
    for field in ('topk', 'scores', 'action'):
        x, y = getattr(a, field), getattr(b, field)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    assert a.confidence == b.confidence and a.library == b.library
    assert a.extras == b.extras


def semantic(d):
    ignored = {'ts', 'conn', 'fit_s', 'method', 'tag', 'wall_ms', 'timing', 'timings', 'ms',
               'decision_index', 'infer_ms', 'stage1_calls'}
    result = {k:v for k,v in d.items() if k not in ignored and k != 'r14' and not k.endswith(('_ms', '_us'))}
    if isinstance(result.get('winner'), str):
        result['winner'] = re.sub(r'^(osplug:(?:synth|blind):r14_cpu:)\d+:', r'\g<1>CONNECTION:', result['winner'])
    return result


class Harness:
    def __init__(self, cell, output, *, pure_c5=False):
        import openpi.cache.config as cc
        self.cc = cc
        self.plugin = plugin
        self.cell = cell
        self.model, self.suite, self.size = cell.split('_')
        rows = json.loads((HERE / 'fits.json').read_text())
        fit = next(r for r in rows if r['config_key'] == key(cell, 'base', None))
        args = ['--os-method', fit['spec'], '--os-kwargs', json.dumps(fit['kwargs']),
                '--os-cell', f'{self.model}_{self.suite}_cache', '--os-root', str(devset.STORE),
                '--os-log-dir', str(output), '--os-tag', 'r14_cpu', '--os-no-shadow-native',
                '--os-tokens', 'off', '--os-r14-audit', '--os-log-inputs', '--os-fit-artifact', fit['path']]
        if pure_c5:
            args += ['--os-log-r4', '--os-judge', 'periodic:1']
        else:
            args += ['--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only']
        opts, rest = plugin.parse_cli(args); assert not rest
        self.rt = plugin.PluginRuntime(opts, self.model)
        self.cfg = cc.load_cache_config(f'exp/trace_dual/config/tr_{self.model}_{"sp" if self.suite == "spatial" else "l10"}_cache.yaml')
        self.cfg.trace = None
        self.shared = cc.build_shared_storage(self.cfg)
        self.lib = store.LibraryView(devset.STORE, f'{self.model}_{self.suite}', devset.PARENT[self.model])
        _, self.episodes, _ = devset.library_episodes(self.model, self.suite, devset.PARENT[self.model])
        self.logs = []
        emit = self.rt.emit
        def capture(row):
            if row.get('ev') == 'dec':
                self.logs.append(copy.deepcopy(row))
            return emit(row)
        self.rt.emit = capture

    def connection(self, method, *, force_guards=False, retries=False):
        from openpi.cache.orchestrator import CacheOrchestrator
        plugin = self.plugin
        harness = self
        lib, cc, rt, cfg = self.lib, self.cc, self.rt, self.cfg
        rt.method = method

        class CPUFake(FakePolicy):
            def __init__(self, *args):
                super().__init__(*args)
                self.looks = self.policy_calls = 0

            def stage1(self, obs):
                self.looks += 1

            def infer(self, obs):
                self.stage1(obs)
                before = self.orch._step_counter
                result = super().infer(obs)
                if result['hit_type'] == 'MISS':
                    self.policy_calls += 1
                assert self.orch._step_counter == before+1
                return result

            def _osp_prepare_blind(self, obs):
                return np.array(lib.rs[obs['_row']], copy=True), None

            def _osp_blind_output(self, action, state):
                return {'actions': action.copy()}

            def _osp_shadow_chunk(self, obs):
                return np.array(lib.action[obs['_row']], np.float32, copy=True)

        def factory(_base, bundle_id='default'):
            comps = cc.build_per_connection_components(cfg, harness.shared, quiet=True)
            comps = rt.attach(comps, cfg, 'r14_cpu')
            kb = FakeKB(NS(key_v0=lib.key_v0, key_v1=lib.key_v1, rs=lib.rs))
            for sess in plugin._TLS.new_sessions:
                sess.kb = kb
            orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
                judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
                write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
                library_stats=comps.get('library_stats'))
            return CPUFake(orch, kb, lib.action)

        conn = plugin._wrap_factory(factory)(None, 'r14_cpu')
        m = conn._osp_sessions[0].method
        if force_guards:
            progress = m.inner._progress
            def forced(owner, q, top1):
                result = progress(q, top1)
                if int(q.step) in (2, 8):
                    owner._noprog_span = owner.noprog_n - 1
                    return owner._noprog_span
                return result
            m.inner._progress = MethodType(forced, m.inner)
        if retries:
            query = m.query
            def repeat(owner, q):
                result = query(q)
                progress = list(owner.inner._vision_progress)
                anchor = copy.deepcopy(owner.inner._policy_gate_anchor)
                equal_result(result, query(q))
                assert progress == owner.inner._vision_progress
                for k in ('step', 'episode', 'task'):
                    assert anchor[k] == owner.inner._policy_gate_anchor[k]
                return result
            m.query = MethodType(repeat, m)
        return conn

    def start(self, conn, task, init=0):
        name = self.lib.meta['tasks'][str(task)]
        conn.on_episode_start(task=name, episode_id=init,
                             extra_metadata=dict(task_uid=f'B:{self.suite}:{task}:{init}', task_id=task,
                                                 orig_init_state_idx=init))

    def obs(self, task, step, *, init=0, executed_steps=5, fixed=False):
        episode = next(i for i,e in enumerate(self.episodes) if e['task_id'] == task and e['orig_init_state_idx'] == init)
        candidates = np.flatnonzero(self.lib.episode == episode)
        r = int(candidates[0 if fixed else min(step, len(candidates)-1)])
        return {'observation/state': np.asarray(self.lib.rs[r, :8], np.float64),
                'prompt': self.lib.meta['tasks'][str(task)], '_row':r,
                'observation/image':np.zeros((2,2,3),np.uint8),
                'observation/wrist_image':np.zeros((2,2,3),np.uint8),
                '__extra__':dict(decision_id=step, executed_steps=executed_steps)}

    def run(self, method, *, tasks=range(10), steps=12, reverse=False, force_guards=False, retries=False):
        conns = [self.connection(method, force_guards=force_guards, retries=retries) for _ in range(2)]
        actions, decs = {}, {}
        begin = len(self.logs)
        for offset in range(0, len(tasks), 2):
            group = list(tasks)[offset:offset+2]
            for i,t in enumerate(group):
                self.start(conns[i],t)
            for step in range(steps):
                for i in (reversed(range(len(group))) if reverse else range(len(group))):
                    t=group[i];conn=conns[i];s=conn._osp_sessions[0]
                    result=conn.infer(self.obs(t,step))
                    assert s.step == s.b_aex.n == len(s.hits) == len(s.has_vision) == step+1
                    assert conn.orch._step_counter == len(conn.orch._action_history) == step+1
                    assert s.b_aex.a[step].tobytes() == np.asarray(result['actions']).tobytes()
                    actions[t,step]=np.asarray(result['actions']).tobytes()
            for i in range(len(group)):
                conns[i].on_episode_end(success=False)
        for d in self.logs[begin:]:
            decs[d['task_id'],d['step']] = json.dumps(semantic(d),sort_keys=True,default=plugin._json_default)
        assert len(decs)==len(actions)==len(tasks)*steps
        return actions,decs,self.logs[begin:]

