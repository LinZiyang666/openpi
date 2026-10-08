"""Real CPU plugin/orchestrator replay, with poisoned tower and call receipts."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.selftest import FakePolicy
from exp.offline_search.harness import store
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.types import CheckpointID
from .build import HERE, STORE, baseline, write
from .encoder import sha


class KB:
    def __init__(self, lib):
        self.lib, self.row, self.calls = lib, 0, 0
    def collect(self, *args, **kwargs):
        pass
    def clear(self):
        pass
    def build(self, *args):
        self.calls += 1
        return dict(vision_0=torch.from_numpy(np.array(self.lib.key_v0[self.row])),
                    vision_1=torch.from_numpy(np.array(self.lib.key_v1[self.row])),
                    robot_state=torch.from_numpy(np.array(self.lib.rs[self.row])))
    cached_data = property(lambda s: {})


def connection(module, size, level, scratch, *, light=True, artifact=None):
    if light:
        path = Path(artifact) if artifact else HERE/f'artifacts/pi05_l10_{size}/{level}.pkl'
        blob = pickle.loads(path.read_bytes())
    else:
        path, blob = baseline(size, level)
    args = ['--os-method', blob['spec'], '--os-kwargs', json.dumps(blob['kwargs']),
        '--os-cell','pi05_l10_cache','--os-root',str(STORE),'--os-log-dir',str(scratch),
        '--os-tag','replay','--os-no-shadow-native','--os-tokens','off','--os-blind',
        '--os-policy-tail','--os-policy-tail-blocks','1','--os-judge','guard_only',
        '--os-fit-artifact',str(path)] + (['--os-light-key'] if light else [])
    opts, _ = module.parse_cli(args)
    exists = Path.exists
    # Default path uses an A identity index for historical debug logs. This
    # regression suppresses that index before any open, and never runs git.
    with patch.object(module, '_git_head', return_value=None), patch.object(Path, 'exists',
            lambda p: False if 'queries' in p.parts else exists(p)):
        rt = module.PluginRuntime(opts, 'pi05')
    records = []
    rt.emit = lambda row: records.append(json.loads(json.dumps(row, default=module._json_default)))
    native = SimpleNamespace(on_episode_start=lambda: None, record_action=lambda a: None,
                             get_search_session_id=lambda: None)
    storage = SimpleNamespace(open_search_session=lambda sid: None, close_search_session=lambda sid: None)
    kb = KB(store.LibraryView(STORE,'pi05_l10','bpool_cs'))
    session = module.PluginSession(rt, native, storage, kb, 'cpu')
    cp = CheckpointID.CP1
    orch = CacheOrchestrator(storage=module.PluginStorage(storage, session), key_builder=kb,
        gates={cp:AlwaysSearchGate()}, judges={cp:module.PluginJudge(session)},
        search_strategies={cp:module.PluginStrategy(session, native)})
    class Model(FakePolicy):
        def __init__(self):
            super().__init__(orch, kb, kb.lib.action)
            self.towers = self.full = self.broadcasts = 0
            self.poison = False
            broadcast = orch.broadcast_action
            def counted(chunk):
                self.broadcasts += 1
                return broadcast(chunk)
            orch.broadcast_action = counted
        def stage1(self, obs):
            if self.poison:
                raise AssertionError('tower called on a light HIT or blind continuation')
            self.towers += 1
        def infer(self, obs):
            self.full += 1
            self.stage1(obs)
            kb.row = int(obs['_row'])
            return super().infer(obs)
        def _osp_prepare_blind(self, obs):
            return np.array(kb.lib.rs[int(obs['_row'])]), None
        def _osp_blind_output(self, action, state):
            return {'actions': action.copy()}
    model = Model()
    conn = module._ConnPolicy(model, [session], 'cpu')
    return conn, session, model, records


def observation(session, row, decision, **receipt):
    lib = session.rt.method
    parent = STORE/'library/pi05_l10/bpool_cs'
    tokrows = np.load(parent/'tok/rows.npy', mmap_mode='r')
    pos = np.flatnonzero(tokrows == row)
    assert len(pos) == 1
    return {'_row':row, 'prompt':session.ep.task if session.ep is not None else session.ident['task'],
        'observation/state': np.zeros(8, np.float32),
        'observation/image': np.array(np.load(parent/'tok/img0.npy',mmap_mode='r')[pos[0]]),
        'observation/wrist_image': np.array(np.load(parent/'tok/img1.npy',mmap_mode='r')[pos[0]]),
        '__extra__':dict(decision_id=decision,executed_steps=5,exec_ok=True) | receipt}


def start(conn, session, task=0, episode='cpu-0'):
    name = next(k for k,v in session.rt.task_map.items() if v==task)
    conn.on_episode_start(task=name,episode_id=0,
        extra_metadata=dict(task_uid=episode,task_id=task,orig_init_state_idx=0,init_pool='B'))


def run():
    results = []
    with tempfile.TemporaryDirectory(prefix='r15qk-replay-') as tmp:
        for size in (50,200):
            conn, s, model, records = connection(plugin,size,'off',Path(tmp)/str(size))
            start(conn,s)
            row = int(s.method.row_subset[0])
            count = [0]
            encode = s.rt.light_encoder.encode
            def counted(look):
                count[0] += 1
                return encode(look)
            s.rt.light_encoder.encode = counted
            model.poison = True
            a = conn.infer(observation(s,row,0))
            assert a['hit_type']=='FULL_HIT' and count[0]==1 and model.towers==model.full==0
            b = conn.infer(observation(s,row,1))
            assert b['hit_type']=='FULL_HIT' and count[0]==1 and model.towers==0
            before = (s.step,count[0],model.broadcasts)
            try:
                conn.infer(observation(s,row,1))
                raise AssertionError('duplicate accepted')
            except ValueError as e:
                assert 'duplicate' in str(e)
            assert (s.step,count[0],model.broadcasts)==before
            # Force one real look/MISS through the unchanged judge boundary.
            model.poison = False
            verdict = s._verdict
            def miss(step, conf, ex):
                return {**verdict(step,conf,ex),'hit':False,'judge':'fixture-forced-miss'}
            s._verdict = miss
            c = conn.infer(observation(s,row,2,executed_steps=3))
            assert c['hit_type']=='MISS' and count[0]==2 and model.towers==model.full==1
            assert model.broadcasts==3 and s.b_aex.n==s.step==3
            model.poison = True
            d = conn.infer(observation(s,row,3))
            assert d['hit_type']=='FULL_HIT' and count[0]==2 and model.towers==1
            np.testing.assert_array_equal(d['actions'][:5],c['actions'][5:10])
            assert records[-1]['source']=='policy_tail'
            # Failed execution receipt must discard a usable blind anchor/tail.
            model.poison = False
            conn.infer(observation(s,row,4,exec_ok=False))
            assert count[0]==3 and model.towers==2
            assert model.broadcasts==5 and s.step==s.b_aex.n==5
            assert [r['qk_work'] for r in records if r['ev']=='dec']==[
                dict(Q=1,T=0,M=0,N=1),dict(Q=0,T=0,M=0,N=1),
                dict(Q=1,T=1,M=1,N=1),dict(Q=0,T=0,M=0,N=1),dict(Q=1,T=1,M=1,N=1)]
            last = [r for r in records if r['ev']=='dec'][-1]
            assert last['IR_new']==(s.method.light_key_spec['epsilon']*3+2)/5
            results.append(dict(size=size, Q=count[0],T=model.towers,M=model.full,N=s.step,
                                broadcasts=model.broadcasts,duplicate_rejected=True,policy_tail=True,failed_receipt=True,
                                artifact=str(HERE/f'artifacts/pi05_l10_{size}/off.pkl'),
                                artifact_sha256=sha(HERE/f'artifacts/pi05_l10_{size}/off.pkl'),
                                student_sha256=s.method.light_key_spec['kwargs']['model_sha256'],
                                completed_epochs=s.method.light_key_spec['completed_epochs'],
                                epsilon=s.method.light_key_spec['epsilon'],
                                all_policy_light_looks_IR=1+s.method.light_key_spec['epsilon']))
            s._verdict = verdict
            start(conn,s,task=1,episode='cpu-reset')
            row = int(s.method.row_subset[s.method.inner.base.tasks[1].rows[0]])
            conn.infer(observation(s,row,0))
            assert s.step==1 and len(s._qk_looks)==1 and s._qk_counts['N']==1
            assert s._qk_last_policy is None
        # Execute byte-preserved pre-change plugin and updated default teacher
        # path against identical library rows, actions and receipt sequences.
        name = 'exp.offline_search.closed_loop._qk_before'
        spec = importlib.util.spec_from_file_location(name,HERE/'before/plugin.py')
        before = importlib.util.module_from_spec(spec)
        sys.modules[name]=before
        spec.loader.exec_module(before)
        snapshots=[]
        for i,module in enumerate((before,plugin)):
            conn,s,model,records=connection(module,50,'off',Path(tmp)/f'default{i}',light=False)
            start(conn,s)
            row=int(s.method.row_subset[0])
            outs=[conn.infer(observation(s,row,j)) for j in range(4)]
            decs=[r for r in records if r['ev']=='dec']
            snapshots.append((outs,decs))
        for old,new in zip(snapshots[0][0],snapshots[1][0],strict=True):
            np.testing.assert_array_equal(old['actions'],new['actions'])
            assert old['hit_type']==new['hit_type']
        keys=('hit','source','vision','topk','scores','extras','rows','weights','served_head','stage1_calls','robot_state')
        for old,new in zip(snapshots[0][1],snapshots[1][1],strict=True):
            for key in keys:
                assert old.get(key)==new.get(key),(key,old.get(key),new.get(key))
        assert not any('qk_work' in r for r in snapshots[1][1])
    report=dict(PASS=True,cells=results,default_path_byte_equal=True,epsilon_zero_IR=2/5,
                all_policy_light_looks_IR='per-cell 1+frozen epsilon',servers=0,GPU=0)
    write(HERE/'plugin_replay.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    run()
