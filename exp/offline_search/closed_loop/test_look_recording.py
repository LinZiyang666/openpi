"""CPU fake-observation integration, B boundary, reader integrity and off parity."""
import importlib.util
import itertools
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from . import plugin, look_recording as recording
from .ops.h100 import assets, control
from .test_devset import fixture  # existing attested ten-task toy B library
from exp.offline_search.harness import api, dims


class FakeMethod:
    name = 'FakeAWM'
    prof = api.NULL_PROFILER

    def reset(self, episode):
        self._anchor = None
        self._noprog_span = 0

    def _dist(self, q):
        pass

    def query(self, q):
        rows = np.arange(16, dtype=np.int64)
        weights = np.arange(1, 17, dtype=np.float32); weights /= weights.sum()
        self._anchor = dict(rows=rows.copy(), weights=weights.copy())
        self._noprog_span += 1
        return api.Result(rows, -np.arange(1, 17, dtype=np.float64), .5,
                          action=np.full((10, 32), q.step + 2., np.float32),
                          extras=dict(os_force_miss=0., noprog=float(self._noprog_span)))


class FakePolicy:
    def __init__(self, session):
        self.s = session
        self.orch = SimpleNamespace(_twins=None)

    def stage1(self, obs):
        return obs

    def _osp_prepare_blind(self, obs):
        return np.zeros(32, np.float32), None

    def _osp_blind_output(self, action, state):
        return {'actions': action[:, :7]}

    def _osp_shadow_chunk(self, obs):
        return np.full((10, 32), 42, np.float32)

    def infer(self, obs):
        s = self.s
        self.stage1(obs)
        ctx = SimpleNamespace(current_step=s.step, task_key='task', checkpoint_id='CP1',
                              query_keys={'vision_0': np.arange(32768, dtype=np.float32),
                                          'vision_1': np.arange(32768, dtype=np.float32) + 1,
                                          'robot_state': np.zeros(32, np.float32)})
        s.on_search(ctx)
        chunk = s._dec['served']
        if obs.get('_miss'):
            s._dec.update(hit=False, judge='fake', tau=0, run=0)
            chunk = np.full((10, 32), 77, np.float32)
        s.on_executed(chunk)
        return {'actions': chunk[:, :7].copy()}


def writer_options(tmp_path, *, tokens=False, images=0, shadow=False):
    contract = dict(schema=recording.SCHEMA, dev=True, init_pool='B', arm='sample', model='pi05',
                    recording_arm_dir=str(tmp_path / 'server'),
                    suite='libero_10', cell='pi05_l10_cache', method='fixture:Recipe',
                    pairs=[[0, 7]], manifest_sha256='a'*64, fit_sha256='b'*64,
                    subset={'excluded_inits_by_task': {'0': [0, 1]}})
    p = tmp_path / 'contract.json'; p.write_text(json.dumps(contract))
    opts, rest = plugin.parse_cli(['--os-method', 'fixture:Recipe', '--os-cell', 'pi05_l10_cache',
         '--os-log-dir', str(tmp_path / 'server'), '--os-record-looks', '--os-look-contract', str(p),
         '--os-look-contract-sha256', recording.devset.sha(p)])
    opts.os_record_look_tokens, opts.os_record_look_images, opts.os_record_look_shadow = tokens, images, shadow
    opts.os_tokens = 'off'  # recording token capture is independent of method access
    assert not rest
    return opts, contract


def connection(module, opts, looks=None):
    records = []
    rt = SimpleNamespace(r4=False, gpu=None, looks=looks, decision_lock=threading.RLock(), _conn_ids=itertools.count(),
        native_mode=False, method=FakeMethod(), api=api, shadow_native=False, sessions=weakref.WeakSet(),
        task_map={'task': 0}, lib_key='pi05_l10', opts=opts, ep_index={}, randomized=False, H=10, dims=dims,
        tag='fake', lib_sizes={'current': 16}, tables={'current': np.zeros((16,10,32),np.float32)},
        cur_action=np.zeros((16,10,32),np.float32), ids=list(map(str,range(16))), judge=None, model='pi05',
        suite='l10', method_name='fake', blind=False, policy_tail=False, decision_count=0, emit=records.append)
    kb = SimpleNamespace(_slice=lambda: {'vision_0': np.zeros((256,2048),np.float16),
                                         'vision_1': np.ones((256,2048),np.float16)})
    s = module.PluginSession(rt, None, None, kb, 'sample')
    s.ident = dict(task='task', episode_id=123, extra=dict(task_id=0, orig_init_state_idx=7,
                   task_uid='sample:eval:0:7', run_id=recording.token(looks.contract) if looks else None))
    conn = module._ConnPolicy(FakePolicy(s), [s], 'sample')
    obs = {'observation/state': np.arange(8,dtype=np.float64),
           'observation/image': np.zeros((224,224,3),np.uint8),
           'observation/wrist_image': np.full((224,224,3),99,np.uint8)}
    return conn, s, obs, records


def test_real_session_records_full_topk_tokens_images_and_shadow(tmp_path):
    opts, _ = writer_options(tmp_path, tokens=True, images=56, shadow=True)
    writer = recording.LookWriter(opts, 'pi05')
    conn, s, obs, _ = connection(plugin, opts, writer)
    output = conn.infer(obs)
    assert s.cur_obs is None  # capture happened before the regular cleanup
    r = list(recording.read_looks(writer.directory))[0]
    assert r['topk'].tolist() == list(range(16))  # ordinary JSONL retains ten
    np.testing.assert_array_equal(r['distances'], np.arange(1,17))
    np.testing.assert_array_equal(r['donor_rows'], r['topk'])
    assert np.isclose(r['weights'].sum(), 1)
    np.testing.assert_array_equal(r['wire_actions'], output['actions'])
    assert r['shadow_policy'].shape == r['action'].shape == (10,32)
    assert np.all(r['shadow_policy'] == 42) and np.all(r['action'] == 2)
    assert r['tok_v0'].shape == (256,2048) and r['tok_v0'].dtype == np.float16
    assert r['img1'].shape == (56,56,3) and np.all(r['img1'] == 99)
    assert r['meta']['episode_id'] == 123 and r['meta']['task_id'] == 0 and r['meta']['init'] == 7
    assert r['guard_state']['method']['0']['_noprog_span'] == 1


def test_miss_label_is_actual_policy_and_blind_never_writes(tmp_path):
    opts,_ = writer_options(tmp_path, shadow=True)
    writer = recording.LookWriter(opts,'pi05')
    conn,s,obs,_ = connection(plugin,opts,writer)
    conn.infer({**obs,'_miss':True})
    r = list(recording.read_looks(writer.directory))[0]
    assert np.all(r['action'] == 77) and np.array_equal(r['shadow_policy'], r['action'])
    assert r['meta']['shadow_source'] == 'served_policy'
    writer.write(s, {'vision':False}, None, None)
    assert len(list(recording.read_looks(writer.directory))) == 1


def test_subset_anchor_maps_to_parent_without_recomputing_distances():
    b = FakeMethod(); b._anchor = dict(rows=np.array([0,1]),weights=np.array([.2,.8],np.float32))
    wrapper=SimpleNamespace(inner=SimpleNamespace(base=b),row_subset=np.array([10,30]))
    s=recording.retrieval_snapshot(wrapper,np.array([30,10]),np.array([-1.,-3.]),{})
    assert s['donor_rows'].tolist()==[10,30]
    np.testing.assert_array_equal(s['weights'],np.array([.8,.2],np.float32))
    assert s['distances'].tolist()==[1.,3.]


@pytest.mark.parametrize('change',['pool','dev','model','cell','method','sha','overlap'])
def test_server_contract_fail_closed(tmp_path,change):
    opts,c=writer_options(tmp_path)
    if change=='sha': opts.os_look_contract_sha256='c'*64
    else:
        if change=='pool': c['init_pool']='A'
        elif change=='dev': c['dev']=False
        elif change=='overlap': c['pairs']=[[0,0]]
        else: c[change]='other'
        Path(opts.os_look_contract).write_text(json.dumps(c))
        opts.os_look_contract_sha256=recording.devset.sha(opts.os_look_contract)
    with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'): recording.LookWriter(opts,'pi05')


def test_episode_A_unstamped_foreign_arm_and_unselected_refused(tmp_path):
    opts,_=writer_options(tmp_path);w=recording.LookWriter(opts,'pi05')
    good=dict(run_id=recording.token(w.contract),task_uid='sample:eval:0:7')
    for extra,t,i in [({},0,7),({**good,'run_id':'A'},0,7),({**good,'task_uid':'foreign:eval:0:7'},0,7),(good,0,8)]:
        with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'):w.validate_episode(extra,t,i)
    w.validate_episode(good,0,7)


def test_log_directory_cannot_be_rebound_to_test_A(tmp_path):
    opts,_=writer_options(tmp_path)
    opts.os_log_dir=str(tmp_path/'test_A/runs/sample/server_1')
    with pytest.raises(ValueError,match='outside the attested dev arm'):
        recording.LookWriter(opts,'pi05')
    assert not Path(opts.os_log_dir).exists()


def test_corrupt_shard_and_foreign_manifest_refused(tmp_path):
    opts,_=writer_options(tmp_path);w=recording.LookWriter(opts,'pi05')
    conn,_,obs,_=connection(plugin,opts,w);conn.infer(obs)
    p=w.directory/w.data['shards'][0]['file'];p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA mismatch'):list(recording.read_looks(w.directory))
    w.data['init_pool']='A';w._flush()
    with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'):list(recording.read_looks(w.directory))


def test_resume_process_uses_distinct_manifests(tmp_path):
    opts,_=writer_options(tmp_path)
    for _ in range(2):
        w=recording.LookWriter(opts,'pi05');conn,_,obs,_=connection(plugin,opts,w);conn.infer(obs)
    assert len(list(recording.read_looks(w.directory)))==2
    assert len(list(w.directory.glob('manifest_*.json')))==2


def test_pi_shadow_private_noise_preserves_global_rng():
    import contextlib
    import torch
    stage1=SimpleNamespace(state=torch.zeros((1,32)))
    s=SimpleNamespace(rt=SimpleNamespace(model='pi05',H=10),_look_stage1=stage1,
                      ep=SimpleNamespace(seed=19),step=2,miss_steps=10)
    p=SimpleNamespace(_stage_config=None,_stage2_fn=lambda x:x,
                      _stage3_fn=lambda x,noise,num_steps:SimpleNamespace(action_chunk=noise))
    adapter=SimpleNamespace(fake=False,policy=p,lock=contextlib.nullcontext())
    before=torch.random.get_rng_state().clone()
    first=recording.shadow_chunk(s,adapter);second=recording.shadow_chunk(s,adapter)
    assert torch.equal(before,torch.random.get_rng_state())
    assert first.tobytes()==second.tobytes() and first.shape==(10,32)


@pytest.mark.parametrize('local',[False,True])
def test_opt_in_plan_adds_bound_contract_and_excludes_queries(fixture,tmp_path,local):
    root,store,row,_,_=fixture
    row['yaml']=str(root/'config.yaml');row['matrix']=str(root/'matrix.yaml')
    # build_plan's other dependencies are already exercised by the h100 suite;
    # here arrange only the small files it opens on this path.
    import yaml
    pkl=store/'native.pkl';pkl.write_bytes(b'payload')
    Path(row['yaml']).write_text(yaml.safe_dump({'backend':{'in_memory':{'preload_path':str(pkl)}}}))
    Path(row['matrix']).write_text('arms: []')
    d=store/'library/pi05_l10/current';d.mkdir()
    for name in ('action.npy','task_id.npy'): (d/name).write_bytes(b'array')
    (d/'ids.json').write_text('[]');(d/'manifest.json').write_text('{}')
    row['plugin_args']+=['--os-record-looks'];(root/'arms.json').write_text(json.dumps([row]))
    base=tmp_path/'h100'
    plan=assets.build_plan(root,['sample'],tmp_path/'plan',store=store,local=local,base=base)
    args=plan['arms'][0]['plugin_args']
    entry=next(f for f in plan['files'] if 'sample: B recording contract' in f['reasons'])
    cp=Path(entry['source'])
    assert recording.devset.sha(cp)==assets.flag(args,'--os-look-contract-sha256')
    assert not any('/queries/' in f['original'] for f in plan['files'])
    contract=json.loads(cp.read_text());assert contract['init_pool']=='B' and len(contract['pairs'])==50
    expected=str(root/'runs/sample') if local else str(base/'runs'/root.name/'runs/sample')
    assert contract['recording_arm_dir']==expected
    if not local:
        assert assets.flag(args,'--os-look-contract')==str(Path(expected)/'look_contract.json')


@pytest.mark.parametrize('flag',recording.FLAGS+recording.CONTRACT_FLAGS)
def test_A_planner_refuses_every_recording_flag(tmp_path,flag):
    root=tmp_path/'A';root.mkdir()
    row=dict(arm='sample',model='pi05',suite='libero_10',mode='plugin',plugin_args=[flag])
    (root/'arms.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'):
        assets.build_plan(root,['sample'],tmp_path/'plan')
    with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'):control.server_spec(root,row,23220)
    with pytest.raises(ValueError,match='LOOK_RECORDING_REFUSED'):recording.prepare(root,tmp_path/'dest')
    assert not (tmp_path/'dest').exists()


def test_worker_stamp_only_after_B_contract():
    from openpi.conductor.task import EpisodeTask
    t=EpisodeTask('sample:eval:0:7','sample','eval','libero_10',0,7,7,'host',1,'sample',extra={})
    class Strategy:
        def _episodes(self,yaml_id,server):return [t]
    run=SimpleNamespace(SweepStrategy=Strategy)
    contract=dict(dev=True,arm='sample',model='pi05',suite='libero_10',manifest_sha256='a',fit_sha256='b')
    with pytest.raises(ValueError):recording.stamp_tasks(run,'A',contract)
    recording.stamp_tasks(run,'B',contract)
    from examples.libero.episode_runner import _episode_extra_metadata
    assert _episode_extra_metadata(Strategy()._episodes('sample',None)[0])['run_id']==recording.token(contract)


def test_default_off_actions_and_decision_log_bytes_match_saved_original(tmp_path,monkeypatch):
    baseline=Path('exp/offline_search/rounds/r12/prep/evidence/plugin_before.py')
    if not baseline.exists():pytest.skip('prep baseline required for historical byte proof')
    spec=importlib.util.spec_from_file_location('look_plugin_before',baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    monkeypatch.setattr(plugin.time,'perf_counter_ns',lambda:1000000000)
    monkeypatch.setattr(plugin.time,'perf_counter',lambda:1.)
    monkeypatch.setattr(plugin.time,'time',lambda:1.)
    opts,_=plugin.parse_cli(['--os-method','fixture:Recipe','--os-cell','pi05_l10_cache','--os-log-dir',str(tmp_path)])
    results=[]
    for module in (old,plugin):
        conn,s,obs,records=connection(module,opts)
        action=conn.infer(obs)['actions']
        s.finish_episode('episode_end',True)
        logbytes=json.dumps(records,sort_keys=True,default=recording.json_value).encode()
        results.append((action.tobytes(),logbytes))
    assert results[0]==results[1]
    assert not (tmp_path/'looks').exists()
