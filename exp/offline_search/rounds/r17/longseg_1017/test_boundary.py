"""Real fitted recipe, periodic/stall clocks and serving, with CPU adapters."""
import copy
import inspect
from pathlib import Path
import numpy as np
import pytest

from exp.r17_integration.tests.ops.test_cache_study import fixture_model,freeze
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from .runtime import installer,source

def test_semantic_policy_seed_uses_heads_with_frozen_delivery_numbers():
    from types import SimpleNamespace
    from unittest.mock import patch
    from exp.r17_integration.common.serving import ConnectionPolicy
    from exp.r17_integration.benchmarks.metaworld import decision_seed
    p=ConnectionPolicy.__new__(ConnectionPolicy)
    p.requires_context=False;p.commit_horizon=10;p._delivery_seq=4;p._decision_seq=2
    p.spec=SimpleNamespace(layout=SimpleNamespace(replan_steps=5))
    p.config=SimpleNamespace(adapter_binding={'benchmark_protocol':'formal-test'},
        recipe=SimpleNamespace(enabled=False,task_table_identity=None))
    p.bundle_id='test';p._episode=dict(task_name='reach-v3',task_id=0,original_init_idx=7)
    with patch('exp.r17_integration.common.benchmarks.get_benchmark',return_value=SimpleNamespace(benchmark='metaworld_mt50')):
        r=p._request(dict(decision_seq=4,explicit_seed=decision_seed('reach-v3',7,4)))
    assert r.decision_seq==2 and r.explicit_seed==decision_seed('reach-v3',7,2)


def test_same_head_observations_have_identical_real_recipe_clocks(tmp_path):
    from exp.r17_integration.search.cache_recipe import CacheStudyRecipe
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    provider=fixture_model('dp_libero')
    policies=[]
    for commit in (5,10):
        config,fit=freeze(provider,tmp_path/str(commit),commit,gated=True)
        config.recipe.periodic_gap=1.5
        model=StallModel.load(tmp_path/str(commit)/'stall')
        # Each real fit must match its serving horizon; the fixture inputs and
        # metric/Q3 calibration are otherwise the same decision-level data.
        recipe=CacheStudyRecipe(fit.fork(),config,model=model)
        recipe.force_look=lambda *a:(_ for _ in ()).throw(AssertionError('look evaluated on delivery'))
        recipe.tail_diagnostics=recipe.force_look
        policy=make_connection_factory(provider,config,build_shared_storage(config),
            recipe_factory=lambda *a,recipe=recipe:recipe)()
        policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
        policies.append(policy)
    traces=[[],[]];deliveries=[0,0]
    fields=('stall_status','stall_call','stall_extra_look','stall_cooldown','stall_anchor',
            'decision_index','knob_call','corrector_computed','corrector_applied','source','donor_rows','donor_weights')
    for decision in range(24):
        obs={'state':np.asarray(fit.state[decision%len(fit.state)],np.float32),**{
            n:np.arange(w,dtype=np.float32) for n,w in provider.model_spec.feature_spec.fields.items() if n!='robot_state'}}
        for index,policy in enumerate(policies):
            seq=deliveries[index];wire=dict(obs,__request__={'decision_seq':seq})
            if seq:wire['__execution__']={'decision_seq':seq-1,'executed_count':5}
            head=policy.infer(wire);meta=head['__hit_meta__'];deliveries[index]+=1
            assert meta['decision_index']==decision and meta['env_step_index']==decision*(5 if index==0 else 10)
            traces[index].append({k:meta.get(k) for k in fields})
            if index:
                before=(dict(policy.adapter.counts),policy._decision_seq,len(policy._states),
                        policy.recipe.anchor,policy.recipe.memo_step,policy.recipe.tracker._last_control)
                extract=policy.adapter.extract_state_only
                policy.adapter.extract_state_only=recipe.force_look
                tail=policy.infer(dict(obs,state=np.full_like(obs['state'],1e6),
                    __request__={'decision_seq':seq+1},__execution__={'decision_seq':seq,'executed_count':5}))
                policy.adapter.extract_state_only=extract;deliveries[index]+=1
                after=(dict(policy.adapter.counts),policy._decision_seq,len(policy._states),
                       policy.recipe.anchor,policy.recipe.memo_step,policy.recipe.tracker._last_control)
                assert before==after
                assert tail['__hit_meta__']['source']==meta['source']+'_tail'
                assert tail['actions'].tobytes()==head['actions'][5:].tobytes()
    assert traces[0]==traces[1]


def test_frozen_worker_keeps_clock_metadata_without_worker_changes():
    import asyncio
    from types import SimpleNamespace
    from exp.r17_integration.common.websocket_policy_server import _ControlSocket
    from exp.r17_integration.benchmarks.execution import cost_row
    from openpi_client import msgpack_numpy
    received=[]
    class Socket:
        async def send(self,value):received.append(msgpack_numpy.unpackb(value))
    sock=_ControlSocket(Socket(),SimpleNamespace(_policy=None,_concurrent=True))
    meta=dict(clock_unit='segment_decision',decision_index=3,delivery_index=7,env_step_index=35,segment_head=False)
    asyncio.run(sock.send(msgpack_numpy.Packer().pack(dict(__hit_meta__=meta,server_timing={'infer_ms':1.}))))
    row=cost_row(received[0],{},7)
    assert row['decision_seq']==7 and row['server_timing']['decision_index']==3
    assert row['server_timing']['env_step_index']==35


@pytest.mark.parametrize('gap',[None,0.,1.,3.])
def test_real_cache_study_recipe_never_interrupts_a_head(tmp_path,gap):
    provider=fixture_model('dp_libero');cfg,fit=freeze(provider,tmp_path,10,gated=True)
    cfg.recipe.periodic_gap=gap
    # This is the exact in-memory patch shipped inside each new bootstrap.
    from .build import pin
    resident=tmp_path/'resident';(resident/'assets').mkdir(parents=True)
    pin(resident,{})
    code=(resident/'assets/ffk_bootstrap.py').read_text().replace(
        "ffk_runtime=dict(__name__='__main__')","ffk_runtime=dict(__name__='cpu_test')",1)
    scope={};exec(code,scope);scope['ffk_runtime']['install']({})
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    obs={'state':np.asarray(fit.state[2],np.float32),**{n:np.arange(w,dtype=np.float32)
         for n,w in provider.model_spec.feature_spec.fields.items() if n!='robot_state'}}
    policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    trace=[]
    for seq in range(40):
        r=policy.infer(dict(obs,**({'__execution__':{'decision_seq':seq-1,'executed_count':5}} if seq else {})))
        m=r['__hit_meta__'];trace.append(m)
        if seq%2:
            assert m['source']==trace[-2]['source']+'_tail'
            assert m['cursor']==5 and m['stage_calls']==(0,0,0)
            assert r['actions'].tobytes()==head[5:].tobytes()
        else:
            assert m['cursor']==0;head=r['actions'].copy()
    assert policy.recipe.decisions==19
    if gap==0:assert all(m['source']=='policy' for m in trace[::2])
    assert [m['decision_index'] for m in trace]==[i for i in range(20) for _ in (0,1)]


def test_audit_counts_consecutive_midpoint_heads_but_not_terminal_heads(tmp_path):
    import json
    from .audit import scan
    root=tmp_path/'lane';(root/'driver').mkdir(parents=True)
    (root/'campaign.json').write_text(json.dumps(dict(model='dp',domain='metaworld')))
    rows=[]
    for uid,sources in [('cut',['library','policy']),('valid',['policy','policy_tail']),('terminal',['library'])]:
        for seq,s in enumerate(sources):rows.append(dict(accepted=True,task_uid=uid,attempt=1,run_id='r',
            decision_seq=seq,step_idx=5*seq,cursor=5 if s.endswith('_tail') else 0,source=s,commit_horizon=10,arm='k10'))
    path=root/'driver/per_step.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    r=scan(path)
    assert r['cuts']==1 and r['tails']==1 and r['eligible_heads']==2 and r['cut_percent']==50


def test_budgeted_archive_roundtrip(tmp_path):
    import tarfile
    from .io_budget import filesystem_budget
    payload=b'bounded staging inputs\n'*1000
    source=tmp_path/'input.bin';archive=tmp_path/'inputs.tar'
    with filesystem_budget(str(tmp_path)):
        source.write_bytes(payload)
        with tarfile.open(archive,'w') as tar:tar.add(source,arcname='input.bin')
        with tarfile.open(archive,'r') as tar:
            assert tar.extractfile('input.bin').read()==payload


def test_server_staging_reuses_immutable_inputs_without_duplicate_numeric_snapshot(tmp_path):
    import json
    from types import SimpleNamespace
    from .staging import metadata_tree,reusable_inputs
    assets=tmp_path/'assets';inputs=assets/'cache_inputs';inputs.mkdir(parents=True)
    (assets/'full.yaml').write_text('enabled: false\n')
    (inputs/'fit.npz').write_bytes(b'frozen parameters')
    manifest=dict(digest='prior-verified-seal',files={'fit.npz':'sealed-fit-pin'})
    (inputs/'manifest.json').write_text(json.dumps(manifest))
    staged=metadata_tree(tmp_path)
    assert (staged/'full.yaml').read_bytes()==(assets/'full.yaml').read_bytes()
    assert not (staged/'cache_inputs').exists()
    calls=[]
    def remote(*args,**kwargs):
        calls.append((args,kwargs));return json.dumps(dict(manifest,present=True))
    assert reusable_inputs(SimpleNamespace(remote=remote),inputs)
    assert len(calls)==1 and calls[0][1]['replay_safe']


def test_two_gpu_transport_maps_are_forwardable_and_locally_disjoint(tmp_path):
    from .ports import route
    from exp.r17_integration.ops.formal_ports import local_ports
    from exp.r17_integration.ops.formal_eval import open_tunnels
    base=dict(server_node='local',public_port=23400,driver_port=23110,replicas=4,
              worker_nodes={'local':48},execution_worker_nodes={'timan1':24,'timan107':24},longseg_1017={})
    with pytest.raises(ValueError,match='inside 23100-23199'):open_tunnels(dict(base),tmp_path)
    local=route(copy.deepcopy(base),tmp_path)
    h100=route(dict(copy.deepcopy(base),server_node='h100',public_port=23180),tmp_path)
    assert local['endpoint']=='ziyanglin.com:23130' and local['driver_public_port']==23139
    assert h100['endpoint']=='149.165.153.233:23180' and h100['driver_public_port']==23199
    assert set(local_ports(local)).isdisjoint(local_ports(h100))


def test_plain_control_executes_model_heads_and_native_audit_accepts_only_its_disabled_cache(tmp_path):
    import json,yaml
    from dataclasses import asdict
    from exp.r17_integration.common.config import RecipeConfig
    from .build import pin
    provider=fixture_model('dp_libero');cfg,fit=freeze(provider,tmp_path/'fit',10,gated=True)
    cfg.enabled=False;cfg.checkpoints={};cfg.recipe=RecipeConfig(corrector=False)
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
    obs={'state':np.asarray(fit.state[2],np.float32),**{n:np.arange(w,dtype=np.float32)
         for n,w in provider.model_spec.feature_spec.fields.items() if n!='robot_state'}}
    for seq in range(6):
        out=policy.infer(dict(obs,**({'__execution__':dict(decision_seq=seq-1,executed_count=5)} if seq else {})))
        meta=out['__hit_meta__']
        assert meta['source']==('policy_tail' if seq%2 else 'policy')
        assert meta['resident_pid'] and meta['resident_generation']
        assert meta['stage_calls'][0]==(0 if seq%2 else 1)
    root=tmp_path/'control';(root/'assets').mkdir(parents=True)
    c=dict(longseg_1017_control_kind='long');pin(root,c)
    (root/'campaign.json').write_text(json.dumps(c))
    (root/'EPISODE_PLAN.json').write_text(json.dumps(dict(tasks=[dict(task_uid='only-control')])))
    (root/'assets/CACHE_STUDY.json').write_text(json.dumps(dict(groups=[dict(group='chrg_pure',file='pure.yaml')])))
    doc=dict(enabled=False,checkpoints={},recipe=asdict(cfg.recipe),adapter_binding=dict(commit_horizon=10))
    (root/'assets/pure.yaml').write_text(yaml.safe_dump(doc))
    ns={};exec((root/'assets/ffk_evidence.py').read_text(),ns)
    assert ns['empty_uids'](root)=={'only-control'}
    doc['recipe']['enabled']=True;(root/'assets/pure.yaml').write_text(yaml.safe_dump(doc))
    with pytest.raises(AssertionError):ns['empty_uids'](root)


def test_final_system_never_substitutes_native_length_pure_for_long_fallback(monkeypatch):
    from . import analyze
    monkeypatch.setattr(analyze,'stats',lambda data,*a:dict(n=len(data),successes=sum(v['success'] for v in data.values())))
    target={('libero_10','task',i):dict(success=True) for i in range(50)}
    selected=dict(tasks={'task':dict(selected='pure')},no_knob=False)
    report,data=analyze.whole_segment_result(selected,target,target,{},'dp','full','libero_10','RTX 4090')
    assert not data and report['n']==0 and report['tolerance_all_tasks'] is None
    control={k:dict(success=k[2]>=3) for k in target}
    report,data=analyze.whole_segment_result(selected,target,target,control,'dp','full','libero_10','RTX 4090')
    assert report['successes']==47 and report['tolerance_all_tasks'] is False
    assert report['tasks']['task']['executed_fallback']=='pure_long'


def test_unaffected_analysis_keeps_existing_formal_output_and_jobs_unchanged(tmp_path,monkeypatch):
    import json
    from . import analyze
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a as formal,formal_a_analysis as native
    original_root,original_models=formal.ROOT,formal.MODELS
    monkeypatch.setattr(analyze,'HERE',tmp_path)
    monkeypatch.setattr(native,'active_jobs',lambda:[])
    def fake_analyze():
        assert formal.ROOT==tmp_path/'unaffected_libero'
        (formal.ROOT/'RESULTS.json').write_text(json.dumps(dict(rows=[dict(model='pi05',step_mode='full')])))
    monkeypatch.setattr(native,'analyze',fake_analyze)
    rows=analyze.unaffected_libero()
    assert rows[0]['gpu']=='H100'
    assert formal.ROOT==original_root and formal.MODELS==original_models
