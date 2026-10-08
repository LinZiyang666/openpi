"""CPU contracts; run pytest with -p no:cacheprovider and external basetemp."""
import copy
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from exp.offline_search.closed_loop import plugin, look_recording
from exp.offline_search.rounds.r02.g3_recovery import g3_core
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger as collected_ledger
from . import replay
from .build import HERE, baseline
from .encoder import sha
from .recording import LightLookWriter
from .runtime import initialize


def test_real_plugin_replay():
    assert replay.run()['PASS']


def test_exact_stable_ties():
    scores=np.zeros(1000)
    valid=np.ones(1000,bool);valid[:30]=False
    np.testing.assert_array_equal(g3_core.topk_pos(scores,16,valid),np.arange(30,46))
    scores[998:]=1
    np.testing.assert_array_equal(g3_core.topk_pos(scores,16,valid),np.r_[998,999,np.arange(30,44)])


def test_wrong_cell_partial_artifact(tmp_path):
    conn,s,model,records=replay.connection(plugin,50,'off',tmp_path/'partial')
    original=s.rt.method
    for mutation in ('cell','epochs','heads','rows'):
        s.rt.method=copy.deepcopy(original)
        method=s.rt.method
        if mutation=='cell':method.light_key_spec['cell']='pi05_l10_200'
        elif mutation=='epochs':method.light_key_spec['completed_epochs']=79
        elif mutation=='heads':method.inner.base.heads.pop('9')
        else:method.row_subset=method.row_subset[:-1]
        with pytest.raises(ValueError):initialize(s.rt)


def test_fixed_periodic_and_collector(tmp_path):
    conn,s,model,records=replay.connection(plugin,50,'ir25',tmp_path/'periodic')
    replay.start(conn,s)
    _,old=baseline(50,'ir25')
    assert s.method.knob_settings==old['method'].knob_settings
    # Isolate the periodic scheduler from the independent no-progress trigger.
    # This alteration exists only in this fixture, never in a served artifact.
    s.method.inner.disabled_mask=15
    row=int(s.method.row_subset[0])
    for step in range(24):conn.infer(replay.observation(s,row,step))
    decs=[r for r in records if r['ev']=='dec']
    assert any(r['policy_full_call'] for r in decs)
    setting=old['method'].knob_settings['setting']
    for r in decs:
        if r['has_look']:
            assert r['fixed_cap'] in (math.floor(setting),math.ceil(setting))
            assert r['extras']['os_r11_setting']==setting
            assert r['parent_donor_rows'][:len(r['topk'])]==r['topk']
            assert len(r['donor_weights'])==16
    agg=collected_ledger(dict(model='pi05',cost_ledger=True,client_overrides={'replan_steps':5}),decs)
    assert agg['Q']==sum(r['light_encoder_ran'] for r in decs)
    assert agg['T']==agg['M']==model.towers==model.full
    assert agg['N']==model.broadcasts==24
    assert agg['IR_new']==(agg['epsilon']*agg['Q']+agg['M'])/24
    zero=copy.deepcopy(decs)
    for r in zero:r['epsilon']=0
    assert collected_ledger(dict(model='pi05',cost_ledger=True),zero)['IR_new']==agg['M']/24
    all_policy=copy.deepcopy(decs)
    for r in all_policy:
        r.update(epsilon=.014,light_encoder_ran=True,teacher_tower_ran=True,policy_full_call=True,miss_k=10)
        r['qk_work']=dict(Q=1,T=1,M=1,N=1)
    assert collected_ledger(dict(model='pi05',cost_ledger=True),all_policy)['IR_new']==1.014


def test_hit_output_preflight_single_miss(tmp_path):
    conn,s,model,records=replay.connection(plugin,50,'off',tmp_path/'preflight')
    replay.start(conn,s)
    output=conn._osp_adapter.output
    conn._osp_adapter.output=lambda *args: (_ for _ in ()).throw(ValueError('fixture output transform rejected'))
    row=int(s.method.row_subset[0])
    result=conn.infer(replay.observation(s,row,0))
    assert result['hit_type']=='MISS' and model.towers==model.full==model.broadcasts==1
    assert s.b_aex.n==s.step==1
    dec=[r for r in records if r['ev']=='dec'][0]
    assert dec['qk_work']==dict(Q=1,T=1,M=1,N=1)
    conn._osp_adapter.output=output


def test_b_recording_light_features_no_teacher(tmp_path):
    conn,s,model,records=replay.connection(plugin,50,'off',tmp_path/'recording')
    contracts=json.loads(Path('/home/weiland/trace_runs/os_closed_loop/r15_qk_pilot_pi05_l10_50/dev_contracts.json').read_text())
    arm=next(k for k in contracts if '_light_off' in k)
    c=dict(contracts[arm],schema=look_recording.SCHEMA,cell='pi05_l10_cache',method=s.rt.opts.os_method,
           recording_arm_dir=str(tmp_path),pairs=contracts[arm].get('pairs',[[0,0]]))
    # Planner normally supplies this final attestation augmentation.
    c['pairs']=json.loads(Path('/home/weiland/trace_runs/os_closed_loop/r15_qk_pilot_pi05_l10_50/dev_selection.json').read_text())['pairs']
    path=tmp_path/'contract.json';path.write_text(json.dumps(c))
    opts=s.rt.opts
    opts.os_look_contract=str(path);opts.os_look_contract_sha256=sha(path);opts.os_record_look_images=56
    writer=LightLookWriter(opts,'pi05')
    writer.validate_episode(dict(run_id=look_recording.token(c),task_uid=arm+':eval:0'),0,0)
    with pytest.raises(ValueError):writer.validate_episode(dict(run_id='A',task_uid='A'),0,0)
    s.rt.looks=writer
    name=next(k for k,v in s.rt.task_map.items() if v==0)
    conn.on_episode_start(task=name,episode_id=0,extra_metadata=dict(task_uid=arm+':eval:0',task_id=0,
        orig_init_state_idx=0,run_id=look_recording.token(c)))
    row=int(s.method.row_subset[0]);model.poison=True
    conn.infer(replay.observation(s,row,0))
    shards=list(look_recording.read_looks(writer.directory))
    assert len(shards)==1 and shards[0]['light_features'].shape==(2,64)
    assert not any(k in shards[0] for k in ('key_v0','key_v1','tok_v0','tok_v1','shadow_policy'))
    assert not bool(shards[0]['label_valid']) and model.towers==0
    c['init_pool']='A';path.write_text(json.dumps(c));opts.os_look_contract_sha256=sha(path)
    with pytest.raises(ValueError):LightLookWriter(opts,'pi05')
