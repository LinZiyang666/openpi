from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import common as C
from .follow_audit import tube, eligibility
from .camera_audit import action_errors
from .stage_value import score, observed_support
from .profile_report import decision_cost, get_key, paired, DEFAULT_KEYS


def test_two_level_ht_exact_enumeration():
    # Dose chosen with unequal probabilities; then a distinct Bernoulli coin.
    pi=[.2,.8];p=[.25,.75];tau=[.4,.1]
    for target,wanted in [('natural',sum(a*b for a,b in zip(pi,tau))),('equal_supported_doses',np.mean(tau))]:
        mean=0.
        for d in range(2):
            for z in [0,1]:
                weight,_=score(z,p[d],pi[d],2,target)
                Y=.2+z*tau[d]
                mean+=pi[d]*(p[d] if z else 1-p[d])*weight*Y
        assert mean==pytest.approx(wanted)


@pytest.mark.parametrize('p',[0,1])
def test_forced_choices_rejected(p):
    with pytest.raises(ValueError):score(1,p)


@pytest.mark.parametrize('treatments',[[0,0],[1,1],[0,1]])
def test_stage_interval_requires_both_observed_branches(treatments):
    r=observed_support(dict(estimate=.1,lo=.05,hi=.2),treatments)
    assert r['estimate']==.1
    if len(set(treatments))==1:assert r['lo'] is None and r['hi'] is None
    else:assert r['lo']==.05 and r['hi']==.2


def test_censored_and_zero_weight_member_not_dropped():
    rs=np.arange(12,dtype=float).reshape(6,2)
    rows=np.array([0,1]);weights=np.array([1.,0.])
    chain=np.array([np.arange(6),[2,-1,4,5,-1,-1]])
    states=np.array([rs[0],rs[2]])
    delta,absolute,lost,support=tube(states,rows,weights,rs,np.ones(2),chain)
    assert not support[1] and np.isnan(delta[1]) and lost[1]==0
    assert np.isnan(absolute[1])


def test_incremental_displacement_and_censoring():
    rs=np.array([[0.,0.],[1.,2.],[2.,4.]])
    states=np.array([[10.,10.],[11.,12.],[np.nan,np.nan]])
    d,a,l,s=tube(states,np.array([0]),np.array([1.]),rs,np.ones(2),np.array([[0,1,2],[1,2,-1],[2,-1,-1]]))
    assert d[1]==0 and a[1]==10 and np.isnan(d[2]) and s[2]


@pytest.mark.parametrize('model,cost',[('pi05',.152),('groot',.148)])
def test_cost_actual_vision_and_blind(model,cost):
    row=dict(vision=True,hit=True,stage1_mode='full')
    c,_=decision_cost(row,model,DEFAULT_KEYS);assert sum(c.values())==cost
    row.update(vision=False,src='policy_tail')
    c,camera=decision_cost(row,model,DEFAULT_KEYS);assert sum(c.values())==0 and camera=='blind'
    row.update(vision=True,hit=False)
    c,_=decision_cost(row,model,DEFAULT_KEYS);assert sum(c.values())==pytest.approx(1)


def test_wrist_completion_and_missing_telemetry():
    row=dict(vision=True,hit=False,stage1_mode='wrist_only',camera_completed=True)
    c,_=decision_cost(row,'pi05',DEFAULT_KEYS)
    assert sum(c.values())==pytest.approx(.055198+.049890+.848)
    del row['camera_completed']
    with pytest.raises(ValueError,match='completion'):decision_cost(row,'pi05',DEFAULT_KEYS)
    row.update(vision=False)
    with pytest.raises(ValueError,match='without vision'):decision_cost(row,'pi05',DEFAULT_KEYS)


def test_cost_counts_actual_completion_forwards():
    row=dict(vision=True,hit=False,camera_mode='wrist_only',camera_completion=True,camera_completion_calls=2)
    c,_=decision_cost(row,'pi05',DEFAULT_KEYS)
    assert c['completion']==pytest.approx(2*.049890)
    row['camera_completion_calls']=.5
    with pytest.raises(ValueError,match='completion count'):decision_cost(row,'pi05',DEFAULT_KEYS)


def test_camera_mode_not_inferred_from_plan():
    row=dict(vision=True,hit=True,extras={'os_sw_next_camera':1.})
    with pytest.raises(ValueError,match='actual camera'):decision_cost(row,'pi05',DEFAULT_KEYS)
    c,_=decision_cost(row,'pi05',DEFAULT_KEYS,allow_full=True)
    assert c['full_looks']==.152


def test_configurable_keys():
    row={'blind_extras':{'new':3}}
    assert get_key(row,{'granted':['extras.absent','blind_extras.new']},'granted')==3


def test_gripper_event_missing_and_timing():
    a=np.array([[0.,0.],[0.,0.],[0.,1.],[0.,1.]])
    b=np.array([[0.,0.],[0.,1.],[0.,1.],[0.,1.]])
    e=action_errors(a,b,np.ones(2),1,np.array([0.,1.]),2)
    assert e['event_time_error']==1 and e['gripper_mode_error']==.25
    e=action_errors(np.zeros_like(a),b,np.ones(2),1,np.array([0.,1.]),2)
    assert e['event_time_error'] is None and e['missing_event']==1


def fixture(root,ds,journals):
    (root/'client').mkdir(parents=True);(root/'server_1').mkdir()
    (root/'client/journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journals))
    (root/'server_1/decisions_a.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ds))


def test_accepted_prefix_reuse_and_duplicates(tmp_path):
    root=tmp_path/'arm'
    ds=[dict(ev='dec',uid='a',attempt=1,step=s,ts=ts,task_id=0,init=3) for s,ts in [(0,1),(1,2),(0,3),(1,4)]]
    fixture(root,ds+[ds[-1]],[dict(accepted=True,status='failed',success=False,task_uid='a',attempt=1,ts=5)])
    eps,audit=C.accepted_arm(root)
    assert audit['discarded_prior_prefix']==2 and len(eps[0]['decisions'])==2
    assert eps[0]['decisions'][0]['ts']==3


def test_reader_rejects_gap_and_conflicting_duplicates(tmp_path):
    for kind in ['gap','conflict']:
        root=tmp_path/kind
        ds=[dict(ev='dec',uid='a',attempt=1,step=0,ts=1,task_id=0,init=3)]
        ds.append(dict(ds[0],step=2 if kind=='gap' else 0,ts=1,init=4 if kind=='conflict' else 3))
        fixture(root,ds,[dict(accepted=True,status='done',success=True,task_uid='a',attempt=1,ts=3)])
        with pytest.raises(ValueError):C.accepted_arm(root)


def test_paired_same_episodes_identity():
    eps=[dict(task=t,init=i,Y=i%2,N=10,IR=.076,components={'full_looks':.76}) for t in [0,1] for i in [0,1]]
    a=dict(arm='A',episodes=eps)
    r=paired(a,dict(a,arm='B'),100)
    assert r['matched']==4 and r['delta_IR']['lo']==r['delta_IR']['hi']==0 and r['delta_SR_descriptive']==0


def test_output_ownership(tmp_path):
    with pytest.raises(ValueError):C.write(Path('/tmp/outside_c4_profile.json'),{})


def test_task_init_bootstrap_clusters_repeats():
    records=[dict(task=0,init=0),dict(task=0,init=0),dict(task=0,init=1)]
    r=C.task_bootstrap(records,[0,2,1],100)
    assert r['clusters']==2 and r['estimate']==1 and r['lo']==r['hi']==1


def test_private_plan_complete_and_manifest_disjoint():
    prepared=C.HERE/'prepared'
    profile=json.loads((prepared/'arms_profile.json').read_text())
    evaluation=json.loads((prepared/'arms_eval500.json').read_text())
    assert len(profile)==44 and len(evaluation)==32
    assert len({r['name'] for r in profile})==44
    assert all(r['name'].endswith('_profile') for r in profile)
    assert all('--os-log-r4' in r['plugin_args'] and '--os-log-inputs' in r['plugin_args'] for r in profile)
    for model in ['pi05','groot']:
        for suite in ['l10','spatial']:
            m=json.loads((prepared/f'manifests/{model}_{suite}_bval20.json').read_text())
            prior=json.loads(Path(m['prior_calibration_manifest']).read_text())
            assert len(m['selected'])==20
            assert not {(r['task'],r['init']) for r in m['selected']} & {(r['task'],r['init']) for r in prior['selected']}
            assert all(sum(r['task']==t for r in m['selected'])==2 for t in range(10))


def test_profile_analyzer_end_to_end(tmp_path,monkeypatch):
    from .profile_report import report_arm
    root=tmp_path/'arm'
    ds=[dict(ev='dec',uid='a',attempt=1,step=0,ts=1,task_id=0,init=3,lib='current',vision=True,hit=True,src='cache',stage1_mode='full',q_us=10,ok=True),
        dict(ev='dec',uid='a',attempt=1,step=1,ts=2,task_id=0,init=3,lib='current',vision=False,hit=True,src='cache_blind',q_us=5,ok=True)]
    fixture(root,ds,[dict(accepted=True,status='done',success=True,task_uid='a',attempt=1,ts=3)])
    (root/'summary.json').write_text(json.dumps(dict(success=1,cost_ledger=dict(decisions=2,vision_decisions=1,misses=0))))
    monkeypatch.setattr(C,'bank',lambda c:(SimpleNamespace(k=16),None,{'exec_steps':5}))
    monkeypatch.setattr(C,'stage_table',lambda *args:None)
    r,ts=report_arm(root,dict(model='pi05',suite='l10'),DEFAULT_KEYS,100,3,False)
    assert r['IR']==.076 and r['IR_components']['blind']==0 and r['SR_descriptive']==1
    assert r['active_control_episodes']==0 and r['IR_active_controls'] is None and len(ts)==2
    (root/'summary.json').write_text(json.dumps(dict(success=1,cost_ledger=dict(decisions=3))))
    with pytest.raises(ValueError,match='ledger'):report_arm(root,dict(model='pi05',suite='l10'),DEFAULT_KEYS,100,3,False)
