"""CPU-only native seed/init and attempt evidence tests; no policy/model load."""
import json

import pytest

from exp.r17_integration.models.cosmos.recover_baseline import audit_bindings, cross_segment_retries, recover
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.formal_resume import sha


@pytest.fixture
def native_segment(tmp_path):
    root=tmp_path/'segment'
    root.mkdir()
    uid='cell0'
    extra=dict(task_name='libero_10/task',num_steps=5,model_identity='model',norm_identity='norm',
               layout_identity='layout',benchmark_protocol='benchmark',yaml_identity='yaml',commit_horizon=16)
    task=dict(task_uid=uid,attempt=1,task_id=34,orig_init_state_idx=42,bundle_id='formal_full_v1',
              experiment='libero_10',extra=extra)
    write_json(root/'campaign.json',{'run_name':'segment'})
    write_json(root/'EPISODE_PLAN.json',dict(tasks=[task]))
    write_json(root/'BUNDLE_ACK.json',{'bundle_identity':'bundle'})
    write_json(root/'assets/benchmark.json',dict(seed=195,seed_mode='fixed'))
    driver=root/'driver';driver.mkdir()
    rows=dict(journal=[dict(task_uid=uid,accepted=True,attempt=1,status='done',success=True,error=None)],
        attempts=[dict(task_uid=uid,accepted=True,attempt=1,success=True,error=None,n_steps=100)],
        per_step=[dict(task_uid=uid,accepted=True,attempt=1,decision_seq=0,**extra,
            actual_nfe=5,stage_calls=[1,1,5],original_init_idx=42,orig_init_state_idx=42,
            task_id=34,seed=195,task_data_hash='native-init-hash',bundle_id='formal_full_v1',
            bundle_identity='bundle',capture_enabled=False,source='policy')])
    for name,items in rows.items():(driver/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in items))
    native=dict(campaign='segment',episode_plan_sha256=sha(root/'EPISODE_PLAN.json'),gpu_used=False,
                native_init_hashes={uid:'native-init-hash'})
    return root,native


def test_actual_native_init_seed_and_k5_audit(native_segment):
    root,native=native_segment
    audit=audit_bindings(root,native)
    assert audit['valid'] and audit['episodes']==audit['successes']==1
    assert audit['cells']['cell0']['original_init_idx']==42
    assert audit['cells']['cell0']['policy_seed']==195


@pytest.mark.parametrize('field,value',[('seed',42),('task_data_hash','other-init'),('orig_init_state_idx',41),
                                      ('task_id',33),('actual_nfe',1),('stage_calls',[1,1,1])])
def test_wrong_seed_init_or_sampler_fails_closed(native_segment,field,value):
    root,native=native_segment
    path=root/'driver/per_step.jsonl'
    row=json.loads(path.read_text());row[field]=value
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):audit_bindings(root,native)


def test_native_evidence_does_not_bind_other_plan(native_segment):
    root,native=native_segment
    native['episode_plan_sha256']='other-plan'
    with pytest.raises(ValueError,match='bind the original'):audit_bindings(root,native)


def test_recovery_refuses_running_or_unclosed_campaign(native_segment):
    root,_=native_segment
    write_json(root/'campaign.json',dict(model='cosmos',domain='libero',official_50=True,
        resume_from='/source',smoke=False))
    write_json(root/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False,
        error='driver incomplete: 2 is not a completed campaign with resolved retry errors'))
    before=(root/'driver/journal.jsonl').read_bytes()
    with pytest.raises(ValueError,match='not a closed Cosmos'):recover(root)
    assert (root/'driver/journal.jsonl').read_bytes()==before


def test_cross_campaign_retry_records_reset_attempt_and_same_original_init(native_segment):
    root,native=native_segment
    from exp.r17_integration.ops.formal_resume import read
    plan=read(root/'EPISODE_PLAN.json')
    source=dict(plan=plan,campaign={'run_name':'source'},evidence={'attempts':[
        dict(task_uid='cell0',accepted=True,attempt=1,error='EGL framebuffer',n_steps=0)]})
    recovered=cross_segment_retries(source,plan,audit_bindings(root,native))
    assert recovered[0]['failed_attempt']==recovered[0]['resolved_attempt']==1
    assert recovered[0]['original_init_idx']==42 and recovered[0]['policy_seed']==195
    assert recovered[0]['sealed_cell_unchanged'] and recovered[0]['final_success']
    changed=json.loads(json.dumps(plan));changed['tasks'][0]['orig_init_state_idx']=41
    with pytest.raises(ValueError,match='changed sealed seed/init/config'):
        cross_segment_retries(source,changed,audit_bindings(root,native))
