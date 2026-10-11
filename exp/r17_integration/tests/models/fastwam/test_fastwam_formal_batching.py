"""Normal launcher contracts: no GPU, remote calls or real campaign creation."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from exp.r17_integration.models.fastwam.batch_runtime import BatchConfig
from exp.r17_integration.models.fastwam import formal_batching as fb
from exp.r17_integration.ops import formal_eval, formal_server


def campaign(**updates):
    return dict(model='fastwam',domain='libero',server_node='h100',checkpoint='/sealed/checkpoint',
        workers=16,worker_nodes={'timan107':16},replicas=1,worker_egl_device=1,
        worker_max_episodes=2,official_50=True,smoke=False,**updates)


def batched(**updates):
    c=campaign(**updates)
    fb.bind(c,8,5.)
    return c


def metadata(c):
    return dict(r17_batching=c['fastwam_batching'],
        fastwam_batch_treatment_identity=c['fastwam_batch_treatment_identity'],
        gpu_family=c['fastwam_batch_treatment']['gpu_family'])


def write(path, doc):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(doc))


def control(tmp_path,c=None):
    c=c or batched()
    root=tmp_path/'control'
    write(root/'campaign.json',c)
    write(root/'RECEIPT.json',dict(status='PASS',all_owned_closed=True,
        fastwam_batch_treatment_identity=c['fastwam_batch_treatment_identity'],
        batching_audit=dict(status='PASS',treatment_identity=c['fastwam_batch_treatment_identity'])))
    write(root/'SUMMARY.json',dict(status='PASS',expected_episodes=2000,completed_episodes=2000))
    write(root/'SERVER_ATTESTATIONS.json',[dict(port=23171,metadata=metadata(c))])
    for name in ('model_manifest.json','model_spec.json','benchmark.json'):
        write(root/'assets'/name,dict(identity=name))
    return root


def test_normal_cli_defaults_preserve_campaign_and_environment():
    args=formal_eval.parse_args(['--model','fastwam_libero','--prepare-only'])
    assert (args.max_batch,args.batch_wait_ms,args.paired_control)==(1,5.,None)
    c=campaign()
    before=json.dumps(c,sort_keys=True).encode()
    fb.bind(c)
    assert json.dumps(c,sort_keys=True).encode()==before
    assert fb.receipt_fields(c)=={}
    assert not any(k.startswith('BATCHING_') for k in formal_eval.server_env(c,'/runtime'))


@pytest.mark.parametrize('node',['local','h100'])
@pytest.mark.parametrize('batch',[8,16,24,32])
def test_normal_cli_accepts_qualified_capacity_range_on_both_gpu_nodes(node,batch):
    args=formal_eval.parse_args(['--model','fastwam_libero','--server-node',node,
        '--max-batch',str(batch),'--batch-wait-ms','5','--prepare-only'])
    assert args.max_batch==batch and args.server_node==node


@pytest.mark.parametrize('extra',[
    ['--max-batch','33'],['--max-batch','0'],['--batch-wait-ms','nan'],
    ['--batch-wait-ms','-1'],['--model','fastwam_mw','--max-batch','8'],
    ['--model','cosmos_libero','--max-batch','8'],['--paired-control','/tmp/control'],
    ['--batch-qualification-observations','/tmp/replay.npz']])
def test_normal_cli_rejects_invalid_batch_modes(extra):
    with pytest.raises(SystemExit):
        formal_eval.parse_args(['--model','fastwam_libero','--prepare-only',*extra])


def test_dp_owned_seam_accepts_32_without_fastwam_control_semantics():
    args=formal_eval.parse_args(['--model','dp_libero','--max-batch','32','--prepare-only'])
    assert args.max_batch==32


def test_normal_server_dispatches_owned_fastwam_entry(monkeypatch,tmp_path):
    from exp.r17_integration.models.fastwam import formal_batch_server
    called=[]
    monkeypatch.setattr(formal_batch_server,'serve',lambda *args:called.append(args))
    c=batched()
    formal_server.serve(c,23171,tmp_path)
    assert called==[(c,23171,tmp_path)]


def test_dp_dispatch_keeps_graph_argument(monkeypatch,tmp_path):
    from exp.r17_integration.models.dp import formal_batch_server
    called=[]
    monkeypatch.setattr(formal_batch_server,'serve',lambda *args,**kwargs:called.append((args,kwargs)))
    c=dict(dp_batching={'max_batch':8})
    formal_server.serve(c,23161,tmp_path,cuda_graph=True)
    assert called==[((c,23161,tmp_path),{'cuda_graph':True})]


@pytest.mark.parametrize('key,value',[('commit_horizon',10),('denoising_steps',2)])
def test_study_requires_a_completed_control(key,value):
    c=campaign()
    c[key]=value
    with pytest.raises(ValueError,match='paired-control'):
        fb.bind(c,8,5.)


def test_completed_same_treatment_control_accepted_and_pinned(tmp_path):
    root=control(tmp_path)
    c=campaign()
    c.update(commit_horizon=20,suites=['libero_spatial','libero_10'])
    fb.bind(c,8,5.,root)
    seal=fb.check_control(c,root/'assets')
    assert seal['source']==str(root) and seal['treatment_identity']==c['fastwam_batch_treatment_identity']
    assert len(seal['source_receipt_identity'])==64


@pytest.mark.parametrize('field,value',[('server_node','local'),('workers',8),('replicas',2),
    ('worker_max_episodes',8),('worker_egl_device',2),('checkpoint','/different/checkpoint'),
    ('worker_egl_devices',[1,2,3]),('worker_vram_mib',512)])
def test_control_refuses_gpu_and_topology_mixing(field,value,tmp_path):
    root=control(tmp_path)
    c=campaign()
    c[field]=value
    fb.bind(c,8,5.,root)
    with pytest.raises(ValueError,match='mismatch'):
        fb.check_control(c,root/'assets')


def test_control_refuses_b1_and_window_mixing(tmp_path):
    root=control(tmp_path)
    old=json.loads((root/'campaign.json').read_text())
    c=campaign()
    fb.bind(c,8,6.,root)
    with pytest.raises(ValueError,match='mismatch'):
        fb.check_control(c,root/'assets')
    for key in list(old):
        if key.startswith('fastwam_batch'):
            old.pop(key)
    write(root/'campaign.json',old)
    fb.bind(c,8,5.,root)
    with pytest.raises(ValueError,match='mismatch'):
        fb.check_control(c,root/'assets')


@pytest.mark.parametrize('receipt',[dict(status='FAIL',all_owned_closed=True),
    dict(status='PASS',all_owned_closed=False)])
def test_failed_or_running_control_rejected(receipt,tmp_path):
    root=control(tmp_path)
    write(root/'RECEIPT.json',receipt)
    c=campaign()
    fb.bind(c,8,5.,root)
    with pytest.raises(ValueError,match='completed'):
        fb.check_control(c,root/'assets')


def test_sealed_model_mismatch_rejected(tmp_path):
    root=control(tmp_path)
    other=tmp_path/'changed_assets'
    for name in ('model_manifest.json','model_spec.json','benchmark.json'):
        write(other/name,dict(identity='wrong'))
    c=campaign()
    fb.bind(c,8,5.,root)
    with pytest.raises(ValueError,match='sealed identity'):
        fb.check_control(c,other)


def test_resume_rejects_changed_batch_and_default_b1(tmp_path):
    root=control(tmp_path)
    fb.check_resume(batched(),root)
    with pytest.raises(ValueError,match='resume batch'):
        fb.check_resume(campaign(),root)
    c=campaign()
    fb.bind(c,4,5.)
    with pytest.raises(ValueError,match='resume batch'):
        fb.check_resume(c,root)


@pytest.mark.parametrize('field,value',[('r17_batching',{}),('gpu_family','different GPU'),
    ('fastwam_batch_treatment_identity','wrong')])
def test_resident_attestation_refuses_mixing(field,value):
    c=batched()
    meta=metadata(c)
    meta[field]=value
    with pytest.raises(ValueError,match='resident'):
        fb.check_attestation(c,meta)


def test_normal_seam_records_identity_and_forces_qualified_stream_settings():
    c=batched()
    fields=fb.receipt_fields(c)
    assert fields['fastwam_batch_treatment_identity']==fb.digest(fields['fastwam_batch_treatment'])
    env=formal_eval.server_env(c,'/runtime')
    assert env['OPENPI_DISABLE_STAGE_STREAMS']=='0'
    assert all(env[f'BATCHING_STAGE{i}_WORKERS']=='1' for i in (1,2,3))


def test_closed_receipt_keeps_code_pin_but_new_resident_refuses_changed_code(monkeypatch):
    c=batched()
    pinned=copy.deepcopy(c['fastwam_batch_treatment'])
    monkeypatch.setattr(fb,'implementation_identity',lambda:'changed')
    assert fb.validate(c)==pinned
    with pytest.raises(ValueError,match='implementation'):
        fb.validate_runtime(c)
    new=batched()
    assert new['fastwam_batch_treatment']!=pinned


def decision(c):
    member=dict(request_id='unique',request=dict(decision_seq=3,task_name='task'))
    stages={stage:dict(stage=stage,size=1,row=0,batch_id=stage,members=[member],
        config_identity=c['fastwam_batching']['identity']) for stage in ('stage1','stage2','stage3')}
    return dict(source='policy',nfe=10,decision_seq=3,task_name='task',
        batching=dict(config=c['fastwam_batching'],stages=stages))


def test_every_decision_composition_is_audited_including_zero_call_tail(tmp_path):
    c=batched()
    row=decision(c)
    path=tmp_path/'replica_0'/'serving.jsonl'
    path.parent.mkdir()
    tail=dict(row,source='policy_tail',nfe=0,batching=dict(config=c['fastwam_batching'],stages={}))
    path.write_text(json.dumps(row)+'\n'+json.dumps(tail)+'\n')
    audit=fb.audit_decisions(c,tmp_path)
    assert audit['decisions']==2 and audit['max_actual_batch_size']['stage3']==1


@pytest.mark.parametrize('defect',['missing_stage','wrong_config','wrong_row','wrong_task','bad_tail'])
def test_malformed_or_mixed_batch_decisions_fail_closed(defect,tmp_path):
    c=batched()
    row=decision(c)
    if defect=='missing_stage':row['batching']['stages'].pop('stage3')
    elif defect=='wrong_config':row['batching']['config']=BatchConfig(4,5.).metadata()
    elif defect=='wrong_row':row['batching']['stages']['stage1']['row']=2
    elif defect=='wrong_task':row['task_name']='other'
    else:row['source']='policy_tail'
    path=tmp_path/'replica_0'/'serving.jsonl'
    path.parent.mkdir()
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):fb.audit_decisions(c,tmp_path)
