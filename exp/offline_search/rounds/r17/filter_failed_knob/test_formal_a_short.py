"""The long-name MPS failure must disappear without changing episode identity."""
from dataclasses import dataclass
import json
from types import SimpleNamespace
import pytest
from . import formal_a_short as s, formal_a_names as n, build as b
from .common import dump,read,sha
from exp.r17_integration.models.fastwam.ops.mps import paths
from exp.offline_search.rounds.r17.filter_failed import reset


def test_short_names_keep_plans_and_reset_only_dead_mps_startups(tmp_path,monkeypatch):
    uuid='GPU-7ea0f760-d3aa-7023-c19b-95f49123311b'
    with pytest.raises(ValueError,match='MPS pipe path too long'):
        paths('/data/mw_finetune/eval/'+s.FAILED,uuid)
    name=n.unit('cosmos','reduced','original','levels')
    assert name=='fa_co_r_u_l' and len(str(paths('/data/mw_finetune/eval/'+name,uuid)[1]).encode())<=85
    source=tmp_path/'long_source';(source/'assets').mkdir(parents=True)
    yaml=source/'assets/cell.yaml';yaml.write_text('fitted_path: /frozen/fit\nperiodic_gap: 7\n')
    @dataclass
    class Task:task_uid:str
    task=Task('original-uid')
    dump(source/'EPISODE_PLAN.json',dict(model={'identity':'same'},tasks=[dict(task_uid=task.task_uid)]))
    dump(source/'campaign.json',dict(run_name='long_source',assets=str(source/'assets'),
        local_assets=str(source/'assets'),server_run=str(source/'servers'),num_steps=2))
    for file in ('runtime.tar','workers.tar','RUNTIME_RELEASE.json','STUDY.json'):(source/file).write_bytes(b'unchanged')
    monkeypatch.setattr(s.f,'EVAL',tmp_path)
    monkeypatch.setattr(b,'cpu_plan',lambda c:(None,{'identity':'same'},[task],None))
    result=s.clone_r17(dict(run=str(source),name='long_source',episodes=1),name)
    root=tmp_path/name
    assert sha(root/'EPISODE_PLAN.json')==sha(source/'EPISODE_PLAN.json')
    assert sha(root/'assets/cell.yaml')==sha(yaml)
    campaign=read(root/'campaign.json')
    assert campaign['run_name']==name and campaign['assets']==str(source/'assets') and campaign['num_steps']==2
    failure=tmp_path/'failed';failure.mkdir()
    dump(failure/'RECEIPT.json',dict(status='PARTIAL',error='ValueError: MPS pipe path too long;',
        owned_remote=[dict(label='mps_guard',node='h100',receipt=dict(child=dict(pid=123,start_ticks='5')))]))
    dump(failure/'LAUNCH.json',dict(pid=999999999))
    calls=[]
    def probe(argv,**kwargs):
        calls.append(argv);return SimpleNamespace(stdout=json.dumps(dict(alive=[],ready=False)))
    monkeypatch.setattr(__import__('subprocess'),'run',probe)
    assert reset.closed_mps_path_failure(failure)['reset_read_only_closure'][0]['alive']==[]
    assert calls[0][:6]==['tether','exec','--safe','--timeout','30s','h100']
    assert calls[0][6:10]==['--','/usr/bin/python3','-B','-S']
    dump(failure/'RECEIPT.json',dict(status='PARTIAL',error='ValueError: MPS pipe path too long;',
        owned_remote=[dict(label='driver',node='h100',receipt={})]))
    with pytest.raises(ValueError,match='not a pre-replica'):reset.closed_mps_path_failure(failure)
