import json
from pathlib import Path
from unittest.mock import patch
import pytest

from . import analyze
from .sources import r17_runs


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_collect_merges_ancestors_and_deduplicates_copied_outcomes(tmp_path):
    source,child=tmp_path/'source',tmp_path/'child'
    tasks=[dict(task_uid=f'u{i}',task_id=0,experiment='libero_spatial',orig_init_state_idx=i,
                extra=dict(cache_group='c',library_size=50,task_name='task')) for i in range(5)]
    for r,c in [(source,{}),(child,dict(cache_resume_selection=dict(source=str(source))))]:
        write(r/'campaign.json',dict(domain='libero',model='fastwam',**c))
    write(source/'EPISODE_PLAN.json',dict(tasks=tasks))
    write(child/'FULL_EPISODE_PLAN.json',dict(tasks=tasks))
    write(child/'EPISODE_PLAN.json',dict(tasks=tasks[3:]))
    old=[dict(task_uid=f'u{i}',accepted=True,status='done',success=bool(i%2),attempt=1,ts=1) for i in range(3)]
    old.append(dict(task_uid='u3',accepted=True,status='failed',error='renderer',success=False,ts=2))
    (source/'driver').mkdir();(source/'driver/journal.jsonl').write_text('\n'.join(map(json.dumps,old))+'\n')
    new=old[:3]+[dict(task_uid=f'u{i}',accepted=True,status='failed' if i==3 else 'done',success=i==4,ts=3) for i in (3,4)]
    (child/'driver').mkdir();(child/'driver/journal.jsonl').write_text('\n'.join(map(json.dumps,new))+'\n')
    r17_runs.cache_clear()
    job=dict(run=str(child),model='fastwam',domain='libero')
    result=analyze.collect([job])
    assert len(result['fastwam/libero_spatial/50'])==5
    assert result['fastwam/libero_spatial/50']['task',3] is False
    assert result['fastwam/libero_spatial/50']['task',4] is True
    assert analyze.collect([dict(job,run=str(source))],[child])==result
    r17_runs.cache_clear()


def test_recorded_continuations_follow_only_the_requested_lane(tmp_path):
    source,child,later,other=[str(tmp_path/n) for n in ('source','child','later','other')]
    write(tmp_path/'CONTINUATIONS.json',dict(jobs=[dict(source=child,run=later),dict(source=source,run=child),dict(source=other,run='unrelated')]))
    with patch.object(analyze,'HERE',tmp_path):
        assert analyze.continuations([dict(run=source)])==[child,later]


@pytest.mark.parametrize('verify_closed',(False,True))
def test_unproved_or_live_source_never_reaches_continuation_clone(tmp_path,verify_closed):
    from . import resume
    write(tmp_path/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False))
    with patch.object(resume,'prove_closed',side_effect=ValueError('owned process still alive')), \
         patch('exp.r17_integration.ops.cache_resume.prepare') as prepare:
        with pytest.raises(ValueError):
            resume.build(tmp_path,'ffk1015_cpu_c1',verify_closed=verify_closed)
    prepare.assert_not_called()
    assert read_receipt(tmp_path)==dict(status='FAIL',all_owned_closed=False)


def read_receipt(root):
    return json.loads((root/'RECEIPT.json').read_text())


def fresh_knob(tmp_path):
    campaign=dict(run_name='ffk1015_cpu',domain='metaworld',model='dp',replicas=4,
        cache_study='unchanged_study',workers=120,worker_nodes={'timan1':40,'timan107':40,'timan108':40},
        dp_batching={'max_batch':16},filter_failed_knob=dict(method='frozen_fit_candidate_rows_only',selection={'fit':[1,2]}))
    write(tmp_path/'campaign.json',campaign);write(tmp_path/'OFFLINE_READY.json',{})
    return campaign


def test_fresh_knob_uses_native_execution_map_and_preserves_canonical_treatment(tmp_path):
    from .resume import topology
    before=fresh_knob(tmp_path)
    topology(tmp_path,'timan107=40,timan1=40')
    after=json.loads((tmp_path/'campaign.json').read_text())
    assert after['execution_worker_nodes']=={'timan107':40,'timan1':40} and after['execution_workers']==80
    for key,value in before.items():assert after[key]==value
    record=json.loads((tmp_path/'WORKER_REDISTRIBUTION.json').read_text())
    assert record['source_total_workers']==120 and record['total_workers']==80
    assert record['unchanged_settings']['dp_batching']=={'max_batch':16}
    from exp.r17_integration.ops.cache_resume import segment_resources
    resource=segment_resources(after,tmp_path)[0]
    assert resource['workers']==80 and resource['worker_nodes']==after['execution_worker_nodes']
    assert resource['canonical_workers']==120


@pytest.mark.parametrize('marker',('PREPARED.json','LAUNCH.json','RECEIPT.json'))
def test_topology_does_not_bypass_prepared_or_launched_seals(tmp_path,marker):
    from .resume import topology
    fresh_knob(tmp_path);original=(tmp_path/'campaign.json').read_bytes()
    write(tmp_path/marker,{})
    with pytest.raises(ValueError,match='unstaged'):
        topology(tmp_path,'timan107=40,timan1=40')
    assert (tmp_path/'campaign.json').read_bytes()==original
    assert not (tmp_path/'WORKER_REDISTRIBUTION.json').exists()


def test_fresh_continuation_stage_keeps_inherited_driver_and_control_pins(tmp_path):
    from contextlib import nullcontext
    from . import stage
    source=tmp_path/'source';source.mkdir()
    write(source/'WORKER_DEPLOYMENT.json',{})
    campaign=dict(run_name='new_continuation',filter_failed_source='unfiltered_ancestor',
        cache_resume_selection=dict(source=str(source)),runtime='frozen',runtime_identity='pin',
        driver_runtime='original_driver',orchestration={'selected':'original_pin'},
        model='fastwam',domain='libero',server_node='h100',replicas=2,ports=[],local_assets=str(tmp_path/'assets'))
    campaign['driver_port']=0  # Bind-only admission chooses a free ephemeral port in this CPU fixture.
    write(tmp_path/'campaign.json',campaign)
    write(tmp_path/'assets/model_spec.json',{'checkpoint_identity':'model'})
    write(tmp_path/'assets/CACHE_STUDY.json',{'benchmark_protocol':'protocol'})
    tasks=[{'task_uid':'only_missing'}]
    write(tmp_path/'EPISODE_PLAN.json',dict(tasks=tasks,expected_episodes=1))
    with patch('exp.r17_integration.ops.formal_runtime.attach',side_effect=AssertionError('rebuild runtime')), \
         patch('exp.r17_integration.ops.cache_control.attach',side_effect=AssertionError('rebuild control')), \
         patch('exp.r17_integration.ops.driver_release.prepare',side_effect=AssertionError('rebuild driver')), \
         patch('exp.r17_integration.ops.driver_release.planned_tasks',return_value=tasks), \
         patch('exp.r17_integration.ops.immutable_release.install'), \
         patch('exp.r17_integration.ops.immutable_release.sha',return_value='same_archive'), \
         patch('exp.r17_integration.ops.formal_eval.prepare_workers',return_value={}), \
         patch.object(stage.worker_resilience,'prepare',side_effect=lambda c,d,native:native(c,d)), \
         patch('exp.r17_integration.ops.formal_eval.retry_scope',return_value=nullcontext()), \
         patch('exp.r17_integration.ops.formal_eval.remote',side_effect=AssertionError('remote forbidden in CPU test')), \
         patch.object(stage,'stage_server_inputs') as servers,patch.object(stage,'stage_worker_inputs') as workers:
        stage._stage(tmp_path)
    actual=json.loads((tmp_path/'campaign.json').read_text())
    assert actual['driver_runtime']=='original_driver' and actual['orchestration']==campaign['orchestration']
    servers.assert_called_once();workers.assert_called_once()
