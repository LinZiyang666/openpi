import json
from unittest.mock import patch

import pytest

from .test_resume import resumed_lane, write
from .common import continued_lanes
from exp.offline_search.rounds.r17.cache import analyze_cache as cache
from exp.offline_search.rounds.r17.knob import common as knob
from exp.offline_search.rounds.r17.filter_failed import analyze, resume, stage


def test_infra_error_is_missing_until_valid_uid_replacement_and_keeps_real_failures(tmp_path):
    source,child=resumed_lane(tmp_path)
    bad=dict(task_uid='u1',accepted=True,status='failed',success=False,attempt=1,ts=1,
             error="FatalEpisodeError: retry budget exhausted; RuntimeError('LIBERO inference failed')")
    legitimate=dict(task_uid='u0',accepted=True,status='done',success=False,attempt=1,ts=1)
    journal=source/'driver/journal.jsonl';journal.write_text('\n'.join(map(json.dumps,[legitimate,bad]))+'\n')
    write(source/'.startup_inputs/STARTUP_INPUTS.json',dict(files={}))
    selection=resume.infra_error_selection(source)
    assert selection['missing_task_uids']==['u1']
    assert set(selection['retained_task_uids'])=={'u0','u2','u3','u4'}
    assert selection['source_closure_required_at_prepare'] is True
    snapshot=cache.receipt_snapshot(source)
    data=cache.load_lane(source,snapshot,cache.Sources(),[250],suites=['mt50'])
    assert len(next(iter(data.values()))['data'])==1
    assert snapshot['missing_infrastructure_error_task_uids']==['u1']
    repaired=dict(bad,success=True,error=None,attempt=2,ts=3)
    write(child/'EPISODE_PLAN.json',dict(tasks=[json.loads((source/'EPISODE_PLAN.json').read_text())['tasks'][1]]))
    write(child/'campaign.json',dict(run_name='isolated_cpu_repair',model='dp',domain='metaworld',filter_failed_source=str(source),
                                   cache_resume_selection=selection))
    (child/'driver/journal.jsonl').write_text(json.dumps(repaired)+'\n')
    # The parent continues appending after the error-only draft was built.
    with journal.open('a') as f:
        for i in range(2,5):f.write(json.dumps(dict(legitimate,task_uid=f'u{i}',ts=4))+'\n')
    with patch.object(analyze,'HERE',tmp_path):lanes=continued_lanes([source])
    assert lanes==[child]
    cells,_=knob.load_knob(lanes,'A');out=cells['DP','mt50',250]['koff']
    assert len(out)==5 and sum(r['success'] for r in out.values())==1
    assert out['mt50','reach-v3',0]['success'] is False
    assert out['mt50','reach-v3',1]['segment']==str(child)
    snapshot=cache.receipt_snapshot(child)
    cache.load_lane(child,snapshot,cache.Sources(),[250],suites=['mt50'])
    assert snapshot['infrastructure_error_task_uids']==['u1']
    assert snapshot['missing_infrastructure_error_task_uids']==[]
    with patch.object(resume,'prove_closed',side_effect=ValueError('source still alive')):
        with pytest.raises(ValueError,match='source still alive'):stage._stage(child)
    assert not (child/'PREPARED.json').exists()
    from .formal_a_analysis import choose
    pure={('suite',str(t),i):dict(success=True) for t in range(10) for i in range(50)}
    arms={k:dict(pure) for k in ('k50','k40','koff')}
    # Prevent the cell-wide off bypass; the ordinary descent must wait at k40.
    arms['koff']['suite','1',0]=dict(success=False)
    del arms['k40']['suite','0',7]
    pending=choose(arms,pure,('k50','k40','koff'))
    assert pending['tasks']['0']['incomplete'] and pending['tasks']['0']['selected'] is None
    assert pending['tasks']['0']['levels']['k40']['missing_inits']==[7]


def test_remaining_draft_refreshes_after_closure_without_repeating_new_parent_acceptances(tmp_path):
    source,child=resumed_lane(tmp_path)
    write(source/'campaign.json',dict(run_name='source',model='dp',domain='metaworld'))
    write(source/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
    write(source/'.startup_inputs/STARTUP_INPUTS.json',dict(files={}))
    bad=dict(task_uid='u1',accepted=True,status='failed',success=False,attempt=1,ts=1,
             error='FatalEpisodeError: retry budget exhausted; infrastructure')
    good=dict(task_uid='u0',accepted=True,status='done',success=False,attempt=1,ts=1)
    journal=source/'driver/journal.jsonl';journal.write_text('\n'.join(map(json.dumps,[good,bad]))+'\n')
    selection=resume.infra_error_selection(source,include_missing=True)
    assert selection['missing_task_uids']==['u1','u2','u3','u4']
    write(child/'campaign.json',dict(run_name='child',model='dp',cache_resume_selection=selection))
    write(child/'WORKER_DEPLOYMENT.json',{})
    write(child/'LOCAL_PLAN_SEAL.json',{'original_draft':True})
    write(child/'CACHE_RESUME.json',selection)
    write(child/'FFK_DEFINITION.json',dict(name='child',episodes=4))
    with journal.open('a') as f:f.write(json.dumps(dict(good,task_uid='u2',success=True,ts=2))+'\n')
    def seal(c,r,d):write(r/'PREPARED.json',dict(files={}))
    with patch('exp.r17_integration.ops.formal_prestage.seal_prepared',side_effect=seal):
        c=resume.refresh_remaining(child,dict(status='PASS',all_owned_closed=True))
    assert c['cache_resume_selection']['missing_task_uids']==['u1','u3','u4']
    assert c['cache_resume_selection']['refresh_after_source_closure'] is False
    assert [t['task_uid'] for t in json.loads((child/'EPISODE_PLAN.json').read_text())['tasks']]==['u1','u3','u4']
    assert json.loads((child/'FFK_DEFINITION.json').read_text())['episodes']==3
    assert list(child.glob('LOCAL_PLAN_SEAL.json.before_closure_*'))
    assert (child/'LOCAL_PLAN_SEAL.json').exists() and not (child/'PREPARED.json').exists()
