import json
from unittest.mock import patch

from .common import continued_lanes
from . import workflow
from exp.offline_search.rounds.r17.filter_failed import analyze as ff_analyze
from exp.offline_search.rounds.r17.knob import common as knob


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def resumed_lane(tmp_path):
    source,child=tmp_path/'source',tmp_path/'child'
    tasks=[dict(task_uid=f'u{i}',task_id=0,experiment='metaworld_mt50',orig_init_state_idx=i,
                extra=dict(cache_group='chrg_koff',library_size=250,task_name='reach-v3')) for i in range(5)]
    for root,extra in ((source,{}),(child,dict(cache_resume_selection=dict(source=str(source))))):
        write(root/'campaign.json',dict(model='dp',domain='metaworld',**extra))
        write(root/'assets/CACHE_STUDY.json',dict(groups=[]))
    write(source/'EPISODE_PLAN.json',dict(tasks=tasks))
    write(child/'FULL_EPISODE_PLAN.json',dict(tasks=tasks))
    write(child/'EPISODE_PLAN.json',dict(tasks=tasks[2:]))
    old=[dict(task_uid=f'u{i}',accepted=True,status='done',success=bool(i%2),attempt=1,ts=1) for i in range(2)]
    new=old+[dict(task_uid=f'u{i}',accepted=True,status='done',success=bool(i%2),attempt=1,ts=2) for i in range(2,5)]
    for root,rows in ((source,old),(child,new)):
        (root/'driver').mkdir()
        (root/'driver/journal.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
    write(tmp_path/'CONTINUATIONS.json',dict(jobs=[dict(source=str(source),run=str(child))]))
    return source,child


def test_registered_knob_tip_uses_native_ancestor_merge_without_counting_copies(tmp_path):
    source,child=resumed_lane(tmp_path)
    with patch.object(ff_analyze,'HERE',tmp_path):
        lanes=continued_lanes([source,child])
    assert lanes==[child]
    cells,planned=knob.load_knob(lanes,'A')
    data=cells['DP','mt50',250]['koff']
    assert len(data)==5 and sum(r['success'] for r in data.values())==2
    assert data['mt50','reach-v3',0]['segment']==str(source)
    assert data['mt50','reach-v3',4]['segment']==str(child)
    assert planned['DP','mt50',250]=={'reach-v3':{'koff'}}


def test_descent_workflow_routes_original_catalog_entry_to_continuation(tmp_path):
    source,child=resumed_lane(tmp_path)
    catalog=dict(jobs=[dict(run=str(source),model='dp',benchmark='mt50',mode='reduced',stage='descent')])
    with patch.object(ff_analyze,'HERE',tmp_path),patch.object(workflow,'read',return_value=catalog), \
         patch.object(workflow,'mt_controls',return_value=[]),patch.object(workflow.descent,'main') as main:
        workflow.dev('dp','mt50','reduced')
    args=main.call_args.args[0]
    assert args[:2]==['--lane',str(child)] and str(source) not in args
