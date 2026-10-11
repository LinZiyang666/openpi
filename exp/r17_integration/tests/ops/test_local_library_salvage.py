"""Missing worker results in shared local bindings never become library rows."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest

from exp.r17_integration.ops import formal_driver,library_salvage as s
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.library_repair import SEAL_FILES,inspect_repair,merge_repair,capture_sources,outcomes
from exp.r17_integration.debug.attempts import capture_counts,json_rows
from exp.r17_integration.debug.validate import file_sha
from exp.r17_integration.tests.search.library_fixture import synthetic_capture
from exp.r17_integration.tests.ops.test_library_retry_repair import write_rows
from exp.r17_integration.search.teacher_library import build_teacher_parent
from openpi.conductor.task import EpisodeTask


@pytest.fixture
def missing_local(tmp_path,monkeypatch):
    original=tmp_path/'original';source=original/'servers/replica_0/capture/binding_cpu'
    b,episodes,audit,names=synthetic_capture(source,episodes=5,suite='libero_10')
    tasks=[EpisodeTask(e['task_uid'],'cpu','eval','libero_10',e['task_id'],e['orig_init_idx'],e['orig_init_idx'],
        'unused',23320,'generation1',extra={k:v for k,v in e.items() if k!='attempt'}) for e in episodes]
    assets=original/'assets';assets.mkdir()
    for name in SEAL_FILES:write_json(assets/name,dict(sealed=name))
    write_json(assets/'model_spec.json',b['model']);write_json(original/'ALIGNMENT.json',audit)
    archive=original/'bpool_inputs.tar';archive.write_bytes(b'B byte identity')
    c=dict(library=True,model='dp',suite='libero_10',server_node='local',smoke=False,run_name='CPU_LOCAL',
        local_assets=str(assets),assets=str(assets),checkpoint='/CPU/teacher',alignment=str(original/'ALIGNMENT.json'),
        bpool_archive=str(archive),bpool_transfer=dict(archive_sha256=file_sha(archive)),synthetic=True)
    write_json(original/'campaign.json',c);write_json(original/'EPISODE_PLAN.json',dict(model=b['model'],tasks=[asdict(t) for t in tasks]))
    write_json(original/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True,error='driver missing worker results',
        owned_local=[dict(pid=900001,pgid=900001,start_ticks='1')],owned_remote=[dict(node='timan107',receipt=dict(child=dict(pid=900002,pgid=900002,start_ticks='1')))]))
    results=[dict(task_uid=e['task_uid'],attempt=1,accepted=True,success=e['success'],error=None,n_steps=10,
                  status='done' if e['success'] else 'failed') for e in episodes[:2]]
    for name,rows in [('journal',results),('attempts',results),
                      ('per_step',[dict(e,accepted=True,decision_seq=i) for e in episodes[:2] for i in (0,1)]),
                      ('dispatch',[dict(task_uid=e['task_uid'],attempt=1) for e in episodes])]:
        write_rows(original/'driver'/(name+'.jsonl'),rows)
    # All5 episodes share one immutable capture block. Missing results can
    # have complete or incomplete acks; neither constitutes an outcome.
    path=source/'execution.jsonl';rows=json_rows(path)
    rows=[r for r in rows if not(r['task_uid']==episodes[3]['task_uid'] and r['terminal'])]
    write_rows(path,rows)
    m=json.loads((source/'r17_capture.json').read_text());m['files']={n:file_sha(source/n) for n in m['files']};write_json(source/'r17_capture.json',m)
    def plan(c):
        chosen=[t for t in tasks if t.task_uid in c['repair_attempts']] if c.get('repair_attempts') else tasks
        return NS(identity=b['benchmark_protocol'],ordered_tasks=names),b['model'],chosen,[]
    monkeypatch.setattr(formal_driver,'plan',plan)
    # /proc closure proof is independently covered; fake nodes here never exec.
    monkeypatch.setattr(s,'prove_closed',lambda *a:dict(status='PASS',all_owned_closed=True,processes={'CPU':'dead'}))
    return original,c,source,b,episodes,tasks,audit


def salvage(fixture):
    original,c,*_=fixture
    return s.salvage(original,model='dp',suite='libero_10',remote=lambda *a:pytest.fail('unexpected remote'),
                     pull=lambda *a:pytest.fail('local salvage attempted a pull'))


def test_local_salvage_separates_missing_results_keeps_failure_and_original_bytes(missing_local):
    original,c,source,b,episodes,tasks,audit=missing_local
    before={str(p):file_sha(p) for p in original.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='missing outcome|missing worker result'):
        inspect_repair(original,model='dp',suite='libero_10')
    result=salvage(missing_local)
    assert result['kept_episodes']==2 and result['missing_episodes']==3
    derived=json.loads((original/'salvage/campaign.json').read_text())
    assert [r['success'] for r in json_rows(original/'salvage/driver/journal.jsonl')]==[False,True]
    selected=capture_sources(derived,original/'salvage');counts,_=capture_counts(selected,binding=b)
    assert sum(counts.values())==4 and {k[0] for k in counts}=={e['task_uid'] for e in episodes[:2]}
    q=json.loads((original/'salvage/QUARANTINE.json').read_text())
    assert len(q['sources'])==3 and sum(d['captured_decisions'] for d in q['dropped_episodes'])==6
    for view in selected:
        block=next((view/'blocks').glob('*.npz'))
        assert block.stat().st_ino==(source/'blocks'/block.name).stat().st_ino
    _,_,pin=inspect_repair(original/'salvage',model='dp',suite='libero_10')
    assert pin['kept_outcomes']==2 and set(pin['repair_attempts'])=={e['task_uid'] for e in episodes[2:]}
    assert set(pin['repair_attempts'].values())=={2}
    assert all(file_sha(p)==sha for p,sha in before.items())
    assert all(r['disposition']=='discarded_missing_worker_result' for rr in pin['prior_attempts'].values() for r in rr)


def test_local_salvage_repair_merge_and_parent_exclude_every_old_missing_row(missing_local,tmp_path):
    original,c,source,b,episodes,tasks,audit=missing_local;salvage(missing_local)
    old,_,pin=inspect_repair(original/'salvage',model='dp',suite='libero_10')
    repair=tmp_path/'repair';repair.mkdir();shutil.copytree(original/'assets',repair/'assets')
    shutil.copyfile(original/'bpool_inputs.tar',repair/'bpool_inputs.tar');write_json(repair/'ALIGNMENT.json',audit)
    fresh=dict(old,run_name='CPU_REPAIR',semantic_campaign=c['run_name'],local_assets=str(repair/'assets'),
        assets=str(repair/'assets'),alignment=str(repair/'ALIGNMENT.json'),bpool_archive=str(repair/'bpool_inputs.tar'),
        repair_pin=pin,repair_attempts=pin['repair_attempts'],repair_prior_attempts=pin['prior_attempts'])
    for name in ('repair_merge','node_failure_salvage','node_failure_prior_attempts'):fresh.pop(name,None)
    _,new,_,_=synthetic_capture(repair/'servers/replica_0/capture/binding_cpu',episodes=3,
                                episode_offset=2,attempt=2,suite='libero_10')
    rows=[dict(task_uid=e['task_uid'],attempt=2,accepted=True,status='done' if e['success'] else 'failed',
               success=e['success'],error=None,n_steps=10) for e in new]
    for name,rr in [('journal',rows),('attempts',rows),
                   ('per_step',[dict(e,accepted=True,decision_seq=i) for e in new for i in (0,1)])]:
        write_rows(repair/'driver'/(name+'.jsonl'),rr)
    write_json(repair/'campaign.json',fresh);write_json(repair/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    merged_c,merged=merge_repair(fresh,repair)
    sources=capture_sources(merged_c,merged);counts,_=capture_counts(sources,binding=b)
    accepted,decisions,histories,missing=outcomes(merged,tasks,counts)
    cohort=[dict(e,attempt=accepted[e['task_uid']]['attempt'],prior_attempts=histories[e['task_uid']]) for e in episodes]
    parent=build_teacher_parent(sources,tmp_path/'parent',b,cohort,audit,expected_episodes=5)
    assert len(parent.rows)==10 and not missing
    assert len(parent.manifest['metadata']['quarantined_missing_worker_result_attempts'])==3
    assert parent.manifest['metadata']['pre_decision_infrastructure_attempts']==[]
    assert [e['attempt'] for e in parent.manifest['metadata']['episodes']]==[1,1,2,2,2]


def test_intent_only_missing_result_is_quarantined_without_a_zero_progress_claim(missing_local):
    original,c,source,b,episodes,tasks,audit=missing_local;uid=episodes[-1]['task_uid']
    for path in source.glob('decisions_*'):
        write_rows(path,[r for r in json_rows(path) if r['task_uid']!=uid])
    path=source/'execution.jsonl';write_rows(path,[r for r in json_rows(path) if r['task_uid']!=uid])
    # Retain both original observation intents: inference might have begun.
    assert sum(r['task_uid']==uid for r in json_rows(source/'decision_intents.jsonl'))==2
    m=json.loads((source/'r17_capture.json').read_text());m['writer_stats'].update(accepted=8,written=8)
    m['files']={n:file_sha(source/n) for n in m['files']};write_json(source/'r17_capture.json',m)
    result=salvage(missing_local);assert result['kept_episodes']==2
    proof=json.loads((original/'salvage/SALVAGE_PROVENANCE.json').read_text())
    discarded=next(r for r in proof['discarded_attempts'] if r['task_uid']==uid)
    assert discarded['capture_decisions']==2 and discarded['disposition']=='discarded_missing_worker_result'
    _,_,pin=inspect_repair(original/'salvage',model='dp',suite='libero_10')
    assert pin['repair_attempts'][uid]==2


@pytest.mark.parametrize('mutation',['error_after_progress','returned_incomplete','quarantine_changed','checkpoint','source_changed'])
def test_local_salvage_and_repair_keep_strict_rejection(missing_local,mutation):
    original,c,source,b,episodes,tasks,audit=missing_local
    if mutation=='error_after_progress':
        path=original/'driver/attempts.jsonl';rows=json_rows(path)
        rows.append(dict(task_uid=episodes[2]['task_uid'],attempt=1,accepted=True,error='teacher error',n_steps=1))
        write_rows(path,rows)
        with pytest.raises(ValueError,match='error after progress'):salvage(missing_local)
        return
    if mutation=='returned_incomplete':
        path=source/'execution.jsonl';rows=[r for r in json_rows(path) if r['task_uid']!=episodes[0]['task_uid']]
        write_rows(path,rows);m=json.loads((source/'r17_capture.json').read_text());m['files']['execution.jsonl']=file_sha(path);write_json(source/'r17_capture.json',m)
        with pytest.raises(ValueError,match='returned teacher outcome'):salvage(missing_local)
        return
    salvage(missing_local)
    if mutation=='quarantine_changed':(original/'salvage/QUARANTINE.json').write_bytes(b'changed')
    elif mutation=='source_changed':next((source/'blocks').glob('*.npz')).write_bytes(b'changed')
    else:
        with pytest.raises(ValueError,match='checkpoint differs'):
            inspect_repair(original/'salvage',model='dp',suite='libero_10',checkpoint='/wrong')
        return
    with pytest.raises(ValueError,match='evidence changed|checksum mismatch'):
        inspect_repair(original/'salvage',model='dp',suite='libero_10')
