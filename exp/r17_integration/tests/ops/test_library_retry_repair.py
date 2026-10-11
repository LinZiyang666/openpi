from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest

from exp.r17_integration.debug.attempts import decision_intent, json_rows
from exp.r17_integration.debug.validate import digest, file_sha
from exp.r17_integration.ops import formal_driver
from exp.r17_integration.ops.formal_eval import driver_completion
from exp.r17_integration.ops.library_campaign import finish, parse_args
from exp.r17_integration.ops.library_repair import (
    SEAL_FILES, inspect_repair, merge_repair, outcomes, verify_repair_seal,
)
from exp.r17_integration.ops.library_retry import configure_library_driver
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.tests.ops.test_library_campaign import assets
from exp.r17_integration.tests.search.library_fixture import synthetic_capture
from openpi.conductor.task import EpisodeTask

EGL = "FatalError('Offscreen framebuffer is not complete, error 0x8cdd')"


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows))


@pytest.mark.parametrize('case,expected', [('zero',2), ('exhausted',3), ('step',1), ('capture',1),
    ('intent',1), ('reported',1), ('missing_evidence',1), ('outcome',1), ('timeout',1)])
def test_actual_campaign_driver_only_retries_proven_predecision_errors(tmp_path, monkeypatch, case, expected):
    """Real P2/stock queue and journal, mock control transport only; no simulator."""
    from exp.r17_integration.qualification.driver import ProbeRecordingDriver
    from exp.r17_integration.common import websocket_policy_server
    campaign,_ = assets(tmp_path, smoke=True)
    campaign.update(server_node='local',server_run=str(tmp_path/'servers'),retries=2,
                    driver_port=0,episode_seconds=1,seconds=60)
    if case != 'missing_evidence':
        for index in range(2): (tmp_path/'servers'/('replica_'+str(index))).mkdir(parents=True)
    _,_,tasks,_ = formal_driver.plan(campaign)
    uid = tasks[0].task_uid
    counts = {}
    def run(driver, **kwargs):
        for stage in driver.scheduler.pending_setups(): driver.scheduler.mark_setup_done(stage.stage_id)
        while True:
            driver.drive_stages_once()
            if driver.scheduler.all_done(): break
            _,wire = driver.handle_pull('unused:23320')
            task = wire['task']; identity = task['task_uid']
            counts[identity] = counts.get(identity,0)+1
            n = counts[identity]
            error = identity == uid and case != 'outcome' and (n == 1 or case == 'exhausted')
            result = dict(task_uid=identity,attempt=task['attempt'],success=False,n_steps=0,
                          error=EGL if error else None,per_step_rows=[])
            if identity == uid:
                path = tmp_path/'servers/replica_0/capture/binding_test'
                if case == 'step': result['n_steps']=1
                if case == 'capture': write_rows(path/'decisions_fake.jsonl',[dict(task_uid=uid,attempt=1)])
                if case == 'intent': decision_intent(path,dict(task_uid=uid,attempt=1))
                if case == 'reported': result['per_step_rows']=[dict(decision_seq=0)]
                if case == 'timeout':
                    driver.scheduler.requeue_timed_out(timeout_s=0,now=float('inf'))
                    continue
            driver.handle_result(result)
        driver.worker_shutdown=dict(status='PASS',workers=['CPU'])
    monkeypatch.setattr(ProbeRecordingDriver,'run',run)
    monkeypatch.setattr(websocket_policy_server,'WebsocketClientPolicy',lambda *a:NS(
        load_cache_config=lambda *a,**k:{},select_bundle=lambda *a:None,close=lambda:None))
    monkeypatch.setattr(formal_driver.signal,'signal',lambda *a:None)
    with pytest.raises(SystemExit) as exited: formal_driver.run(campaign,tmp_path/'driver')
    assert exited.value.code==0 and counts[uid]==expected
    history = [r for r in json_rows(tmp_path/'driver/attempts.jsonl') if r['task_uid']==uid]
    assert [r['attempt'] for r in history]==list(range(1,expected+1))
    terminals = [r for r in json_rows(tmp_path/'driver/journal.jsonl') if r['task_uid']==uid and r['accepted']]
    assert len(terminals)==1 and terminals[0]['attempt']==expected
    assert (terminals[0].get('error') is None)==(case in ('zero','outcome'))
    if case=='zero': assert history[0]['capture_decisions']==0 and history[0]['pre_decision_retry_allowed']


def test_repair_plan_keeps_original_uids_rng_b_coordinates_and_only_missing_tasks(tmp_path):
    campaign,_ = assets(tmp_path)
    _,_,all_tasks,_ = formal_driver.plan(campaign)
    chosen = all_tasks[341:345]
    changed = dict(campaign,run_name='NEW_REPAIR',semantic_campaign=campaign['run_name'],
                   endpoint_host='new_transport',repair_attempts={t.task_uid:2 for t in chosen})
    _,_,tasks,_ = formal_driver.plan(changed)
    assert len(tasks)==4
    for old,new in zip(chosen,tasks,strict=True):
        assert old.task_uid==new.task_uid and old.yaml_id==new.yaml_id and old.extra==new.extra
        assert new.orig_init_state_idx==old.orig_init_state_idx and new.server_host=='new_transport'
    assert [t.extra['library_episode_id'] for t in tasks]==[341,342,343,344]


def test_repair_attempt_floor_and_nonlibrary_timeout_policy_unchanged(tmp_path):
    from exp.r17_integration.tests.conductor.test_dispatch import setup
    _,tasks,_,_,queue = setup(tmp_path,names=['reach-v3'],indices=(30,))
    driver = NS(_scheduler=queue)
    original = queue.requeue_timed_out
    configure_library_driver(driver,dict(library=False))
    assert queue.requeue_timed_out==original
    configure_library_driver(driver,dict(library=True,repair_attempts={tasks[0].task_uid:2}))
    assert queue.next_task('local:1').attempt==2


def repair_fixture(tmp_path, monkeypatch):
    original = tmp_path/'original'; original.mkdir()
    repair = tmp_path/'repair'; repair.mkdir()
    b,good,audit,names = synthetic_capture(original/'servers/replica_0/capture/binding_cpu',episodes=496)
    _,rerun,_,_ = synthetic_capture(repair/'servers/replica_0/capture/binding_cpu',episodes=4,episode_offset=496,attempt=2)
    all_episodes = good+[dict(e,attempt=1) for e in rerun]
    tasks = [EpisodeTask(e['task_uid'],'cpu','eval','libero_spatial',e['task_id'],e['orig_init_idx'],
        e['orig_init_idx'],'unused',23320,'generation1',extra={k:v for k,v in e.items() if k != 'attempt'}) for e in all_episodes]
    root = original/'assets'; root.mkdir()
    for name in SEAL_FILES: (root/name).write_text(json.dumps(dict(sealed=name)))
    write_json(root/'model_spec.json',b['model'])
    write_json(original/'ALIGNMENT.json',audit)
    archive = original/'bpool_inputs.tar'; archive.write_bytes(b'CPU B archive identity fixture')
    campaign = dict(library=True,model='dp',suite='libero_spatial',domain='libero',server_node='local',
        smoke=False,run_name='CPU_ORIGINAL',local_assets=str(root),assets=str(root),checkpoint='/CPU/teacher',
        bpool_archive=str(archive),bpool_transfer=dict(archive_sha256=file_sha(archive),manifest={'CPU':'B-only'}),
        alignment=str(original/'ALIGNMENT.json'),library_parent=str(tmp_path/'parent'),synthetic=True)
    write_json(original/'campaign.json',campaign)
    write_json(original/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True,error='driver incomplete'))
    write_json(original/'EPISODE_PLAN.json',dict(model=b['model'],tasks=[asdict(t) for t in tasks]))
    terminals,attempts,steps = [],[],[]
    for i,e in enumerate(all_episodes):
        row = dict(task_uid=e['task_uid'],attempt=1,accepted=True,success=e['success'] if i<496 else False,
                   error=None if i<496 else EGL,n_steps=10 if i<496 else 0)
        attempts.append(row)
        terminals.append(dict(row,status='done' if row['success'] else 'failed'))
        if i<496:
            steps.extend(dict(task_uid=e['task_uid'],attempt=1,accepted=True,decision_seq=s,
                capture_enabled=True,source='policy',actual_nfe=5,stage_calls=[1,1,5]) for s in (0,1))
    for name,rows in [('journal',terminals),('attempts',attempts),('per_step',steps)]:
        write_rows(original/'driver'/(name+'.jsonl'),rows)
    def plan(c):
        selected = [t for t in tasks if t.task_uid in c['repair_attempts']] if c.get('repair_attempts') else tasks
        return NS(identity=b['benchmark_protocol'],ordered_tasks=names),b['model'],selected,[]
    monkeypatch.setattr(formal_driver,'plan',plan)
    return original,repair,campaign,b,rerun,tasks


def prepared_repair(original, repair, campaign):
    _,audit,pin = inspect_repair(original,model='dp',suite='libero_spatial')
    shutil.copytree(original/'assets',repair/'assets')
    shutil.copy2(original/'bpool_inputs.tar',repair/'bpool_inputs.tar')
    write_json(repair/'ALIGNMENT.json',audit)
    c = dict(campaign,run_name='CPU_REPAIR',semantic_campaign=campaign['run_name'],
        local_assets=str(repair/'assets'),assets=str(repair/'assets'),
        alignment=str(repair/'ALIGNMENT.json'),bpool_archive=str(repair/'bpool_inputs.tar'),
        repair_attempts=pin['repair_attempts'],repair_prior_attempts=pin['prior_attempts'],repair_pin=pin)
    return c,pin


def test_cpu_repair_only_four_missing_merge500_build_subsets_and_fit(tmp_path, monkeypatch):
    original,repair,campaign,b,rerun,tasks = repair_fixture(tmp_path,monkeypatch)
    before = {str(p):file_sha(p) for p in original.rglob('*') if p.is_file()}
    c,pin = prepared_repair(original,repair,campaign)
    assert pin['kept_outcomes']==496 and len(pin['repair_attempts'])==4
    assert set(pin['repair_attempts'].values())=={2}
    verify_repair_seal(c)
    rows = [dict(task_uid=e['task_uid'],attempt=2,accepted=True,status='done' if e['success'] else 'failed',
                 success=e['success'],error=None,n_steps=10) for e in rerun]
    for name in ('journal','attempts'): write_rows(repair/'driver'/(name+'.jsonl'),rows)
    write_rows(repair/'driver/per_step.jsonl',[dict(task_uid=e['task_uid'],attempt=2,accepted=True,decision_seq=s,
        capture_enabled=True,source='policy',actual_nfe=5,stage_calls=[1,1,5]) for e in rerun for s in (0,1)])
    write_json(repair/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    write_json(repair/'campaign.json',c)
    write_json(repair/'EPISODE_PLAN.json',dict(tasks=[asdict(t) for t in tasks if t.task_uid in c['repair_attempts']]))
    write_json(repair/'driver/DRIVER_RECEIPT.json',dict(status='PASS'))
    assert driver_completion(repair,0)['accepted_episodes']==4
    merged_c,merged = merge_repair(c,repair)
    result = finish(merged_c,merged)
    assert result['status']=='PASS' and result['episodes']==500 and result['rows']==1000
    assert result['failed_episodes']==250 and set(result['fits'])=={'50','100','200','300','400','500'}
    parent = json.loads((Path(result['parent'])/'READY.json').read_text())
    assert len(parent['metadata']['pre_decision_infrastructure_attempts'])==4
    assert sum(e['attempt']==2 for e in parent['metadata']['episodes'])==4
    assert {str(p):file_sha(p) for p in original.rglob('*') if p.is_file()}==before
    assert finish(merged_c,merged)==result


@pytest.mark.parametrize('mutation', ['checkpoint','seal','alignment','B','original_journal','capture','post_step','post_decision','ordinary_outcome','unreported','missing_prior','budget'])
def test_repair_fail_closed_on_changed_inputs_or_unsafe_original_attempt(tmp_path,monkeypatch,mutation):
    original,repair,campaign,b,rerun,tasks = repair_fixture(tmp_path,monkeypatch)
    if mutation in ('checkpoint','seal','alignment','B','original_journal','capture'):
        c,pin = prepared_repair(original,repair,campaign)
        if mutation=='checkpoint': c['checkpoint']='/OTHER/teacher'
        elif mutation=='seal': (repair/'assets/full.yaml').write_text('different bundle')
        elif mutation=='alignment':
            a=json.loads((repair/'ALIGNMENT.json').read_text()); a['rows'][0]['episode_id']=49
            a['digest']=digest({k:v for k,v in a.items() if k!='digest'}); write_json(repair/'ALIGNMENT.json',a)
        elif mutation=='B': (repair/'bpool_inputs.tar').write_bytes(b'other B states')
        elif mutation=='original_journal': (original/'driver/journal.jsonl').write_text('{}\n')
        else: next((original/'servers').rglob('r17_capture.json')).write_text('{}')
        with pytest.raises(ValueError): verify_repair_seal(c)
        return
    attempts=json_rows(original/'driver/attempts.jsonl'); uid=attempts[-1]['task_uid']
    if mutation=='post_step': attempts[-1]['n_steps']=1
    elif mutation=='post_decision':
        rows=json_rows(original/'driver/per_step.jsonl'); rows.append(dict(task_uid=uid,attempt=1,accepted=True,decision_seq=0))
        write_rows(original/'driver/per_step.jsonl',rows)
    elif mutation=='ordinary_outcome':
        attempts[-1]['error']=None
    elif mutation=='unreported':
        attempts.pop(); write_rows(original/'driver/dispatch.jsonl',[dict(task_uid=uid,attempt=1)])
    elif mutation=='missing_prior': attempts[-1]['attempt']=2
    else: attempts.extend(dict(attempts[-1],attempt=a) for a in (2,3))
    write_rows(original/'driver/attempts.jsonl',attempts)
    with pytest.raises(ValueError): inspect_repair(original,model='dp',suite='libero_spatial')


def test_repair_cli_is_explicit_mode_and_rejects_run_plus_repair():
    args=['--model','dp_libero','--suite','libero_spatial','--repair','/closed/failed']
    assert parse_args(args).repair==Path('/closed/failed')
    with pytest.raises(SystemExit): parse_args(args+['--run'])


def test_fresh_seal_mismatch_aborts_prepare_before_worker_deployment(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval
    original,repair,campaign,_,_,_ = repair_fixture(tmp_path,monkeypatch)
    c,_ = prepared_repair(original,repair,campaign)
    c.update(model='dp',server_node='local')
    def fresh_seal(*args,**kwargs):
        (repair/'assets/load.json').write_text('{"expected_weight_sha256":"other"}')
        return '{}'
    deployed=[]
    monkeypatch.setattr(formal_eval,'execute',fresh_seal)
    monkeypatch.setattr(formal_eval,'prepare_workers',lambda *a:deployed.append(a))
    with pytest.raises(ValueError,match='seal/checkpoint/weight/code/norm'):
        formal_eval.prepare(c,repair)
    assert not deployed


def test_live_repair_cli_runs_only_missing_and_finishes_merged_collection(tmp_path,monkeypatch):
    from exp.r17_integration.ops import library_campaign,formal_eval
    original,fixture,campaign,_,rerun,tasks = repair_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(library_campaign,'LIBRARY_ROOT',tmp_path/'library')
    calls=[]
    def execute(argv,**kwargs):
        assert argv[:2]==['tether','expose']
        if argv[2]=='rm': return '{}'
        return json.dumps(dict(node=argv[2],name=argv[argv.index('--name')+1],public_host='CPU_transport',port=12345))
    def prepare(c,d):
        shutil.copytree(original/'assets',d/'assets')
        verify_repair_seal(c)
        write_json(d/'campaign.json',c)
        return {}
    def collect(c,d,deployment):
        assert c['retries']==2 and c['checkpoint']==campaign['checkpoint']
        chosen=formal_driver.plan(c)[2]
        assert len(chosen)==4
        calls.extend(t.task_uid for t in chosen)
        shutil.copytree(fixture/'servers',d/'servers')
        rows=[dict(task_uid=e['task_uid'],attempt=2,success=e['success'],error=None,n_steps=10,accepted=True,status='done') for e in rerun]
        for name in ('journal','attempts'): write_rows(d/'driver'/(name+'.jsonl'),rows)
        write_rows(d/'driver/per_step.jsonl',[dict(task_uid=e['task_uid'],attempt=2,accepted=True,decision_seq=s) for e in rerun for s in (0,1)])
        write_json(d/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
        return 0
    def finish_merged(c,d):
        assert c['repair_merge'] and d.name=='merged'
        assert len([r for r in json_rows(d/'driver/journal.jsonl') if r['accepted']])==500
        return dict(status='PASS',cpu_mock_live_launch=True)
    monkeypatch.setattr(formal_eval,'execute',execute)
    monkeypatch.setattr(formal_eval,'prepare',prepare)
    monkeypatch.setattr(formal_eval,'run_campaign',collect)
    monkeypatch.setattr(library_campaign,'finish',finish_merged)
    assert library_campaign.main(['--model','dp_libero','--suite','libero_spatial','--repair',str(original),
        '--replicas','2','--workers','4','--run-name','CPU_CLI_REPAIR'])==0
    assert set(calls)=={e['task_uid'] for e in rerun}


def test_fastwam_repair_reuses_original_g2_receipts_and_still_verifies_gate(tmp_path,monkeypatch):
    from exp.r17_integration.ops import library_campaign,library_repair
    from exp.r17_integration.ops.library_alignment import load_alignment
    audit=load_alignment(library_campaign.AUDIT_ROOT/'libero_spatial.json')
    original=dict(checkpoint='/sealed/Fast',g2_a_receipt='/sealed/G2_A.json',g2_b_receipt='/sealed/G2_B.json')
    monkeypatch.setattr(library_repair,'inspect_repair',lambda *a,**k:(original,audit,{}))
    called=[]
    def gate(a,b,checkpoint,**kwargs):
        called.append((a,b,checkpoint))
        raise ValueError('CPU fixture: strict G2 gate was invoked')
    monkeypatch.setattr(library_campaign,'require_fastwam_g2',gate)
    with pytest.raises(ValueError,match='strict G2 gate'):
        library_campaign.main(['--model','fastwam_libero','--suite','libero_spatial','--repair',str(tmp_path/'failed')])
    assert called==[('/sealed/G2_A.json','/sealed/G2_B.json','/sealed/Fast')]
