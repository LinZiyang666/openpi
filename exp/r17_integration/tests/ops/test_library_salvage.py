"""Actual CPU captures: node loss never turns partial episodes into outcomes."""
from dataclasses import asdict
import ast
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest

from exp.r17_integration.ops import formal_driver, library_salvage as s, library_staging as staging
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.library_repair import SEAL_FILES, inspect_repair, merge_repair, verify_repair_seal
from exp.r17_integration.ops.library_campaign import parse_args
from exp.r17_integration.debug.attempts import json_rows, validate_prior_attempts
from exp.r17_integration.debug.validate import digest, file_sha
from exp.r17_integration.search.teacher_library import build_teacher_parent
from exp.r17_integration.tests.search.library_fixture import synthetic_capture
from exp.r17_integration.tests.ops.test_library_retry_repair import write_rows
from openpi.conductor.task import EpisodeTask


@pytest.fixture
def failed(tmp_path, monkeypatch):
    original=tmp_path/'original';original.mkdir();sources=[];episodes=[]
    for index in range(5):
        source=original/'servers/replica_0/capture/shards'/digest(index)/'binding_cpu'
        b,eps,audit,names=synthetic_capture(source,episodes=1,episode_offset=index)
        sources.append(source);episodes+=eps
    # 0,1: accepted successes/ordinary failure. 2: terminal success missing final
    # ack. 3: inference error after partial progress hidden by worker n_steps=0.
    # 4: torn writer. 5: never attempted (no capture).
    _,last,_,_=synthetic_capture(tmp_path/'unused',episodes=1,episode_offset=5)
    episodes+=last
    path=sources[2]/'execution.jsonl';write_rows(path,json_rows(path)[:-1]);reseal(sources[2])
    path=sources[4]/next(sources[4].glob('decisions_*')).name;path.write_bytes(path.read_bytes().rstrip(b'\n'));reseal(sources[4])
    tasks=[EpisodeTask(e['task_uid'],'cpu','eval','libero_spatial',e['task_id'],e['orig_init_idx'],e['orig_init_idx'],
        'unused',23360,'generation1',extra={k:v for k,v in e.items() if k!='attempt'}) for e in episodes]
    root=original/'assets';root.mkdir()
    for name in SEAL_FILES:(root/name).write_text(json.dumps(dict(sealed=name)))
    write_json(root/'model_spec.json',b['model']);write_json(original/'ALIGNMENT.json',audit)
    archive=original/'bpool_inputs.tar';archive.write_bytes(b'B byte identity')
    c=dict(library=True,model='dp',suite='libero_spatial',server_node='local',smoke=False,run_name='CPU_ORIGINAL',
        local_assets=str(root),assets=str(root),checkpoint='/CPU/teacher',alignment=str(original/'ALIGNMENT.json'),
        bpool_archive=str(archive),bpool_transfer=dict(archive_sha256=file_sha(archive)),synthetic=True)
    write_json(original/'campaign.json',c);write_json(original/'EPISODE_PLAN.json',dict(model=b['model'],tasks=[asdict(t) for t in tasks]))
    write_json(original/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False,error='node_offline: status=STALE'))
    attempts=[];journal=[];steps=[]
    for i,e in enumerate(episodes[:5]):
        r=dict(task_uid=e['task_uid'],attempt=1,accepted=True,success=e['success'],error=None,n_steps=10)
        if i==3:r.update(success=False,error='LIBERO inference failed',n_steps=0)
        attempts.append(r);journal.append(dict(r,status='done' if r['success'] else 'failed'))
        if i!=3:steps.extend(dict(e,accepted=True,decision_seq=seq) for seq in (0,1))
    for name,rows in [('attempts',attempts),('journal',journal),('per_step',steps),('dispatch',attempts)]:write_rows(original/'driver'/(name+'.jsonl'),rows)
    def plan(c):
        chosen=[t for t in tasks if t.task_uid in c['repair_attempts']] if c.get('repair_attempts') else tasks
        return NS(identity=b['benchmark_protocol'],ordered_tasks=names),b['model'],chosen,[]
    monkeypatch.setattr(formal_driver,'plan',plan)
    return original,c,sources,episodes,tasks,b,audit


def reseal(source):
    m=json.loads((source/'r17_capture.json').read_text())
    m['files']={n:file_sha(source/n) for n in m['files']}
    write_json(source/'r17_capture.json',m)


def reconstruct(failed):
    original,c,sources,episodes,tasks,b,audit=failed
    return s.reconstruct(original,original/'salvage',c,sources,dict(all_owned_closed=True,status='PASS'),{})


def test_drop_entire_torn_missing_ack_error_episodes_keep_ordinary_failures_and_originals(failed):
    original,c,sources,episodes,tasks,b,audit=failed
    before={str(p):file_sha(p) for p in original.rglob('*') if p.is_file()}
    result=reconstruct(failed)
    assert result['kept_episodes']==2 and result['missing_episodes']==4
    assert result['dropped_attempted_episodes']==3 and result['unattempted_episodes']==1
    derived=json.loads((original/'salvage/campaign.json').read_text())
    assert len(derived['repair_merge']['capture_sources'])==2
    assert [r['success'] for r in json_rows(original/'salvage/driver/journal.jsonl')]==[False,True]
    assert all(file_sha(p)==sha for p,sha in before.items())
    proof=json.loads((original/'salvage/SALVAGE_PROVENANCE.json').read_text())
    assert len(proof['bad_sources'])==1 and len(proof['discarded_attempts'])==3
    assert any(d['reason'].startswith('missing/disagreeing') for d in proof['dropped_episodes'])
    assert proof['discarded_attempts'][1]['capture_decisions']==2


@pytest.mark.parametrize('h100',[False,True])
def test_salvage_repair_recollect_missing_only_preserve_seal_then_build_complete_parent(failed,tmp_path,h100):
    original,c,sources,episodes,tasks,b,audit=failed
    if h100:
        c.update(server_node='h100',h100_staging=dict(staging.DEFAULT_POLICY))
        write_json(original/'campaign.json',c)
    reconstruct(failed)
    derived=original/'salvage'
    old,_,pin=inspect_repair(derived,model='dp',suite='libero_spatial')
    assert pin['kept_outcomes']==2 and len(pin['repair_attempts'])==4
    assert list(pin['repair_attempts'].values()).count(1)==1
    assert list(pin['repair_attempts'].values()).count(2)==3
    repair=tmp_path/'repair';repair.mkdir();shutil.copytree(original/'assets',repair/'assets')
    shutil.copy2(original/'bpool_inputs.tar',repair/'bpool_inputs.tar');write_json(repair/'ALIGNMENT.json',audit)
    fresh=dict(old,run_name='CPU_REPAIR',semantic_campaign=c['run_name'],local_assets=str(repair/'assets'),
        assets=str(repair/'assets'),alignment=str(repair/'ALIGNMENT.json'),bpool_archive=str(repair/'bpool_inputs.tar'),
        repair_pin=pin,repair_attempts=pin['repair_attempts'],repair_prior_attempts=pin['prior_attempts'])
    # Live repair campaign does not inherit a merged source list.
    fresh.pop('repair_merge');fresh.pop('node_failure_salvage');fresh.pop('node_failure_prior_attempts')
    verify_repair_seal(fresh)
    newepisodes=[];rows=[];cost=[]
    for index in range(2,6):
        gen=pin['repair_attempts'][episodes[index]['task_uid']]
        path=repair/'servers'/('replica_'+str(index))/'capture'
        if h100:path=path/'shards'/digest(index)
        _,eps,_,_=synthetic_capture(path/'binding_cpu',episodes=1,episode_offset=index,attempt=gen)
        ep=eps[0];newepisodes.append(ep)
        rows.append(dict(task_uid=ep['task_uid'],attempt=gen,accepted=True,status='done',success=ep['success'],error=None,n_steps=10))
        cost.extend(dict(ep,accepted=True,decision_seq=seq) for seq in (0,1))
    for name,rr in [('journal',rows),('attempts',rows),('per_step',cost)]:write_rows(repair/'driver'/(name+'.jsonl'),rr)
    transfers={} if not h100 else dict(h100_servers=dict(verified=dict(status='verified')),
        h100_shards=dict(status='verified',shards={digest(index):{} for index in range(2,6)}))
    write_json(repair/'campaign.json',fresh);write_json(repair/'RECEIPT.json',dict(status='PASS',all_owned_closed=True,transfers=transfers))
    merged_c,merged=merge_repair(fresh,repair)
    from exp.r17_integration.ops.library_repair import outcomes,capture_sources
    from exp.r17_integration.debug.attempts import capture_counts
    all_sources=capture_sources(merged_c,merged);counts,_=capture_counts(all_sources,binding=b)
    accepted,decisions,histories,missing=outcomes(merged,tasks,counts)
    cohort=[dict(e,attempt=accepted[e['task_uid']]['attempt'],prior_attempts=histories[e['task_uid']]) for e in episodes]
    parent=build_teacher_parent(all_sources,tmp_path/'parent',b,cohort,audit,expected_episodes=6)
    assert len(parent.rows)==12 and not missing
    assert len(parent.manifest['metadata']['quarantined_node_failure_attempts'])==3
    assert parent.manifest['metadata']['pre_decision_infrastructure_attempts']==[]
    if h100:assert len(json.loads((merged/'RECEIPT.json').read_text())['transfers']['h100_shards']['shards'])==6


@pytest.mark.parametrize('mutation',['original','seal','proof','unquarantined'])
def test_salvage_and_discard_seals_fail_closed(failed,mutation):
    original,c,sources,episodes,tasks,b,audit=failed;reconstruct(failed)
    derived=json.loads((original/'salvage/campaign.json').read_text())
    row=next(iter(derived['node_failure_prior_attempts'].values()))[0]
    if mutation=='original':(original/'driver/journal.jsonl').write_text('{}\n')
    elif mutation=='seal':(original/'assets/load.json').write_text('changed')
    elif mutation=='proof':(original/'salvage/SALVAGE_PROVENANCE.json').write_text('{}')
    else:
        with pytest.raises(ValueError):validate_prior_attempts(dict(task_uid=row['task_uid'],attempt=2,prior_attempts=[row]),{(row['task_uid'],1):1})
        return
    with pytest.raises((ValueError,KeyError)):s.verify_salvage(derived)


def test_salvage_cli_is_cpu_only_early_dispatch_and_modes_exclusive(monkeypatch,tmp_path):
    from exp.r17_integration.ops import library_campaign,formal_eval
    argv=['--model','cosmos_libero','--suite','libero_10','--salvage',str(tmp_path)]
    assert parse_args(argv).salvage==tmp_path
    with pytest.raises(SystemExit):parse_args(argv+['--repair',str(tmp_path)])
    monkeypatch.setattr(s,'salvage',lambda *a,**k:dict(status='PASS',cpu_only=True))
    for method in ('prepare','run_campaign','execute'):
        monkeypatch.setattr(formal_eval,method,lambda *a,**k:pytest.fail('GPU/launch/tunnel invoked'))
    assert library_campaign.main(argv)==0


def test_live_or_unreachable_owned_process_refuses_before_transfer(monkeypatch):
    monkeypatch.setattr(s,'alive',lambda e:True)
    with pytest.raises(ValueError,match='still alive'):s.process_probe([dict(pid=123,start_ticks='1',pgid=123)],'CPU')


@pytest.mark.parametrize('partial',[False,True])
def test_actual_bounded_stage_sha_install_ack_cleanup_with_partial_quarantine(tmp_path,monkeypatch,partial):
    from exp.r17_integration.qualification import archive_stage
    from exp.r17_integration.ops.sync_back import install_snapshot
    from exp.r17_integration.tests.ops.test_library_staging import closed_shard
    monkeypatch.setattr(staging,'TRANSFER_ROOT',tmp_path/'tmp')
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'qtmp')
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda p:NS(free=100*staging.GIB))
    c=dict(library=True,server_node='h100',run_name='CPU',h100_staging=dict(staging.DEFAULT_POLICY),server_run=str(tmp_path/'remote/servers'))
    shard,leaf,b,eps,audit,names=closed_shard(c)
    if partial:
        path=leaf/'binding_cpu/r17_capture.json';m=json.loads(path.read_text());m['status']='partial';write_json(path,m)
        path=leaf/'binding_cpu/unfinished.part';path.write_bytes(b'torn')
        state=json.loads((Path(c['server_run'])/'SHARD_BUDGET.json').read_text());state['shards'][shard]['status']='writing';write_json(Path(c['server_run'])/'SHARD_BUDGET.json',state)
    transfer=s.remote_shards(c,'stage',shard)
    target=tmp_path/'verified'/shard;verified=install_snapshot(transfer['archive'],target,expected_sha256=transfer['archive_sha256'])
    receipt=dict(schema='r17.salvage_shard_receipt.v1',status='verified',campaign='CPU',shard_id=shard,
        archive_sha256=transfer['archive_sha256'],snapshot_digest=verified['digest'],partials=verified['partials'],build_host='weilandserver',destination=str(target))
    sources=[target/'binding_cpu'];receipts={shard:dict(receipt=receipt)}
    s.verify_received(sources,c,receipts)
    with pytest.raises(ValueError):s.verify_received(sources,c,{shard:dict(receipt=dict(receipt,campaign='OTHER'))})
    with pytest.raises(ValueError):s.remote_shards(c,'ack',shard,dict(receipt,archive_sha256='0'*64))
    assert leaf.exists() and Path(transfer['archive']).exists()
    result=s.remote_shards(c,'ack',shard,receipt)
    assert result['removed_after_verified_receipt'] and not leaf.exists()
    assert not Path(transfer['archive']).exists() and not Path(transfer['source_archive']).exists()
    assert s.remote_shards(c,'inventory')['shards']=={}
    assert s.remote_shards(c,'ack',shard,receipt)['already_cleaned']
    assert bool(verified['partials'])==partial


def test_unknown_torn_driver_tail_is_explicit_and_never_accepted(failed):
    original,c,sources,episodes,tasks,b,audit=failed
    path=original/'driver/journal.jsonl';path.write_bytes(path.read_bytes()+b'{"torn":')
    result=reconstruct(failed)
    assert result['kept_episodes']==0
    proof=json.loads((original/'salvage/SALVAGE_PROVENANCE.json').read_text())
    assert len(proof['driver_issues'])==1


@pytest.mark.parametrize('mutation',['checksum','cost_missing','coordinate','executed_steps','duplicate_ack','quarantine_corrupt'])
def test_incomplete_or_inconsistent_evidence_is_never_kept(failed,mutation):
    original,c,sources,episodes,tasks,b,audit=failed
    source=sources[1]
    if mutation=='checksum':
        block=next((source/'blocks').glob('*.npz'));block.write_bytes(b'changed')
    elif mutation=='cost_missing':
        path=original/'driver/per_step.jsonl';write_rows(path,[r for r in json_rows(path) if r['task_uid']!=episodes[1]['task_uid']])
    elif mutation=='coordinate':
        path=next(source.glob('decisions_*'));rows=json_rows(path);rows[0]['init_state_sha256']='0'*64;write_rows(path,rows);reseal(source)
    elif mutation=='executed_steps':
        path=original/'driver/attempts.jsonl';rows=json_rows(path);rows[1]['n_steps']=9;write_rows(path,rows)
    elif mutation=='duplicate_ack':
        path=source/'execution.jsonl';rows=json_rows(path);write_rows(path,rows+[rows[-1]]);reseal(source)
    else:
        reconstruct(failed);derived=json.loads((original/'salvage/campaign.json').read_text())
        next((sources[3]/'blocks').glob('*.npz')).write_bytes(b'changed quarantine')
        with pytest.raises(ValueError,match='quarantine evidence'):s.verify_salvage(derived)
        return
    assert reconstruct(failed)['kept_episodes']==1


def test_remote_probe_does_not_mistake_its_inline_helper_for_a_worker():
    assert s.process_probe([], 'CPU')==[]


def test_remote_free_space_failure_and_foreign_campaign_never_delete(tmp_path,monkeypatch):
    from exp.r17_integration.qualification import archive_stage
    from exp.r17_integration.tests.ops.test_library_staging import closed_shard
    monkeypatch.setattr(staging,'TRANSFER_ROOT',tmp_path/'tmp')
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'qtmp')
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda p:NS(free=100*staging.GIB))
    c=dict(library=True,server_node='h100',run_name='CPU',h100_staging=dict(staging.DEFAULT_POLICY),server_run=str(tmp_path/'remote/servers'))
    shard,leaf,*_=closed_shard(c)
    with pytest.raises(ValueError,match='foreign'):s.remote_shards(dict(c,run_name='OTHER'),'stage',shard)
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda p:NS(free=staging.GIB))
    with pytest.raises(ValueError,match='free-space'):s.remote_shards(c,'stage',shard)
    assert leaf.exists()


def test_empty_interrupted_shard_uses_original_manifest_attempt_identity(failed):
    original,c,sources,episodes,tasks,b,audit=failed
    source=sources[3]
    for path in source.glob('decisions_*'):path.write_bytes(b'')
    # Truly empty: surviving intent fences/acks are progress evidence and
    # now have their own regression, rather than being called zero rows.
    for name in ('decision_intents.jsonl','execution.jsonl'):(source/name).write_bytes(b'')
    write_json(source.parent/'manifest.json',dict(schema='r17.teacher_shard.v1',campaign=c['run_name'],
        episode_identity=[c['run_name'],'full','generation1',b['benchmark_protocol'],episodes[3]['task_name'],3,
                          episodes[3]['task_data_hash'],episodes[3]['task_uid'],1]))
    reconstruct(failed)
    proof=json.loads((original/'salvage/SALVAGE_PROVENANCE.json').read_text())
    row=next(r for r in proof['discarded_attempts'] if r['task_uid']==episodes[3]['task_uid'])
    assert row['attempt']==1 and row['capture_decisions']==0


def test_salvage_recovers_closed_batch_after_node_failure_without_redeploy(failed,tmp_path,monkeypatch):
    from exp.r17_integration.ops import library_batch
    from exp.r17_integration.qualification import archive_stage
    original,c,sources,episodes,tasks,b,audit=failed
    remote_root=tmp_path/'h100_servers';(original/'servers').rename(remote_root)
    c.update(server_node='h100',server_run=str(remote_root),runtime='/existing/sealed/runtime',
             h100_staging=dict(staging.DEFAULT_POLICY),h100_batch=True)
    write_json(original/'campaign.json',c);write_json(original/'H100_SHARD_TRANSFERS.json',dict(shards={}))
    monkeypatch.setattr(staging,'TRANSFER_ROOT',tmp_path/'tmp')
    monkeypatch.setattr(archive_stage,'STAGE_ROOT',tmp_path/'qtmp')
    monkeypatch.setattr(staging.shutil,'disk_usage',lambda p:NS(free=100*staging.GIB))
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    monkeypatch.setattr(s,'prove_closed',lambda *a:dict(status='PASS',all_owned_closed=True))
    budget=staging.ShardBudget(c)
    for i,e in enumerate(episodes[:5]):
        leaf=remote_root/'replica_0/capture/shards'/digest(i)
        write_json(leaf/'manifest.json',dict(schema='r17.teacher_shard.v1',campaign=c['run_name'],
            episode_identity=[c['run_name'],'full','generation1',b['benchmark_protocol'],e['task_name'],i,
                              e['task_data_hash'],e['task_uid'],1]))
        budget.reserve(leaf.name,leaf);budget.charge(leaf.name,2*sum(p.stat().st_size for p in leaf.rglob('*') if p.is_file()))
        budget.closed(leaf.name)
    transfer=library_batch.next_batch(c)['transfer'];assert len(transfer['shards'])==5
    state=json.loads(budget.state_path.read_text());state['failure']='node_offline: status=STALE';write_json(budget.state_path,state)
    calls=[];pulls=[]
    def remote(node,argv):
        assert node=='h100';expr=ast.parse(argv[-1]).body[-1].value.args[0].args[0]
        assert expr.func.id=='remote_shards'
        args=[ast.literal_eval(a) for a in expr.args];calls.append(args[1])
        return json.dumps(s.remote_shards(*args))
    def pull(source,target):pulls.append(source);shutil.copyfile(source,target)
    result=s.salvage(original,model='dp',suite='libero_spatial',remote=remote,pull=pull)
    assert result['kept_episodes']==2 and result['missing_episodes']==4
    assert len(pulls)==1 and calls==['inventory','batch_ack','inventory','inventory']
    assert not Path(transfer['archive']).exists() and s.remote_shards(c,'inventory')['shards']=={}
    _,_,pin=inspect_repair(original/'salvage',model='dp',suite='libero_spatial')
    assert len(pin['repair_attempts'])==4
