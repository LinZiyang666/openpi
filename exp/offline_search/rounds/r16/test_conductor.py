"""CPU mechanism tests; synthetic outcomes only, scratch outside the repository."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpi.conductor.protocol import result_to_wire, task_from_wire
from openpi.conductor.task import EpisodeResult, ServerEndpoint
from . import conductor as c, conductor_analysis as ca, analyze as a
from .test_r16 import episode, synthetic, jl
from .resident import ArrayPool, Registry


def fixture_group(root):
    cell='pi05_l10_100_k1';pairs=[(t,i) for t in range(10) for i in range(20,50)]
    manifest=root/'manifest.json';c.write_json(manifest,pairs)
    ref=root/'reference.json';c.write_json(ref,{cell:dict(y=np.ones((10,30),int).tolist())})
    entries=[]
    for level in ('c10',*c.ORDER):
        row=dict(arm='r15d_r16_synthetic_'+level,r16_cell=cell,r14_descent_level=level,
                 suite='libero_10',manifest=str(manifest),init_pool='B')
        entries.append(dict(row=row,pairs=pairs))
    return dict(root=str(root),phase='dev',model='pi05',k=1,entries=entries,references=str(ref),group_sha256='synthetic')


def driver(group):
    strategy=c.R16Strategy(group)
    endpoints={'a:1':['libero_10'],'b:2':['libero_10']}
    return c.R16Driver(strategy,endpoints,yaml_weights={e['row']['arm']:300 for e in group['entries']},
                       servers=[ServerEndpoint('a',1),ServerEndpoint('b',2)],
                       journal_path=str(Path(group['root'])/'journal.jsonl'),ctl_factory=lambda ep:NS())


def pull(d,key='a:1'):
    _,wire=d.handle_pull(key)
    return task_from_wire(wire['task']) if 'task' in wire else None


def report(d,ep,success=True,error=None):
    d.handle_result(result_to_wire(EpisodeResult(ep.task_uid,success,5,error=error,attempt=ep.attempt)))


def successes(level,task):
    # Different first failures, including recovery below failure. Task 9 reaches off.
    levels=list(c.ORDER)
    failure=task//2 if task<9 else len(levels)
    return 28 if failure<len(levels) and levels.index(level)==failure else 30


def simulate(d):
    done=[]
    for _ in range(4000):
        d.drive_stages_once()
        if d._scheduler.all_done():break
        ep=pull(d)
        if ep is None:continue
        level=ep.yaml_id.rsplit('_',1)[1]
        count=30 if level=='c10' else successes(level,ep.task_id)
        report(d,ep,ep.episode_idx-20<count)
        done.append(ep)
    else:raise AssertionError('Queue stalled')
    return done


def test_adaptive_matches_full_ladder_and_skips_recovery(tmp_path):
    group=fixture_group(tmp_path);d=driver(group);episodes=simulate(d)
    accepted=c.completed_rows(tmp_path/'journal.jsonl')
    for task in range(10):
        full={level:a.paired(np.r_[np.ones(successes(level,task)),np.zeros(30-successes(level,task))],np.ones(30)) for level in c.ORDER}
        full['c10']=a.paired(np.ones(30),np.ones(30))
        expected=a.choose(full,5)['level']
        observed={level:stats for level,stats in full.items() if level=='c10' or
                  any(ep.task_id==task and ep.yaml_id.endswith('_'+level) for ep in episodes)}
        assert ca.partial_choose(observed,5)['level']==expected
        for tol in (0,2):assert ca.partial_choose(observed,tol)['level']==a.choose(full,tol)['level']
        failed=a.choose(full,5)['first_failure']
        if failed:
            for below in c.ORDER[c.ORDER.index(failed)+1:]:
                assert not any(ep.task_id==task and ep.yaml_id.endswith('_'+below) for ep in episodes)
        fallback=[ep for ep in episodes if ep.task_id==task and ep.yaml_id.endswith('_c10')]
        assert len(fallback)==(30 if expected=='c10' else 0)
    assert len(accepted)==len(episodes)<1800


def test_thirty_episode_gate_and_restart_fence(tmp_path):
    group=fixture_group(tmp_path);d=driver(group);d.drive_stages_once()
    dispatched=[]
    for _ in range(300):
        ep=pull(d);assert ep is not None;dispatched.append(ep)
    assert {ep.server.key for ep in dispatched}=={'a:1','b:2'} # one worker default steals across endpoints
    held=next(ep for ep in dispatched if ep.task_id==0)
    for ep in dispatched:
        if ep!=held:report(d,ep)
    d.drive_stages_once()
    assert d._scheduler.stage_state('r15d_r16_synthetic_ir40/task00').value=='blocked'
    before=set(c.completed_rows(tmp_path/'journal.jsonl'))
    resumed=driver(group);resumed.drive_stages_once()
    todo=[]
    for _ in range(301):
        ep=pull(resumed)
        if ep is None:break
        assert ep.task_uid not in before
        todo.append(ep)
    repeat=next(ep for ep in todo if ep.task_uid==held.task_uid)
    assert repeat.attempt>held.attempt
    report(resumed,held,False) # stale pre-crash result cannot overwrite new dispatch
    assert held.task_uid not in c.completed_rows(tmp_path/'journal.jsonl')
    report(resumed,repeat,True);resumed.drive_stages_once();resumed.drive_stages_once()
    assert resumed._scheduler.stage_state('r15d_r16_synthetic_ir40/task00').value!='blocked'


def test_transport_error_is_retried_not_sr_failure(tmp_path):
    d=driver(fixture_group(tmp_path));d.drive_stages_once();ep=pull(d)
    report(d,ep,False,'network disconnected')
    assert ep.task_uid not in c.completed_rows(tmp_path/'journal.jsonl')
    for _ in range(1000):
        retry=pull(d)
        if retry.task_uid==ep.task_uid:break
        report(d,retry)
        d.drive_stages_once()
    assert retry.attempt==ep.attempt+1


def test_sparse_analysis_primary_identical_and_unknown_not_imputed():
    full=episode(np.ones((10,30),int),k=10)
    complete={l:episode(full.y.copy()) for l in ('c10',*c.ORDER)}
    complete['ir40'].y[0,:2]=0
    sparse=copy.deepcopy(complete)
    for level in ('ir32','ir25','off'):sparse[level].y[0]=-1;sparse[level].n[0]=sparse[level].v[0]=sparse[level].m[0]=sparse[level].cost[0]=0
    result=ca.adaptive_cell('pi05_l10_100_k1',sparse,full,draws=50)
    legacy=a.dev_cell('pi05_l10_100_k1',complete,full,draws=50)
    for tol in (0,2,5):assert result['tolerances'][str(tol)]['levels']==legacy['tolerances'][str(tol)]['levels']
    assert result['tolerances']['5']['selected']==legacy['tolerances']['5']['selected']
    assert 'uniform_stats' not in result['tolerances']['5']
    sparse['ir45'].y[1]=-1
    with pytest.raises(ValueError,match='Missing level'):ca.adaptive_cell('pi05_l10_100_k1',sparse,full,draws=2)


def test_array_interning_byte_exact_and_containers_isolated():
    pool=ArrayPool();arr=np.arange(100,dtype=np.float32)
    one=NS(a=arr);two=NS(a=arr.copy())
    # Only fitted method objects are traversed; dicts exercise the same leaf rule.
    x=pool.intern({'fit':arr,'mutable':[1]});y=pool.intern({'fit':arr.copy(),'mutable':[2]})
    assert x['fit'] is y['fit'] and x['fit'].tobytes()==arr.tobytes()
    assert x['mutable'] is not y['mutable'] and pool.bytes==arr.nbytes
    registry=Registry({'arm':NS(api=NS(),attach=lambda comps,cfg,yaml_id:yaml_id)})
    assert registry.attach({},None,'arm')=='arm'
    with pytest.raises(ValueError,match='unregistered'):registry.attach({},None,'wrong')


def test_gr_checkpoint_capability_and_runner_argv(tmp_path,monkeypatch):
    from . import conductor_runner as runner
    from exp.offline_search.closed_loop.ops.h100 import local_server
    group=dict(root=str(tmp_path/'r16_smoke'),model='groot',k=1,entries=[])
    for suite in ('libero_10','libero_spatial'):
        group['entries'].append(dict(row=dict(arm=suite,suite=suite,method='x:Y',kwargs={},cell='groot_l10_cache',
            plugin_args=['--os-fit-artifact','/fit'],yaml='/config')))
    specs=runner.server_specs(group,[23170,23171],1)
    monkeypatch.setattr(runner,'BASE_COMMAND',lambda spec:(['taskset','-c','22-37','python','/serve_groot.py','--port',str(spec['port'])],{}))
    argv,env=runner.resident_command(specs[0])
    assert argv[argv.index('--server-tag')+1]==str(specs[0]['port'])
    assert argv[argv.index('--resident-suite')+1]=='libero_10'
    with pytest.raises(ValueError):c.endpoints_from_args('a:1,b:2','groot')


def test_adaptive_freeze_refuses_tamper_and_missing_cells(tmp_path,monkeypatch):
    # This verifies only synthetic tables; it does not emit A arms.
    cells={cell:dict(tolerances={str(t):dict(levels={str(j):'ir45' for j in range(10)}) for t in (0,2,5)}) for cell in c.CELLS}
    result=dict(adaptive=True,uniform_tables=False,cells=cells,inputs={})
    for tol in (0,2,5):c.write_json(tmp_path/f'levels_tol{tol}pp.json',{cell:r['tolerances'][str(tol)]['levels'] for cell,r in cells.items()})
    c.write_json(tmp_path/'levels.json',json.loads((tmp_path/'levels_tol5pp.json').read_bytes()))
    c.write_json(tmp_path/'dev.json',result);(tmp_path/'DESCENT.md').write_text('synthetic')
    monkeypatch.setattr(a,'verify',lambda reader:{})
    ca.freeze(tmp_path)
    monkeypatch.setattr('exp.offline_search.rounds.r16.prepare.HERE',c.HERE)
    from .prepare import frozen_tables
    levels,uniform,frozen=frozen_tables(tmp_path)
    assert len(levels)==32 and set(uniform.values())=={None} and frozen['adaptive']
    (tmp_path/'levels.json').write_text('{}')
    from .prepare import frozen_tables
    with pytest.raises(ValueError,match='Changed frozen'):frozen_tables(tmp_path)


@pytest.mark.parametrize('model',['pi05','groot'])
def test_resident_loader_matches_legacy_and_rejects_cross_arm(tmp_path,model):
    seal,name,pairs,original=synthetic(tmp_path/'legacy',model=model)
    legacy=a.load(seal,name,pairs,'B',1,a.Reader())
    root=tmp_path/'dispatch';row=json.loads((Path(seal['root'])/'arms.json').read_bytes())[0]
    row.update(r16_cell=model+'_l10_50_k1',suite='libero_10')
    contract=json.loads((Path(seal['root'])/'dev_contracts.json').read_bytes())[name]
    entry=dict(row=row,pairs=pairs,pool_contract=contract,fit_sha256=seal['fit_sha256'][row['plugin_args'][1]])
    group=dict(root=str(root),model=model,k=1,phase='dev',group_sha256='synthetic')
    c.write_json(root/'launch.json',dict(server_host='local',worker_host='timan107',STAGE1_ONLY=0,
                                      group_sha256='synthetic',smoke_episodes=None))
    journal=[dict(json.loads(l),yaml_id=name) for l in (original/'client/journal.jsonl').read_text().splitlines()]
    jl(root/'journal.jsonl',journal)
    rows=[json.loads(l) for l in (original/'server_synthetic/decisions_0.jsonl').read_text().splitlines()]
    for r in rows:
        if r.get('ev')=='dec':r.update(arm=name,yaml_id=name,bundle=name)
    path=root/'runs'/name/'server_0/decisions_synth.jsonl';jl(path,rows)
    actual=ca.load_entry(group,entry,a.Reader())
    for field in ('y','n','v','m','cost'):assert np.array_equal(getattr(actual,field),getattr(legacy,field))
    for r in rows:
        if r.get('ev')=='dec':r['arm']='other_arm';break
    jl(path,rows)
    with pytest.raises(ValueError,match='Cross-arm'):ca.load_entry(group,entry,a.Reader())


def test_collector_summary_rounding_only(tmp_path):
    seal,name,pairs,run=synthetic(tmp_path/'legacy')
    run=Path(seal['root'])/'runs'/name
    journal=[json.loads(l) for l in (run/'client/journal.jsonl').read_text().splitlines()]
    # Both duplicate accepted terminal rows of this UID must agree.
    uid=journal[0]['task_uid']
    for r in journal:
        if r['task_uid']==uid and r.get('accepted'):r['success']=False;r['status']='failed'
    jl(run/'client/journal.jsonl',journal)
    summary=json.loads((run/'summary.json').read_bytes());summary.update(success=299,sr=round(299/300,4))
    summary['per_task']['0']['success']=29;c.write_json(run/'summary.json',summary)
    assert a.load(seal,name,pairs,'B',1,a.Reader()).y.sum()==299
    summary['sr']=.998;c.write_json(run/'summary.json',summary)
    with pytest.raises(ValueError,match='Summary SR'):a.load(seal,name,pairs,'B',1,a.Reader())


def test_legacy_pure_receipts_recover_whole_episode_gap_only(tmp_path):
    from .conductor_reuse import load_completed
    seal,name,pairs,run=synthetic(tmp_path/'legacy')
    row=json.loads((Path(seal['root'])/'arms.json').read_bytes())[0]
    row.update(r14_descent_level='c10',r16_k=1);c.write_json(Path(seal['root'])/'arms.json',[row])
    c.write_json(run/'h100_launch.json',dict(server_host='local',worker_host='timan107',servers=[dict(env=dict(STAGE1_ONLY='0'))]))
    seal['definitions_sha256']['arms.json']=a.sha((Path(seal['root'])/'arms.json').read_bytes());c.write_json(Path(seal['root'])/'ROOT.json',seal)
    baseline=a.load(seal,name,pairs,'B',1,a.Reader())
    journal=[json.loads(l) for l in (run/'client/journal.jsonl').read_text().splitlines()]
    for r in journal:r['run_id']='synthetic'
    jl(run/'client/journal.jsonl',journal)
    receipts=[]
    for t,i in pairs:
        uid=f'{name}:eval:{t}:{i}'
        for step in (0,1):
            hit=step==1
            receipts.append(dict(task_uid=uid,yaml_id=name,task_id=t,orig_init_state_idx=i,
                accepted=True,attempt=2,run_id='synthetic',step_idx=5*step,
                hit_type='FULL_HIT' if hit else 'MISS',searched=not hit,
                winner_id=('osplug:blind:' if hit else 'osplug:synth:')+name+'_1:0:'+str(step),
                factor_outputs=dict(osplug=dict(os_hit=hit,os_judge='blind' if hit else 'force:62'))))
    jl(run/'client/per_step.jsonl',receipts)
    path=run/'server_synthetic/decisions_0.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    kept=[r for r in rows if r.get('task_id')!=0];jl(path,kept)
    summary=json.loads((run/'summary.json').read_bytes());ledger=summary['cost_ledger']
    ledger.update(decisions=540,vision_decisions=270,misses=270,controls=2700);c.write_json(run/'summary.json',summary)
    actual=load_completed(seal,name,pairs,'B',1,a.Reader())
    assert actual.audit['historical_log_gap']['episodes']==30
    for field in ('y','n','v','m','cost'):assert np.array_equal(getattr(actual,field),getattr(baseline,field))
    # A hole inside a retained episode is never filled by an invented decision.
    jl(path,[r for r in kept if not (r.get('task_id')==1 and r.get('init')==20 and r.get('step')==0)])
    with pytest.raises(ValueError,match='Partial historical'):load_completed(seal,name,pairs,'B',1,a.Reader())


def test_detached_agent_lease_expires_and_rejects_old_controller(tmp_path):
    path=tmp_path/'lease.json';c.write_json(path,dict(token='current',active=True,ts=100))
    assert c.lease_alive(path,'current',90,now=120)
    assert not c.lease_alive(path,'previous',90,now=120)
    assert not c.lease_alive(path,'current',90,now=191)
    c.write_json(path,dict(token='current',active=False,ts=120))
    assert not c.lease_alive(path,'current',90,now=121)


@pytest.mark.parametrize('interrupted',[False,True])
@pytest.mark.parametrize('dynamic',[False,True])
def test_coordinator_owned_lifecycle_and_lease_without_launches(tmp_path,monkeypatch,interrupted,dynamic):
    """All remote/GPU/process operations are replaced; exercise the real control flow."""
    from . import conductor_runner as runner
    from exp.offline_search.closed_loop.ops.h100 import local_server,control
    group=fixture_group(tmp_path)
    for entry in group['entries']:
        entry['row'].update(cell='pi05_l10_cache',method='synthetic:Method',kwargs={},
                            plugin_args=['--os-fit-artifact','/synthetic/fit'],yaml='/synthetic/yaml')
    monkeypatch.setattr(runner,'read_group',lambda path:group)
    operations={'remote':[],'start':[],'stop':[]}
    from exp.offline_search.rounds.r17.autoscale_1017 import legacy_release, switch
    monkeypatch.setattr(switch, 'enabled', lambda: dynamic)
    monkeypatch.delenv('OPENPI_AUTOSCALE_REMOTE', raising=False)
    monkeypatch.setenv('OPENPI_AUTOSCALE_REMOTE', '')
    staged=[]
    monkeypatch.setattr(legacy_release, 'stage', lambda *a, **kw: staged.append((a,kw)) or dict(remote='/sealed/new_ops'))
    monkeypatch.setattr(control,'remote',lambda host,argv,**kw:operations['remote'].append(argv) or '')
    monkeypatch.setattr(local_server,'choose_ports',lambda **kw:[23170,23171,23172])
    monkeypatch.setattr(local_server,'status',lambda spec:dict(running=True,listening=True,dead=False))
    monkeypatch.setattr(local_server,'start',lambda spec:operations['start'].append(spec.copy()) or 'OWNED')
    monkeypatch.setattr(local_server,'stop',lambda spec:operations['stop'].append(spec.copy()) or 'OWNED_STOP')
    args=NS(group=tmp_path/'group.json',workers=4,port=23179,ports=None,smoke_episodes=None,
            agent_island='/scratch/zixuans8/openpi_r16_conductor_synthetic')
    def finished(args):c.write_json(tmp_path/'DONE.json',dict(group_sha256=group['group_sha256']))
    if interrupted:
        def interrupt(args):raise KeyboardInterrupt
        monkeypatch.setattr(runner,'run_driver',interrupt)
        with pytest.raises(KeyboardInterrupt):runner._run(args)
        assert not operations['stop'] and not (tmp_path/'DONE.json').exists()
        assert json.loads((tmp_path/'runner_state.json').read_bytes())['generation']==1
    monkeypatch.setattr(runner,'run_driver',finished)
    original_command=local_server.command
    runner._run(args)
    assert local_server.command is original_command
    assert len(operations['stop'])==3 and (tmp_path/'DONE.json').exists()
    for argv in operations['remote']:
        if argv[:2]==['bash','-c']:
            assert 'nohup env' in argv[2] and '--lease-file' in argv[2] and '--lease-token' in argv[2]
            assert str(args.agent_island) in argv[2]
            if dynamic:
                assert 'OPENPI_AUTOSCALE_CAMPAIGN='+group['group_sha256'] in argv[2]
                assert '/autoscale' in argv[2] and '/sealed/new_ops' in argv[2]
            else:
                assert 'OPENPI_AUTOSCALE' not in argv[2] and '/sealed/new_ops' not in argv[2]
    assert bool(staged) is dynamic
    assert any('active=False' in str(argv) for argv in operations['remote'])
    if interrupted:
        assert operations['start'][:3]==operations['start'][3:] # surviving launch receipts stay identical
    before=(len(operations['start']),len(operations['remote']))
    runner._run(args) # recover a crash after DONE but before owned-server cleanup
    assert before==(len(operations['start']),len(operations['remote'])) and len(operations['stop'])==6
