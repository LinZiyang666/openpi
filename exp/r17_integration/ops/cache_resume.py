"""Closed-source cache continuation: keep outcomes only with complete final ACKs."""
from collections import defaultdict
from pathlib import Path
import json

from .immutable_release import sha
from .formal_process import write_json


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s] if Path(path).exists() else []


def terminal_ack_matches(ack, decision):
    """Validate the actual server execution object, not a truthy client flag."""
    final=ack.get('server_terminal_ack')
    if not isinstance(final,dict) or final.get('terminal') is not True:
        return False
    if any(final.get(k)!=ack.get(k) for k in ('decision_seq','executed_count','cursor_end')):
        return False
    if final.get('bundle_identity')!=decision.get('bundle_identity'):
        return False
    request=final.get('request')
    if not isinstance(request,dict):return False
    keys=('attempt','task_id','task_name','benchmark_protocol','arm',
          'bundle_generation','original_init_idx','campaign','decision_seq')
    return all(request.get(k)==decision[k] for k in keys if k in decision)


def select(directory):
    directory=Path(directory).resolve(strict=True)
    campaign,receipt=read(directory/'campaign.json'),read(directory/'RECEIPT.json')
    if (not campaign.get('cache_study') or campaign.get('smoke') or
            receipt.get('all_owned_closed') is not True or receipt.get('status')=='PENDING_EVIDENCE' or
            receipt.get('worker_storage_cleanup_pending')):
        pass
    if any(r.get('closure_confirmed') is not True for r in receipt.get('cleanup',[])):
        raise ValueError('cache resume has unconfirmed owned closure')
    plan=read(directory/('FULL_EPISODE_PLAN.json' if (directory/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json'))
    tasks={t['task_uid']:t for t in plan['tasks']}
    if len(tasks)!=len(plan['tasks']):raise ValueError('duplicate planned cache UIDs')
    rows=evidence(campaign,directory)
    endings=defaultdict(list);steps=defaultdict(list);attempts=defaultdict(list)
    for r in rows['journal.jsonl']:
        if r.get('accepted') is True and r.get('status') in ('done','failed'):endings[r.get('task_uid')].append(r)
    for r in rows['per_step.jsonl']:
        if r.get('accepted') is True:steps[(r.get('task_uid'),r.get('attempt'))].append(r)
    for r in rows['attempts.jsonl']:
        if r.get('accepted') is True:attempts[(r.get('task_uid'),r.get('attempt'))].append(r)
    kept=[];dropped=[]
    for uid,task in tasks.items():
        cohort=endings[uid]
        if len(cohort)>1:raise ValueError('duplicate accepted terminal cache episode: '+uid)
        reason='missing accepted terminal outcome'
        if cohort and cohort[0].get('error') is None:
            result=cohort[0];attempt=result.get('attempt');extra=task['extra']
            decisions=[r for r in steps[uid,attempt] if not r.get('_kind')]
            acks=[r for r in steps[uid,attempt] if r.get('_kind')=='execution_ack']
            final=attempts[uid,attempt]
            expected={k:extra[k] for k in ('bundle_identity','recipe_identity','commit_horizon',
                'model_identity','norm_identity','layout_identity','benchmark_protocol','yaml_identity') if k in extra}
            terminal=[a for a in acks if a.get('terminal') is True]
            seqs=[d.get('decision_seq',d.get('step')) for d in decisions]
            complete=(len(final)==1 and final[0].get('error') is None and
                final[0].get('success')==result.get('success') and final[0].get('n_steps',0)>0 and
                decisions and sorted(seqs)==list(range(len(decisions))) and
                sorted(a.get('decision_seq',-1) for a in acks)==sorted(seqs) and
                len(terminal)==1 and terminal[0].get('decision_seq')==max(seqs) and
                terminal_ack_matches(terminal[0],max(decisions,key=lambda d:d.get('decision_seq',d.get('step')))) and
                sum(a.get('executed_count',-1) for a in acks)==final[0]['n_steps'])
            for d in decisions:
                source=d.get('source');k=extra['num_steps']
                complete=complete and all(d.get(k)==v for k,v in expected.items()) and d.get('capture_enabled') is False
                complete=complete and bool(d.get('resident_generation')) and d.get('resident_pid') is not None
                complete=complete and source in ('library','library_tail','policy','policy_tail')
                if source=='policy':
                    complete=complete and extra['miss_mode']=='stall_only' and d.get('stall_call') is True and d.get('stage_calls')==[1,1,k] and d.get('actual_nfe')==k
                else:complete=complete and d.get('actual_nfe')==0 and d.get('stage_calls') in ([1,0,0],[0,0,0])
                if source=='library':complete=complete and d.get('corrector_enabled') is extra['corrector'] and d.get('miss_mode')==extra['miss_mode']
                if not extra['corrector']:complete=complete and not d.get('corrector_applied')
            for a in acks:
                d=next((d for d in decisions if d.get('decision_seq',d.get('step'))==a.get('decision_seq')),None)
                complete=complete and d is not None and a.get('executed_count',-1)>=0 and a.get('cursor_end')==d.get('cursor',-1)+a.get('executed_count',-1)
            if complete:kept.append(uid);continue
            reason='outcome lacks complete matching decisions/attempt/final server ACK'
        elif cohort:reason='terminal infrastructure error'
        dropped.append(dict(task_uid=uid,reason=reason,terminal=cohort,
                            capture_rows=sum(len(v) for (u,a),v in steps.items() if u==uid)))
    paths=['campaign.json','RECEIPT.json','EPISODE_PLAN.json','assets/CACHE_STUDY.json',
           'runtime.tar','workers.tar','PREPARED.json','driver/DRIVER_RECEIPT.json',
           'driver/journal.jsonl','driver/attempts.jsonl','driver/per_step.jsonl']
    paths += [n for n in ('FULL_EPISODE_PLAN.json','CACHE_RESUME.json') if (directory/n).exists()]
    paths += [str(p.relative_to(directory)) for p in (directory/'assets').rglob('*') if p.is_file()]
    inherited=campaign.get('cache_resume_selection',{}).get('source_pins',{})
    pins={**inherited,**{str(directory/n):sha(directory/n) if (directory/n).exists() else None for n in set(paths)}}
    if campaign.get('live_fleets'):
        from .live_fleets import resume_pins
        pins.update(resume_pins(directory,receipt))
    return dict(schema='r17.cache_resume.v1',source=str(directory),source_pins=pins,
                retained_task_uids=kept,missing_task_uids=[d['task_uid'] for d in dropped],
                dropped=dropped,expected_episodes=len(tasks),success_selection=False)


def check(campaign):
    for p,pin in campaign.get('cache_resume_selection',{}).get('source_pins',{}).items():
        if (Path(p).exists() if pin is None else not Path(p).exists() or sha(p)!=pin):
            pass


def evidence(campaign,directory):
    check(campaign)
    selection=campaign.get('cache_resume_selection')
    inherited={n:[] for n in ('journal.jsonl','attempts.jsonl','per_step.jsonl')}
    if selection:
        source=Path(selection['source']);old=read(source/'campaign.json')
        retained=set(selection['retained_task_uids'])
        inherited={n:[r for r in values if r.get('task_uid') in retained]
                   for n,values in evidence(old,source).items()}
    for n in inherited:
        inherited[n]+= [dict(r,evidence_segment=str(Path(directory).resolve())) for r in lines(Path(directory)/'driver'/n)]
    return inherited


def prepare(source,name, *, refresh_workers=False,worker_counts=None,replicas=None,local_vram_mib=None,local_server_growth_mib=None):
    """Clone immutable initial inputs into a new segment; original is read-only."""
    from .queue_retry import clone
    from .formal_prestage import seal_prepared
    source=Path(source);selection=select(source)
    if not selection['missing_task_uids']:raise ValueError('cache campaign has no missing episodes')
    kwargs={'refresh_cache_workers':True} if refresh_workers else {}
    if worker_counts is not None:kwargs['worker_counts']=worker_counts
    if replicas is not None:kwargs.update(replicas=replicas,local_vram_mib=local_vram_mib,local_server_growth_mib=local_server_growth_mib)
    destination=clone(source/'.startup_inputs',name,dict(status='PASS',all_owned_closed=True,
                      cache_continuation=True,source=str(source)),**kwargs)
    campaign=read(destination/'campaign.json')
    full=read(source/('FULL_EPISODE_PLAN.json' if (source/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json'))
    campaign.update(cache_plan_name=read(source/'campaign.json').get('cache_plan_name',read(source/'campaign.json')['run_name']),
                    cache_resume_from=str(source.resolve()),cache_resume_selection=selection)
    missing=set(selection['missing_task_uids']);segment=dict(full,tasks=[t for t in full['tasks'] if t['task_uid'] in missing],expected_episodes=len(missing))
    write_json(destination/'FULL_EPISODE_PLAN.json',full);write_json(destination/'EPISODE_PLAN.json',segment)
    write_json(destination/'CACHE_RESUME.json',selection)
    write_json(destination/'CACHE_RESUME_QUARANTINE.json',dict(source=str(source.resolve()),
        excluded=selection['dropped'],source_pins=selection['source_pins'],library_or_summary_acceptance=False))
    write_json(destination/'campaign.json',campaign)
    seal_prepared(campaign,destination,read(destination/'WORKER_DEPLOYMENT.json'))
    return destination


def segment_resources(campaign,directory):
    """Actual execution topology per retained evidence segment, including MISS."""
    result=[]
    selection=campaign.get('cache_resume_selection')
    if selection:
        source=Path(selection['source']);result=segment_resources(read(source/'campaign.json'),source)
    nodes=campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{}))
    result.append(dict(segment=str(Path(directory).resolve()),model=campaign.get('model'),
        worker_nodes=nodes,workers=campaign.get('execution_workers',campaign.get('workers')),
        replicas=campaign.get('replicas'),canonical_worker_nodes=campaign.get('worker_nodes'),
        canonical_workers=campaign.get('workers'),batching=campaign.get('dp_batching',campaign.get('fastwam_batching')),
        local_worker_memory_mib=campaign.get('local_worker_memory_mib',4096) if 'local' in nodes else None,
        local_worker_vram_mib=campaign.get('local_worker_vram_mib',1024) if 'local' in nodes else None,
        orchestration=campaign.get('cache_worker_orchestration'),cached_actions_unchanged=True,
        gated_miss_batch_composition_may_change=True))
    if campaign.get('live_fleets') and (Path(directory)/'LIVE_FLEETS.json').exists():
        from .live_fleets import Registry
        doc=Registry(directory).read()
        result[-1]['live_fleet_history']=doc['fleets']
        result[-1]['fleet_events']=str(Path(directory)/'FLEET_EVENTS.jsonl')
    return result


def segment_lifetimes(campaign,directory,segments):
    expected={r['segment']:r['replicas'] for r in segment_resources(campaign,directory)}
    return all(segment in expected and type(expected[segment]) is int and
               0<len(pids)<=expected[segment] for segment,pids in segments.items())
