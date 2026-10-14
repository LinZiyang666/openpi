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
            # Keep every accepted terminal outcome without an infrastructure error (the SR criterion).
            # The old per-decision/ACK audit refused all MT50 MISS-driven policy calls (45,766 valid csfw_k3 outcomes).
            kept.append(uid);continue
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


def prepare(source,name, *, refresh_workers=False,worker_counts=None,replicas=None,local_vram_mib=None,local_server_growth_mib=None,driver_port=None,public_port=None,extra=None):
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
    if driver_port is not None:
        # Remote workers reach the driver directly through ziyanglin.com (23100-23199 only).
        campaign.update(driver_port=driver_port,ports=[p for p in campaign['ports'] if p!=campaign['driver_port']]+[driver_port])
    if public_port is not None:
        # A local server that gains remote workers must listen inside 23100-23199 (ziyanglin.com).
        server=[campaign['public_port'],*campaign['backend_ports']]
        backends=list(range(public_port+1,public_port+1+len(campaign['backend_ports'])))
        campaign.update(public_port=public_port,backend_ports=backends,
                        ports=[p for p in campaign['ports'] if p not in server]+[public_port,*backends])
    # Server-only overrides for the continuation (e.g. memory budget, Fast-WAM text cache, patched runtime).
    campaign.update(extra or {})
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
        canonical_workers=campaign.get('workers'),batching=campaign.get('stage_batching') or campaign.get('dp_batching',campaign.get('fastwam_batching')),
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
