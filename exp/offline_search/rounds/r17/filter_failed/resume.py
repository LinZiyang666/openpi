"""Build a native cache continuation or choose an unstaged execution topology.

cache_resume.prepare owns episode selection, inherited evidence and full/segment
plans. Its ordinary clone also provisions remote nodes, so substitute only the
local file-copy part here. filter_failed.stage later provisions the new paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
from unittest.mock import patch

from .audit_counts import HERE,read
from .build import write
from . import timan1_gpu2


@timan1_gpu2.coordinator()
def local_clone(snapshot,name,failure,*,worker_counts=None,campaign_overrides=None,**options):
    from exp.r17_integration.ops import formal_eval,queue_retry
    from exp.r17_integration.ops.worker_topology import redistribute
    if any(options.values()):raise ValueError('Only the existing worker-count override is supported')
    campaign,_=queue_retry.validate_inputs(snapshot)
    prefix='ffk1015_' if campaign.get('filter_failed_knob') else 'ff1015_'
    prefixes=(prefix,'fa_') if campaign.get('filter_failed_knob') else (prefix,)
    if not name.startswith(prefixes) or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError('A new '+prefix+' continuation name is required')
    root=formal_eval.FORMAL_ROOT/name;root.mkdir(exist_ok=False)
    seal=read(snapshot/'PREPARED.json')
    for relative in dict.fromkeys([*seal['files'],'runtime.tar','workers.tar']):
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(snapshot/relative,dest)
    campaign.update(run_name=name,prepare_only=True,local_assets=str(root/'assets'),
                    assets=str(root/'assets'),server_run=str(root/'servers'))
    if campaign['server_node']=='h100':
        campaign['server_campaign']='/tmp/r17_formal/'+name+'/server/campaign.json'
    campaign.update(campaign_overrides or {})
    if worker_counts is not None:redistribute(campaign,worker_counts,root)
    write(root/'campaign.json',campaign)
    return root


def prove_closed(source):
    """Reuse the native read-only ownership probe; never repair a FAIL receipt."""
    from exp.r17_integration.ops.library_salvage import prove_closed as probe
    from exp.r17_integration.ops.formal_eval import remote
    from exp.r17_integration.ops.immutable_release import sha
    def readonly(node, argv):
        return remote(node, argv, seconds=90)
    proof=probe(read(source/'RECEIPT.json'),read(source/'campaign.json'),readonly)
    proof.update(source=str(source),source_receipt_sha256=sha(source/'RECEIPT.json'),
                 source_campaign_sha256=sha(source/'campaign.json'),
                 checked_at=datetime.now(timezone.utc).isoformat())
    return proof


def infra_error_selection(source, *, include_missing=False):
    """Select errors (optionally every missing UID) from terminal outcomes only.

    The parent may still be running. Its closure is checked by normal staging
    before this continuation can be prepared, rather than asserted at build.
    """
    from exp.offline_search.rounds.r17.cache.analyze_cache import Sources, infrastructure_error, resume_segments
    from exp.r17_integration.ops.immutable_release import sha
    plan=read(source/('FULL_EPISODE_PLAN.json' if (source/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json'))
    tasks=[t['task_uid'] for t in plan['tasks']]
    terminals={}
    # Every segment of the continuation chain counts (analysis reads them all,
    # resume_segments); only the direct parent's journal re-ran ancestors' work.
    sources=Sources()
    for run,_ in resume_segments(source,sources):
        for r in sources.rows(run/'driver/journal.jsonl'):
            if r.get('accepted') is True and r.get('status') in ('done','failed'):
                uid=r['task_uid']
                if uid in terminals and terminals[uid].get('error') is None and not infrastructure_error(terminals[uid]):
                    continue
                terminals[uid]=r
    errors={u:r for u,r in terminals.items() if infrastructure_error(r)}
    missing=(set(tasks)-{u for u,r in terminals.items() if r.get('error') is None}
             if include_missing else set(errors))
    if not missing or not missing<=set(tasks):raise ValueError('No planned missing/error UIDs')
    # Mutable parent outcome files cannot be pinned while it is appending.
    # Its sealed inputs and the explicit error rows remain immutable evidence.
    snapshot=source/'.startup_inputs'
    pins={str(snapshot/n):pin for n,pin in read(snapshot/'STARTUP_INPUTS.json')['files'].items()}
    plan_path=source/('FULL_EPISODE_PLAN.json' if (source/'FULL_EPISODE_PLAN.json').exists() else 'EPISODE_PLAN.json')
    pins[str(plan_path)]=sha(plan_path)
    return dict(schema='r17.cache_resume.v1',source=str(source),source_pins=pins,
        retained_task_uids=[u for u in tasks if u not in missing],
        missing_task_uids=[u for u in tasks if u in missing],
        dropped=[dict(task_uid=u,reason='terminal infrastructure error' if u in terminals else 'missing accepted terminal outcome',
                      terminal=[terminals[u]] if u in terminals else []) for u in tasks if u in missing],
        expected_episodes=len(tasks),success_selection=False,infra_errors_only=not include_missing,
        refresh_after_source_closure=include_missing,
        source_closure_required_at_prepare=True)


def build(source,name,worker_counts=None,*,verify_closed=False,infra_errors=False,remaining=False):
    from exp.r17_integration.ops import cache_resume,queue_retry
    from exp.r17_integration.ops.immutable_release import sha
    source=Path(source).resolve(strict=True)
    receipt=read(source/'RECEIPT.json')
    if not (infra_errors or remaining) and receipt.get('status') not in ('FAIL','PARTIAL'):
        raise ValueError('Continuation requires a closed partial source')
    closure=prove_closed(source) if verify_closed else None
    if not (infra_errors or remaining) and receipt.get('all_owned_closed') is not True and closure is None:
        raise ValueError('Continuation requires a closed partial source; --verify-closed probes original owners read-only')
    from contextlib import nullcontext
    selection_context=(patch.object(cache_resume,'select',return_value=infra_error_selection(source,include_missing=remaining))
                       if infra_errors or remaining else nullcontext())
    with patch.object(queue_retry,'clone',local_clone),selection_context:
        root=cache_resume.prepare(source,name,worker_counts=worker_counts)
    # Native prepare seals local files. Remote deployment is deliberately not
    # claimed; normal stage must take its fresh-prepare branch for this run.
    (root/'PREPARED.json').rename(root/'LOCAL_PLAN_SEAL.json')
    selection=read(root/'CACHE_RESUME.json')
    assert sha(root/'runtime.tar')==sha(source/'runtime.tar')
    assert sha(root/'workers.tar')==sha(source/'workers.tar')
    campaign=read(root/'campaign.json');original=read(source/'campaign.json')
    key='filter_failed_knob' if original.get('filter_failed_knob') else 'filter_failed'
    assert campaign[key]==original[key]
    if closure:
        write(root/'SOURCE_CLOSURE.json',closure)
        selection['source_closure_proof']=closure
        campaign['cache_resume_selection']=selection
        write(root/'CACHE_RESUME.json',selection);write(root/'campaign.json',campaign)
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    apply(campaign, root)
    write(root/'campaign.json', campaign)
    if key=='filter_failed_knob':
        job=read(source/'FFK_DEFINITION.json')
        job.update(name=name,run=str(root),episodes=len(selection['missing_task_uids']),
                   workers=campaign.get('execution_worker_nodes',campaign['worker_nodes']),
                   continuation_source=str(source))
        write(root/'FFK_DEFINITION.json',job)
    write(root/'CONTINUATION_LOCAL_ONLY.json',dict(source=str(source),run=str(root),
        retained=len(selection['retained_task_uids']),missing=len(selection['missing_task_uids']),
        expected=selection['expected_episodes'],remote_staged=False,server_started=False,workers_started=False))
    index=HERE/'CONTINUATIONS.json'
    records=read(index)['jobs'] if index.exists() else []
    records=[r for r in records if r['run']!=str(root)]
    records.append(dict(source=str(source),run=str(root),name=name))
    from exp.r17_integration.ops.formal_process import write_json
    write_json(index,dict(jobs=records))
    print(str(root))
    return root


def refresh_remaining(root, closure):
    """Finalize a live-source draft once, using native cache_resume.prepare."""
    from exp.r17_integration.ops import cache_resume,queue_retry
    from exp.r17_integration.ops.immutable_release import sha
    root=Path(root);campaign=read(root/'campaign.json')
    if not campaign['cache_resume_selection'].get('refresh_after_source_closure'):return campaign
    source=Path(campaign['cache_resume_selection']['source'])
    selection=infra_error_selection(source,include_missing=True)
    selection.update(refresh_after_source_closure=False,source_closure_proof=closure)
    for name in ('RECEIPT.json','driver/journal.jsonl'):
        selection['source_pins'][str(source/name)]=sha(source/name)
    suffix='.before_closure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for name in ('EPISODE_PLAN.json','CACHE_RESUME.json','CACHE_RESUME_QUARANTINE.json','LOCAL_PLAN_SEAL.json'):
        p=root/name
        if p.exists():p.rename(p.with_name(p.name+suffix))
    # The immutable clone already exists. Reuse native selection/plan/seal
    # construction without a second clone or any remote provisioning.
    with patch.object(queue_retry,'clone',return_value=root),patch.object(cache_resume,'select',return_value=selection):
        cache_resume.prepare(source,root.name)
    (root/'PREPARED.json').rename(root/'LOCAL_PLAN_SEAL.json')
    job=root/'FFK_DEFINITION.json'
    if job.exists():write(job,dict(read(job),episodes=len(selection['missing_task_uids'])))
    write(root/'CONTINUATION_LOCAL_ONLY.json',dict(source=str(source),run=str(root),
        retained=len(selection['retained_task_uids']),missing=len(selection['missing_task_uids']),
        expected=selection['expected_episodes'],remote_staged=False,server_started=False,workers_started=False))
    return read(root/'campaign.json')


@timan1_gpu2.coordinator()
def topology(root,counts):
    """Use the native knob on a continuation or a fresh, unstaged ffk draft.

    Canonical workers/batch treatment stay sealed; only the recorded execution
    map changes. Prepared or launched runs must never take this path.
    """
    from exp.r17_integration.ops.worker_topology import redistribute
    root=Path(root).resolve(strict=True)
    campaign=read(root/'campaign.json')
    fresh_knob=(campaign['run_name'].startswith(('ffk1015_','fa_')) and
                campaign.get('filter_failed_knob',{}).get('method')=='frozen_fit_candidate_rows_only' and
                (root/'OFFLINE_READY.json').exists())
    fresh_knob=fresh_knob or (campaign['run_name'].startswith('ff1015_') and
                campaign.get('filter_failed',{}).get('method')=='frozen_fit_candidate_rows_only' and
                (root/'OFFLINE_READY.json').exists())
    if (not ((root/'CONTINUATION_LOCAL_ONLY.json').exists() or fresh_knob) or any(
            (root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))):
        raise ValueError('Choose topology only on an unstaged continuation or fresh ffk draft')
    redistribute(campaign,counts,root)
    write(root/'campaign.json',campaign)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--run',type=Path,help='Configure an unstaged continuation or fresh ffk draft')
    parser.add_argument('--worker-counts',help='Existing execution-topology knob, e.g. timan1=16,timan107=45')
    parser.add_argument('--verify-closed',action='store_true',help='Read-only local/remote ownership checks when the failed receipt could not prove closure')
    parser.add_argument('--infra-errors',action='store_true',help='Build only accepted retry-exhaustion error UIDs; parent closure is required at prepare')
    parser.add_argument('--remaining',action='store_true',help='Draft every unaccepted/error UID; refresh the plan once after parent closure at prepare')
    args=parser.parse_args()
    if args.run:
        if not args.worker_counts:parser.error('--run requires an explicit --worker-counts choice')
        topology(args.run,args.worker_counts)
    else:
        if not args.source or not args.name:parser.error('building requires --source and --name')
        build(args.source,args.name,args.worker_counts,verify_closed=args.verify_closed,infra_errors=args.infra_errors,remaining=args.remaining)
