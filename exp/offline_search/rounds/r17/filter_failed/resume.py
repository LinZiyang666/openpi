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


def local_clone(snapshot,name,failure,*,worker_counts=None,campaign_overrides=None,**options):
    from exp.r17_integration.ops import formal_eval,queue_retry
    from exp.r17_integration.ops.worker_topology import redistribute
    if any(options.values()):raise ValueError('Only the existing worker-count override is supported')
    campaign,_=queue_retry.validate_inputs(snapshot)
    prefix='ffk1015_' if campaign.get('filter_failed_knob') else 'ff1015_'
    if not name.startswith(prefix) or not all(c.isalnum() or c in '_-' for c in name):
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
    proof=probe(read(source/'RECEIPT.json'),read(source/'campaign.json'),remote)
    proof.update(source=str(source),source_receipt_sha256=sha(source/'RECEIPT.json'),
                 source_campaign_sha256=sha(source/'campaign.json'),
                 checked_at=datetime.now(timezone.utc).isoformat())
    return proof


def build(source,name,worker_counts=None,*,verify_closed=False):
    from exp.r17_integration.ops import cache_resume,queue_retry
    from exp.r17_integration.ops.immutable_release import sha
    source=Path(source).resolve(strict=True)
    receipt=read(source/'RECEIPT.json')
    if receipt.get('status') not in ('FAIL','PARTIAL'):
        raise ValueError('Continuation requires a closed partial source')
    closure=prove_closed(source) if verify_closed else None
    if receipt.get('all_owned_closed') is not True and closure is None:
        raise ValueError('Continuation requires a closed partial source; --verify-closed probes original owners read-only')
    with patch.object(queue_retry,'clone',local_clone):
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
    records.append(dict(source=str(source),run=str(root),name=name))
    write(index,dict(jobs=records))
    print(str(root))
    return root


def topology(root,counts):
    """Use the native knob on a continuation or a fresh, unstaged ffk draft.

    Canonical workers/batch treatment stay sealed; only the recorded execution
    map changes. Prepared or launched runs must never take this path.
    """
    from exp.r17_integration.ops.worker_topology import redistribute
    root=Path(root).resolve(strict=True)
    campaign=read(root/'campaign.json')
    fresh_knob=(campaign['run_name'].startswith('ffk1015_') and
                campaign.get('filter_failed_knob',{}).get('method')=='frozen_fit_candidate_rows_only' and
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
    args=parser.parse_args()
    if args.run:
        if not args.worker_counts:parser.error('--run requires an explicit --worker-counts choice')
        topology(args.run,args.worker_counts)
    else:
        if not args.source or not args.name:parser.error('building requires --source and --name')
        build(args.source,args.name,args.worker_counts,verify_closed=args.verify_closed)
