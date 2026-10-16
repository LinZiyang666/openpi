"""Build a native cache continuation locally; the coordinator stages/launches it.

cache_resume.prepare owns episode selection, inherited evidence and full/segment
plans. Its ordinary clone also provisions remote nodes, so substitute only the
local file-copy part here. filter_failed.stage later provisions the new paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from unittest.mock import patch

from .audit_counts import HERE,read
from .build import write


def local_clone(snapshot,name,failure,*,worker_counts=None,campaign_overrides=None,**options):
    from exp.r17_integration.ops import formal_eval,queue_retry
    from exp.r17_integration.ops.worker_topology import redistribute
    if any(options.values()):raise ValueError('Only the existing worker-count override is supported')
    if not name.startswith('ff1015_') or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError('A new ff1015_ continuation name is required')
    campaign,_=queue_retry.validate_inputs(snapshot)
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


def build(source,name,worker_counts=None):
    from exp.r17_integration.ops import cache_resume,queue_retry
    from exp.r17_integration.ops.immutable_release import sha
    source=Path(source).resolve(strict=True)
    receipt=read(source/'RECEIPT.json')
    if receipt.get('all_owned_closed') is not True or receipt.get('status') not in ('FAIL','PARTIAL'):
        raise ValueError('Continuation requires a closed partial source')
    with patch.object(queue_retry,'clone',local_clone):
        root=cache_resume.prepare(source,name,worker_counts=worker_counts)
    # Native prepare seals local files. Remote deployment is deliberately not
    # claimed; normal stage must take its fresh-prepare branch for this run.
    (root/'PREPARED.json').rename(root/'LOCAL_PLAN_SEAL.json')
    selection=read(root/'CACHE_RESUME.json')
    assert sha(root/'runtime.tar')==sha(source/'runtime.tar')
    assert sha(root/'workers.tar')==sha(source/'workers.tar')
    assert read(root/'campaign.json')['filter_failed']==read(source/'campaign.json')['filter_failed']
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
    """Apply an explicitly chosen native orchestration knob to this new draft."""
    from exp.r17_integration.ops.worker_topology import redistribute
    root=Path(root).resolve(strict=True)
    if not (root/'CONTINUATION_LOCAL_ONLY.json').exists() or any(
            (root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json')):
        raise ValueError('Choose topology only on the new, unstaged continuation draft')
    campaign=read(root/'campaign.json')
    redistribute(campaign,counts,root)
    write(root/'campaign.json',campaign)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--run',type=Path,help='Configure the new, unstaged draft rather than build another one')
    parser.add_argument('--worker-counts',help='Existing execution-topology knob, e.g. timan1=16,timan107=45')
    args=parser.parse_args()
    if args.run:
        if not args.worker_counts:parser.error('--run requires an explicit --worker-counts choice')
        topology(args.run,args.worker_counts)
    else:
        if not args.source or not args.name:parser.error('building requires --source and --name')
        build(args.source,args.name,args.worker_counts)
