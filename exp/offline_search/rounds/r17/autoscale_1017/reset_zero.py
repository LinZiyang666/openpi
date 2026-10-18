"""Owner-invoked retry of a closed failed campaign with no accepted episodes.

The entire original directory is renamed to an explicit backup. Scientific
inputs are copied serially into a fresh directory at the original pathname.
No process is stopped, no file is deleted, and this module never launches.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .compare import wait_io
from .node import atomic, read

SCIENCE=('campaign.json','EPISODE_PLAN.json','assets','runtime.tar','workers.tar',
         'RUNTIME_RELEASE.json','STUDY.json','FFK_DEFINITION.json','OFFLINE_READY.json',
         'longseg_server_assets','control_inputs')


def reset(root, backup, *, remote_backup=None, remote_stage_backup=None):
    root,backup=Path(root).resolve(),Path(backup).resolve()
    if backup.parent!=root.parent or backup==root or backup.exists():
        raise ValueError('backup must be a new explicit sibling directory')
    with (root/'.prepared.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=read(root/'RECEIPT.json',{})
        if receipt.get('status')!='FAIL' or receipt.get('all_owned_closed') is not True:
            raise ValueError('requires FAIL receipt and all_owned_closed=true')
        journal=root/'driver/journal.jsonl'
        pre_driver = (not journal.exists() and receipt.get('owned_local') == [] and
                      receipt.get('owned_remote') == [] and receipt.get('backend_evidence') == {})
        if not pre_driver and (not journal.is_file() or journal.stat().st_size):
            raise ValueError('requires an existing empty accepted journal; use continuation for nonzero progress')
        if (root/'DONE.json').exists():raise ValueError('completed campaign cannot reset')
        campaign=read(root/'campaign.json',{})
        if campaign.get('resume_from') or campaign.get('cache_resume_selection'):
            raise ValueError('fixed/source-pinned continuations cannot use zero-progress reset')
        server_root=Path(campaign.get('server_run',str(root/'servers'))).parent
        remote_needed=campaign.get('server_node')=='h100'
        if remote_needed and (server_root!=root or remote_backup is None):
            raise ValueError('H100 reset requires --remote-backup and a campaign-owned server_run')
        if remote_backup is not None and not remote_needed:
            raise ValueError('remote backup requested for a non-H100 campaign')
        if remote_stage_backup is not None and not remote_needed:
            raise ValueError('remote stage backup requested for a non-H100 campaign')
        archives=[]
        if remote_needed:
            remote_backup=Path(remote_backup)
            if (remote_backup.parent!=server_root.parent or
                not remote_backup.name.startswith(root.name+'.')):
                raise ValueError('remote backup must be an explicit sibling with the campaign name prefix')
            server_stage=Path('/tmp/r17_formal')/root.name
            stage_backup=Path(remote_stage_backup) if remote_stage_backup else server_stage.with_name(remote_backup.name)
            if stage_backup.parent!=server_stage.parent or not stage_backup.name.startswith(root.name+'.'):
                raise ValueError('remote stage backup must be an explicit sibling with the campaign name prefix')
            archives=[('h100',server_root,remote_backup),('h100',server_stage,stage_backup)]
        # A zero-progress retry keeps the same name. Native spawn and fleet main
        # both require fresh directories, so retain all prior node evidence too.
        suffix=backup.name[len(root.name):] if backup.name.startswith(root.name+'.') else '.'+backup.name
        deployment=read(root/'LAUNCH_WORKER_DEPLOYMENT.json',read(root/'WORKER_DEPLOYMENT.json',{}))
        for node,item in deployment.items():
            if node not in ('local','timan1','timan107'):
                raise ValueError('unsupported worker node; never reset timan108')
            stage_parent=Path('/scratch/zixuans8/mw_r17/formal/stage' if node=='timan1' else '/tmp/r17_formal')
            expected_stage=stage_parent/root.name/node
            worker_parent=Path('/data/mw_finetune/qualification/local_workers/runs' if node=='local'
                               else '/scratch/zixuans8/mw_r17/formal/runs')
            expected_run=worker_parent/(root.name+'_'+node)
            if item.get('stage')!=str(expected_stage) or item.get('worker_run')!=str(expected_run):
                raise ValueError('worker paths do not match this campaign; inspect manually')
            for owned_root in (expected_stage.parent,expected_run):
                archives.append((node,owned_root,owned_root.with_name(owned_root.name+suffix)))
        if archives:
            from .reset_remote import run as archive_remote
            # Check EVERY node before renaming any tree. Each rename rechecks
            # ownership; interrupted multi-node resets replay the archive markers.
            for args in archives:archive_remote(*args,check_only=True)
        wait_io();plan=(root/'EPISODE_PLAN.json').read_bytes()
        if not json.loads(plan).get('tasks'):raise ValueError('nonempty frozen episode plan required')
        digest=hashlib.sha256(plan).hexdigest()
        staging=root.with_name(root.name+'.retry_build_'+uuid.uuid4().hex)
        staging.mkdir()
        def copy(source,target):
            source=Path(source);relative=source.relative_to(root)
            wait_io()
            # Sol's frozen numeric/view inputs are deliberately hard-linked.
            # Keep those inodes; mutable metadata remains independent of backup.
            if 'cache_inputs' in relative.parts and source.name != 'manifest.json':
                os.link(source,target)
            else:
                from exp.offline_search.rounds.r17.longseg_1017.io_budget import copy as bounded_copy
                bounded_copy(source,target)
                shutil.copystat(source,target)
            return str(target)
        for name in SCIENCE:
            source=root/name
            if source.is_dir():shutil.copytree(source,staging/name,symlinks=True,copy_function=copy)
            elif source.exists():copy(source,staging/name)
        if (staging/'EPISODE_PLAN.json').read_bytes()!=plan:
            raise ValueError('copied plan mismatch; original untouched, staging retained')
        proof=dict(status='RESET_ZERO',backup=str(backup),plan_sha256=digest,
                   accepted=0,all_owned_closed=True,launches=0,pre_driver_failure=pre_driver)
        if archives:
            proof['node_backups']=[]
            for args in archives:
                proof['node_backups'].append(dict(node=args[0],**archive_remote(*args)))
            if remote_needed:
                proof['remote_backup']=proof['node_backups'][0]
                proof['remote_stage_backup']=proof['node_backups'][1]
        atomic(staging/'RESET_ZERO_RECEIPT.json',proof)
        root.rename(backup)  # preserve all evidence, including LAUNCH/RECEIPT and logs
        staging.rename(root)
        return proof


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--reprepare',action='store_true')
    p.add_argument('--remote-backup',type=Path,help='Archive the entire CLOSED H100 campaign directory, including mps_control')
    p.add_argument('--remote-stage-backup',type=Path,
                   help='H100 /tmp/r17_formal sibling backup; defaults to the --remote-backup basename')
    from .formal_trial import counts,prepare
    p.add_argument('--worker-counts',type=counts)
    a=p.parse_args(argv)
    import inspect
    from openpi.conductor.worker import WorkerLoop
    from .switch import enabled
    if enabled() and '__generation_' not in inspect.getsource(WorkerLoop.__init__):
        raise ValueError('generation identity fix is not installed; no reset performed')
    proof=reset(a.run,a.backup,remote_backup=a.remote_backup,remote_stage_backup=a.remote_stage_backup)
    if a.reprepare:
        prepare(a.run,a.worker_counts,reprepare=True)
        wait_io()
        if hashlib.sha256((a.run/'EPISODE_PLAN.json').read_bytes()).hexdigest()!=proof['plan_sha256']:
            raise ValueError('reprepare changed the frozen plan; do not launch')
        proof['status']='RESET_PREPARED'
        atomic(a.run/'RESET_ZERO_RECEIPT.json',proof)
    print(json.dumps(proof,sort_keys=True))


if __name__=='__main__':main()
