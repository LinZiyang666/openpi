"""Owner-only local reset of closed, zero-progress R15 arms; never contacts nodes.

Remote evidence stays in place. Native retry appends its log and uses a new
launch_id. Nothing is deleted and no process is signalled. The explicit owner
attestation is necessary because local failed runs may have no collected journal.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from .compare import wait_io
from .node import atomic, read
from .switch import enabled

SCIENCE=('ROOT.json','arms.json','arms_in.json','config','eval500.json')


def reset_many(roots,suffix,*,owner_confirmed_closed_zero=False):
    if enabled():raise ValueError('this recovery command requires the global switch OFF')
    if not owner_confirmed_closed_zero:raise ValueError('owner must attest all local/remote owned processes closed and accepted=0')
    if not suffix.startswith('.') or '/' in suffix:raise ValueError('backup suffix must start with dot and contain no slash')
    roots=[Path(p).resolve() for p in roots]
    backups=[p.with_name(p.name+suffix) for p in roots]
    if any(p.exists() for p in backups):raise ValueError('backup already exists; no changes made')
    staged=[]
    with ExitStack() as stack:
        for root in roots:
            if not root.is_dir() or not (root/'arms.json').is_file():raise ValueError('missing native arm definition: '+str(root))
            lockpath=root/'state/h100_chain.lock';lockpath.parent.mkdir(exist_ok=True)
            lock=stack.enter_context(lockpath.open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if list((root/'state').glob('*.DONE')):raise ValueError('completed arm present: '+str(root))
            # Reject any locally collected successful OR unsuccessful accepted episode.
            for journal in root.glob('runs/**/journal.jsonl'):
                with journal.open() as stream:
                    for line in stream:
                        if not line.strip():continue
                        row=json.loads(line)
                        if row.get('status') in ('done','failed') and row.get('accepted') is not False:
                            raise ValueError('accepted episode present: '+str(journal))
            owner=read(root/'state/h100_chain.owner.json',{})
            if owner.get('pid'):
                stat=Path('/proc')/str(owner['pid'])/'stat'
                if stat.exists() and stat.read_text().rsplit(')',1)[1].split()[19]==str(owner.get('starttime')):
                    raise ValueError('local coordinator is still alive: '+str(root))
        # Validate all eight before any rename; serial bounded copies only.
        for root,backup in zip(roots,backups):
            fresh=root.with_name(root.name+'.native_retry_'+uuid.uuid4().hex);fresh.mkdir()
            def copy(a,b):wait_io();return shutil.copy2(a,b)
            pins={}
            for name in SCIENCE:
                source=root/name
                if source.is_dir():shutil.copytree(source,fresh/name,symlinks=True,copy_function=copy)
                elif source.is_file():copy(source,fresh/name)
            for p in fresh.rglob('*'):
                if p.is_file():pins[str(p.relative_to(fresh))]=hashlib.sha256(p.read_bytes()).hexdigest()
            atomic(fresh/'RESET_ZERO_RECEIPT.json',dict(backup=str(backup),accepted=0,
                owner_confirmed_closed_zero=True,native_science_sha256=pins,remote_evidence='untouched',launches=0))
            staged.append((root,backup,fresh))
        for root,backup,fresh in staged:root.rename(backup);fresh.rename(root)
    return [dict(run=str(root),backup=str(backup)) for root,backup,_ in staged]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit',required=True,choices=['fa_pi_f_u_o'])
    p.add_argument('--backup-suffix',required=True)
    p.add_argument('--owner-confirmed-closed-zero',action='store_true')
    a=p.parse_args(argv)
    roots=[Path('/data/mw_finetune/eval')/(a.unit+'_'+suite+str(size)) for suite in ('s','l') for size in (50,100,200,500)]
    print(json.dumps(reset_many(roots,a.backup_suffix,owner_confirmed_closed_zero=a.owner_confirmed_closed_zero),indent=2))

if __name__=='__main__':main()
