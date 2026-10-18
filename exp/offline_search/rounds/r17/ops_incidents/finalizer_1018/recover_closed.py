"""CPU-only native recovery and both analysis-loader probes for closed runs.

One /data reader; pause on IO PSI, cap physical reads at 4 MiB/s. Memoize only
immutable driver evidence in RAM so the two native analyzers need no extra
disk scans. Every open rechecks device/inode/size/mtime/ctime; no row is skipped.
"""
import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from exp.offline_search.rounds.r17.longseg_1017.io_budget import filesystem_budget,dump
from exp.offline_search.rounds.r17.longseg_1017.audit import io_ready
from exp.r17_integration.ops.formal_finalize import retry_closed
from exp.offline_search.rounds.r17.filter_failed_knob.formal_a_analysis import load_r17
from exp.offline_search.rounds.r17.longseg_1017.analyze import lane_data

HERE=Path(__file__).resolve().parent


def signature(path):
    s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def physical_reads():
    with open('/proc/self/io') as stream:
        return int(next(line.split(':')[1] for line in stream if line.startswith('read_bytes:')))


@contextmanager
def closed_evidence_cache(roots):
    allowed=set();cache={};native=Path.open;read_bytes=0;disk_bytes=0
    for root in roots:
        r=json.loads((root/'RECEIPT.json').read_text())
        if r.get('all_owned_closed') is not True:raise ValueError('never cache a running campaign')
        allowed.update(root/'driver'/name for name in ('journal.jsonl','attempts.jsonl','per_step.jsonl'))
    def opened(path,mode='r',*args,**kw):
        nonlocal read_bytes,disk_bytes
        if path not in allowed or mode not in ('r','rb') or args:
            return native(path,mode,*args,**kw)
        now=signature(path)
        if path not in cache:
            blocks=[];fd=os.open(path,os.O_RDONLY)
            try:
                while True:
                    io_ready();start=time.monotonic();before=physical_reads()
                    block=os.read(fd,256*1024);cold=physical_reads()-before
                    disk_bytes+=cold
                    time.sleep(max(0,cold/2**22-(time.monotonic()-start)))
                    if not block:break
                    blocks.append(block);read_bytes+=len(block)
                    if len(blocks)%1024==0:
                        print('read',str(path),len(blocks)//4,'MiB; physical',disk_bytes//2**20,'MiB',flush=True)
            finally:os.close(fd)
            if signature(path)!=now:raise ValueError('closed driver evidence changed')
            cache[path]=(now,b''.join(blocks))
        old,data=cache[path]
        if old!=now:raise ValueError('closed evidence signature differs')
        stream=io.BytesIO(data);stream.name=str(path)
        return stream if 'b' in mode else io.TextIOWrapper(stream,encoding=kw.get('encoding') or 'utf-8',
            errors=kw.get('errors'),newline=kw.get('newline'))
    metrics={}
    with patch.object(Path,'open',opened):yield metrics
    metrics.update(files=len(cache),logical_bytes=read_bytes,physical_bytes=disk_bytes,
        physical_read_cap_mib_per_second=4,PSI_full_avg10_limit=20,all_closed_signatures_unchanged=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,nargs='+',required=True)
    args=p.parse_args();roots=set(args.run)
    with filesystem_budget():
        for root in list(roots):
            current=root
            while parent:=json.loads((current/'campaign.json').read_text()).get('cache_resume_selection',{}).get('source'):
                current=Path(parent);roots.add(current)
        proofs=[]
        with closed_evidence_cache(roots) as io_proof:
            for root in args.run:
                started=time.monotonic();print('RETRY',str(root),flush=True)
                receipt=retry_closed(root)
                if 'completed_episodes' not in receipt:
                    receipt['completed_episodes']=json.loads((root/'SUMMARY.json').read_text())['completed_episodes']
                print('SEALED',json.dumps(receipt),flush=True)
                formal=load_r17(dict(run=str(root)))
                longseg=lane_data(root)
                a={(s,n,l,k):v['success'] for (s,n),arms in formal.items() for l,data in arms.items() for k,v in data.items()}
                b={(s,n,l,k):v['success'] for (_,s,n),arms in longseg.items() for l,data in arms.items() for k,v in data.items()}
                assert a==b and len(a)==receipt['completed_episodes']
                proof=dict(run=str(root),receipt=receipt,status='PASS',formal_a_accepted=len(a),
                    longseg_accepted=len(b),same_unique_task_init_level_keys=True,
                    seconds=time.monotonic()-started,driver_returncode=json.loads((root/'RECEIPT.json').read_text())['driver_completion']['returncode'])
                proofs.append(proof);dump(HERE/'RECOVERY_PROOF.json',dict(rows=proofs,cpu_only=True))
                print('ANALYZERS',json.dumps(proof),flush=True)
        dump(HERE/'RECOVERY_PROOF.json',dict(status='PASS',rows=proofs,cpu_only=True,io=io_proof))


if __name__=='__main__':main()
