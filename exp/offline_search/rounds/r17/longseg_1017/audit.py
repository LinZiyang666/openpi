"""Stream R17 decision outcomes; a new head at the midpoint proves a cut."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

try:
    from ujson import loads
except ImportError:
    loads = json.loads

HERE=Path(__file__).resolve().parent
EVAL=Path('/data/mw_finetune/eval')
PROTECTED={'fa_co_r_u_l','fa_pi_r_u_l'}

def classify(r):
    r['cut_affected']=r.get('cuts',0)>0
    r['clock_coupled_without_cuts']=not r['cut_affected'] and r.get('model') in ('dp','fastwam','groot') and any(
        g.rsplit('_',1)[-1] in ('ch','chr') and v.get('eligible_heads',0)>0 for g,v in r.get('groups',{}).items())
    r['affected']=r['cut_affected'] or r['clock_coupled_without_cuts']
    r['withdrawal_reason']='midpoint cuts' if r['cut_affected'] else ('delivery-count recipe clock on long ch/chr heads' if r['clock_coupled_without_cuts'] else None)
    return r


def io_ready(limit=20.):
    """Owner 10-18: no blocking I/O gates; reads proceed (rsync keeps its bwlimit)."""
    return
    waits=0
    while True:
        full=next(s for s in Path('/proc/pressure/io').read_text().splitlines() if s.startswith('full '))
        avg=float(dict(v.split('=') for v in full.split()[1:])['avg10'])
        if avg<=limit:return
        if waits%6==0:print(f'IO_PAUSE full_avg10={avg:.2f} > {limit}',flush=True)
        time.sleep(5);waits+=1


def scan(path, *, max_bytes=16*1024*1024, max_lines=16000, rate_mib=4):
    path=Path(path);root=path.parent.parent
    data_disk=path.is_relative_to(EVAL)
    if data_disk:io_ready()
    campaign=json.loads((root/'campaign.json').read_text())
    # Each accepted episode's decisions are written together by formal_driver.
    # Keep the latest decision by attempt identity as well, so interleaving or
    # duplicate episode reports cannot manufacture an adjacent pair.
    previous={};counts=Counter();by_group=defaultdict(Counter);examples=[]
    before=path.stat();read_bytes=0;start=time.monotonic();checkpoint=0;samples=[]
    windows=1 if before.st_size<=max_bytes else 8
    width=max_bytes//windows;line_limit=max_lines//windows
    offsets=[round(i*(max(0,before.st_size-width))/(windows-1)) for i in range(windows)] if windows>1 else [0]
    def sampled(stream):
        nonlocal read_bytes,checkpoint
        for offset in offsets:
            if data_disk:io_ready()
            stream.seek(offset)
            if offset:stream.readline()  # incomplete first line is not evidence
            sample=dict(offset=stream.tell(),bytes=0,rows=0);samples.append(sample)
            while sample['rows']<line_limit:
                line=stream.readline()
                if not line or sample['bytes']+len(line)>width:break
                sample['rows']+=1;sample['bytes']+=len(line);read_bytes+=len(line)
                if data_disk and read_bytes-checkpoint>=1024*1024:
                    elapsed=time.monotonic()-start
                    delay=read_bytes/(rate_mib*1024*1024)-elapsed
                    if delay>0:time.sleep(min(delay,.25))
                    io_ready();checkpoint=read_bytes
                yield line
    with path.open('rb',buffering=64*1024) as stream:
        for line in sampled(stream):
            try:r=loads(line)
            except ValueError:
                counts['malformed_lines']+=1;continue
            counts['rows']+=1
            if r.get('accepted') is not True:continue
            uid=r.get('task_uid');seq=r.get('decision_seq');step=r.get('step_idx')
            if uid is None or seq is None or step is None:continue
            key=(uid,r.get('attempt'),r.get('run_id'))
            source=r.get('source','');cursor=r.get('cursor');commit=r.get('commit_horizon',0)
            group=r.get('arm',r.get('yaml_id','unknown'))
            old=previous.get(key)
            if old is not None:
                oq,ost,os,oc,oh,og=old
                # Only heads with a recorded successor are eligible. Episode
                # ends and incomplete/truncated logs are never called cuts.
                if seq==oq+1 and oc==0 and oh>0 and step>ost and step-ost<oh:
                    counts['eligible_heads']+=1;by_group[og]['eligible_heads']+=1
                    if source==os+'_tail' and cursor==step-ost:
                        counts['tails']+=1;by_group[og]['tails']+=1
                    elif cursor==0 and not source.endswith('_tail'):
                        counts['cuts']+=1;by_group[og]['cuts']+=1
                        if len(examples)<8:examples.append(dict(task_uid=uid,head_seq=oq,
                            next_seq=seq,head_step=ost,next_step=step,head_source=os,next_source=source,
                            commit_horizon=oh,midpoint=step-ost,group=og))
                    else:
                        counts['other_successors']+=1;by_group[og]['other_successors']+=1
            previous[key]=(seq,step,source,cursor,commit,group)
    after=path.stat()
    def stats(c):
        out=dict(c);out['cut_percent']=100*c['cuts']/c['eligible_heads'] if c['eligible_heads'] else 0.
        return out
    return dict(run=root.name,model=campaign.get('model'),domain=campaign.get('domain'),
        server_node=campaign.get('server_node'),file=str(path),bytes=before.st_size,
        sample_bytes=read_bytes,sample_rows=counts['rows'],sample_limit_bytes=max_bytes,
        sample_limit_lines=max_lines,scan_kind='bounded_windows',samples=samples,
        inode=before.st_ino,mtime_ns=before.st_mtime_ns,live=root.name in PROTECTED,
        changed_during_scan=(before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns),
        episodes=len(previous),affected=counts['cuts']>0,**stats(counts),
        groups={k:stats(v) for k,v in sorted(by_group.items())},examples=examples)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--max-mib',type=int,default=16);p.add_argument('--max-lines',type=int,default=16000)
    p.add_argument('--rate-mib',type=float,default=4);p.add_argument('--pause',type=float,default=5)
    p.add_argument('--run',action='append');args=p.parse_args()
    if args.workers!=1:p.error('Exactly one reader is authorized')
    out=HERE/'audit';out.mkdir(exist_ok=True)
    paths=[p for p in sorted(EVAL.glob('*/driver/per_step.jsonl')) if (p.parent.parent/'campaign.json').is_file()]
    if args.run:paths=[p for p in paths if p.parent.parent.name in args.run]
    pending=[];results=[]
    for path in paths:
        cached=out/(path.parent.parent.name+'.json')
        if cached.exists():
            r=json.loads(cached.read_text());s=path.stat()
            if r['bytes']==s.st_size and r['mtime_ns']==s.st_mtime_ns and (r.get('affected') or r.get('scan_kind')!='bounded_prefix'):
                r.setdefault('scan_kind','full_completed_before_IO_correction')
                r.setdefault('sample_bytes',r['bytes']);r.setdefault('sample_rows',r.get('rows',0))
                results.append(r);continue
        pending.append(path)
    print(f'{len(paths)} files; reuse {len(results)} completed audits; one reader, <= {args.max_mib} MiB/file at {args.rate_mib} MiB/s',flush=True)
    start=time.monotonic()
    for path in pending:
        from .io_budget import reader
        with reader():
            r=scan(path,max_bytes=args.max_mib*1024*1024,max_lines=args.max_lines,rate_mib=args.rate_mib)
        results.append(r)
        (out/(r['run']+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        print(f'{r["run"]}: {r.get("cuts",0)}/{r.get("eligible_heads",0)} cuts; sample {r["sample_rows"]} rows/{r["sample_bytes"]/2**20:.1f} MiB; {time.monotonic()-start:.0f}s',flush=True)
        time.sleep(args.pause)
    existing={r['run'] for r in results}
    for path in sorted(EVAL.glob('*/campaign.json')):
        if path.parent.name not in existing:
            results.append(dict(run=path.parent.name,affected=False,status='no_per_step_outcomes'))
    results=[classify(r) for r in results];results.sort(key=lambda r:r['run'])
    report=dict(schema='r17.long_segment_audit.v1',created_at=datetime.now(timezone.utc).isoformat(),
        criterion='Sampled accepted head whose consecutive next delivery is a new head before commit_horizon; terminal heads excluded; zero cuts means none observed in sample',
        rows=results,affected=[r['run'] for r in results if r['affected']])
    (HERE/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'AUDIT.tsv').write_text('run\tmodel\tbenchmark\tsample_rows\tsample_MiB\teligible_heads\tcuts\tcut_percent\taffected\tscan_kind\n'+''.join(
        f'{r["run"]}\t{r.get("model")}\t{r.get("domain")}\t{r.get("sample_rows",0)}\t{r.get("sample_bytes",0)/2**20:.2f}\t{r.get("eligible_heads",0)}\t{r.get("cuts",0)}\t{r.get("cut_percent",0):.4f}\t{r["affected"]}\t{r.get("scan_kind",r.get("status"))}\n' for r in results))


if __name__=='__main__':main()
