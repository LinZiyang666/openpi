"""Emit eight direct-PCA arms, then fit at most four three-thread CPU workers."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, sources

PREFIX=['taskset','-c','18-25,62-69','env','OMP_NUM_THREADS=3','OPENBLAS_NUM_THREADS=3','MKL_NUM_THREADS=3',
        'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src','.venv/bin/python']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',action='store_true')
    p.add_argument('--parallel',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    assert 1<=a.parallel<=4
    rows=[]; provenance=[]; jobs=[]
    for (model,suite,scale,kind),(path,src) in sources().items():
        if kind!='A': continue
        r=copy.deepcopy(src)
        r['name']=f"r6p2_direct_{'p' if model=='pi05' else 'g'}_{'sp' if suite=='spatial' else suite}_{scale}"
        r['method']='exp.offline_search.rounds.r06.p2_ablations.token_pca:TokenPCAAWM'
        r['kwargs']['pooling_grid']=16
        r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1]=f"<RUN>/fits/{r['name']}.pkl"
        rows.append(r); provenance.append(dict(arm=r['name'],source=str(path),source_row=src))
        args=[x.replace('<RUN>',str(RUN)) for x in r['plugin_args']]
        cmd=PREFIX+['-m','exp.offline_search.closed_loop.plugin','--os-method',r['method'],'--os-kwargs',json.dumps(r['kwargs']),
                    '--os-cell',f'{model}_{suite}_cache','--os-log-dir',str(RUN/'pca_logs'/r['name']),'--os-tag',r['name']]
        if '--os-root' not in args: cmd+=['--os-root',STORE]
        jobs.append(dict(arm=r['name'],argv=cmd+args))
    assert len(rows)==8
    (HERE/'arms_pca.json').write_text(json.dumps(rows,indent=1)+'\n')
    (HERE/'results/pca/provenance.json').write_text(json.dumps(provenance,indent=1)+'\n')
    (HERE/'results/pca/prefit_commands.json').write_text(json.dumps(jobs,indent=1)+'\n')
    if not a.fit: return
    (RUN/'fits').mkdir(parents=True,exist_ok=True)
    (RUN/'pca_logs').mkdir(parents=True,exist_ok=True)
    pending=[]
    for job in jobs:
        if (RUN/'fits'/f"{job['arm']}.pkl").exists():
            if a.resume: continue
            raise FileExistsError(job['arm'])
        pending.append(job)
    running=[]; failed=[]; peak_rss=0
    while pending or running:
        while pending and len(running)<a.parallel:
            job=pending.pop(0); log=(RUN/'pca_logs'/f"{job['arm']}.log").open('w')
            running.append((job['arm'],subprocess.Popen(job['argv'],stdout=log,stderr=subprocess.STDOUT),log))
        total=0
        for name,proc,log in list(running):
            if proc.poll() is not None:
                log.close(); print(name,proc.returncode,flush=True)
                if proc.returncode: failed.append(name)
                running.remove((name,proc,log))
            else:
                try:
                    vals=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    total+=int(next(v for v in vals if v.startswith('VmRSS:')).split()[1])*1024
                except (OSError,StopIteration): pass
        peak_rss=max(peak_rss,total)
        if total>60*1024**3:
            # Only our explicit child PIDs; no shared-job process matching.
            for _,proc,_ in running: proc.terminate()
            raise MemoryError('P2 children exceeded 60 GiB RSS safety threshold')
        if running: time.sleep(2)
    (HERE/'results/pca/resources.json').write_text(json.dumps(dict(peak_children_rss_bytes=peak_rss,
         sample_period_s=2, workers=a.parallel,blas_threads_per_worker=3,coordinator_threads=1,failed=failed),indent=1)+'\n')
    assert not failed,failed


if __name__=='__main__': main()
