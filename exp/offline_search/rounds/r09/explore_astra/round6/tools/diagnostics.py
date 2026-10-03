"""Descriptive train coverage and local single-thread correction timing."""
import json
import time
from types import SimpleNamespace
import numpy as np
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from .data import HERE, RUN, CELLS, VARIANTS, dataset, combine, dump, artifact


def main():
    specs={a['arm']:a for a in json.loads((RUN/'arms.json').read_text())}
    result={}
    for cell in CELLS:
        d=combine([dataset(cell,v,'train') for v in VARIANTS])
        row=specs[f'r9a6_{cell}_taskfree']
        with open(artifact(row),'rb') as f:
            base=FitUnpickler(f).load()['method'].base
        mass=np.bincount(d['rows'].ravel(),weights=d['weights'].ravel(),minlength=len(base.act))
        eval_data=dataset(cell,'A','eval')
        q=SimpleNamespace(key_v0=np.zeros(base.B0T.shape[1],np.float32),
            key_v1=np.zeros(base.B1T.shape[1],np.float32),rs=np.zeros(8,np.float32),step=4)
        a=np.array(base.act[0],copy=True)
        rows,w=d['rows'][0],d['weights'][0]
        for _ in range(30): base._ar6_action(q,a,rows,w)
        samples=[]
        for _ in range(300):
            start=time.perf_counter_ns()
            base._ar6_action(q,a,rows,w)
            samples.append((time.perf_counter_ns()-start)/1e6)
        result[cell]=dict(train_anchors=len(d['init']),train_arm_episodes=len(np.unique(d['episode'])),
            library_rows=len(base.act),rows_with_labels=int((mass>0).sum()),
            eval_A_unseen_weight=float((eval_data['weights']*(mass[eval_data['rows']]==0)).sum(1).mean()),
            correction_latency_ms_median=float(np.median(samples)),
            correction_latency_ms_p95=float(np.quantile(samples,.95)),
            head_bytes=sum(v.nbytes for v in base.ar6_head.values()), table_bytes=base.ar6_table.nbytes,
            artifact_bytes=(HERE/'artifacts'/f'{cell}.npz').stat().st_size)
    dump(HERE/'results/diagnostics.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
