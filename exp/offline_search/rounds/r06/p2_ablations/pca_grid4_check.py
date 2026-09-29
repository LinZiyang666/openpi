"""Exact deployed-A control: compare every query payload field, including visual diagnostics."""
import json

import numpy as np

from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import view
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, STORE, artifact, sources
from exp.offline_search.rounds.r06.p2_ablations.pca_check import CacheView, load, payload_equal
from exp.offline_search.rounds.r06.p2_ablations.token_pca import TokenPCAAWM


def main():
    results=[]
    for (model,suite,scale,kind),(_,src) in sources().items():
        if kind!='A': continue
        baseline=load(artifact(src))
        control,_=clone_method(baseline)
        control.__class__=TokenPCAAWM
        control.pooling_grid=4
        control._previous_tokens=None
        counts={}
        for stream in ('cache','inf'):
            qc=store.QueryCell(STORE,f'{model}_{suite}_{stream}')
            arrays=api.QueryArrays(qc)
            token_rows=np.asarray(qc.tok_rows)
            ids=sorted(set(token_rows[np.linspace(0,len(token_rows)-1,300,dtype=int)].tolist()+
                           [e['start'] for e in qc.episodes if qc.tok_index[e['start']]>=0]))
            for i in ids:
                q=CacheView(view(qc,arrays,int(i)))
                baseline.reset(q.episode); control.reset(q.episode)
                a=baseline.query(q); b=control.query(q)
                payload_equal(a,b)
                assert a.extras==b.extras
            counts[stream]=len(ids)
        results.append(dict(model=model,suite=suite,scale=scale,queries=counts))
    out=dict(PASS=True,all_extras_including_still_equal=True,cells=results,
             queries=sum(sum(r['queries'].values()) for r in results))
    (HERE/'results/pca/grid4_exact_payloads.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__': main()
