"""Cross-check opus's shared model on astra's identical B-held-out input table."""
import json
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, dump, install, sha
from exp.offline_search.rounds.r11.astra.experiment import CELLS, predictor_score
from exp.offline_search.rounds.r11.astra.signals import dag, probability, simulate


def main():
    install()
    from exp.offline_search.rounds.r11.opus import ir_model as shared
    comparisons=[]
    for m,s,n in CELLS:
        tag=f'{m}_{s}_{n}'
        with np.load(HERE/'data'/tag/'signals.npz') as z:
            a={k:np.array(z[k]) for k in z.files}
        with np.load(HERE/'data'/tag/'pack.npz') as z:
            pack={k:np.array(z[k]) for k in z.files}
        analysis=json.loads((HERE/'data'/tag/'analysis.json').read_text())
        guards=np.zeros(len(a['row']),bool);anchor=np.zeros(len(a['row']),bool)
        for j,L in enumerate(pack['length']):
            ii=np.arange(0,L,2)
            rows=pack['idx'][ii,j];anchor[rows]=True
            guards[rows]=shared.guard_flags(ii,pack['prog'][ii,j],pack['den'][ii,j]+1)
        V=anchor.sum();N=pack['length'].sum();g=float(guards.sum()/V)
        maxdiff=0.
        for name in ('distance','predicted_error','disagreement'):
            for dose in (0.,.2,.4,.6,.8,1.):
                for beta in (1.,.5):
                    p=probability(a[name],dose,beta)[0]
                    expected=shared.owner_ir(m,V,np.sum(guards[anchor]+(~guards[anchor])*p[anchor]),N)
                    direct=dag(pack,p,m)
                    maxdiff=max(maxdiff,abs(expected-direct['owner_ir']))
                    assert abs(expected-direct['owner_ir'])<1e-12
                    assert abs(direct['looks']-V)<1e-10
        for dose in (0.,.2,.4,.6,.8,1.):
            expected=shared.random_closed_form(m,V/N,g,dose)
            actual=dag(pack,np.full(len(a['row']),dose),m)['owner_ir']
            assert abs(expected-actual)<1e-12
            maxdiff=max(maxdiff,abs(expected-actual))
        chosen=next(c for c in analysis['calibrations'] if c['method']=='predicted_error' and c['beta']==.5)
        sim=simulate(pack,a['predicted_error'],chosen['dose'],chosen['target'],m,
                     seeds=128,seed=20261005)
        error=sim['owner_ir_mean']-chosen['owner_ir']
        assert abs(error)<max(.003,5*sim['seed_ir_std']/np.sqrt(128))
        # This is a distribution-transfer diagnostic, not an honest held-out value score.
        head=json.loads((HERE/'data'/tag/'predictor.json').read_text())
        head={k:np.asarray(head[k]) for k in ('mean','std','coef')}
        finalscore=predictor_score(head,a['X'])
        q=chosen['dose'];t=chosen['threshold'];tie=chosen['tie_probability']
        crossfit_p=probability(a['predicted_error'],q,.5)[0]
        final_p=.5*q+.5*((finalscore>t)+(finalscore==t)*tie)
        transfer_ir=shared.owner_ir(m,V,np.sum(guards[anchor]+(~guards[anchor])*final_p[anchor]),N)
        comparisons.append(dict(cell=tag,max_shared_model_absolute_error=maxdiff,
            actual_looks=float(V),nominal_looks=float(V),base_guard_fraction=g,
            dag_ir=chosen['owner_ir'],simulation=sim,simulation_error=error,
            final_head_on_OOF_features_ir=float(transfer_ir),
            final_head_score_mean=float(finalscore.mean()),crossfit_score_mean=float(a['predicted_error'].mean())))
        print(tag,'shared diff',maxdiff,'MC diff',error,'transfer IR',transfer_ir,flush=True)
    dump(HERE/'crosscheck.json',dict(passed=True,shared_model=str(shared.__file__),
        shared_model_sha256=sha(shared.__file__),inputs='astra B-held-out arrays only; no opus table reads',
        normal_cadence='all reachable looks are even; span=1 cache hits are unreachable',comparisons=comparisons))


if __name__=='__main__':
    main()
