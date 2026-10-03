"""Meaningful independent checks of the analysis arithmetic and boundaries."""
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, check, dump, install
from exp.offline_search.rounds.r11.astra.signals import probability, dag, VISION
from exp.offline_search.rounds.r11.astra.controller import MissController, Ledger


def brute(prog, den, probs, model):
    """Enumerate complete paths independently of the vectorized DAG."""
    n=len(prog);out=np.zeros(2)
    def visit(i,prev,span,mass,v,m):
        if i>=n:
            out[:]+=mass*np.array([v,m]);return
        if i==0 or (prog[i]-prog[prev])*den[i]>.5:
            newspan=0
        else:
            newspan=span+i-prev
        p=1. if newspan>=2 else probs[i]
        visit(i+2,i,newspan,mass*p,v+1,m+1)
        visit(i+(1 if newspan else 2),i,newspan,mass*(1-p),v+1,m)
    visit(0,0,0,1.,0,0)
    return out, (VISION[model]*out[0]+(1-VISION[model])*out[1])/n


def main():
    install();checks=[]
    score=np.array([0.,1.,1.,3.,4.,4.]);weights=np.array([1,2,3,1,2,1.])
    for beta in (1.,.5):
        for rate in np.linspace(0,1,21):
            p,_,_=probability(score,rate,beta,weights)
            assert np.isclose(np.average(p,weights=weights),rate)
            assert np.all((p>=(1-beta)*rate)&(p<=(1-beta)*rate+beta+1e-12))
    checks.append('weighted quantile tie arithmetic, endpoints, hybrid random floor')
    prog=np.array([0.,.1,.1,.1,.3,.2,.2,.7,.7])
    den=np.full(len(prog),10.)
    probs=np.linspace(.05,.95,len(prog))
    pack=dict(idx=np.arange(len(prog))[:,None],length=np.array([len(prog)]),
              prog=prog[:,None],den=den[:,None],risk=np.ones((len(prog),1)))
    for model in VISION:
        for p in (probs,np.zeros_like(probs),np.ones_like(probs)):
            counts,ir=brute(prog,den,p,model);r=dag(pack,p,model)
            np.testing.assert_allclose([r['looks'],r['misses']],counts,atol=1e-12)
            assert abs(r['owner_ir']-ir)<1e-12
        assert np.isclose(dag(pack,np.ones_like(probs),model)['owner_ir'],5/9)
        assert np.isclose(dag(pack,np.zeros_like(probs),model,False)['owner_ir'],VISION[model]*5/9)
    checks.append('independent path enumeration agrees with cost DAG and terminal truncation')
    c=MissController(model='pi05',method='adaptive',dose=.5,target=.32,score_reference=score)
    r=c.propose(step=0,score=1.,guard=True,ledger=Ledger(0,0,0));assert r['miss']
    assert r==c.propose(step=0,score=1.,guard=True,ledger=Ledger(0,0,0))
    old=c.q
    c.propose(step=2,score=1.,guard=True,ledger=Ledger(2,1,1));assert c.q<old
    old=c.q
    c.propose(step=4,score=1.,guard=False,ledger=Ledger(4,2,1));assert c.q>old
    c.reset('other');assert c.q==.5 and c.last_step==-1
    checks.append('adaptive feedback sign, mandatory guard, retry idempotence, episode reset')
    for dose,score,want in ((0.,1e6,False),(1.,-1e6,True)):
        c=MissController(model='pi05',method='threshold',dose=dose,target=.32,threshold=1.)
        assert c.propose(step=0,score=score,guard=False,ledger=Ledger(0,0,0))['miss']==want
    checks.append('static off/all-call endpoints hold outside the fitted score support')
    for path,write in (('/home/weiland/trace_runs/os_closed_loop/r10_x/fits/foo',False),
                       ('/tmp/r11_outside',True),
                       (str(HERE.parent/'opus'/'bad'),True)):
        try:check(path,write)
        except PermissionError:pass
        else:raise AssertionError(path)
    checks.append('test-data reads and writes outside astra fail closed')
    dump(HERE/'selfcheck.json',dict(passed=True,checks=checks))
    print('\n'.join(checks))


if __name__=='__main__':
    main()
