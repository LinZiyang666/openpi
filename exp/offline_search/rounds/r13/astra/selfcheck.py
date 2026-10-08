from common import *
fence()
import numpy as np
from controller import DistanceFeedback, Ledger, replay
from replay_study import library_pack

def main():
    rng=np.random.default_rng(20261004);proposals=retries=0
    for model in COSTS:
        for target in (.25,.32,.4,.45):
            for width in (.5,1.,2.):
                for irregular in (False,True):
                    steps=np.cumsum(np.r_[0,rng.integers(1,3,99) if irregular else np.full(99,2)])
                    guards=rng.random(100)<.2;ds=np.exp(rng.normal(2,1,100))
                    p=dict(distance=ds[:,None],step=steps[:,None],guard=guards[:,None],alive=np.ones((100,1),bool),N=np.array([steps[-1]+1]))
                    cfg=dict(model=model,target=target,lo=1.,hi=3.,width=width,bias=.37)
                    offline=replay(p,**cfg)
                    c=DistanceFeedback(**cfg);V=M=0
                    for j,s in enumerate(steps):
                        ledger=Ledger(int(s),V,M)
                        out=c.propose(step=int(s),distance=ds[j],guard=bool(guards[j]),ledger=ledger)
                        again=c.propose(step=int(s),distance=ds[j]+10,guard=not bool(guards[j]),ledger=ledger)
                        assert out==again
                        assert out['miss']==offline['calls'][j,0]
                        assert out['threshold']==offline['thresholds'][j,0]
                        V+=1;M+=int(out['miss']);proposals+=1;retries+=1
                    c.reset();first=c.propose(step=0,distance=ds[0],guard=bool(guards[0]),ledger=Ledger(0,0,0))
                    assert first['miss']==offline['calls'][0,0]
    c=DistanceFeedback(model='pi05',target=.32,lo=1,hi=3,width=1,bias=.5)
    assert c.propose(step=0,distance=np.nan,guard=False,ledger=Ledger(0,0,0))['miss']
    d=DistanceFeedback(model='pi05',target=.32,lo=1,hi=3,width=1,bias=.5)
    assert d.propose(step=0,distance=0,guard=True,ledger=Ledger(0,0,0))['miss']
    # Bounded-score no-guard debt invariant; forced guards can violate upper bound.
    bounds=0
    for target in (.25,.32,.4,.45):
        cv,cm=COSTS['pi05'];q=(2*target-cv)/cm;w=1.;b=.5
        for level in (0.,1.,100.):
            p=dict(distance=np.full((200,1),level),step=2*np.arange(200)[:,None],guard=np.zeros((200,1),bool),
                   alive=np.ones((200,1),bool),N=np.array([400]))
            r=replay(p,model='pi05',target=target,lo=1,hi=3,width=w,bias=b)
            ds=np.cumsum(r['calls'][:,0]-q)
            assert ds.min()>=min(0,-w*b-q)-1e-9
            assert ds.max()<=max(0,w*(1-b)+1-q)+1e-9
            bounds+=1
    out=dict(pass_all=True,proposals=proposals,retries=retries,bound_scenarios=bounds,
             scope='CPU arithmetic, committed ledger, reset, retries, nonfinite fallback, guard precedence; no plugin or closed-loop claim')
    dump(HERE/'selfcheck.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__': main()
