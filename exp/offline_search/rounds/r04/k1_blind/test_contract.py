"""Method-side fake driver and edge cases; no policy, server, GPU, or LIBERO worker."""
import copy, json, pathlib, pickle, time
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r03.h3_judge.judge import MixedJudge
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM, BlindResult, LookReason
from exp.offline_search.rounds.r04.k1_blind.control_step import ControlStepLibrary
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge, MemoResetMixedJudge
from exp.offline_search.rounds.r04.k1_blind.checks import ROOT, OUT, adapt, equal, get_method


def toy(H=10):
    n=80
    m=BlindAWM(lib='current',kref=5,budget=4,gates='budget_only')
    m.model,m.H,m.cand_name='pi05',H,'current'
    m.act=np.zeros((n,H,32),np.float32)
    m.act[:,:,0]=np.arange(n)[:,None]
    m.act[:,:,6]=-1
    m.lib_step=np.arange(n,dtype=np.int32)%40
    m.lib_ep=np.arange(n,dtype=np.int32)//40
    m.blind_rs=np.repeat(m.lib_step[:,None]*.1,8,axis=1).astype(np.float32)
    m.blind_next=np.arange(1,n+1,dtype=np.int32);m.blind_next[[39,79]]=-1
    m.blind_event=np.zeros(n,bool);m.blind_terminal=m.lib_step>=38
    m.state_scale_by_task={0:np.ones(8,np.float32)};m.motion10={0:.01}
    m.reset(NS(uid='toy'))
    q=NS(step=0,task_id=0,episode=NS(uid='toy'),rs=np.zeros(8,np.float32))
    rows=np.r_[np.arange(8),np.arange(40,48)].astype(np.int64)
    w=np.ones(16,np.float32)/16
    m._remember_anchor(q,rows,w,np.tensordot(w,m.act[rows],1))
    return m,q,rows,w


def bq(q,step,rs=None,**kw):
    v=dict(vars(q),step=step,blind_age=step-1,prev_hit=True,
           rs=np.ones(8,np.float32)*(.1*step) if rs is None else rs,
           hist_rs=np.ones((step,8),np.float32)*np.arange(step)[:,None]*.1)
    v.update(kw)
    return NS(**v)


def edge_cases():
    total=0
    for H in (10,16):
        for serving in ('phase_particles','kernel_clock','top1_clock','anchor_tail'):
            for budget in range(5):
                m,q,rows,w=toy(H);m.serving=serving;m.budget=budget
                prev=m.lib_step[rows]
                for h in range(1,6):
                    result=m.blind_step(bq(q,h))
                    cap=min(budget,H//5-1) if serving=='anchor_tail' else budget
                    if h<=cap:
                        assert isinstance(result,BlindResult)
                        assert result.action.shape==(H,32) and result.action.dtype==np.float32
                        assert result.rows.dtype==np.int64 and result.weights.dtype==np.float32
                        assert np.isfinite(result.action).all() and np.all(result.action[:,7:]==0)
                        if serving=='phase_particles':
                            phase=m.lib_step[result.rows]
                            assert np.all(phase>=prev) and np.all(phase<=prev+2)
                            assert np.array_equal(m.lib_ep[result.rows],m.lib_ep[rows])
                            assert np.array_equal(result.weights,w)
                            prev=phase
                        if serving=='anchor_tail':
                            np.testing.assert_array_equal(result.action[:5],m._anchor['action'][5*h:5*(h+1)])
                    else:
                        assert isinstance(result,LookReason) and result.code in (1,6)
                        break
                    total+=1
    for field,value,name in [('step',0,'first_decision'),('prev_hit',False,'after_miss'),
                              ('task_id',1,'episode_or_task_change'),('executed_steps',4,'executed_steps'),
                              ('blind_age',2,'decision_discontinuity')]:
        m,q,_,_=toy();values={field:value};r=m.blind_step(bq(q,1,**values)) if field!='step' else m.blind_step(bq(q,0))
        assert r.code==6 and r.name==name and m._anchor is None,r
        total+=1
    for gate in (2,3,4,5):
        m,q,rows,w=toy();m.gates='all'
        z=bq(q,1)
        if gate==2:m.blind_event[m._advance(rows,1)]=True
        if gate==3:m.blind_terminal[m._advance(rows,1)]=True
        if gate==4:
            m._anchor['step']=-1;m._anchor['last_step']=1
            z=bq(q,2,rs=np.zeros(8,np.float32),blind_age=2,hist_rs=np.zeros((2,8),np.float32))
        if gate==5:z.rs[:]=2
        r=m.blind_step(z);assert isinstance(r,LookReason) and r.code==gate,(gate,r)
        total+=1
    m,q,rows,w=toy();m.reset(q.episode);assert m.blind_step(bq(q,1)).name=='invalid_anchor';total+=1
    m,q,_,_=toy();legacy=BlindMixedJudge(progress_guard='noprog_n');legacy.base=m
    assert legacy.blind_step(bq(q,1)).name=='noprog_n_requires_vision';total+=1
    # The virtual chord selects nearest control offset, with lower-offset ties.
    c=ControlStepLibrary(lib='current',kref=5)
    c.H=10;c.act=np.arange(4*10*32,dtype=np.float32).reshape(4,10,32)
    c.control_next=np.array([1,2,3,-1]);c.edge_l2=np.array([25,25,25,0],np.float32)
    T=NS(rows=np.arange(4));d2=np.array([6.25,6.25,56.25,156.25])
    _,o=c._virtual(T,d2);assert o[0]==2 and o[-1]==0,o
    chunk=c.aligned_chunks(np.array([0]),np.array([3]))[0]
    np.testing.assert_array_equal(chunk[:5,:7],np.concatenate([c.act[0,3:5,:7],c.act[1,:3,:7]]))
    np.testing.assert_array_equal(chunk[5:10,:7],np.concatenate([c.act[1,3:5,:7],c.act[2,:3,:7]]))
    np.testing.assert_array_equal(c.aligned_chunks(np.array([0]),np.array([0]))[0],c.act[0])
    tail=c.aligned_chunks(np.array([2]),np.array([4]))[0]
    np.testing.assert_array_equal(tail[6:,:7],np.repeat(c.act[3,4:5,:7],4,axis=0))
    return total+4


class Driver:
    """CPU plugin-facing driver: query only on vision, one dense commit per executed action."""
    def __init__(self,m,qc):
        self.m,self.qc=m,qc;self.ep=qc.episodes[0];self.step=0;self.age=0
        self.episode=api.EpisodeView(self.ep['uid'],self.ep['task'],self.ep['task_id'],self.ep['init'],0,0)
        m.reset(self.episode)
        self.actions=[];self.hits=[];self.vision=[];self.states=[];self.keys0=[];self.keys1=[]
    def query(self):
        i=self.ep['start']+self.step
        return NS(step=self.step,task_id=self.ep['task_id'],episode=self.episode,rs=self.qc.rs[i],raw_state=self.qc.raw_state[i],
                  model=self.qc.model,key_v0=self.qc.key_v0[i],key_v1=self.qc.key_v1[i],
                  hist_key_v0=np.asarray(self.keys0),hist_key_v1=np.asarray(self.keys1),
                  hist_rs=np.asarray(self.states).reshape(-1,self.qc.rs.shape[1]),
                  prev_hit=self.hits[-1] if self.hits else None,prev_a_exec=self.actions[-1] if self.actions else None,
                  hist_a_exec=np.asarray(self.actions),hist_hit=np.asarray(self.hits,np.int8),
                  hist_has_vision=np.asarray(self.vision,bool),blind_age=self.age)
    def run(self,force_vision=False,hit=True):
        q=self.query()
        stateq=NS(**{k:v for k,v in vars(q).items() if 'key' not in k and k!='model'})
        blind=self.m.blind_step(stateq)
        vision=force_vision or isinstance(blind,LookReason)
        result=self.m.query(q) if vision else blind
        action=result.action.copy() if hit else np.array(self.qc.a_inf[self.ep['start']+self.step])
        self.actions.append(action);self.hits.append(hit);self.vision.append(vision);self.states.append(q.rs)
        self.keys0.append(q.key_v0 if vision else np.full_like(q.key_v0,np.nan))
        self.keys1.append(q.key_v1 if vision else np.full_like(q.key_v1,np.nan))
        if not hit:self.m.invalidate_anchor()
        self.age=0 if vision else self.age+1;self.step+=1
        return blind,result


def mixed_sequence(key,scale):
    qc=store.QueryCell(ROOT,f'{key}_cache')
    lib=store.LibraryView(ROOT,key,'current')
    kw=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,budget=2,gates='budget_only')
    m=BlindMixedJudge(base_kwargs=kw,guards=False,ncal=64)
    ctx=api.Context(root=ROOT,cell=f'{key}_cache',seed=0,scratch=OUT/'scratch_contract')
    m.fit(lib,ctx)
    d=Driver(m,qc)
    first,_=d.run();assert first.code==6
    _,_=d.run();_,_=d.run()
    budget,_=d.run();assert budget.code==1
    _,_=d.run(force_vision=True,hit=False);assert m.base._anchor is None
    miss,fresh=d.run();assert miss.code==6 and miss.name=='after_miss'
    assert d.vision==[True,False,False,True,True,True]
    assert d.hits==[True,True,True,True,False,True]
    assert len(d.states)==len(d.actions)==6 and m.base._anchor['step']==5
    # Fresh branch must use the actually executed policy tail.
    base=get_method(key,scale)
    q=d.query();q.step=5;q.prev_hit=False;q.prev_a_exec=d.actions[4]
    q.rs=qc.rs[d.ep['start']+5];q.key_v0=qc.key_v0[d.ep['start']+5];q.key_v1=qc.key_v1[d.ep['start']+5]
    np.testing.assert_array_equal(fresh.action,base.query(q).action)
    # Span accumulates elapsed decisions between real anchors, never blind phases.
    m.reset(d.episode);row=int(fresh.topk[0]);m.guards=True
    a=NS(step=0,prev_hit=None);b=NS(step=3,prev_hit=True)
    assert m._progress(a,row)==0 and m._progress(b,row)==3
    m.memo_reset_after_miss=True
    assert m._progress(NS(step=4,prev_hit=False),row)==0
    # A single nonadvancing transition forces another look before its MISS threshold.
    m._progress(NS(step=5,prev_hit=True),row)
    m.base._remember_anchor(NS(step=5,task_id=0,episode=d.episode,rs=q.rs),fresh.topk,
                            np.ones(16,np.float32)/16,fresh.action)
    z=NS(step=6,task_id=0,episode=d.episode,rs=q.rs,blind_age=0,prev_hit=True)
    assert m.blind_step(z).name=='noprog_span'
    # Reset and pickle round trip contain no stale anchor.
    m.reset(d.episode);assert m.base._anchor is None and not m._vision_progress
    p=pickle.loads(pickle.dumps(m));assert p.base._anchor is None
    return dict(key=key,scale=scale,sequence='vision blind blind vision MISS vision',vision=d.vision,
                hits=d.hits,look_reasons=[first.code,budget.code,miss.code],fresh_action_equal=True)


def main():
    count=edge_cases();report={'edge_case_checks':count,'sequences':[]}
    for key in ('pi05_l10','groot_l10'):
        for scale in (50,500):
            result=mixed_sequence(key,scale);report['sequences'].append(result);print(result,flush=True)
    (OUT/'contract.json').write_text(json.dumps(report,indent=2));print('CONTRACT PASS',count,flush=True)

if __name__=='__main__':main()
