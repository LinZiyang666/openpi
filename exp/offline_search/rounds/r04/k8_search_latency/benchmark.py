"""Warm real-method replays; all measurements in milliseconds. No fit or source mutation."""
import argparse, os
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',type=int,default=1);p.add_argument('--n',type=int,default=1024);p.add_argument('--profile',action='store_true');a=p.parse_args() if __name__=='__main__' else None
if __name__=='__main__':os.sched_setaffinity(0,{34})
from common import *
from types import SimpleNamespace
from exp.offline_search.closed_loop.blind import BlindQueryView,BlindResult
import collections,contextlib,functools,gc,threading

def queries(cell):
    result=[]
    for arm in ('cache','inf'):
        qc=store.QueryCell(STORE,cell.rsplit('_',1)[0]+'_'+arm)
        for task in range(10):
            ix=next(i for i,e in enumerate(qc.episodes) if e['task_id']==task)
            e=qc.episodes[ix];lo,hi=e['start'],min(e['end'],e['start']+64)
            ep=api.EpisodeView(e['uid'],e['task'],task,e['init'],ix,plugin.ep_seed(0,e['uid']))
            s=SimpleNamespace(rt=SimpleNamespace(model=qc.model,api=api,opts=SimpleNamespace(os_tokens='off')),hits=np.asarray(qc.exec_hit_flag[lo:hi]).tolist(),has_vision=[True]*(hi-lo),last_vision_step=0,_age_before=0)
            for name,field in [('b_v0','key_v0'),('b_v1','key_v1'),('b_rs','rs'),('b_raw','raw_state'),('b_aex','a_exec')]:
                ar=np.array(getattr(qc,field)[lo:hi],copy=True);ar.flags.writeable=False
                buf=plugin._Buf(ar.shape[1:],ar.dtype);buf.a=ar;buf.n=len(ar);setattr(s,name,buf)
            qs=[plugin.OnlineQueryView(s,j,task,ep) for j in range(hi-lo)]
            result.append((ep,qs))
    return result

def reset_prof(m,prof):
    m.prof=prof
    if hasattr(m,'base'):reset_prof(m.base,prof)

class Recorder:
    def __init__(self):self.stack=[];self.acc=collections.defaultdict(float);self.rows=[]
    @contextlib.contextmanager
    def section(self,name):
        t=time.perf_counter_ns();frame=[name,t,0];self.stack.append(frame)
        try:yield
        finally:
            elapsed=time.perf_counter_ns()-t;self.stack.pop();self.acc[name]+=(elapsed-frame[2])/1e6
            if self.stack:self.stack[-1][2]+=elapsed
    def finish(self):self.rows.append(dict(self.acc));self.acc.clear()
    def wrap(self,obj,name,label):
        old=getattr(obj,name)
        @functools.wraps(old)
        def timed(*x,**kw):
            with self.section(label):return old(*x,**kw)
        setattr(obj,name,timed)
        return lambda:setattr(obj,name,old)

def profile_hooks(m,rec):
    undo=[]
    for name in ('argpartition','partition','argsort','lexsort'):
        undo.append(rec.wrap(np,name,'topk'))
    for target in ([m,base(m)] if m is not base(m) else [m]):
        for name,label in [('_mix','kernel'),('_insure','insurance'),('_features','judge_features'),('_v7','judge_calibration'),('_self_change','judge_motion'),('_virtual','control_geometry'),('aligned_chunks','control_synthesis'),('dense_motion','dense_motion'),('_progress','guard_progress'),('_remember_anchor','anchor_capture'),('_advance','blind_advance')]:
            if hasattr(target,name):undo.append(rec.wrap(target,name,label))
    return undo

def vision(m,seq,n,rec=None,capture=False):
    times=[];digests=[];rounds=0
    while len(times)<n:
        for ep,qs in seq:
            m.reset(ep)
            for q in qs:
                q._s.has_vision[:]=[True]*len(q._s.has_vision)
                start=time.perf_counter_ns();r=m.query(q);times.append((time.perf_counter_ns()-start)/1e6)
                if rec:rec.finish()
                if capture and len(digests)<64:
                    digests.append(hashlib.sha256(np.asarray(r.topk).tobytes()+np.asarray(r.action).tobytes()).hexdigest())
                if len(times)>=n:return times,digests
        rounds+=1
    return times,digests

def blind(m,seq,n):
    import copy
    yes=[];no=[];reason=collections.Counter();templates=[]
    owner=base(m)
    cache=[(ep,qs) for ep,qs in seq if '_cache:' in ep.uid]
    for ep,qs in cache:
        m.reset(ep);age=0
        for q in qs:
            s=q._s;s._age_before=age
            bq=BlindQueryView(q.step,q.task_id,q.episode,q.rs,q.raw_state,q.prev_hit,q.prev_a_exec,q.hist_a_exec,q.hist_hit,q.hist_rs,q.hist_has_vision,age)
            anchor=copy.deepcopy(owner._anchor);span=getattr(m,'_noprog_span',None)
            t=time.perf_counter_ns();r=m.blind_step(bq);dt=(time.perf_counter_ns()-t)/1e6
            if isinstance(r,BlindResult):
                templates.append((bq,anchor,span,hashlib.sha256(r.rows.tobytes()+r.action.tobytes()).hexdigest()))
                s.has_vision[q.step]=False;age+=1
            else:
                no.append(dt);reason[r.name]+=1;s.has_vision[q.step]=True;s.last_vision_step=q.step;age=0;m.query(q)
    digests=[]
    if templates:
        for i in range(n+128):
            bq,anchor,span,expected=templates[i%len(templates)]
            owner._anchor=copy.deepcopy(anchor)
            if span is not None:m._noprog_span=span
            t=time.perf_counter_ns();r=m.blind_step(bq);dt=(time.perf_counter_ns()-t)/1e6
            assert isinstance(r,BlindResult)
            if i>=128:
                yes.append(dt)
                if len(digests)<64:
                    dg=hashlib.sha256(r.rows.tobytes()+r.action.tobytes()).hexdigest();assert dg==expected;digests.append(dg)
    return dict(success_ms=stats(yes),look_ms=stats(no),samples_ms=yes,look_samples_ms=no,reasons=reason,successful_input_templates=len(templates),digests=digests)

def main():
    cfg=next(c for c in json.loads((OUT/'configs.json').read_text()) if c['id']==a.config)
    b=load(cfg['path']);m=b['method']
    if cfg.get('extract_base'):m=base(m)
    reset_prof(m,api.NULL_PROFILER);seq=queries(cfg['cell']);a.n=max(a.n,sum(len(qs) for _,qs in seq));before=cpu();load0=os.getloadavg();t0=time.time()
    vision(m,seq,a.n) # warm every input and actual method state path
    ts,dig=vision(m,seq,a.n,capture=True)
    result=dict(config=cfg,run=a.run,n=a.n,affinity=sorted(os.sched_getaffinity(0)),load_start=load0,vision_ms=stats(ts),samples_ms=ts,digests=dig)
    if hasattr(m,'blind_step'):
        result['blind']=blind(m,seq,a.n)
    if a.profile:
        rec=Recorder();reset_prof(m,rec);undo=profile_hooks(m,rec)
        try:pt,pdig=vision(m,seq,a.n,rec=rec,capture=True)
        finally:
            for fn in reversed(undo):fn()
            reset_prof(m,api.NULL_PROFILER)
        assert dig==pdig,'profile wrappers changed output'
        result.update(profile_total_ms=stats(pt),components_ms={k:stats([r.get(k,0.) for r in rec.rows]) for k in set().union(*rec.rows)},profile_samples=rec.rows)
    result.update(load_end=os.getloadavg(),idle_percent=idle(before,cpu()),wall_s=time.time()-t0,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(OUT/f"bench_{a.config}_r{a.run}.json",result)
    print(a.config,'run',a.run,'vision',result['vision_ms'],'blind',result.get('blind',{}).get('success_ms'),flush=True)
if __name__=='__main__':main()
