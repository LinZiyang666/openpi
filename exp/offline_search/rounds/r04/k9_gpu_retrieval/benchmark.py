"""Run ONE fitted cell in ONE process, release GPU at exit; parity + timing."""
import argparse, collections, gc, time
from common import *
import torch
from gpu_awm import GPUAWM, verdict_packet


def inputs(q,device='cuda',batch=1,stuck=0):
    vals=[q.key_v0,q.key_v1,q.rs,np.int64(q.task_id),np.int64(q.step),np.int64(-1 if q.prev_hit is None else int(q.prev_hit)),
          q.prev_a_exec if q.step else np.zeros((10,32),np.float32),
          q.hist_key_v0[-1] if q.step else np.zeros(32768,np.float32),
          q.hist_key_v1[-1] if q.step else np.zeros(32768,np.float32),
          q.hist_rs[-1] if q.step else np.zeros(32,np.float32),np.int64(stuck)]
    return [torch.as_tensor(np.array(v,copy=True),device=device).unsqueeze(0).repeat((batch,)+(1,)*np.ndim(v)) for v in vals]


def capture(fn,watch):
    watch.check(admit=True)
    s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    return g,out


def timing(fn,watch,reps=80,warm=10):
    watch.check(admit=True)
    sample_start=len(watch.samples)-1
    for _ in range(warm):fn()
    torch.cuda.synchronize()
    ev=[];wall=[]
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for i in range(reps):
        t=time.perf_counter_ns();start.record();fn();end.record();end.synchronize()
        wall.append((time.perf_counter_ns()-t)/1e6);ev.append(start.elapsed_time(end))
    watch.check()
    return dict(event=stats(ev),wall=stats(wall),event_samples=ev,wall_samples=wall,
                gpu_samples=watch.samples[sample_start:])


def parity(m,mod,seq,watch,limit=0):
    ii=inputs(seq[0][1][0]);g,out=capture(lambda:mod(*ii),watch)
    mismatch=[];agg=collections.defaultdict(int);mx=collections.defaultdict(float);all_deltas=collections.defaultdict(list)
    regime=collections.Counter();top1_by_reg=collections.Counter();n=0;maxcase={};source=[]
    by_reg={r:collections.defaultdict(float) for r in (0,1,2)}
    for ep,qs in seq:
        m.reset(ep);gpu_stuck=0
        source.append(dict(uid=ep.uid,task_id=ep.task_id,n=len(qs)))
        for q in qs:
            cp=m.query(q)
            for dst,src in zip(ii,inputs(q,device='cpu',stuck=gpu_stuck)):dst.copy_(src)
            g.replay();got={k:v.detach().cpu().numpy()[0] for k,v in out.items()}
            if mod.judge:gpu_stuck=int(got['stuck_n'])
            n+=1;r=int(got['regime']);regime[r]+=1
            one=cp.topk[0]==got['topk'][0];st=np.array_equal(np.sort(cp.topk),np.sort(got['topk']));order=np.array_equal(cp.topk,got['topk'])
            agg['top1_agree']+=int(one);agg['top16_set_agree']+=int(st);agg['top16_order_agree']+=int(order)
            if one:top1_by_reg[r]+=1
            ref={'action':cp.action,'scores':cp.scores,'confidence':cp.confidence}
            if mod.judge:
                w=np.exp(cp.scores-cp.scores[0])
            else:
                rel=cp.scores[0]-cp.scores
                w=np.exp(-(rel/max(float(rel[mod.kref-1]),1e-6))**2)
            ref['weights']=(w/w.sum()).astype(np.float32)
            ref.update({k:v for k,v in cp.extras.items() if k in got and (q.step or k not in ('motion','vself'))})
            # Wrapper hides AWM extras, audit these independently against the actual fitted base.
            if mod.judge:
                br=m.base.query(q)
                ref.update({k:br.extras[k] for k in ('d1','d1_rel','disp5','dst','w_eff')})
                ref['awm_confidence']=br.confidence
            bad=[]
            for k,v in ref.items():
                delta=float(np.max(np.abs(np.asarray(v,dtype=float)-np.asarray(got[k],dtype=float))))
                all_deltas[k].append(delta)
                by_reg[r][k]=max(by_reg[r][k],delta)
                if delta>mx[k]:mx[k]=delta;maxcase[k]=dict(uid=ep.uid,step=q.step,delta=delta)
                tol=1e-4 if k=='action' else 1e-3 if k not in ('vself','vis','motion') else 1e-5
                if delta>tol:bad.append(k);agg[k+'_over_tol']+=1
            if not order or bad:
                rank_detail={}
                if not order:
                    base=getattr(m,'base',m);tt,*_,dd=base._dist(q)
                    pos=np.searchsorted(tt.rows,cp.topk);gp=np.searchsorted(tt.rows,got['topk'])
                    sorted_d=np.sort(dd)
                    rank_detail=dict(cpu_top1_gap=float(sorted_d[1]-sorted_d[0]),
                                     cpu_rank16_gap=float(sorted_d[16]-sorted_d[15]),
                                     gpu_chosen_top1_cpu_delta=float(dd[gp[0]]-dd[pos[0]]),
                                     cpu_dt_at_cpu_selection=dd[pos].tolist(),cpu_dt_at_gpu_selection=dd[gp].tolist())
                mismatch.append(dict(uid=ep.uid,step=q.step,regime=r,top1=bool(one),set=bool(st),order=bool(order),
                                     cpu_topk=cp.topk.tolist(),gpu_topk=got['topk'].tolist(),over_tolerance=bad,
                                     deltas={k:all_deltas[k][-1] for k in ref},rank_detail=rank_detail,
                                     scores_relative_linf=all_deltas['scores'][-1]/max(float(np.abs(cp.scores).max()),1e-12)))
            if limit and n>=limit:break
        if limit and n>=limit:break
    return dict(n=n,agreement=dict(agg),regimes=dict(regime),top1_by_regime=dict(top1_by_reg),max_abs=dict(mx),
                delta_stats={k:stats(v) for k,v in all_deltas.items()},max_abs_by_regime={r:dict(v) for r,v in by_reg.items()},
                worst_cases=maxcase,mismatches=mismatch,sources=source)


def append_check(mod,q,watch):
    ii=inputs(q);g,out=capture(lambda:mod(*ii),watch);g.replay();torch.cuda.synchronize()
    task=q.task_id;topid=int(out['topk'][0,0]);pos=int((mod.rows[task,:mod._counts[task]]==topid).nonzero()[0,0])
    before={n:b.data_ptr() for n,b in mod.named_buffers()}
    fields={n:getattr(mod,n)[task,pos:pos+1].clone() for n in mod.row_fields}
    new_id=int(mod.rows.max())+1;fields['rows'].fill_(new_id)
    mod.append_prepared(task,fields);g.replay();torch.cuda.synchronize()
    captured={k:v.clone() for k,v in out.items()};eager=mod(*ii)
    checks=dict(addresses_unchanged=before=={n:b.data_ptr() for n,b in mod.named_buffers()},
                appended_row_seen=bool((captured['topk']==new_id).any()),
                graph_eager_topk_equal=torch.equal(captured['topk'],eager['topk']),
                graph_eager_action_max_abs=float((captured['action']-eager['action']).abs().max()),
                count=mod._counts[task],capacity=mod.capacity)
    assert checks['addresses_unchanged'] and checks['appended_row_seen'] and checks['graph_eager_topk_equal']
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--precision',choices=['float32','float64'],default='float32');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--reps',type=int,default=80);a=p.parse_args()
    cfg=next(c for c in configs() if c['id']==a.config);m=load(cfg['path']);suite=cfg['cell'].split('_')[1]
    seq=sequences(suite);b=getattr(m,'base',m);lib=store.LibraryView(STORE,f'pi05_{suite}',b.cand_name)
    mod=GPUAWM(m,lib,precision=a.precision).eval()
    watch=GPUWatch(f'{a.config}_{a.run}_{a.precision}')
    record=dict(config=cfg,run=a.run,precision=a.precision,torch=torch.__version__,cuda=torch.version.cuda,
                command=sys.argv,affinity=sorted(os.sched_getaffinity(0)),store=str(STORE),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},
                capacity=mod.capacity,counts=mod._counts.copy(),resident_bytes=mod.resident_bytes())
    with torch.inference_mode():
        mod.cuda()
        record['parity']=parity(m,mod,seq,watch,a.limit)
        print(a.config,a.run,a.precision,'parity',record['parity']['n'],record['parity']['agreement'], 'action',record['parity']['max_abs']['action'],flush=True)
        record['latency']={}
        q=seq[0][1][min(2,len(seq[0][1])-1)]
        for batch in (1,8):
            ii=inputs(q,batch=batch);g,out=capture(lambda:mod(*ii),watch)
            record['latency'][str(batch)]=dict(eager=timing(lambda:mod(*ii),watch,a.reps),graph=timing(g.replay,watch,a.reps))
            if batch==8:
                one=mod(*inputs(q));g.replay();torch.cuda.synchronize()
                record['batch8_same_query_parity']=dict(topk=bool(torch.all(out['topk']==one['topk'])),
                    action_max_abs=float((out['action']-one['action']).abs().max()))
            print('latency',batch,record['latency'][str(batch)]['eager']['wall']['p50'],record['latency'][str(batch)]['graph']['wall']['p50'],flush=True)
            del g,out,ii;gc.collect();torch.cuda.empty_cache()
        record['append']=append_check(mod,q,watch)
        record['gpu']=watch.finish()
    dump(OUT/f'bench_{a.config}_{a.precision}_r{a.run}.json',record)

if __name__=='__main__':main()
