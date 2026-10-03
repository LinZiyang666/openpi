"""Library-only metric trained for the actual ten-control commitment.

Uses the deployed PCA space and changes only action-supervised neighbor labels
from five to ten steps. Tests on A's recorded fresh states against shadow actions.
Inverse reconstruction of library PCA features is checked numerically.
"""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .common import RUN, HERE, DERIVED, load_compact, dump
from .shadows import errors, episode_mean
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric, _kernel_w
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler


def nearest_mix(query,codes,rows,actions,kref):
    result=np.empty((len(query),10,7),np.float32)
    for start in range(0,len(query),256):
        q=query[start:start+256]
        d2=np.maximum((q*q).sum(1)[:,None]+(codes*codes).sum(1)[None]-2*q@codes.T,0)
        order=np.argsort(d2,axis=1,kind='stable')[:,:16]
        d=np.sqrt(np.take_along_axis(d2,order,1))
        w=_kernel_w(d-d[:,:1],kref);w/=w.sum(1,keepdims=True)
        result[start:start+len(q)]=np.einsum('nk,nkha->nha',w,actions[rows[order],:10,:7])
    return result


def one(spec):
    name=spec['arm'];z=load_compact(name)
    with open(spec['r8']['source']['artifact'],'rb') as f:
        base=FitUnpickler(f).load()['method']
    take=z['vision']
    x=np.concatenate([z['shadow_look_keys_pca_third'],z['shadow_look_keys_pca_wrist'],z['state_norm']],axis=1)[take]
    task,init,seq=z['task'][take],z['init'][take],z['seq'][take]
    teacher=z['policy_shadow_chunk'][take]
    pred={label:np.empty_like(teacher) for label in ['reconstructed_5','metric_10','metric_10_motion']}
    reconstruction=0.
    for t in range(10):
        T=base.tasks[t];rows=T.rows
        X=(np.asarray(T.Z,np.float64)+T.shift)@np.linalg.inv(np.asarray(T.Wf,np.float64))
        reconstruction=max(reconstruction,float(np.max(np.abs(X@T.Wf-T.shift-T.Z))))
        ep=base.lib_ep[rows]
        for label in pred:
            if label=='reconstructed_5':
                normal=lambda xx:xx@T.Wf-T.shift
                early=lambda xx:xx@T.W0f-T.c0
            else:
                dims=6 if label=='metric_10_motion' else 7
                H=(base.act[rows,:10,:dims]/base.sig[:dims]).reshape(len(rows),-1)
                mu,sd,w=fit_metric(X,H,ep,nn=base.nn,lam=base.lam)
                normal=lambda xx:((xx-mu)/sd)@w
                ii=base.lib_step[rows]<=2
                m0,s0,w0=fit_metric(X[ii],H[ii],ep[ii],nn=base.nn,lam=base.lam)
                early=lambda xx:((xx-m0)/s0)@w0
            for is_early in [False,True]:
                test=(task==t)&((seq==0)==is_early)
                transform=early if is_early else normal
                pred[label][test]=nearest_mix(transform(x[test]),transform(X),rows,base.act,base.kref)
    b=episode_mean(errors(z['cache_chunk'][take],teacher)[0],task,init)
    results={}
    for label,a in pred.items():
        m,g=errors(a,teacher)
        em=episode_mean(m,task,init)
        results[label]=dict(mse=float(em.mean()),relative_mse=float(em.mean()/b.mean()-1),
                            gripper_disagreement=float(episode_mean(g,task,init).mean()),
                            task_relative=(em.mean(1)/b.mean(1)-1).tolist())
    return dict(arm=name,n=int(take.sum()),feature_reconstruction_max_abs=reconstruction,
                baseline_mse=float(b.mean()),results=results)


def main():
    specs=[s for s in json.loads((RUN/'arms.json').read_text()) if s['r8']['variant']=='A']
    result=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,s) for s in specs]):
            r=f.result();result.append(r);print(json.dumps(r),flush=True)
    dump(HERE/'results/horizon_metric.json',result)


if __name__=='__main__':
    main()
