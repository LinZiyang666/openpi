"""Owner-priority library pseudoqueries: exclude entire own donor episode.

Representation is the final library-trained representation, exactly as in AWM
confidence calibration. This is not encoder-held-out validation.
"""
import argparse
import json
import pickle
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.prototype import grids,predict_ridge,evaluate,teacher_metric,reference_codes,task_metric
from exp.offline_search.rounds.r15.qk.astra.kernel import predict as kernel_predict

def run(suite,size):
    out=HERE/f'pi05_{suite}_{size}';tr=dict(np.load(DERIVED/f'pi05_{suite}_{size}/train.npz'))
    _,base,_,_,_=load_cell(suite,size);P=np.c_[tr['P0'],tr['P1']];lx=np.c_[P,tr['rs'][:,:8]]
    dev={**tr,'label':tr['action'],'label_valid':np.ones(len(P),bool),'uid':np.array([f'library:{e}' for e in tr['episode']])}
    ref,rd=evaluate('deployed_key_geometry',lx,lx,teacher_metric(base),tr,dev,base,libcodes=reference_codes(base,tr),exclude_episode=True)
    results=[]
    def save(name,s,r):
        assert not np.any(tr['episode'][r['top']]==tr['episode'][:,None])
        progress=tr['progress'][r['top'][:,0]]
        s['known_query_phase_mae']=float(np.abs(progress-tr['progress']).mean())
        s['phase_wrong_third_vs_query']=float(np.mean(np.abs(progress-tr['progress'])>1/3))
        ep_values=[]
        for t in range(10):
            vals=[r['err10'][tr['episode']==e].mean() for e in np.unique(tr['episode'][tr['task_id']==t])]
            ep_values.append(np.mean(vals))
        s['episode_task_balanced_err10']=float(np.mean(ep_values));results.append(s)
        np.savez(out/f'loeo_{name}.npz',**r)
        print(suite,size,name,s['err10'],s['episode_task_balanced_err10'],s['top16_overlap'],flush=True)
    save('deployed_key_geometry',ref,rd)
    G=[grids(tr[f'img{c}']) for c in range(2)];x=np.c_[*G,tr['rs'][:,:8]]
    for f in sorted(out.glob('asymmetric_ridge_*.pkl')):
        model=pickle.load(f.open('rb'));q=np.c_[predict_ridge(model,x,tr['task_id']),tr['rs'][:,:8]]
        s,r=evaluate(f.stem,lx,q,teacher_metric(base),tr,dev,base,rd['top'],libcodes=reference_codes(base,tr),exclude_episode=True);save(f.stem,s,r)
    model=pickle.load((out/'symmetric_rgb_spatial_awm.pkl').open('rb'))
    p=np.c_[*[(g-mu)@v for g,(mu,v) in zip(G,model['pca'])],tr['rs'][:,:8]]
    name='symmetric_rgb_spatial_awm';s,r=evaluate(name,p,p,model['metric'],tr,dev,base,rd['top'],exclude_episode=True);save(name,s,r)
    for obj in ('distill','action','action_cpu'):
        f=out/f'cnn_{obj}_features.npz'
        if not f.exists():continue
        p=np.c_[np.load(f)['library'],tr['rs'][:,:8]]
        name=f'asymmetric_cnn_{obj}'
        s,r=evaluate(name,lx,p,teacher_metric(base),tr,dev,base,rd['top'],libcodes=reference_codes(base,tr),exclude_episode=True);save(name,s,r)
        met=pickle.load((out/f'cnn_{obj}_symmetric_metric.pkl').open('rb'));name=f'symmetric_cnn_{obj}'
        s,r=evaluate(name,p,p,met,tr,dev,base,rd['top'],exclude_episode=True);save(name,s,r)
    for f in sorted(out.glob('local_residual_krr_*.pkl')):
        model=pickle.load(f.open('rb'));q=np.c_[kernel_predict(model,np.c_[*G],tr['rs'][:,:8],tr['task_id']),tr['rs'][:,:8]]
        s,r=evaluate(f.stem,lx,q,teacher_metric(base),tr,dev,base,rd['top'],libcodes=reference_codes(base,tr),exclude_episode=True);save(f.stem,s,r)
    h=(tr['action'][:,:10,:7]/tr['sig']).reshape(len(x),70)
    for f in sorted(out.glob('action_qk_*.pkl')):
        model=pickle.load(f.open('rb'));pred=np.empty_like(h);rs=tr['rs'][:,:8].copy()
        for t,p in model['ridge'].items():
            rr=tr['task_id']==t;pred[rr]=((x[rr]-p['xm'])/p['xs'])@p['coef']+p['ym'];norm=model['state_norm'][t];rs[rr]=(rs[rr]-norm['mean'])/norm['std']
        l=np.c_[h/np.sqrt(70),model['stateweight']*rs/np.sqrt(8)].astype(np.float32);q=np.c_[pred/np.sqrt(70),model['stateweight']*rs/np.sqrt(8)].astype(np.float32)
        met={t:[{'wf':np.eye(78,dtype=np.float32),'shift':np.zeros(78,np.float32)}]*2 for t in range(10)}
        s,r=evaluate(f.stem,l,q,met,tr,dev,base,rd['top'],exclude_episode=True);save(f.stem,s,r)
    p=tr['rs'][:,:8];met=task_metric(p,tr);name='state_only_awm'
    s,r=evaluate(name,p,p,met,tr,dev,base,rd['top'],exclude_episode=True);save(name,s,r)
    dump(out/'loeo_results.json',{'scope':'All library rows queried against other episodes of same task. Final representation fitted on full cell library, as in deployed confidence calibration. Query episode excluded from donors, not encoder/metric training. In-library diagnostic, not unseen validation.','rows':len(P),'episodes':len(set(tr['episode'])),'methods':results,'own_episode_donors':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suite',default='l10');ap.add_argument('--size',type=int,default=50);a=ap.parse_args();run(a.suite,a.size)
