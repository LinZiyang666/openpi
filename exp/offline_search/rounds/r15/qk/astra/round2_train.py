"""Episode-held-out student selection and final own-library fits; no B-dev reads."""
import argparse
import json
import pickle
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump,sha
from exp.offline_search.rounds.r15.qk.astra.prototype import task_metric
from exp.offline_search.rounds.r15.qk.astra.round2_core import (CONFIGS,Student,setup,image_tensor,take,
    task_normalization,augmentation,encode,query_data,metric_eval,blindness)


def run(size,name,fold,epochs):
    setup();seed=2026100506+max(fold,0);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    cell=f'pi05_l10_{size}';root=HERE/'round2'/cell/name;root.mkdir(parents=True,exist_ok=True)
    full=dict(np.load(DERIVED/cell/'train.npz'));_,base,_,_,provenance=load_cell('l10',size)
    selected=[];held=[]
    for t in range(10):
        es=np.unique(full['episode'][full['task_id']==t]);np.random.default_rng(2026100507+t).shuffle(es)
        n=max(1,len(es)//5);held.extend(es[fold*n:(fold+1)*n] if fold>=0 else [])
    fit=np.flatnonzero(~np.isin(full['episode'],held));valid=np.flatnonzero(np.isin(full['episode'],held))
    tr=take(full,fit);tr['sig']=np.maximum(tr['action'][:,:5,:7].std((0,1)),1e-6)
    validation=take(full,valid) if len(valid) else None
    if validation is not None:validation['sig']=tr['sig']
    assert not set(tr['episode']) & set(held)
    P=np.c_[tr['P0'],tr['P1']];state=tr['rs'][:,:8];pm,ps=task_normalization(P,tr['task_id'],.01)
    if CONFIGS[name].get('global_normalization',False):
        pm=np.tile(P.mean(0),(10,1));ps=np.tile(np.maximum(P.std(0),.01),(10,1))
    model=Student(CONFIGS[name],pm,ps,state.mean(0),np.maximum(state.std(0),.01))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=CONFIGS[name].get('weight_decay',.01))
    im=image_tensor(tr);rr=torch.from_numpy(state.copy());tt=torch.from_numpy(tr['task_id'].astype(np.int64));yy=torch.from_numpy(P)
    aa=torch.from_numpy((tr['action'][:,:5,:7]/tr['sig']).reshape(len(P),35))
    teacher_metric=task_metric(np.c_[P,state],tr)
    wf=torch.tensor(np.stack([teacher_metric[t][0]['wf'] for t in range(10)]))
    scale=torch.tensor([max(np.mean((np.c_[P,state][tr['task_id']==t]@teacher_metric[t][0]['wf']-teacher_metric[t][0]['shift'])**2),1e-6) for t in range(10)])
    token=None
    if CONFIGS[name]['tokens']:
        with np.load(DERIVED/'round2'/cell/'spatial_tokens.npz') as a:
            assert np.array_equal(a['rows'],full['rows']);target=a['target'][fit]
        tm,ts=task_normalization(target,tr['task_id'],.02)
        token=torch.from_numpy((target-tm[tr['task_id']])/ts[tr['task_id']])
    positives=np.zeros((len(P),3),np.int64);taskrows={}
    for t in range(10):
        rows=np.flatnonzero(tr['task_id']==t);h=aa[rows].numpy();d=((h[:,None]-h[None])**2).mean(2)
        d[tr['episode'][rows,None]==tr['episode'][rows][None]]=np.inf
        positives[rows]=rows[np.argsort(d,axis=1,kind='stable')[:,:3]];taskrows[t]=rows
    if validation is not None:
        vi=image_tensor(validation);vr=torch.from_numpy(validation['rs'][:,:8].copy());vt=torch.from_numpy(validation['task_id'].astype(np.int64))
        vd=query_data(validation);vp=np.c_[validation['P0'],validation['P1']]
        reference,ra=metric_eval('fold_teacher',P,vp,teacher_metric,tr,vd,base)
    start=time.perf_counter();trace=[];checks=[]
    for epoch in range(1,epochs+1):
        model.train();sums=np.zeros(5);count=0;order=rng.permutation(len(P))
        for lo in range(0,len(order),96):
            ix=order[lo:lo+96];n=len(ix);task=tt[ix];batch=augmentation(im[ix]) if CONFIGS[name].get('augmentation',True) else im[ix]
            pred,spatial,act=model.components(batch,rr[ix],task)
            lp=(((pred-yy[ix])/model.ps[task])**2).mean()
            delta=torch.bmm((pred-yy[ix]).unsqueeze(1),wf[task,:128]).squeeze(1)
            lm=(delta.square().mean(1)/scale[task]).mean()
            la=F.smooth_l1_loss(act,aa[ix]);lt=pred.new_tensor(0.)
            if token is not None:lt=F.mse_loss(spatial,token[ix])
            ip=np.array([positives[i,rng.integers(3)] for i in ix]);inn=np.array([rng.choice(taskrows[int(tr['task_id'][i])]) for i in ix])
            hp=((aa[ix].numpy()-aa[ip].numpy())**2).mean(1);hn=((aa[ix].numpy()-aa[inn].numpy())**2).mean(1)
            valid_pairs=(tr['episode'][inn]!=tr['episode'][ix])&(hn>hp+.05)
            dp=torch.bmm(torch.cat([pred-yy[ip],rr[ix]-rr[ip]],1).unsqueeze(1),wf[task]).squeeze(1).square().mean(1)/scale[task]
            dn=torch.bmm(torch.cat([pred-yy[inn],rr[ix]-rr[inn]],1).unsqueeze(1),wf[task]).squeeze(1).square().mean(1)/scale[task]
            rank=F.relu(.1+dp-dn)[torch.from_numpy(valid_pairs)].mean() if valid_pairs.any() else pred.new_tensor(0.)
            loss=lp+lm+.1*la+.5*rank+CONFIGS[name]['tokens']*lt
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            sums+=np.array([float(x.detach()) for x in (lp,lm,la,rank,lt)])*n;count+=n
        trace.append({'epoch':epoch,'losses':(sums/count).tolist(),'seconds':time.perf_counter()-start})
        if epoch==1 or epoch%20==0:print(size,name,'fold',fold,'train',trace[-1],flush=True)
        if validation is not None and (epoch in (40,80,120) or epoch==epochs):
            lp=encode(model,im,rr,tt);qp=encode(model,vi,vr,vt);metric=task_metric(np.c_[lp,state],tr)
            s,a=metric_eval(name,lp,qp,metric,tr,vd,base,ra['top'])
            audit=blindness(model,vi,vr,vt,lp,metric,tr,vd,base,s,a)
            c={'epoch':epoch,'clean':s,'reference':reference,'ratio':s['balanced_err10']/reference['balanced_err10'],'blindness':audit}
            checks.append(c);print(size,name,'fold',fold,'check',epoch,'ratio',c['ratio'],'shuffle',audit['shuffle_mean_relative_rise'],'zero',audit['zero']['relative_rise'],flush=True)
    ck={'format':'r15_qk_student_v2','state_dict':model.state_dict(),'config':CONFIGS[name],'name':name,
        'size':size,'fold':fold,'epochs':epochs,'seed':seed,'parameters':sum(p.numel() for p in model.parameters())}
    tag=f'cv{fold}' if fold>=0 else 'final'
    torch.save(ck,root/f'{tag}.pt')
    metadata={'cell':cell,'name':name,'fold':fold,'fit_episodes':sorted(map(int,np.unique(tr['episode']))),
        'heldout_episodes':sorted(map(int,held)),'fit_parent_rows_sha256':__import__('hashlib').sha256(tr['rows'].tobytes()).hexdigest(),
        'training_source_sha256':sha(DERIVED/cell/'train.npz'),'provenance':provenance,'trace':trace,'checks':checks,
        'cpu_only':True,'seconds':time.perf_counter()-start,'teacher_target_qualification':'fixed full-cell teacher PCA coordinates; fold AWM, sigma and all student normalizers fitted to fit episodes only'}
    if fold<0:
        features=encode(model,im,rr,tt);metric=task_metric(np.c_[features,state],tr)
        np.savez(root/'library_features.npz',features=features,rows=tr['rows'])
        with (root/'metric.pkl').open('wb') as f:pickle.dump(metric,f)
        ld=query_data(tr);ref,ra=metric_eval('library_teacher_fold',P,P,teacher_metric,tr,ld,base,exclude=True)
        s,a=metric_eval(name,features,features,metric,tr,ld,base,ra['top'],exclude=True)
        assert not np.any(tr['episode'][a['top']]==tr['episode'][:,None])
        np.savez(root/'loeo_eval.npz',**a);metadata['library_loeo']=s
        # Library pseudoqueries are completed before any B-dev assessment.
    dump(root/f'{tag}_train.json',metadata)
    print('DONE',size,name,fold,metadata['seconds'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,required=True);ap.add_argument('--name',choices=CONFIGS,required=True);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--epochs',type=int,default=120);a=ap.parse_args();run(a.size,a.name,a.fold,a.epochs)
