"""Library-only tiny CNN. Optional action-pair ranking; no pretrained weights."""
import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.prototype import evaluate,teacher_metric,reference_codes,task_metric

class TinySpatial(nn.Module):
    def __init__(self,pm,ps,rm,rs):
        super().__init__()
        for k,v in [('pm',pm),('ps',ps),('rm',rm),('rs',rs)]:self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
        self.conv=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,24,3,2,1),nn.SiLU(),nn.Conv2d(24,32,3,2,1),nn.SiLU())
        self.task=nn.Embedding(10,8)
        self.fc=nn.Sequential(nn.Linear(32*7*7*2+16,256),nn.SiLU(),nn.Linear(256,128))
        self.action=nn.Linear(128,35)
    def forward(self,im,rs,task):
        n=im.shape[0]
        z=self.conv(im.reshape(n*2,3,56,56)*2-1).reshape(n,-1)
        p=self.fc(torch.cat([z,torch.clamp((rs-self.rm)/self.rs,-8,8),self.task(task)],1))
        return p*self.ps+self.pm,self.action(p)

def device_setup(device):
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if device=='cpu':return {'device':'cpu'}
    reason=os.environ.get('R15_QK_GPU_CPU_INFEASIBLE','').strip()
    if not reason:
        raise RuntimeError('OWNER_ADDENDUM: CPU is default; GPU requires a recorded CPU-infeasibility reason, not a speed preference')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]
    idx,total,free=map(int,line.split(','));assert idx==0 and free>=12288, f'Insufficient GPU headroom: {line}'
    totalbytes=torch.cuda.get_device_properties(0).total_memory
    # Stricter allocator cap leaves room for CUDA context under the 4-GiB total rule.
    torch.cuda.set_per_process_memory_fraction(3*1024**3/totalbytes,0)
    torch.backends.cudnn.benchmark=False
    return {'device':'cuda:0','cpu_infeasible_reason':reason,'nvidia_smi_before':line,'allocator_cap_bytes':3*1024**3,'process_limit_bytes':4*1024**3}

def images(d):
    return torch.from_numpy(np.stack([d['img0'],d['img1']],1).transpose(0,1,4,2,3).copy()).float()/255.

def train(suite,size,objective,device,epochs,maxseconds,tag=None):
    guard=device_setup(device);torch.manual_seed(20261005);rng=np.random.default_rng(20261005)
    tr=dict(np.load(DERIVED/f'pi05_{suite}_{size}/train.npz',allow_pickle=False))
    _,base,_,_,provenance=load_cell(suite,size)
    p=np.c_[tr['P0'],tr['P1']];rs=tr['rs'][:,:8]
    model=TinySpatial(p.mean(0),np.maximum(p.std(0),.01),rs.mean(0),np.maximum(rs.std(0),.01)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    im=images(tr);rr=torch.from_numpy(rs.copy());tt=torch.from_numpy(tr['task_id'].astype(np.int64));yy=torch.from_numpy(p)
    aa=torch.from_numpy((tr['action'][:,:5,:7]/tr['sig']).reshape(len(p),35))
    wf=torch.tensor(np.stack([base.tasks[t].Wf[:128] for t in range(10)]),device=device)
    # Unit mean code energy per task, using the same cell's teacher keys.
    scale=torch.tensor([max(np.mean(base.tasks[t].Z**2),1e-6) for t in range(10)],device=device)
    positives=np.zeros((len(p),3),int);negatives={}
    for t in range(10):
        rows=np.flatnonzero(tr['task_id']==t);h=aa[rows].numpy();dh=((h[:,None]-h[None])**2).mean(2)
        dh[tr['episode'][rows,None]==tr['episode'][rows][None]]=np.inf
        positives[rows]=rows[np.argsort(dh,axis=1,kind='stable')[:,:3]]
        negatives[t]=rows
    # Fixed offline teacher features; pairs always other episodes, same task.
    start=time.perf_counter();trace=[];completed=0
    for epoch in range(epochs):
        model.train();sums=np.zeros(4);count=0
        order=rng.permutation(len(p))
        for lo in range(0,len(order),96):
            ix=order[lo:lo+96];n=len(ix);batch=im[ix].to(device);state=rr[ix].to(device);task=tt[ix].to(device);target=yy[ix].to(device)
            pred,act=model(batch,state,task)
            lp=(((pred-target)/model.ps)**2).mean()
            delta=torch.bmm((pred-target).unsqueeze(1),wf[task]).squeeze(1)
            lm=(delta.square().mean(1)/scale[task]).mean()
            la=F.smooth_l1_loss(act,aa[ix].to(device))
            rank=pred.new_tensor(0.)
            if objective=='action':
                ip=np.array([positives[i,rng.integers(3)] for i in ix])
                # Random same-task other-episode negatives; discard negatives
                # whose action is as close as the nominated positive.
                inn=np.array([rng.choice(negatives[int(tr['task_id'][i])]) for i in ix])
                valid=(tr['episode'][inn]!=tr['episode'][ix])
                hp=((aa[ix].numpy()-aa[ip].numpy())**2).mean(1)
                hn=((aa[ix].numpy()-aa[inn].numpy())**2).mean(1)
                valid &= hn>hp+.05
                xp=torch.from_numpy(p[ip]).to(device);xn=torch.from_numpy(p[inn]).to(device)
                # Full query-key distances include exact free state.
                wf_all=torch.tensor(np.stack([base.tasks[int(t)].Wf for t in tr['task_id'][ix]]),device=device)
                dp=torch.bmm(torch.cat([pred-xp,state-rr[ip].to(device)],1).unsqueeze(1),wf_all).squeeze(1).square().mean(1)/scale[task]
                dn=torch.bmm(torch.cat([pred-xn,state-rr[inn].to(device)],1).unsqueeze(1),wf_all).squeeze(1).square().mean(1)/scale[task]
                if valid.any():rank=F.relu(.1+dp-dn)[torch.from_numpy(valid).to(device)].mean()
            loss=lp+lm+.1*la+(0.5*rank if objective=='action' else 0.)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            sums+=np.array([float(lp.detach()),float(lm.detach()),float(la.detach()),float(rank.detach())])*n;count+=n
        completed=epoch+1;trace.append({'epoch':completed,'pca_mse':sums[0]/count,'metric_mse':sums[1]/count,'action_huber':sums[2]/count,'rank_hinge':sums[3]/count,'seconds':time.perf_counter()-start})
        if completed%10==0 or completed==1:print(suite,size,objective,trace[-1],flush=True)
        if time.perf_counter()-start>maxseconds:break
    model.eval();out=HERE/f'pi05_{suite}_{size}';out.mkdir(exist_ok=True)
    name=tag or (f'cnn_{objective}' if device!='cpu' or epochs>1 else 'cnn_cpu_smoke')
    torch.save({'state_dict':model.cpu().state_dict(),'objective':objective,'epochs':completed,'seed':20261005,'parameters':sum(p.numel() for p in model.parameters())},out/f'{name}.pt')
    if device!='cpu':guard.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(out/f'{name}_train.json',{'guard':guard,'trace':trace,'fit_seconds':time.perf_counter()-start,'provenance':provenance,'train_rows':len(p),'train_episodes':len(set(tr['episode']))})
    return name

def evaluate_model(suite,size,name,devroot,device):
    # Called after train's device setup in the same process, or on CPU.
    out=HERE/f'pi05_{suite}_{size}'
    tr=dict(np.load(DERIVED/f'pi05_{suite}_{size}/train.npz',allow_pickle=False))
    dp=DERIVED/f'dev_pi05_{suite}_{devroot}';mask=np.load(dp/f'unseen_{size}.npy')
    dev={k:v[mask] for k,v in dict(np.load(dp/'dev.npz',allow_pickle=False)).items()}
    _,base,_,_,_=load_cell(suite,size)
    ck=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True);sd=ck['state_dict']
    model=TinySpatial(sd['pm'],sd['ps'],sd['rm'],sd['rs']);model.load_state_dict(sd);model.to(device).eval()
    def encode(d):
        im=images(d);rs=torch.from_numpy(d['rs'][:,:8].copy());task=torch.from_numpy(d['task_id'].astype(np.int64));out=[]
        with torch.inference_mode():
            for lo in range(0,len(im),128):out.append(model(im[lo:lo+128].to(device),rs[lo:lo+128].to(device),task[lo:lo+128].to(device))[0].cpu().numpy())
        return np.concatenate(out)
    lp=encode(tr);qp=encode(dev)
    lx=np.c_[tr['P0'],tr['P1'],tr['rs'][:,:8]];qx=np.c_[qp,dev['rs'][:,:8]]
    ref=np.load(out/'eval_reference.npz')['top'];metric=teacher_metric(base);zc=reference_codes(base,tr)
    results=[]
    s,r=evaluate(f'asymmetric_{name}',lx,qx,metric,tr,dev,base,ref,libcodes=zc);results.append(s);np.savez(out/f'eval_asymmetric_{name}.npz',**r)
    sx=np.c_[lp,tr['rs'][:,:8]];sm=task_metric(sx,tr)
    s,r=evaluate(f'symmetric_{name}',sx,qx,sm,tr,dev,base,ref);results.append(s);np.savez(out/f'eval_symmetric_{name}.npz',**r)
    with (out/f'{name}_symmetric_metric.pkl').open('wb') as f:pickle.dump(sm,f)
    dump(out/f'{name}_results.json',{'methods':results,'checkpoint':str(out/f'{name}.pt'),'devroot':devroot})
    np.savez(out/f'{name}_features.npz',library=lp,query=qp)
    for s in results: print(s['name'],s['err5'],s['err10'],s['top1_agreement'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--suite',default='l10');ap.add_argument('--size',type=int,default=50);ap.add_argument('--objective',choices=['distill','action'],default='distill');ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--maxseconds',type=float,default=240);ap.add_argument('--devroot',default='r12_devlooks_50L');ap.add_argument('--tag');a=ap.parse_args()
    name=train(a.suite,a.size,a.objective,a.device,a.epochs,a.maxseconds,a.tag)
    if a.epochs>1:evaluate_model(a.suite,a.size,name,a.devroot,a.device)
