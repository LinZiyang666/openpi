"""Batch-one raw-RGB -> CPU retrieval code, CPU and capped local GPU."""
import argparse
import json
import pickle
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,REPO,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.neural import TinySpatial,device_setup

class LinearKeys(nn.Module):
    def __init__(self,model,kind,task,base):
        super().__init__();self.kind=kind
        if kind=='symmetric':
            for c,(mu,v) in enumerate(model['pca']):
                self.register_buffer(f'm{c}',torch.tensor(mu));self.register_buffer(f'v{c}',torch.tensor(v))
            met=model['metric'][task][0]
        else:
            p=model[task]
            for k,v in p.items():
                self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
            met={'wf':base.tasks[task].Wf,'shift':base.tasks[task].shift}
        self.register_buffer('wf',torch.tensor(met['wf']));self.register_buffer('shift',torch.tensor(met['shift']))
    def forward(self,im,rs,task):
        g=F.avg_pool2d(im.reshape(2,3,56,56),4).permute(0,2,3,1).reshape(2,-1)
        if self.kind=='symmetric':p=torch.cat([(g[0]-self.m0)@self.v0,(g[1]-self.m1)@self.v1])[None]
        elif self.kind=='ridge':
            x=torch.cat([g.reshape(1,-1),rs],1);p=((x-self.xm)/self.xs)@self.coef+self.ym
        else:
            x=(g.reshape(1,-1)-self.xm)/self.xs
            d=torch.clamp(x.square().sum(1,keepdim=True)+self.center2[None]-2*x@self.centers.T,min=0)
            k=torch.exp(-d/self.ell)
            r=torch.cat([torch.ones_like(rs[:,:1]),(rs-self.rm)/self.rs],1)
            p=r@self.rb+k@self.coef
        return torch.cat([p,rs],1)@self.wf-self.shift

class CNNCode(nn.Module):
    def __init__(self,model,base,task):
        super().__init__();self.model=model
        self.register_buffer('wf',torch.tensor(base.tasks[task].Wf));self.register_buffer('shift',torch.tensor(base.tasks[task].shift))
    def forward(self,im,rs,task):
        p,_=self.model(im,rs,task)
        return torch.cat([p,rs],1)@self.wf-self.shift

def stats(v):
    return {'n':len(v),'median_ms':float(np.median(v)),'q1_ms':float(np.quantile(v,.25)),'q3_ms':float(np.quantile(v,.75)),'p95_ms':float(np.quantile(v,.95)),'samples_ms':v}

def run(device,size,reps):
    guard=device_setup(device);out=HERE/f'pi05_l10_{size}';m,base,lib,rows,_=load_cell('l10',size)
    prices=json.loads((REPO/'exp/offline_search/rounds/r12/prep/price_table.json').read_text())
    tr=np.load(DERIVED/f'pi05_l10_{size}/train.npz')
    names=[('symmetric_rgb_spatial_awm','symmetric'),('asymmetric_ridge_0.1' if size==50 else 'asymmetric_ridge_0.01','ridge'),('local_residual_krr_l1_a0.01','kernel'),('cnn_distill','cnn')]
    result=[]
    for name,kind in names:
        if not (out/f'{name}.pt' if kind=='cnn' else out/f'{name}.pkl').exists():continue
        modeldata=torch.load(out/f'{name}.pt',map_location='cpu',weights_only=True) if kind=='cnn' else pickle.load((out/f'{name}.pkl').open('rb'))
        for nthreads in ([1,4] if device=='cpu' else [1]):
            torch.set_num_threads(nthreads);samples=[];core=[]
            for task in (0,2,5,8):
                r=int(np.flatnonzero(tr['task_id']==task)[0]);pr=int(rows[r])
                raw=np.stack([np.asarray(np.load(lib/f'tok/img{c}.npy',mmap_mode='r')[pr]) for c in range(2)])
                state=np.array(tr['rs'][r:r+1,:8]);tid=np.array([task],np.int64)
                if kind=='cnn':
                    sd=modeldata['state_dict'];cnn=TinySpatial(sd['pm'],sd['ps'],sd['rm'],sd['rs']);cnn.load_state_dict(sd);model=CNNCode(cnn,base,task)
                else:model=LinearKeys(modeldata,kind,task,base)
                model.to(device).eval()
                def prep():
                    im=torch.from_numpy(raw[:,::4,::4].transpose(0,3,1,2).copy()).float().unsqueeze(0)/255.
                    return im.to(device),torch.from_numpy(state).to(device),torch.from_numpy(tid).to(device)
                with torch.inference_mode():
                    inp=prep()
                    for _ in range(20):model(*inp).cpu().numpy()
                    if device!='cpu':torch.cuda.synchronize()
                    for _ in range(reps):
                        t=time.perf_counter_ns();args=prep();y=model(*args).cpu().numpy()
                        if device!='cpu':torch.cuda.synchronize()
                        samples.append((time.perf_counter_ns()-t)/1e6)
                    for _ in range(reps):
                        if device!='cpu':torch.cuda.synchronize()
                        t=time.perf_counter_ns();y=model(*inp)
                        if device!='cpu':torch.cuda.synchronize()
                        core.append((time.perf_counter_ns()-t)/1e6)
                del model
            s={'name':name,'device':device,'cpu_threads':nthreads,'size':size,'end_to_end':stats(samples),'device_resident':stats(core)}
            ms=s['end_to_end']['median_ms']
            s['epsilon_pi05_graph']=.152*ms/10.26
            s['epsilon_pi05_r12_eager']=.152*ms/prices['pi05']['normalization_full_ms']
            s['epsilon_groot_l10_r12_eager']=.148*ms/prices['groot_l10']['normalization_full_ms']
            s['epsilon_groot_spatial_r12_eager']=.148*ms/prices['groot_spatial']['normalization_full_ms']
            result.append(s);print(name,device,nthreads,ms,flush=True)
    if device!='cpu':guard.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(HERE/f'latency_{device}_{size}.json',{'guard':guard,'batch':1,'cameras':2,'input':'224x224 uint8 CPU RGB each; includes nearest56, float conversion, host-device copy, light encoder, task metric, code copy to CPU; excludes neighbor synthesis shared by baseline','warmups_per_task':20,'tasks':[0,2,5,8],'results':result})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--size',type=int,default=50);ap.add_argument('--reps',type=int,default=100);a=ap.parse_args();run(a.device,a.size,a.reps)
