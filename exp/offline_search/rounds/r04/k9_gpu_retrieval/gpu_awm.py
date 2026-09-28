"""Fixed-shape GPU AWM and guard-only MixedJudge features. No serving hooks.

forward inputs all have a leading batch dimension; task_id/step/prev_hit are
int64 tensors (prev_hit=0 means MISS, -1 unknown). History inputs are the last
real vision keys/state, previous executed full chunk, and prior stuck count.
A caller owns episode reset and accepted-execution history. Outputs stay on GPU.

The supported fits are pi05 joint AWM k=16, full-rank early map, no insurance,
and guard-only MixedJudge without recovery/blend/events/burst. Unsupported fits
fail at construction. Default precision follows CPU float32 distance arithmetic
and float64 kernel/confidence arithmetic. precision='float64' additionally
promotes projection and distance math; this is a diagnostic variant, not bitwise
CPU emulation. Disable TF32 before capture (benchmark does this explicitly).

Stable full sort orders exact ties by task-row position, hence global row for
existing ascending blocks. MixedJudge guarantees this also at rank 16. Pure CPU
AWM argpartition does not guarantee that boundary tie set; report such cases.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _dot(a,b):
    return (a*b).sum(-1)


def _mv(a,b):
    return torch.bmm(a,b.unsqueeze(-1)).squeeze(-1)


class GPUAWM(nn.Module):
    @classmethod
    def from_pickle(cls,path,library=None,**kwargs):
        """Read a deployed plugin fit, never refit or mutate it (returns CPU buffers)."""
        from exp.offline_search.rounds.r04.k9_gpu_retrieval.common import load
        return cls(load(path),library,**kwargs)

    def __init__(self, method, library=None, *, spare_fraction=.25, spare_rows=64,
                 precision='float32'):
        super().__init__()
        b=getattr(method,'base',method)
        assert b.model=='pi05' and b.features=='joint' and b.feat0=='joint'
        assert b.k==16 and b.early and not b.codes and not b.norm_cap and not b.hyst
        assert precision in ('float32','float64')
        self.judge=hasattr(method,'base');self.precision=precision
        self.calc_dtype=torch.float64 if precision=='float64' else torch.float32
        if self.judge:
            assert not method.blend and not method.recover and not method.events and not method.burst
            assert method.guards
        self.k=16;self.kref=b.kref;self.lam_c=b.lam_c;self.H=b.H
        self.task_ids=sorted(b.tasks)
        assert self.task_ids==list(range(len(self.task_ids)))
        nt=len(self.task_ids);maxn=max(len(t.rows) for t in b.tasks.values())
        self.capacity=int(math.ceil(max(maxn*(1+spare_fraction),maxn+spare_rows)/64)*64)
        self._counts=[len(b.tasks[t].rows) for t in self.task_ids]
        assert min(self._counts)>=self.k
        for name in ('B0T','B1T','mu0','mu1','muB0','muB1','sig'):
            self._buf(name,getattr(b,name),self.calc_dtype if name!='sig' else torch.float32)
        for name in ('Wf','shift','W0f','c0','A0') + (('Vm0','Vm1') if self.judge else ()):
            self._buf(name,np.stack([getattr(b.tasks[t],name) for t in self.task_ids]),self.calc_dtype)
        assert all(b.tasks[t].As0 is None and b.tasks[t].Z0 is None for t in self.task_ids)
        self.row_fields=['Z','z2','n20','HD','h2','RS','rs2',
                         'actions','rows','lib_step','lib_ep','progress','ep_len','next_row']
        if self.judge:self.row_fields+=['V0','V1']
        for name in ('Z','z2','n20','HD','h2','RS','rs2') + (('V0','V1') if self.judge else ()):
            self._padded(name,[getattr(b.tasks[t],name) for t in self.task_ids],self.calc_dtype)
        for name,value,dt in [('actions',b.act,torch.float32),('rows',np.arange(len(b.act)),torch.int64),
                              ('lib_step',b.lib_step,torch.int64),('lib_ep',b.lib_ep,torch.int64)]:
            self._padded(name,[value[b.tasks[t].rows] for t in self.task_ids],dt)
        c=method.C if self.judge else library
        if c is None: raise ValueError('Pure AWM requires the existing LibraryView for exact progress/length/next metadata')
        for name,field,dt in [('progress','prog' if self.judge else 'progress',torch.float64),
                              ('ep_len','ep_len',torch.int64),('next_row','nxt' if self.judge else 'next',torch.int64)]:
            self._padded(name,[np.asarray(getattr(c,field))[b.tasks[t].rows] for t in self.task_ids],dt)
        self._buf('valid',np.arange(self.capacity)[None,:]<np.array(self._counts)[:,None],torch.bool)
        self._buf('counts',self._counts,torch.int64)
        for name in ('s_c','s_d'):
            self._buf(name,[getattr(b.tasks[t],name) for t in self.task_ids],torch.float64)
        for name in ('zmu','zsd'):
            self._buf(name,[getattr(b,name)[k] for k in ('d1','disp','dst')],torch.float64)
        self._buf('zs_sd',b.zs_sd,torch.float64);self._buf('s_a',b.s_a,torch.float64)
        # All kernels have immutable shape after construction, including calibration.
        if self.judge:
            m=method
            for name in ('M0','M1'):
                self._buf(name,np.stack([getattr(m,name)[t] for t in self.task_ids]),torch.float32)
            self._buf('sigma',m.sigma,torch.float64)
            self._padded('judge_HD',[m.C.HD[b.tasks[t].rows] for t in self.task_ids],torch.float32)
            self.row_fields.append('judge_HD')
            self._buf('med_len',[m.med_len[t] for t in self.task_ids],torch.float64)
            self._buf('m_thr',m.m_thr,torch.float64);self._buf('c_thr',m.c_thr,torch.float64)
            self._buf('pairs_i',np.triu_indices(5,1)[0],torch.int64)
            self._buf('pairs_j',np.triu_indices(5,1)[1],torch.int64)
            for reg in (0,1,2):
                cal=m.cal.get(reg) or m.cal[2]
                for name in ('mu','sd','w','kx','ky'):
                    self._buf(f'cal{reg}_{name}',cal[name],torch.float64)
            self.fresh_borrowed=m.cal.get(1) is None

    def _buf(self,name,value,dtype):
        self.register_buffer(name,torch.as_tensor(np.array(value,copy=True),dtype=dtype))

    def _padded(self,name,arrays,dtype):
        dst=torch.zeros((len(arrays),self.capacity,*arrays[0].shape[1:]),dtype=dtype)
        for i,a in enumerate(arrays):dst[i,:len(a)].copy_(torch.as_tensor(np.array(a,copy=True),dtype=dtype))
        self.register_buffer(name,dst)

    @torch.no_grad()
    def append_prepared(self,task:int,fields:dict[str,torch.Tensor]):
        """Append already-projected/fitted rows with copy_, publish valid mask last.

        Host-side maintenance ONLY, outside capture, after prior graph completion.
        Same task metric/means/calibration remain fixed; this is not a refit. All
        row fields are mandatory, global IDs must extend ascending row order.
        Synchronize serving streams before publishing; caller owns this boundary.
        No buffer is replaced, hence captured graph addresses remain valid.
        """
        if set(fields)!=set(self.row_fields):raise ValueError('all row_fields required')
        n=fields['rows'].shape[0];lo=self._counts[task];hi=lo+n
        if hi>self.capacity:raise ValueError('fixed capacity exhausted; new graph required')
        # Validate every shape BEFORE any write. Ordering/metric consistency is the caller's contract.
        for name in self.row_fields:
            target=getattr(self,name)[task,lo:hi]
            if fields[name].shape!=target.shape:raise ValueError(name+' shape mismatch')
        for name in self.row_fields:getattr(self,name)[task,lo:hi].copy_(fields[name])
        self.counts[task].fill_(hi);self.valid[task,lo:hi].fill_(True);self._counts[task]=hi

    def _calibrate(self,reg,features):
        mu=getattr(self,f'cal{reg}_mu');sd=getattr(self,f'cal{reg}_sd');w=getattr(self,f'cal{reg}_w')
        x=torch.where(torch.isfinite(features),features,mu)
        z=(((x-mu)/sd)*w).sum(-1)
        kx=getattr(self,f'cal{reg}_kx');ky=getattr(self,f'cal{reg}_ky')
        # searchsorted is capture-safe, and returns fixed [B] output.
        j=torch.searchsorted(kx,z.contiguous(),right=True).clamp(1,len(kx)-1)
        f=(z-kx[j-1])/(kx[j]-kx[j-1]).clamp_min(1e-300)
        pred=ky[j-1]+f.clamp(0,1)*(ky[j]-ky[j-1])
        return -pred+1e-6*z,z,pred

    def forward(self,pooled_key_v0,pooled_key_v1,rs,task_id,step,prev_hit,prev_a_exec,
                prev_key_v0,prev_key_v1,prev_rs,stuck_prev):
        t=task_id;valid=self.valid[t];cnt=self.counts[t]
        k0=pooled_key_v0.float().to(self.calc_dtype);k1=pooled_key_v1.float().to(self.calc_dtype)
        p0=F.linear(k0,self.B0T)-self.muB0;p1=F.linear(k1,self.B1T)-self.muB1
        # The deployed QueryView casts state to f32 even when stage1.state is f64.
        rs32=rs[:,:8].float();rs8=rs32.to(self.calc_dtype)
        x=torch.cat((p0,p1,rs8),-1)
        z=_mv(self.Wf[t].transpose(-1,-2),x)-self.shift[t]
        y=_mv(self.W0f[t].transpose(-1,-2),x)-self.c0[t]
        Z=self.Z[t]
        dmain=self.z2[t]-2.0*_mv(Z,z)+_dot(z,z)[:,None]
        dearly=self.n20[t]-2.0*_mv(Z,_mv(self.A0[t],y))+_dot(y,y)[:,None]
        d=torch.sqrt(torch.where((step==0)[:,None],dearly,dmain).clamp_min(0))
        d=d.masked_fill(~valid,float('inf'))
        ds=torch.sort(d,dim=1).values
        med=(ds.gather(1,((cnt-1)//2)[:,None])+ds.gather(1,(cnt//2)[:,None]))[:,0]*.5
        # CPU median is f32; adding 1e-12 then division casts scalar back to f32.
        med=med+1e-12
        tail=(prev_a_exec[:,5:10,:7]/self.sig).flatten(1).to(self.calc_dtype)
        c=torch.sqrt((self.h2[t]-2.0*_mv(self.HD[t],tail)+_dot(tail,tail)[:,None]).clamp_min(0)/35)
        fresh=(step>0)&(prev_hit==0)
        dt=torch.where(fresh[:,None],d/med[:,None]+self.lam_c*c/self.s_c[t].to(self.calc_dtype)[:,None],d)
        idx=torch.argsort(dt,dim=1,stable=True)[:,:self.k]
        dk=dt.gather(1,idx).double()
        rel=dk-dk[:,:1];ref=rel[:,self.kref-1:self.kref].clamp_min(1e-6)
        score=-(rel/ref).square();w=score.exp();wn64=w/w.sum(1,keepdim=True);wn=wn64.float()
        chunks=self.actions[t[:,None],idx]
        action=(chunks*wn[:,:,None,None]).sum(1)
        hd=self.HD[t[:,None],idx[:,:5]]
        head=(action[:,:5,:7]/self.sig).flatten(1).to(self.calc_dtype)
        disp5=torch.sqrt((hd-head[:,None,:]).square().mean((1,2))).double()
        ds2=self.rs2[t]-2.0*_mv(self.RS[t],rs8)+_dot(rs8,rs8)[:,None]
        dst=ds2.masked_fill(~valid,float('inf')).min(1).values.clamp_min(0).double().sqrt()/self.s_d[t]
        d1=d.min(1).values.double();c0=c.gather(1,idx[:,:1])[:,0].double()
        zd=(-disp5-self.zmu[1])/self.zsd[1]
        stale_conf=(-d1-self.zmu[0])/self.zsd[0]+zd+(-dst-self.zmu[2])/self.zsd[2]
        awm_conf=torch.where(step==0,self.zs_sd*zd,torch.where(fresh,-c0/self.s_c[t]-disp5/self.s_a,stale_conf))
        regime=torch.where(step==0,0,torch.where(fresh,1,2))
        top=idx[:,0]
        out=dict(action=action,topk=self.rows[t[:,None],idx],scores=score if self.judge else -dk,
                 confidence=awm_conf,awm_confidence=awm_conf,d1=d1,d1_rel=d1/med.double(),disp5=disp5,dst=dst,
                 c0=c0,regime=regime,w_eff=1/wn.double().square().sum(1),weights=wn,
                 lib_ep=self.lib_ep[t,top],lib_step=self.lib_step[t,top],
                 top1_prog=self.progress[t,top],term1=(self.next_row[t,top]<0),
                 ep_len=self.ep_len[t,top])
        still=[]
        for cur,prv,mu in ((pooled_key_v0,prev_key_v0,self.mu0),(pooled_key_v1,prev_key_v1,self.mu1)):
            a=cur.float()-mu.float();b=prv.float()-mu.float()
            # AWM diagnostic multiplies squared norms in f32 before sqrt;
            # MixedJudge below deliberately uses Python/f64 scalar products.
            still.append(_dot(a,b).double()/(_dot(a,a)*_dot(b,b)).sqrt().double().clamp_min(1e-12))
        out['still']=still[0]+still[1]
        if self.judge:
            # Raw task-centred camera cosine is distinct from candidate PCA cosine.
            cs=[]
            for cur,prv,mu in ((pooled_key_v0,prev_key_v0,self.M0[t]),(pooled_key_v1,prev_key_v1,self.M1[t])):
                a=cur.float()-mu;b=prv.float()-mu
                cs.append(_dot(a,b).double()/(_dot(a,a).double()*_dot(b,b).double()).sqrt().clamp_min(1e-12))
            vself=torch.minimum(cs[0],cs[1])
            motion=(rs32-prev_rs[:,:8].float()).square().sum(1).sqrt().double()
            stuck=torch.where((step>0)&(motion<self.m_thr)&(vself>=self.c_thr),stuck_prev+1,0)
            vis=[]
            for pp,vm,v in ((p0,self.Vm0[t],self.V0[t]),(p1,self.Vm1[t],self.V1[t])):
                pc=pp-vm;pc=pc/_dot(pc,pc).sqrt().clamp_min(1e-12)[:,None]
                vis.append(_mv(v,pc).masked_fill(~valid,-float('inf')).max(1).values.double())
            vis=vis[0]+vis[1]
            # V7 uses direct state differences, NOT AWM's expanded squared distance.
            dnn=(self.RS[t].float()-rs32[:,None,:]).square().sum(-1).masked_fill(~valid,float('inf')).min(1).values.sqrt().double()
            heads=self.judge_HD[t[:,None],idx[:,:5]]
            diff=heads[:,self.pairs_i]-heads[:,self.pairs_j]
            disp=(diff.square().sum(-1)/35).sqrt().mean(1).double()
            lag=step.double()-self.lib_step[t[:,None],idx[:,:5]].double().mean(1)
            overtime=step.double()/self.med_len[t]
            h=(action[:,:5,:7].double()/self.sigma).flatten(1)
            ta=(prev_a_exec[:,5:10,:7].double()/self.sigma).flatten(1)
            cont=(h-ta).square().mean(1).sqrt()
            fs=torch.stack((dnn,disp,overtime,vis,stuck.clamp_max(5).double(),lag.abs()),-1)
            ff=fs if self.fresh_borrowed else torch.stack((cont,disp,vis),-1)
            cf=[self._calibrate(0,fs),self._calibrate(1,ff),self._calibrate(2,fs)]
            for j,name in enumerate(('confidence','zsum','pred_err')):
                out[name]=torch.where(step==0,cf[0][j],torch.where(fresh,cf[1][j],cf[2][j]))
            out.update(dnn=dnn,disp=disp,vis=vis,motion=motion,vself=vself,stuck_n=stuck,
                       lag=lag,overtime=overtime,cont=cont,
                       vote=(wn64*torch.where(self.judge_HD[t[:,None],idx][:,:,6]>=0,1.,-1.)).sum(1))
        return out

    def resident_bytes(self):
        return sum(b.numel()*b.element_size() for b in self.buffers())


def verdict_packet(out):
    """One contiguous f64 buffer of verdict inputs for one asynchronous D2H copy.

    Fixed names; integer metadata is exactly represented at these small ranges.
    Full diagnostics/top-16 are kept on GPU and excluded from serving transfers.
    """
    names=['confidence','d1','disp5','dst','regime','lib_ep','lib_step','top1_prog','term1','ep_len']
    if 'stuck_n' in out:
        names+=['dnn','disp','vis','motion','vself','stuck_n','lag','overtime','cont','vote','zsum','pred_err']
    return torch.stack([out[n].double() for n in names],-1)
