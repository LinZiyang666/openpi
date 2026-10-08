"""CPU-only timing of the final CPU-trained primary and direct-action control."""
import argparse
import pickle
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,load_cell,dump
from exp.offline_search.rounds.r15.qk.astra.neural import TinySpatial,device_setup
from exp.offline_search.rounds.r15.qk.astra.latency import CNNCode,stats

class ActionCode(nn.Module):
    def __init__(self,m,t):
        super().__init__()
        for k,v in m['ridge'][t].items():self.register_buffer(k,torch.tensor(v))
        self.register_buffer('sm',torch.tensor(m['state_norm'][t]['mean']));self.register_buffer('ss',torch.tensor(m['state_norm'][t]['std']));self.sw=m['stateweight']
    def forward(self,im,rs,task):
        g=F.avg_pool2d(im.reshape(2,3,56,56),4).permute(0,2,3,1).reshape(1,-1)
        x=torch.cat([g,rs],1);a=((x-self.xm)/self.xs)@self.coef+self.ym
        return torch.cat([a/(70**.5),self.sw*(rs-self.sm)/self.ss/(8**.5)],1)

def run(size):
    device_setup('cpu');out=HERE/f'pi05_l10_{size}';_,base,lib,rows,_=load_cell('l10',size);tr=np.load(DERIVED/f'pi05_l10_{size}/train.npz');results=[]
    ck=torch.load(out/'cnn_action_cpu.pt',map_location='cpu',weights_only=True);sd=ck['state_dict'];met=pickle.load((out/'cnn_action_cpu_symmetric_metric.pkl').open('rb'))
    aname='action_qk_a1_s1' if size==50 else 'action_qk_a0.1_s1';action=pickle.load((out/f'{aname}.pkl').open('rb'))
    for name in ('symmetric_cnn_action_cpu',aname):
        for mode in (('eager','torchscript') if name.startswith('symmetric') else ('eager',)):
            for threads in (1,4):
                torch.set_num_threads(threads);vals=[];parity=[]
                for task in (0,2,5,8):
                    i=int(np.flatnonzero(tr['task_id']==task)[0]);row=int(rows[i])
                    raw=np.stack([np.asarray(np.load(lib/f'tok/img{c}.npy',mmap_mode='r')[row]) for c in range(2)]);state=tr['rs'][i:i+1,:8].copy();tid=np.array([task],np.int64)
                    def prep():return (torch.from_numpy(raw[:,::4,::4].transpose(0,3,1,2).copy()).float()[None]/255.,torch.from_numpy(state),torch.from_numpy(tid))
                    if name.startswith('symmetric'):
                        cnn=TinySpatial(sd['pm'],sd['ps'],sd['rm'],sd['rs']);cnn.load_state_dict(sd);net=CNNCode(cnn,base,task)
                        net.wf.copy_(torch.tensor(met[task][0]['wf']));net.shift.copy_(torch.tensor(met[task][0]['shift']))
                    else:net=ActionCode(action,task)
                    net.eval();inp=prep()
                    with torch.inference_mode():
                        reference=net(*inp)
                        if mode=='torchscript':net=torch.jit.freeze(torch.jit.trace(net,inp,check_trace=True))
                        delta=float(torch.max(torch.abs(net(*inp)-reference)));assert delta<1e-4;parity.append(delta)
                        for _ in range(20):net(*inp)
                        for _ in range(100):
                            start=time.perf_counter_ns();y=net(*prep()).numpy();vals.append((time.perf_counter_ns()-start)/1e6)
                s={'name':name,'device':'cpu','mode':mode,'cpu_threads':threads,'size':size,'end_to_end':stats(vals),'max_abs_export_parity':max(parity)};s['epsilon_pi05_graph']=.152*s['end_to_end']['median_ms']/10.26;results.append(s);print(size,name,mode,threads,s['end_to_end']['median_ms'],flush=True)
    dump(HERE/f'latency_cpu_extra_{size}.json',{'device':'cpu','gpu_used':False,'samples_per_variant':400,'input':'both 224x224 CPU RGB -> nearest56/conversion -> encoder and main metric -> CPU code; excludes common retrieval synthesis','results':results})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=50);a=ap.parse_args();run(a.size)
