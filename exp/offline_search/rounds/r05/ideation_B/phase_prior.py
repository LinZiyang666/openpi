"""Moment fit of a local clock-offset prior from demo action-matched cross-episode pairs.
This is a latent-phase PROXY, not identified real phase, and does not justify changing blind control.
Gaussian state residual plus categorical offset prior implies MSE penalty 2*sigma2/8*log(p0/pdelta).
"""
import pathlib,json,numpy as np
S=pathlib.Path('/home/weiland/trace_runs/offline_search_store');O=pathlib.Path(__file__).parent
def a(p,k):return np.load(p/(k+'.npy'),mmap_mode='r')
out=[]
for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
 for scale in [50,500]:
  lib='current' if scale==50 else ('bpool_cs' if cell.startswith('pi05') else 'bpool_all');p=S/'library'/cell/lib
  ep=a(p,'episode');task=a(p,'task_id');nxt=a(p,'next');rs=np.array(a(p,'rs')[:,:8],float);act=np.array(a(p,'action')[:,:5,:7],float);sig=act.reshape(-1,7).std(0);H=(act/sig).reshape(len(ep),-1);tt=[]
  for t in range(10):
   rr=np.flatnonzero(task==t);ht=H[rr];dh=np.maximum(np.sum(ht**2,1)[:,None]+np.sum(ht**2,1)[None,:]-2*ht@ht.T,0);dh[ep[rr,None]==ep[rr][None,:]]=np.inf
   ix=rr[np.argsort(dh,axis=1)[:,:3]];q=np.repeat(rr,3);c=ix.ravel();c1=nxt[c];c2=nxt[np.maximum(c1,0)];qn=nxt[q];ok=(qn>=0)&(c1>=0)&(c2>=0);q=q[ok];c=c[ok];cn=np.stack([c[...],c1[ok],c2[ok]],axis=1);qn=qn[ok]
   hd=np.mean((H[cn]-H[qn,None])**2,axis=2);choice=np.argmin(hd,axis=1);counts=np.bincount(choice,minlength=3);prob=counts/counts.sum();std=np.maximum(rs[rr].std(0),.05);res=(rs[cn[np.arange(len(cn)),choice]]-rs[qn])/std;var=float(np.mean(res**2))
   pen=(2*var/8*np.log(prob[1]/prob)).tolist();tt.append({'task':t,'pairs':len(q),'counts':counts.tolist(),'offsets':[-1,0,1],'prob':prob.tolist(),'state_sigma2':var,'penalties':pen,'symmetric_lambda':float((pen[0]+pen[2])/2)})
  out.append({'cell':cell,'scale':scale,'tasks':tt,'median_symmetric_lambda':float(np.median([t['symmetric_lambda'] for t in tt])),'tasks_negative_penalty':sum(t['symmetric_lambda']<0 for t in tt)})
  print(cell,scale,out[-1]['median_symmetric_lambda'],flush=True)
(O/'phase_results.json').write_text(json.dumps(out,indent=2))
