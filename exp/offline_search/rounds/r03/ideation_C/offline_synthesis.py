"""Top-10-only counterfactual synthesis screen; not a closed-loop experiment."""
import json
import numpy as np
from diagnose import OUT, STORE, RESULTS, arr, mean

def metrics(a, gt, sig, oracle, conf, floor):
    err=np.sqrt(np.mean(((a-gt)/sig)**2,axis=(1,2)))
    order=np.argsort(-conf,kind='stable')
    out={'err_mean':mean(err),'err_median':float(np.median(err)),
         'regret_mean':mean(err-oracle),'grip_mis':mean(np.mean((a[:,:,6]>=0)!=(gt[:,:,6]>=0),axis=1)),
         'aurc_fixed_base_conf':mean(np.cumsum(err[order])/np.arange(1,len(err)+1)),
         'bad_rate_global_p90':mean(err>floor)}
    return out,err

def main():
    out={}
    for method,lib,kr in [('AWM_joint_cur_fcur_kr5','current',5),('AWM_joint_cur_fbig','current',8),('AWM_joint_big_fbig','big',8)]:
      for model in ['pi05','groot']:
       for suite in ['spatial','l10']:
        for arm in ['cache','inf']:
         cell=f'{model}_{suite}_{arm}'
         p=RESULTS/method/(cell+'.npz')
         if not p.exists():continue
         z=np.load(p); jj=json.loads(p.with_suffix('.json').read_text());sig=np.array(jj['sigma'])
         ln=lib if lib!='big' else ('bpool_cs' if model=='pi05' else 'bpool_all')
         lp=STORE/'library'/f'{model}_{suite}'/ln
         a=arr(lp,'action'); ep=arr(lp,'episode'); rows=z['topk']; scores=z['topk_scores']
         gt=np.array(arr(STORE/'queries'/cell,'a_inf')[z['row'],:5,:7])
         heads=np.array(a[rows,:5,:7]); ee=ep[rows]
         w=np.exp(-((scores-scores[:,:1])/np.maximum(scores[:,0]-scores[:,kr-1],1e-6)[:,None])**2)
         w=w/w.sum(1,keepdims=True)
         base=np.einsum('nk,nktd->ntd',w,heads)
         ew=np.sum((ee[:,:,None]==ee[:,None,:])*w[:,None,:],axis=2)
         ep_eff=1/np.sum(w*ew,axis=1)
         grip=np.where(heads[:,:,:,6]>=0,1.,-1.)
         vote=np.einsum('nk,nkt->nt',w,grip)
         signcode=((grip>0)*2**np.arange(5)).sum(2)
         classw=np.sum((signcode[:,:,None]==signcode[:,None,:])*w[:,None,:],axis=2)
         mode=signcode[np.arange(len(w)),np.argmax(classw,axis=1)]
         schemes={'top10_base':w}
         for gamma in [.5,1.]:schemes[f'epbalance_{gamma}']=w/np.maximum(ew,1e-30)**gamma
         schemes['mode5']=w*(signcode==mode[:,None])
         schemes['mode5_ambig08']=np.where((np.min(abs(vote),axis=1)<.8)[:,None],schemes['mode5'],w)
         m={'base_exact':{k:jj['metrics'][k] for k in ['err_mean','err_median','regret_mean','aurc','bad_c100','grip_mis']},
            'ep_eff_mean':mean(ep_eff),'ep_eff_lt15':mean(ep_eff<1.5),'split08':mean(abs(vote[:,0])<.8),
            'reconstruction_rms_sigma':mean(np.sqrt(np.mean(((base-z['synth_seg'])/sig)**2,axis=(1,2)))),
            'variants':{}}
         for name,vw in schemes.items():
            vw=vw/vw.sum(1,keepdims=True)
            syn=np.einsum('nk,nktd->ntd',vw,heads)
            mm,err=metrics(syn,gt,sig,z['oracle_err'],z['confidence'],jj['metrics']['floor_p90'])
            mm['changed_frac']=mean(np.any(abs(syn-base)>1e-6,axis=(1,2)))
            mm['per_task_err']={str(t):mean(err[z['task_id']==t]) for t in np.unique(z['task_id'])}
            mm['per_task_grip_mis']={str(t):mean(np.mean((syn[z['task_id']==t,:,6]>=0)!=(gt[z['task_id']==t,:,6]>=0),axis=1)) for t in np.unique(z['task_id'])}
            m['variants'][name]=mm
         out[f'{method}/{cell}']=m
         print(method,cell,'base',round(m['variants']['top10_base']['err_mean'],4),
               'epbalance',round(m['variants']['epbalance_0.5']['err_mean'],4),
               'mode',round(m['variants']['mode5']['err_mean'],4),flush=True)
    (OUT/'offline_synthesis.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
