"""Diagnose virtual control-step entries: chord geometry and repeated transition chunks."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.rounds.r04.ideation_C.library_diagnostic import OUT,load_library,summary
from exp.offline_search.rounds.r04.ideation_C.log_diagnostic import records

def work(args):
    model,suite,scale=args; lib,x=load_library(*args)
    data=np.load(OUT/f'codes_{model}_{suite}_{scale}.npz')
    z=np.empty_like(x)
    for t in lib.tasks():
        ix=data[f't{t}_rows']; z[ix]=data[f't{t}_z']
    prev=np.asarray(lib.prev);nxt=np.asarray(lib.next)
    ix=np.flatnonzero((prev>=0)&(nxt>=0));a=z[prev[ix]];b=z[nxt[ix]];q=z[ix];d=b-a
    lam=np.clip(np.sum((q-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-12),0,1)
    chord=a+lam[:,None]*d
    nearest=np.minimum(np.linalg.norm(q-a,axis=1),np.linalg.norm(q-b,axis=1))
    ratio=np.linalg.norm(q-chord,axis=1)/np.maximum(nearest,1e-9)
    midratio=np.linalg.norm(q-(a+b)/2,axis=1)/np.maximum(nearest,1e-9)
    sg=np.sign(lib.action[:,:5,6]); changes=(sg[:,1:]!=sg[:,:-1]).sum(1)
    boundary=sg[prev[ix],-1]!=sg[ix,0]
    grp=changes[ix]>0
    geometry=[]
    for name,mask in [('all',np.ones(len(ix),bool)),('gripper_event',grp|boundary),('no_gripper_event',~(grp|boundary))]:
        geometry.append(dict(group=name,n=int(mask.sum()),ratio=summary(ratio[mask]),midpoint_ratio=summary(midratio[mask]),
                             segment_parameter=summary(lam[mask]),ratio_lt_half=float((ratio[mask]<.5).mean())))
    # Raw PCA metric, independent of action whitening, checks that fit labels did not fabricate smoothness.
    vv=x[:,:128]
    dv=vv[nxt[ix]]-vv[prev[ix]]
    lv=np.clip(np.sum((vv[ix]-vv[prev[ix]])*dv,axis=1)/np.maximum(np.sum(dv*dv,axis=1),1e-12),0,1)
    vc=vv[prev[ix]]+lv[:,None]*dv
    vr=np.linalg.norm(vv[ix]-vc,axis=1)/np.maximum(np.minimum(np.linalg.norm(vv[ix]-vv[prev[ix]],axis=1),np.linalg.norm(vv[ix]-vv[nxt[ix]],axis=1)),1e-9)
    # Adjacent rows' action head is observed execution. Splices at offsets 1..4 contain only observed controls.
    observed_switches=int(changes.sum()); intra_rows=int((changes>0).sum())
    mletter='p' if model=='pi05' else 'g';ss='sp' if suite=='spatial' else 'l10'
    arm=f'oscl{scale}_{mletter}_{ss}_cl2';j,rec=records(f'r02_g{scale}',arm)
    rr={}
    for (u,s),v in sorted(rec.items()):rr.setdefault(u,[]).append(v)
    logs=[]
    for success in [False,True]:
        n=internal=repeat=repeat_event=0;eps_event=set();spells=spell_event=0
        for u,rows in rr.items():
            if bool(j[u]['success'])!=success:continue
            heads=[];top=[]
            for r in rows:
                kk=np.asarray(r['topk']);s=np.asarray(r['scores']);kr=5 if scale==50 else 8
                w=np.exp(-((s-s[0])/max(s[0]-s[kr-1],1e-6))**2);w/=w.sum()
                head=(w[:,None]*lib.action[kk,:5,6]).sum(0)>=0
                heads.append(head);top.append(r['top1'])
            heads=np.array(heads);inside=(heads[:,1:]!=heads[:,:-1]).any(1)
            n+=len(rows);internal+=int(inside.sum())
            i=0
            while i<len(rows):
                k=i+1
                while k<len(rows) and top[k]==top[i]: k+=1
                if k-i>=3:
                    spells+=1;repeat+=k-i
                    repeat_event+=int(inside[i:k].sum())
                    if inside[i:k].sum()>=2: spell_event+=1;eps_event.add(u)
                i=k
        logs.append(dict(success=success,decisions=n,intra_event_decisions=internal,spell_decisions=repeat,
                         repeated_intra_event_decisions=repeat_event,spells=spells,
                         spells_with_repeated_events=spell_event,episodes_with_repeated_events=len(eps_event)))
    result=dict(model=model,suite=suite,scale=scale,L=lib.L,virtual_entries=lib.L+4*int((nxt>=0).sum()),
                library_intra_switches=observed_switches,library_intra_switch_rows=intra_rows,
                geometry=geometry,raw_visual_chord_ratio=summary(vr),logs=logs)
    (OUT/f'segment_{model}_{suite}_{scale}.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    jobs=[(m,s,n) for m in ['pi05','groot'] for s in ['spatial','l10'] for n in [50,500]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for r in pool.map(work,jobs):print(json.dumps(r),flush=True)
