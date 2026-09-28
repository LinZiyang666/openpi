"""Array census, task capacities, actual MISS yields, and focused arm usage."""
from pathlib import Path
from collections import Counter
import json, math
import numpy as np
from exp.offline_search.rounds.r05.ideation_C.library_study import OUT,ROOT,lib,method,CELLS

def arrays(obj, path='', seen=None):
    if seen is None:seen=set()
    if id(obj) in seen:return []
    seen.add(id(obj))
    if isinstance(obj,np.ndarray):return [(path,obj)]
    if isinstance(obj,dict):items=obj.items()
    elif isinstance(obj,(list,tuple)):items=enumerate(obj)
    elif hasattr(obj,'__dict__'):items=vars(obj).items()
    else:return []
    out=[]
    for k,v in items:out+=arrays(v,f'{path}.{k}',seen)
    return out

results=[]
deployed={'pi05_spatial':430792483,'pi05_l10':1103155631,'groot_spatial':429351282,'groot_l10':1068314575}
for key in CELLS:
    for scale in (50,500):
        L=lib(key,scale);M,p=method(key,scale);aa=arrays(M)
        total=sum(v.nbytes for _,v in aa);act7=L.L*L.H*7*4
        sizes=[len(t.rows) for t in M.tasks.values()]
        capacities=[math.ceil(max(x*1.25,x+64)/64)*64 for x in sizes]
        uniform=max(capacities)*10;ragged=sum(capacities)
        fixed_pca=2*(32768+32768*64)*4
        # Explicit stand-alone minimal blind-AWM serving schema: main code136,
        # rs8, early squared norm1, two extra main/state squared norms,
        # head35+h2, identity/progress/next/prev/step/ep/length (7 int32),
        # two centered camera vectors64 for judge compatibility, valid byte.
        row_noaction=136*4+8*4+4+8+35*4+4+7*4+2*64*4+1
        row_bytes=row_noaction+L.H*7*4
        perrow_paths=[];fixed_arrays=0
        # Exact retained in-memory arrays with only full action padding stripped.
        packed_existing=total-M.act.nbytes+act7
        r=dict(cell=key,scale=scale,rows=L.L,episodes=len(L.episodes),H=L.H,deployed_bytes=deployed[key],
               pickle_bytes=p.stat().st_size,array_bytes=total,full32_action_bytes=M.act.nbytes,valid7_action_bytes=act7,
               packed_existing_arrays_bytes=packed_existing,packed_existing_deployed_fraction=packed_existing/deployed[key],
               fixed_pca_bytes=fixed_pca,task_rows=sizes,uniform_capacity_rows=uniform,task_capacities=capacities,
               ragged_capacity_rows=ragged,padding_occupancy=L.L/uniform,
               ragged_vs_uniform_row_fraction=ragged/uniform,
               explicit_proposed_schema_row_bytes=row_bytes,
               arrays=[dict(path=n,shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes) for n,v in aa])
        results.append(r)
(OUT/'memory.json').write_text(json.dumps(results,indent=2))

audit=json.loads((OUT/'log_audit.json').read_text());selected=[];npz_info=[]
for r in audit:
    for pp in r['input_npz']:
        with np.load(pp,allow_pickle=True) as a:
            npz_info.append(dict(path=pp,arrays={k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype)) for k in a.files}))
    if r['arm'].endswith('_cl2') or r['root']=='r04_k7' or r['arm'] in ('r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_sp_g'):
        cell='_'.join(r['meta']['cell'].split('_')[:2]);scale=500 if any(x.startswith('bpool') for x in r['libnames']) else 50
        L=lib(cell,scale);data=np.load(OUT/'usage'/f'{r["root"]}__{r["arm"]}.npz');i=data['info'];top=data['top']
        va=(top>=0)&(top<L.L);use=np.zeros(L.L,bool);use[top[va&(i[:,4]==1)[:,None]]]=True
        train=np.zeros(L.L,bool);train[top[va&(i[:,4]==1)[:,None]&(i[:,1]<25)[:,None]]]=True
        t=top[(i[:,1]>=25)&(i[:,4]==1)];valid=t>=0;covered=np.where(valid,train[np.maximum(t,0)],True)
        row=dict(arm=r['arm'],cell=cell,scale=scale,SR=r['SR'],IR=r['IR'],N=r['decisions'],V=r['V'],M=r['M'],
                 episodes=r['represented_episodes'],used_rows=int(use.sum()),L=L.L,train_rows=int(train.sum()),
                 heldout_queries=len(t),heldout_changed_members=int((~covered).any(1).sum()),
                 growth_yield=r['growth_yield'],counts=r['counts'])
        if r['M']:
            sm=r['counts'].get('success_M',0);e=r['represented_episodes'];delta=lib(cell,500).L-lib(cell,50).L
            row['successful_miss_per_deployment_episode']=sm/e
            row['row_gap_fill_episodes_at_stationary_success_yield']=delta/(sm/e) if sm else None
            row['row_gap_fill_episodes_at_stationary_all_yield']=delta/(r['M']/e)
            row['failed_miss_share']=r['counts'].get('failed_M',0)/r['M']
            # Read-only proxy for redundant adjacent MISS entries. Vision unavailable:
            # near head+state is a necessary descriptive check, NOT safe visual dedup.
            h=data['miss_head'];rs=data['miss_rs'];info=data['miss_info'];uid=data['miss_uid']
            same=(uid[1:]==uid[:-1])&(info[1:,2]==info[:-1,2]+1)
            sigma=np.std(lib(cell,50).action[:,:5,:7],axis=(0,1))
            head=np.sqrt(np.mean(((h[1:]-h[:-1])/sigma)**2,axis=(1,2)))
            state=np.linalg.norm(rs[1:]-rs[:-1],axis=1)
            good=np.flatnonzero(L.next>=0);motion=np.linalg.norm(L.rs[L.next[good],:8]-L.rs[good,:8],axis=1)
            threshold=float(np.quantile(motion,.1))
            row['adjacent_miss_pairs']=int(same.sum());row['near_head_pairs']=int((same&(head<=.1)).sum())
            row['state_available_pairs']=int((same&np.isfinite(state)).sum())
            row['near_head_state_pairs']=int((same&(head<=.1)&(state<=threshold)).sum())
            row['motion10_threshold']=threshold
            # Exactly repeated f32 heads are insufficient to certify identical keys.
            row['unique_miss_heads']=len({x.tobytes() for x in h})
        selected.append(row)
(OUT/'focused_usage_yield.json').write_text(json.dumps(selected,indent=2))
(OUT/'input_archives.json').write_text(json.dumps(npz_info,indent=2))
for r in results:print(json.dumps({k:v for k,v in r.items() if k!='arrays'}))
for r in selected:print(json.dumps(r))
