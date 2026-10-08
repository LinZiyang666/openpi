"""Compact accepted B traces, reconstruct true distance and cache/shadow labels.

Reads only explicit B roots. Recorded `distances` are kernel scores, so never
uses them as geometric distance. Cache reconstruction is checked against hits.
"""
from common import *
fence()
import argparse
import pickle
from types import SimpleNamespace
import numpy as np

def pair(uid):
    p = uid.split(':')
    return tuple(map(int, p[-2:]))

def accepted(arm):
    out = {}
    for line in (arm/'client/journal.jsonl').open():
        r = json.loads(line)
        if 'success' in r and r.get('error') in (None, '') and r.get('accepted', True):
            assert r['init_pool'] == 'B'
            out[pair(r['task_uid'])] = r
    return out

def extract(root, a, with_looks):
    arm = RUNS/root/'runs'/a['arm']
    out = HERE/'data'/f'{root}__{a["arm"]}'
    jp, sp = arm/'client/journal.jsonl', arm/'summary.json'
    subset=a.get('kwargs',{}).get('episode_subset')
    size=a.get('r10_size',sum(map(len,subset.values())) if subset else 'current')
    if not jp.exists() or not sp.exists(): return None
    if out.with_suffix('.json').exists():
        meta = read(out.with_suffix('.json'))
        if meta.get('r14_projected',False) and meta['looks_processed'] == with_looks and meta['summary_sha'] == sha(sp):
            meta['size']=size; meta['cell']=f'{a["model"]}_{a["suite_short"]}_{size}'
            dump(out.with_suffix('.json'),meta)
            return meta
    assert a['dev'] and a['init_pool']=='B'
    contract = read(arm/'pool_contract.json')
    assert contract['dev'] and contract['init_pool']=='B'
    ys = accepted(arm); summary = read(sp)
    assert len(ys)==summary['complete'] and sum(y['success'] for y in ys.values())==summary['success']
    excluded = contract['subset']['excluded_inits_by_task']
    assert all(i not in excluded[str(t)] for t,i in ys)
    records = {}; rejected = duplicate = 0
    stamps = []
    for p in sorted(arm.glob('server_*/decisions_*.jsonl')):
        stamps.append(dict(path=str(p), sha256=sha(p)))
        for line in p.open():
            r = json.loads(line)
            if r.get('ev')!='dec': continue
            k = pair(r['uid']); y=ys.get(k)
            if y is None or int(r.get('attempt',1))!=int(y.get('attempt',1)):
                rejected+=1; continue
            ix=(*k,int(r['step']))
            if ix in records:
                duplicate+=1
                if records[ix]['ts']>r['ts']: continue
            records[ix]=r
    order=sorted(records)
    keys=sorted(ys); ki={k:i for i,k in enumerate(keys)}
    n=len(order); mat={name:np.full(n,np.nan) for name in
        ('ep','step','task','init','vision','miss','guard','knob','distance','progress',
         'risk','grip_error','shadow','cache_exact_error','grip_event','infer_ms')}
    for j,ix in enumerate(order):
        r=records[ix]; e=r.get('extras',{}); v=bool(r.get('vision',r.get('searched',False)))
        g=e.get('os_r11_guard',e.get('r11_guard',bool(not r['hit']) if a.get('r11_method','off')=='off' else np.nan))
        d=e.get('os_r11_score',e.get('r11_score',np.nan)) if a.get('r11_method')=='distance' else np.nan
        vals=dict(ep=ki[ix[:2]],step=ix[2],task=ix[0],init=ix[1],vision=v,miss=not bool(r['hit']),
                  guard=g if v else 0,knob=e.get('os_r11_knob_call',0) if v else 0,
                  distance=d if v else np.nan, progress=e.get('top1_prog',np.nan),
                  shadow=0,infer_ms=r.get('infer_ms',np.nan))
        for k,value in vals.items(): mat[k][j]=value
    mat['keys']=np.array(keys); mat['success']=np.array([ys[k]['success'] for k in keys],bool)
    mat['Xq']=np.full((n,136),np.nan,np.float32)
    mat['policy_chunk']=np.full((n,10,7),np.nan,np.float32)
    for k, weights in [('epN',np.ones(n)),('epV',mat['vision']),('epM',mat['miss'])]:
        mat[k]=np.bincount(mat['ep'].astype(int),weights=weights,minlength=len(keys))
    led=summary['cost_ledger']
    assert [n,int(mat['vision'].sum()),int(mat['miss'].sum())]==[led[k] for k in ('decisions','vision_decisions','misses')]
    assert all(np.array_equal(mat['step'][mat['ep']==i],np.arange(mat['epN'][i])) for i in range(len(keys)))
    looks=0; fit_sha=None; logged_kernel_zero=0; distance_checks=[]; shard_stamps=[]
    sig=None
    if with_looks:
        args=a['plugin_args']; fit=Path(args[args.index('--os-fit-artifact')+1]); fit_sha=sha(fit)
        assert fit_sha==contract['fit_sha256']
        method=pickle.load(fit.open('rb'))['method']; base=method.inner.base
        sig=np.asarray(base.sig).copy()
        assert base.hyst==0 and base.norm_cap==0
        index={ix:j for j,ix in enumerate(order)}
        for mp in sorted(arm.glob('server_*/looks/manifest_*.json')):
            manifest=read(mp)
            assert manifest['schema']=='offline_search.dev_looks.v1' and manifest['dev'] and manifest['init_pool']=='B'
            shard_stamps.append(dict(path=str(mp),sha256=sha(mp)))
            for sh in manifest['shards']:
                p=mp.parent/sh['file']
                assert p.resolve().parent==mp.parent.resolve() and sha(p)==sh['sha256']
                # The official reader loads images/tokens too. Preserve its hash/pool
                # validation but selectively decode numerical inputs for CPU replay.
                with np.load(p,allow_pickle=False) as z:
                    meta=json.loads(str(z['meta'])); key=(*pair(meta['uid']),int(z['step']))
                    assert meta['dev'] and meta['init_pool']=='B' and meta['camera_mode']=='full'
                    if key not in index or int(meta['attempt'])!=int(ys[key[:2]].get('attempt',1)): continue
                    j=index[key]; r=records[key]; gs=json.loads(str(z['guard_state']))
                    assert int(z['task_id'])==key[0] and int(z['init'])==key[1]
                    prev=records.get((*key[:2],key[2]-1))
                    # True inherited geometry supports fresh fallback looks too.
                    q=SimpleNamespace(step=key[2],task_id=key[0], key_v0=z['key_v0'],key_v1=z['key_v1'],rs=z['rs'],
                        prev_hit=bool(prev['hit']) if prev else None,
                        prev_a_exec=np.zeros((10,32),np.float32), episode=meta['uid'])
                    T,_,_,_,_,_,_,d,_,_,_=base._dist(q)
                    mat['Xq'][j]=np.r_[base.B0T@q.key_v0-base.muB0,
                        base.B1T@q.key_v1-base.muB1,q.rs[:8]]
                    true_d=float(np.min(d))
                    if np.isfinite(mat['distance'][j]): distance_checks.append(abs(mat['distance'][j]-true_d))
                    mat['distance'][j]=true_d
                    logged_kernel_zero+=int(float(z['distances'][0])==0.)
                    rows=np.asarray(z['donor_rows'],int); pos=np.searchsorted(method.row_subset,rows)
                    assert np.array_equal(method.row_subset[pos],rows)
                    # Replay actual recorded donor weights (normalized float32),
                    # then the same deployed distance-attenuated motion head.
                    base._anchor=None
                    cache=base.os_synth(q,pos,z['donor_weights'])
                    assert bool(z['hit'])==bool(r['hit'])
                    if bool(z['hit']):
                        err=float(np.max(np.abs(cache[:10,:7]-z['action'][:10,:7])))
                        mat['cache_exact_error'][j]=err
                    if 'shadow_policy' in z.files:
                        policy=z['shadow_policy']; mat['shadow'][j]=1
                        mat['policy_chunk'][j]=policy[:10,:7]
                        mat['risk'][j]=np.mean(((policy[:10,:6]-cache[:10,:6])/sig[:6])**2)
                        mat['grip_error'][j]=np.mean((policy[:10,6]>=0)!=(cache[:10,6]>=0))
                    grip=cache[:10,6]>=0
                    mat['grip_event'][j]=np.any(grip!=grip[0])
                    looks+=1
    cv,cm=COSTS[a['model']]
    cell=f'{a["model"]}_{a["suite_short"]}_{size}'
    maxerr=float(np.nanmax(mat['cache_exact_error'])) if np.isfinite(mat['cache_exact_error']).any() else None
    meta=dict(root=root,arm=a['arm'],cell=cell,model=a['model'],suite=a['suite_short'],size=size,
        method=a.get('r11_method','off'),target=a.get('target_ir'),summary_sha=sha(sp),journal_sha=sha(jp),
        n=len(keys),N=n,V=int(mat['vision'].sum()),M=int(mat['miss'].sum()),sr=float(mat['success'].mean()),
        ir=float((cv*mat['vision'].sum()+cm*mat['miss'].sum())/n),rejected=rejected,duplicates=duplicate,
        fit_sha=fit_sha,sigma=sig,looks_processed=with_looks,looks=looks,recorded_kernel_min_zero=logged_kernel_zero,
        replay_hit_max_abs_error=maxerr,distance_log_reconstruction_max=max(distance_checks) if distance_checks else None,
        log_sources=stamps,look_manifests=shard_stamps,r14_projected=True)
    if with_looks:
        assert looks==int(mat['vision'].sum()), (cell,looks,int(mat['vision'].sum()))
        assert maxerr is not None and maxerr<2e-4, (cell,maxerr)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),**mat)
    dump(out.with_suffix('.json'),meta)
    print(cell, root, a.get('r11_method','off'), 'episodes',len(keys),'IR',round(meta['ir'],4),
          'looks',looks,'cache_error',maxerr,flush=True)
    return meta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',action='append'); ap.add_argument('--no-looks',action='store_true'); args=ap.parse_args()
    roots=args.root or ['r11_devknob_50','r12_devlooks_50L','r12_devlooks_50S','r12_devlooks_200','r12_devlooks_knob']
    results=[]
    for root in roots:
        assert root in ROOTS
        for a in read(RUNS/root/'arms.json'):
            results.append(extract(root,a,root.startswith('r12_') and not args.no_looks))
    dump(HERE/'extraction_last.json',results); provenance('extraction')

if __name__=='__main__': main()
