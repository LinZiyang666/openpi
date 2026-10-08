"""Prepare library-only train arrays and hashed B-only look evaluation arrays.

No training code has a B-dev input. Selected rows and prefit subsets are checked.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
STORE = Path('/home/weiland/trace_runs/offline_search_store')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
DERIVED = STORE/'derived/r15_qk'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while b:=f.read(8<<20): h.update(b)
    return h.hexdigest()

def dump(p,v): p.write_text(json.dumps(v,indent=2))

def load_cell(suite,size):
    subpath=REPO/f'exp/offline_search/rounds/r10/subsets/pi05_{suite}_{size}.json'
    sub=json.loads(subpath.read_text())
    art=REPO/f'exp/offline_search/rounds/r10/recipe/artifacts/r10_recipe_pi05_{suite}_{size}.pkl'
    b=pickle.load(art.open('rb')); m=b['method']
    assert m.episode_subset==sub['episode_ids_by_task']
    p=STORE/f'library/pi05_{suite}/bpool_cs'
    ep=np.load(p/'episode.npy'); ids=sum(m.episode_subset.values(),[])
    rows=np.flatnonzero(np.isin(ep,ids))
    assert np.array_equal(rows,m.row_subset)
    return m,m.inner.base,p,rows,{'subset':str(subpath),'subset_sha256':sha(subpath),'artifact':str(art),'artifact_sha256':sha(art),'fit_info':m.fit_info}

def train(suite,size):
    m,b,p,rows,provenance=load_cell(suite,size)
    out=DERIVED/f'pi05_{suite}_{size}';out.mkdir(parents=True,exist_ok=True)
    z={k:np.asarray(np.load(p/f'{k}.npy',mmap_mode='r')[rows]) for k in ('action','rs','episode','task_id','step','progress')}
    z['rows']=rows;z['sig']=b.sig
    tokrows=np.load(p/'tok/rows.npy');inv=np.full(len(np.load(p/'episode.npy')),-1)
    inv[tokrows]=np.arange(len(tokrows));assert np.all(inv[rows]>=0)
    for cam in range(2):
        a=np.load(p/f'tok/img{cam}.npy',mmap_mode='r')
        # Match LookRecorder's nearest-neighbor resize, 224 -> 56.
        z[f'img{cam}']=np.stack([np.array(a[inv[r],::4,::4]) for r in rows])
        key=np.load(p/f'key_v{cam}.npy',mmap_mode='r')
        bt=getattr(b,f'B{cam}T'); shift=getattr(b,f'muB{cam}')
        proj=[]
        for lo in range(0,len(rows),256):
            proj.append(np.asarray(key[rows[lo:lo+256]])@bt.T-shift)
        z[f'P{cam}']=np.concatenate(proj)
    np.savez(out/'train.npz',**z)
    provenance.update(cell=f'pi05_{suite}_{size}',rows=len(rows),episodes=len(set(z['episode'].tolist())),train_file=str(out/'train.npz'),image_transform='224->56 nearest indices 4*j; both cameras',created=time.time())
    dump(out/'train_meta.json',provenance)
    print('train',provenance['cell'],len(rows),flush=True)

def dev(suite,rootname):
    root=RUNS/rootname
    assert rootname.startswith('r12_devlooks')
    arm=f'r11_dev_pi05_{suite}_'+('50' if '50' in rootname else '200')
    run=root/'runs'/arm
    accepted={}
    for line in (run/'client/journal.jsonl').read_text().splitlines():
        r=json.loads(line)
        assert r.get('init_pool')=='B'
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
            accepted[r['task_uid']]=r
    fields=['img0','img1','key_v0','key_v1','rs','action','step','task_id','init','hit','topk','weights']
    records={};prov=[]
    for f in sorted(run.glob('server_*/looks/manifest_*.json')):
        man=json.loads(f.read_text());c=man['contract']
        assert man['dev'] is True and man['init_pool']=='B' and c['model']=='pi05'
        for sh in man['shards']:
            uid=sh['uid'];at=int(sh['attempt'])
            if uid not in accepted or at!=int(accepted[uid].get('attempt',1)): continue
            p=f.parent/sh['file'];assert p.name==sh['file'] and p.parent==f.parent
            assert sha(p)==sh['sha256']
            with np.load(p,allow_pickle=False) as a:
                meta=json.loads(str(a['meta']));assert meta['dev'] is True and meta['init_pool']=='B'
                r={k:a[k].copy() for k in fields}
                r['label']=a['shadow_policy'].copy() if 'shadow_policy' in a else a['action'].copy()
                r['label_valid']=('shadow_policy' in a or not bool(a['hit']))
                r['label_source']='shadow' if 'shadow_policy' in a else ('miss' if not bool(a['hit']) else 'unlabelled_hit')
                r['uid']=uid;r['attempt']=at
                r['guard_state']=str(a['guard_state'])
            records[(uid,int(sh['step']))]=r
            prov.append({'file':str(p),'sha256':sh['sha256']})
    data=list(records.values());data.sort(key=lambda r:(r['uid'],int(r['step'])))
    arrays={k:np.stack([r[k] for r in data]) for k in data[0]}
    out=DERIVED/f'dev_pi05_{suite}_{rootname}';out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'dev.npz',**arrays)
    overlaps={}
    lib=STORE/f'library/pi05_{suite}/bpool_cs';eps=json.loads((lib/'episodes.json').read_text());epid=np.load(lib/'episode.npy')
    for size in (50,100,200,500):
        sub=json.loads((REPO/f'exp/offline_search/rounds/r10/subsets/pi05_{suite}_{size}.json').read_text())
        ids=set(sum(sub['episode_ids_by_task'].values(),[]))
        pairs={(e['task_id'],e['orig_init_state_idx']) for e in eps if int(epid[e['start']]) in ids}
        unseen=np.array([(int(r['task_id']),int(r['init'])) not in pairs for r in data])
        arrays_name=out/f'unseen_{size}.npy';np.save(arrays_name,unseen)
        overlaps[str(size)]={'looks':int(unseen.sum()),'labelled_looks':int((unseen&arrays['label_valid']).sum()),'episodes':len(set(arrays['uid'][unseen])),'per_task_episodes':{str(t):len(set(arrays['uid'][unseen&(arrays['task_id']==t)])) for t in range(10)}}
    meta={'root':str(root),'arm':arm,'looks':len(data),'accepted_episodes':len(accepted),'recorded_episodes':len(set(arrays['uid'])),'labels':dict(Counter(arrays['label_source'])),'unseen_by_size':overlaps,'input_shards':prov}
    dump(out/'dev_meta.json',meta)
    print('dev',rootname,suite,{k:v for k,v in meta.items() if k!='input_shards'},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','dev']);ap.add_argument('--suite',default='l10');ap.add_argument('--size',type=int,default=50);ap.add_argument('--root',default='r12_devlooks_50L');a=ap.parse_args()
    if a.mode=='train':train(a.suite,a.size)
    else:dev(a.suite,a.root)
