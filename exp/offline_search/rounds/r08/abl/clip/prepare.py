"""Audit exact source images, encode libraries sequentially, and prefit on CPU.

GPU work is only performed by --encode. Never launches policy/simulator jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import h5py
import numpy as np

from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm import awm
from .make_arms import HERE, RUN, SPEC, build
from .method import ClipAWM

ROOT = store.FULL_ROOT
DERIVED = ROOT/'derived/clip_vitb32'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=1)+'\n')
    os.replace(tmp,path)


def library(row):
    name = 'current' if row['kwargs']['lib']=='current' else awm.BIG[row['model']]
    return store.LibraryView(ROOT,f'{row["model"]}_{row["suite"]}',name)


def audit():
    """Inspect all store metadata and representative source HDF5s, never substitute rollouts."""
    report = []
    for row in build()[0]:
        L = library(row)
        eps = json.loads((L.dir/'episodes.json').read_text())
        files = [Path(eps[i]['file']) for i in sorted({0,len(eps)//2,len(eps)-1})]
        checks = []
        for p in files:
            with h5py.File(p,'r') as h:
                g = h['step_0000']
                checks.append({'file':str(p),'step0_keys':list(g),
                               'input_image_keys':list(g['input_images']) if 'input_images' in g else []})
        has = all((L.dir/'tok'/f'img{i}.npy').exists() for i in (0,1))
        if has:
            rr = np.load(L.dir/'tok/rows.npy',mmap_mode='r')
            assert np.array_equal(rr,np.arange(L.L))
            imgs = [np.load(L.dir/'tok'/f'img{i}.npy',mmap_mode='r') for i in (0,1)]
            assert all(x.shape==(L.L,224,224,3) and x.dtype==np.uint8 for x in imgs)
            source_equal = []
            for p in files:
                ep = next(e for e in eps if e['file']==str(p))
                ids = json.loads((L.dir/'ids.json').read_text())
                r = ids.index(ep['stem']+':0')
                with h5py.File(p,'r') as h:
                    for i,k in enumerate(('base_0_rgb','left_wrist_0_rgb')):
                        assert np.array_equal(imgs[i][r],h['step_0000/input_images/'+k][()])
                        source_equal.append({'file':str(p),'row':r,'camera':i,'max_abs_diff':0})
            # Recorded actual server wire images vs the same post-transform inputs
            # A's vision key builder used. Discovery starts only.
            wire_checks = []
            qc = store.QueryCell(ROOT,f'{row["model"]}_{row["suite"]}_cache')
            for ep in [e for e in qc.episodes if e['init']==0]:
                with h5py.File(ep['file'],'r') as h:
                    g=h['step_0000']
                    for raw,slot in [('observation%2Fimage','base_0_rgb'),
                                     ('observation%2Fwrist_image','left_wrist_0_rgb')]:
                        x,y=g['trace/raw_images/'+raw][()],g['input_images/'+slot][()]
                        assert x.shape==(224,224,3) and np.array_equal(x,y)
                        wire_checks.append({'file':ep['file'],'camera':slot,'max_abs_diff':0})
            status, reason = 'ready', None
        else:
            source_equal,wire_checks = [],[]
            status,reason = 'blocked','No tok/img0/img1 arrays; sampled source build HDF5s contain no input_images or raw_images. Exact library frames unavailable; tokens cannot reconstruct RGB.'
        report.append({'arm':row['name'],'library':str(L.dir),'rows':L.L,'episodes':len(eps),
                       'status':status,'blocker':reason,'source_checks':checks,
                       'library_source_equality':source_equal,'logged_server_wire_equality':wire_checks})
    write_json(HERE/'results/source_audit.json',report)
    print(json.dumps([{'arm':r['arm'],'rows':r['rows'],'status':r['status'],'blocker':r['blocker']} for r in report]),flush=True)
    return report


def encode():
    from .encoder import Encoder, RECIPE, WEIGHT_SHA256, guard_local_gpu
    import torch
    # Exclusive lock across all CLIP encoding/latency jobs created by this task.
    import fcntl
    DERIVED.mkdir(parents=True,exist_ok=True)
    with (DERIVED/'.gpu_job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard = guard_local_gpu()
        print(json.dumps({'event':'gpu_guard',**guard}),flush=True)
        enc=Encoder('cuda:0')
        audits={r['arm']:r for r in json.loads((HERE/'results/source_audit.json').read_text())}
        for row in build()[0]:
            if audits[row['name']]['status']!='ready':
                print(json.dumps({'event':'skip_blocked','arm':row['name']}),flush=True)
                continue
            L=library(row)
            d=DERIVED/f'{row["model"]}_{row["suite"]}'/L.dir.name
            d.mkdir(parents=True,exist_ok=True)
            ids_hash=hashlib.sha256((L.dir/'ids.json').read_bytes()).hexdigest()
            expected={'n':L.L,'ids_sha256':ids_hash,'encoder_recipe':RECIPE,'weight_sha256':WEIGHT_SHA256}
            if (d/'encoded.json').exists():
                done=json.loads((d/'encoded.json').read_text())
                assert all(done[k]==v for k,v in expected.items())
                print(json.dumps({'event':'encoded_resume','arm':row['name']}),flush=True)
                continue
            imgs=[np.load(L.dir/'tok'/f'img{i}.npy',mmap_mode='r') for i in (0,1)]
            paths=[d/f'emb{i}.npy' for i in (0,1)]
            out=[np.lib.format.open_memmap(p.with_suffix('.partial.npy'),mode='w+',dtype=np.float32,shape=(L.L,512)) for p in paths]
            t=time.perf_counter()
            for lo in range(0,L.L,4):
                hi=min(L.L,lo+4)
                x=enc.encode([a[r] for r in range(lo,hi) for a in imgs])
                out[0][lo:hi],out[1][lo:hi]=x[::2],x[1::2]
                if lo%400==0:
                    print(json.dumps({'event':'encoding','arm':row['name'],'rows':hi,'total':L.L,
                                      'elapsed_s':time.perf_counter()-t}),flush=True)
            for p,a in zip(paths,out):
                a.flush()
                os.replace(p.with_suffix('.partial.npy'),p)
            del out
            write_json(d/'encoded.json',{**expected,'library':str(L.dir),'camera_order':['base_0_rgb','left_wrist_0_rgb'],
                                        'input_shape':[224,224,3],'wall_s':time.perf_counter()-t,
                                        'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**guard})
            print(json.dumps({'event':'encoded','arm':row['name'],'elapsed_s':time.perf_counter()-t}),flush=True)


def fit():
    from .encoder import RECIPE, WEIGHT_SHA256
    reports=[]
    for row in build()[0]:
        L=library(row)
        key=f'{row["model"]}_{row["suite"]}'
        d=DERIVED/key/L.dir.name
        artifact=RUN/'fits'/f'{row["name"]}.pkl'
        if not (d/'encoded.json').exists():
            reports.append({'arm':row['name'],'status':'blocked','artifact':None})
            continue
        if artifact.exists():
            reports.append({'arm':row['name'],'status':'ready','artifact':str(artifact),'resumed':True})
            continue
        t=time.perf_counter()
        for i,v in enumerate(('v0','v1')):
            vd=d/v
            if (vd/'meta.json').exists():
                continue
            vd.mkdir(exist_ok=True)
            x=np.load(d/f'emb{i}.npy',mmap_mode='r')
            mu,B,P=awm.pca_fit(x)
            for name,a in [('mean',mu),('basis',B),('proj',P)]:
                np.save(vd/f'{name}.npy',a)
            write_json(vd/'meta.json',{'n':L.L,'ids_sha256':hashlib.sha256((L.dir/'ids.json').read_bytes()).hexdigest(),
                                      'encoder_recipe':RECIPE,'weight_sha256':WEIGHT_SHA256,
                                      'recipe':awm.PCA_RECIPE,'camera':i,'dims':512,'keep':64})
        m=ClipAWM(**row['kwargs'])
        m.prof=api.NULL_PROFILER
        ctx=api.Context(root=ROOT,cell=key+'_cache',seed=0,scratch=RUN/'clip_logs'/row['name'])
        ctx.scratch.mkdir(parents=True,exist_ok=True)
        current=store.LibraryView(ROOT,key,'current')
        m.fit(current,ctx)
        artifact.parent.mkdir(parents=True,exist_ok=True)
        tmp=artifact.with_suffix('.pkl.tmp')
        with tmp.open('wb') as f:
            pickle.dump({'method':m,'registered':ctx.registered,'spec':SPEC,'kwargs':row['kwargs'],
                         'cell':key+'_cache','fit_s':time.perf_counter()-t},f,protocol=4)
        os.replace(tmp,artifact)
        report={'arm':row['name'],'status':'ready','artifact':str(artifact),'bytes':artifact.stat().st_size,
                'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'fit_s':time.perf_counter()-t}
        reports.append(report)
        print(json.dumps(report),flush=True)
    write_json(HERE/'results/fits.json',reports)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',action='store_true');p.add_argument('--encode',action='store_true');p.add_argument('--fit',action='store_true')
    a=p.parse_args()
    if a.audit: audit()
    if a.encode: encode()
    if a.fit: fit()


if __name__=='__main__':
    main()
