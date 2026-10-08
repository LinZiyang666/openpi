"""Dataset, export/geometry/correction parity, latency and artifact rejection checks."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r10.recipe import fitting
from exp.offline_search.rounds.r10.recipe.recipe import predict
from exp.offline_search.closed_loop import devset
from .build import HERE, STORE, write
from .encoder import FrozenEncoder, sha
from .recipe import FeatureGeometry
from .runtime import initialize
from .boundary import library_boundary


def check(size):
    manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
    blob=pickle.loads(Path(manifest['records']['off']['path']).read_bytes())
    method=blob['method'];base=method.inner.base
    data=dict(np.load(manifest['library_features'],allow_pickle=False))
    subset=json.loads(Path(manifest['subset']).read_text())
    ep=np.load(STORE/'library/pi05_l10/bpool_cs/episode.npy',mmap_mode='r')
    expected_rows=np.flatnonzero(np.isin(ep,sum(subset['episode_ids_by_task'].values(),[])))
    assert np.array_equal(expected_rows,method.row_subset) and np.array_equal(expected_rows,data['rows'])
    assert sha(manifest['library_features'])==manifest['library_features_sha256']
    assert len(expected_rows)==manifest['rows'] and len(np.unique(data['episode']))==size
    np.testing.assert_array_equal(base.sig,np.maximum(data['action'][:,:5,:7].std((0,1)),1e-6))
    assert all('/library/pi05_l10/bpool_cs/' in p for p in manifest['fit_reads'])
    rejected=[]
    directory=STORE/'library/pi05_l10/bpool_cs'
    with library_boundary(STORE,directory):
        for forbidden in (STORE/'queries/forbidden_probe.npy',STORE/'library/pi05_spatial/bpool_cs/forbidden_probe.npy',
                           STORE/'derived/r15_qk/forbidden_probe.npy',(STORE/'queries').resolve()/'forbidden_probe.npy'):
            try:
                with forbidden.open('rb') as f:f.read(1)
                raise AssertionError('forbidden fit read accepted')
            except ValueError:
                rejected.append(str(forbidden))
    # Exact live/library input parity, all tasks and early/stale/fresh branches.
    encoder=FrozenEncoder(method.light_key_spec)
    parent=STORE/'library/pi05_l10/bpool_cs'
    tokrows=np.load(parent/'tok/rows.npy',mmap_mode='r')
    inv=np.full(len(ep),-1,int);inv[tokrows]=np.arange(len(tokrows))
    ims=[np.load(parent/f'tok/img{cam}.npy',mmap_mode='r') for cam in range(2)]
    math=[];latencies=[];blindness=[]
    for task in range(10):
        rows=np.flatnonzero(data['task_id']==task)
        i=int(rows[len(rows)//2]);row=int(data['rows'][i])
        images=[np.array(im[inv[row]]) for im in ims]
        look=SimpleNamespace(img0=images[0],img1=images[1],rs=data['rs'][i],task_id=task)
        features=encoder.encode(look)
        np.testing.assert_array_equal(features,data['features'][i])
        other=int(rows[0]) if i!=int(rows[0]) else int(rows[-1])
        alternative=[np.array(im[inv[int(data['rows'][other])]]) for im in ims]
        comparisons=[]
        current_code=np.r_[features.ravel(),look.rs[:8]] @ base.tasks[task].Wf-base.tasks[task].shift
        def donors(f):
            code=np.r_[f.ravel(),look.rs[:8]] @ base.tasks[task].Wf-base.tasks[task].shift
            d=np.sqrt(np.maximum(base.tasks[task].z2-2*(base.tasks[task].Z@code)+float(code@code),0))
            return base.tasks[task].rows[np.argsort(d,kind='stable')[:16]]
        current_donors=donors(features)
        for mode,imgs in (('same_task_image_swap_fixed_state',alternative),
                          ('agentview_removed',[np.zeros_like(images[0]),images[1]]),
                          ('wrist_removed',[images[0],np.zeros_like(images[1])])):
            changed=encoder.encode(SimpleNamespace(img0=imgs[0],img1=imgs[1],rs=look.rs,task_id=task))
            code=np.r_[changed.ravel(),look.rs[:8]] @ base.tasks[task].Wf-base.tasks[task].shift
            comparisons.append(dict(mode=mode,code_change_norm=float(np.linalg.norm(code-current_code)),
                feature_change_norm=float(np.linalg.norm(changed-features)),
                top16_overlap=len(set(current_donors)&set(donors(changed)))/16))
        blindness.append(dict(task=task,comparisons=comparisons))
        for step,prev in ((0,None),(2,True),(2,False)):
            q=SimpleNamespace(step=step,task_id=task,model='pi05',episode=api.EpisodeView(f'check-{task}','',task,0,-1,0),
                light_features=features,hist_light_features=np.repeat(features[None],step,0),
                rs=data['rs'][i],has_look=True,hist_has_look=np.ones(step,bool),
                hist_hit=np.ones(step,np.int8),prev_hit=prev,
                prev_a_exec=np.pad(data['action'][i],((0,40),(0,25))),
                hist_a_exec=np.zeros((step,50,32),np.float32),hist_rs=np.repeat(data['rs'][i][None],step,0))
            geom=FeatureGeometry(q);method.reset(q.episode)
            T,_,regime,_,_,xv,rs8,ds,_,_,dt=base._dist(geom)
            x=np.r_[xv,rs8]
            if regime==0:
                y=x @ T.W0f-T.c0
                cross=T.Z @ (T.A0 @ y)
                expected=np.sqrt(np.maximum(T.n20-2*cross+float(y@y),0))
            else:
                z=x @ T.Wf-T.shift
                expected=np.sqrt(np.maximum(T.z2-2*(T.Z@z)+float(z@z),0))
            np.testing.assert_array_equal(ds,expected)
            result=method.query(q)
            order=np.argsort(dt,kind='stable')[:16]
            np.testing.assert_array_equal(result.topk,method.row_subset[T.rows[order]])
            w=awm._kernel_w(dt[order].astype(np.float64)-float(dt[order][0]),base.kref)
            wn=(w/w.sum()).astype(np.float32)
            np.testing.assert_allclose(base._anchor['weights'],wn,rtol=1e-6,atol=1e-7)
            raw=base._mix(T.rows[order],w)[2]
            corr=base._correction(geom,raw)
            if not step:
                assert not np.any(corr)
            else:
                ratio=float(np.min(ds[order]))/base.distance_scale
                factor=float(np.clip((2-ratio)/1.25,0,1))
                X=base._features(geom,xv,rs8,raw)
                want=predict(base.heads[str(task)],X)[0].reshape(10,6)*base.sig_head[:6]*np.float32(factor)
                np.testing.assert_allclose(corr,want,rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(result.action,base._apply(raw,corr),rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(result.action[:,6:],raw[:,6:])
            math.append(dict(task=task,branch=regime,neighbors=16,weights=True,correction=True,parent_mapping=True))
        # 400 measured looks total over tasks 0/2/5/8, plus warmup.
        if task in (0,2,5,8):
            for _ in range(20):encoder.encode(look)
            for _ in range(100):
                started=time.perf_counter_ns()
                f=encoder.encode(look).ravel()
                np.r_[f,look.rs[:8]] @ base.tasks[task].Wf-base.tasks[task].shift
                latencies.append((time.perf_counter_ns()-started)/1e6)
    latency=dict(samples=len(latencies),warmups=80,threads=torch.get_num_threads(),
        cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
        median_ms=float(np.median(latencies)),quartiles_ms=np.quantile(latencies,[.25,.75]).tolist(),
        p95_ms=float(np.quantile(latencies,.95)),graph_epsilon_measured=.152*float(np.median(latencies))/10.26,
        graph_epsilon_p95=.152*float(np.quantile(latencies,.95))/10.26,
        frozen_artifact_epsilon=method.light_key_spec['epsilon'],
        host='local CPU, not H100 deployment-host timing',cost_gate_h100='unmeasured',
        includes='raw224 resize/conversion, CPU affinity, export, library projection if any, usable task code')
    write(HERE/f'latency_{size}.json',latency)
    assert all(all(c['feature_change_norm']>1e-5 for c in t['comparisons']) for t in blindness)
    assert any(t['comparisons'][0]['top16_overlap']<1 for t in blindness)
    assert not torch.cuda.is_initialized()
    return dict(size=size,PASS=True,rows=len(expected_rows),episodes=size,selected_sigma_exact=True,
        fit_reads_boundary=True,rejected_fit_reads=rejected,all_task_live_library_feature_parity=True,
        replay=math,blindness=dict(PASS=True,state_fixed=True,all_tasks_both_camera_sensitivity=True,
            real_donor_changes=True,evidence=blindness),latency=latency,CUDA_initialized=False)


def verify():
    result=dict(PASS=True,cells=[check(50),check(200)])
    write(HERE/'verification.json',result)
    print('DATA_GEOMETRY_PASS',[(c['size'],c['latency']['median_ms']) for c in result['cells']],flush=True)
    return result


if __name__=='__main__':verify()
