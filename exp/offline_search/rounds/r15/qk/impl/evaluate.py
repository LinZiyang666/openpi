"""All-row donor LOEO and accepted unseen B-dev proposal replay of refitted layers."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r02.g3_recovery import g3_core as core
from exp.offline_search.rounds.r10.recipe.fitting import _Library
from .build import HERE, STORE, DERIVED, FeatureLibrary, baseline, write
from .encoder import FrozenEncoder, sha


class LightPseudo:
    def __init__(self,q):
        self.q=q
    def __getattr__(self,name):
        return getattr(self.q,name)
    light_features=property(lambda s:np.stack([s.q.key_v0,s.q.key_v1]))
    hist_light_features=property(lambda s:np.stack([s.q.hist_key_v0,s.q.hist_key_v1],1))
    has_look=property(lambda s:True)
    hist_has_look=property(lambda s:np.ones(s.q.step,bool))


def restricted(task, mask):
    out=copy.copy(task)
    for name in ('rows','Z','z2','HD','h2','V0','V1','RS','rs2','Z0','n20'):
        a=getattr(task,name,None)
        if a is not None:
            setattr(out,name,a[mask])
    return out


def error(action,target,sigma,k):
    return float(np.sqrt(np.mean(((np.asarray(action)[:k,:7]-np.asarray(target)[:k,:7])/sigma)**2)))


def summarize(z):
    task_reports=[]
    episodes=[]
    for t in range(10):
        rows=np.flatnonzero(z['task']==t)
        eps=[]
        for ep in np.unique(z['episode'][rows]):
            ix=rows[z['episode'][rows]==ep]
            rec=dict(task=t,episode=str(ep),rows=len(ix),student_rms5=float(z['student5'][ix].mean()),
                reference_rms5=float(z['reference5'][ix].mean()),student_rms10=float(z['student10'][ix].mean()),
                reference_rms10=float(z['reference10'][ix].mean()))
            eps.append(rec);episodes.append(rec)
        s=float(np.mean([e['student_rms10'] for e in eps]));r=float(np.mean([e['reference_rms10'] for e in eps]))
        task_reports.append(dict(task=t,episodes=len(eps),rows=len(rows),student_rms10=s,reference_rms10=r,
                                 excess=s/r-1))
    s=float(np.mean([t['student_rms10'] for t in task_reports]))
    r=float(np.mean([t['reference_rms10'] for t in task_reports]))
    rng=np.random.default_rng(20261005)
    draws=[]
    for _ in range(2000):
        ss,rr=[],[]
        for t in range(10):
            e=[e for e in episodes if e['task']==t]
            ix=rng.integers(0,len(e),len(e))
            ss.append(np.mean([e[i]['student_rms10'] for i in ix]))
            rr.append(np.mean([e[i]['reference_rms10'] for i in ix]))
        draws.append(np.mean(ss)/np.mean(rr)-1)
    return dict(rows=len(z['task']),episodes=len(episodes),student_rms10_balanced=s,reference_rms10_balanced=r,
        excess=s/r-1,excess_95_interval=np.quantile(draws,[.025,.975]).tolist(),
        student_rms5_look_weighted=float(z['student5'].mean()),reference_rms5_look_weighted=float(z['reference5'].mean()),
        student_rms10_look_weighted=float(z['student10'].mean()),reference_rms10_look_weighted=float(z['reference10'].mean()),
        student_rms10_p90=float(np.quantile(z['student10'],.9)),reference_rms10_p90=float(np.quantile(z['reference10'],.9)),
        top1_agreement=float(np.mean(z['student_top'][:,0]==z['reference_top'][:,0])),
        top16_overlap=float(np.mean([len(set(a)&set(b))/16 for a,b in zip(z['student_top'],z['reference_top'])])),
        per_task=task_reports,per_episode=episodes,bootstrap='2000 stratified paired episode draws seed20261005')


def loeo(size):
    manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
    data=dict(np.load(manifest['library_features'],allow_pickle=False))
    student=pickle.loads(Path(manifest['records']['off']['path']).read_bytes())['method']
    _,refblob=baseline(size,'off');reference=refblob['method']
    libs=[FeatureLibrary(STORE,'pi05_l10','bpool_cs',student.episode_subset,features=data['features']),
          _Library(STORE,'pi05_l10','bpool_cs',reference.episode_subset)]
    table=student.inner.C
    out=dict(task=[],episode=[],step=[],student5=[],student10=[],reference5=[],reference10=[],
             student_top=[],reference_top=[],student_guard=[],reference_guard=[])
    for erows in core.episode_rows(table).values():
        t=int(table.task[erows[0]]);ep=int(table.ep[erows[0]])
        originals=[m.inner.base.tasks[t] for m in (student,reference)]
        for method,task in zip((student,reference),originals):
            mask=method.inner.base.lib_ep[task.rows]!=ep
            method.inner.base.tasks[t]=restricted(task,mask)
            method.reset(api.EpisodeView(f'loeo-{ep}','',t,-1,-1,0))
        try:
            for pos,i in enumerate(erows):
                results=[]
                for j,(method,lib) in enumerate(zip((student,reference),libs)):
                    q=core.PseudoQuery(lib,table,erows,pos,'pi05',prev_hit=True)
                    result=method.query(LightPseudo(q) if j==0 else q)
                    local=np.searchsorted(method.row_subset,result.topk)
                    assert np.all(lib.episode[local]!=ep) and np.all(lib.task_id[local]==t)
                    assert len(result.topk)==16 and np.isfinite(result.action).all()
                    results.append(result)
                out['task'].append(t);out['episode'].append(ep);out['step'].append(int(table.step[i]))
                for label,result,method in zip(('student','reference'),results,(student,reference)):
                    for k in (5,10):
                        out[f'{label}{k}'].append(error(result.action,libs[0].action[i],method.inner.base.sig,k))
                    out[f'{label}_top'].append(result.topk)
                    out[f'{label}_guard'].append(int(result.extras.get('os_force_miss',0)))
        finally:
            for method,task in zip((student,reference),originals):
                method.inner.base.tasks[t]=task
    z={k:np.asarray(v) for k,v in out.items()}
    dest=DERIVED/f'pi05_l10_{size}/loeo_refitted.npz';np.savez(dest,**z)
    report=summarize(z)
    report.update(all_rows=len(z['task'])==manifest['rows'],own_episode_donors=0,cross_task_donors=0,
        frozen_encoder_trained_on_query_episode=True,encoder_episode_cross_fitted=False,
        interpretation='sequential library proposal replay with refitted guard/corrector; not closed-loop SR',
        data=str(dest),data_sha256=sha(dest),student_artifact_sha256=manifest['records']['off']['sha256'],
        representation_sha256=student.light_key_spec['representation_sha256'])
    write(HERE/f'evaluation/loeo_{size}.json',report)
    print('LOEO',size,report['student_rms10_balanced'],report['reference_rms10_balanced'],flush=True)
    return report


def dev(size):
    manifest=json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
    student=pickle.loads(Path(manifest['records']['off']['path']).read_bytes())['method']
    _,refblob=baseline(size,'off');reference=refblob['method']
    encoder=FrozenEncoder(student.light_key_spec)
    directory=STORE/'derived/r15_qk/dev_pi05_l10_r12_devlooks_50L'
    meta=json.loads((directory/'dev_meta.json').read_text())
    mask=np.load(directory/f'unseen_{size}.npy',allow_pickle=False)
    data={k:v[mask] for k,v in dict(np.load(directory/'dev.npz',allow_pickle=False)).items()}
    assert np.all(data['label_valid']) and np.all(data['label_source']=='shadow')
    # Verify the accepted B-only input provenance once again, before evaluation.
    for shard in meta['input_shards']:
        path=Path(shard['file'])
        assert path.is_relative_to(Path('/home/weiland/trace_runs/os_closed_loop/r12_devlooks_50L'))
        assert sha(path)==shard['sha256']
    subset=__import__('exp.offline_search.closed_loop.devset',fromlist=['']).r10_subset('pi05','l10',size)
    assert all(int(i) not in subset['excluded_inits_by_task'][str(int(t))] for t,i in zip(data['task_id'],data['init']))
    dense={};decision_sources={}
    run=Path('/home/weiland/trace_runs/os_closed_loop/r12_devlooks_50L/runs/r11_dev_pi05_l10_50')
    for path in sorted(run.glob('server_*/decisions_*.jsonl')):
        decision_sources[str(path)]=sha(path)
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r.get('ev')=='dec' and r.get('ok'):
                key=(r['uid'],int(r.get('attempt',1)),int(r['step']))
                if key in dense:
                    assert dense[key]['served_head']==r['served_head']
                dense[key]=r
    out=dict(task=[],episode=[],step=[],student5=[],student10=[],reference5=[],reference10=[],
             student_top=[],reference_top=[],student_guard=[],reference_guard=[])
    old_uid=None;state_hist=[];feature_hist=[];key_hist=[[],[]];exec_hist=[];hit_hist=[];look_hist=[]
    blind_veto=[0,0];last_look_step=None;last_look_action=None
    for i in range(len(data['step'])):
        uid=str(data['uid'][i]);t=int(data['task_id'][i]);step=int(data['step'][i])
        if uid!=old_uid:
            old_uid=uid;state_hist=[];feature_hist=[];key_hist=[[],[]];exec_hist=[];hit_hist=[];look_hist=[]
            episode=api.EpisodeView(uid,'',t,int(data['init'][i]),-1,0)
            for method in (student,reference):method.reset(episode)
        while len(hit_hist)<step:
            from exp.offline_search.closed_loop.blind import policy_tail_chunk, BlindQueryView
            missing=len(hit_hist)
            r=dense[uid,int(data['attempt'][i]),missing]
            assert not r['vision'] and r['hit'] and last_look_action is not None
            action=policy_tail_chunk(last_look_action,5*(missing-last_look_step))
            if r['source']=='cache_blind':
                action[:,7:]=0
            else:
                assert r['source']=='policy_tail'
            np.testing.assert_array_equal(action[:5,:7],np.asarray(r['served_head'],np.float32))
            state=np.asarray(r['robot_state'],np.float32)
            bq=BlindQueryView(missing,t,episode,state,np.full(8,np.nan,np.float32),
                bool(hit_hist[-1]),exec_hist[-1],np.asarray(exec_hist),np.asarray(hit_hist,np.int8),
                np.asarray(state_hist),np.asarray(look_hist,bool),missing-last_look_step-1)
            for j,method in enumerate((student,reference)):
                from exp.offline_search.closed_loop.blind import LookReason
                blind_veto[j]+=int(isinstance(method.policy_tail_step(bq) if r['source']=='policy_tail' else method.blind_step(bq),LookReason))
            state_hist.append(state);feature_hist.append(np.full((2,64),np.nan,np.float32))
            for cam in range(2):key_hist[cam].append(np.full(32768,np.nan,np.float32))
            exec_hist.append(action);hit_hist.append(1);look_hist.append(False)
        # Recorded 56x56 is EXACTLY the default student's nearest input grid.
        # Expand only to transport that grid through the same encoder API.
        # This does not recover 224x224 RGB and is not valid for a larger-input student.
        images=[np.repeat(np.repeat(data[f'img{cam}'][i],4,0),4,1) for cam in range(2)]
        features=encoder.encode(SimpleNamespace(img0=images[0],img1=images[1],rs=data['rs'][i],task_id=t))
        q=SimpleNamespace(step=step,task_id=t,episode=episode,model='pi05',rs=data['rs'][i],
            raw_state=np.full(8,np.nan,np.float32),has_look=True,has_vision=True,
            hist_has_look=np.asarray(look_hist,bool),hist_has_vision=np.asarray(look_hist,bool),
            light_features=features,hist_light_features=np.asarray(feature_hist,np.float32).reshape(-1,2,64),
            key_v0=data['key_v0'][i],key_v1=data['key_v1'][i],
            hist_key_v0=np.asarray(key_hist[0],np.float32).reshape(-1,32768),
            hist_key_v1=np.asarray(key_hist[1],np.float32).reshape(-1,32768),
            hist_rs=np.asarray(state_hist,np.float32),hist_a_exec=np.asarray(exec_hist,np.float32),
            hist_hit=np.asarray(hit_hist,np.int8),prev_hit=None if not hit_hist else bool(hit_hist[-1]),
            prev_a_exec=None if not exec_hist else exec_hist[-1])
        if len(hit_hist)!=step:
            raise ValueError('B-dev history reconstruction failed')
        results=[student.query(q),reference.query(q)]
        for label,result,method in zip(('student','reference'),results,(student,reference)):
            assert np.isfinite(result.action).all()
            for k in (5,10):out[f'{label}{k}'].append(error(result.action,data['label'][i],method.inner.base.sig,k))
            out[f'{label}_top'].append(result.topk)
            out[f'{label}_guard'].append(int(result.extras.get('os_force_miss',0)))
        out['task'].append(t);out['episode'].append(uid);out['step'].append(step)
        state_hist.append(data['rs'][i]);feature_hist.append(features)
        for cam in range(2):key_hist[cam].append(data[f'key_v{cam}'][i])
        exec_hist.append(data['action'][i]);hit_hist.append(int(data['hit'][i]));look_hist.append(True)
        last_look_step=step;last_look_action=data['action'][i]
    z={k:np.asarray(v) for k,v in out.items()}
    dest=DERIVED/f'pi05_l10_{size}/dev_refitted.npz';np.savez(dest,**z)
    report=summarize(z)
    report.update(source=str(directory),source_sha256=sha(directory/'dev.npz'),all_shadow_labels=True,
        student_artifact_sha256=manifest['records']['off']['sha256'],representation_sha256=student.light_key_spec['representation_sha256'],
        dense_decision_sources=decision_sources,dense_blind_reconstruction_head_parity=True,
        observed_blind_veto_counts=dict(student=blind_veto[0],reference=blind_veto[1]),
        no_training_overlap=True,interpretation='off-policy proposal replay; observed trajectories are old system B-dev, not new SR')
    write(HERE/f'evaluation/dev_{size}.json',report)
    print('DEV',size,report['student_rms10_balanced'],report['reference_rms10_balanced'],flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,choices=(50,200),required=True)
    ap.add_argument('--mode',choices=('loeo','dev'),required=True);a=ap.parse_args()
    (loeo if a.mode=='loeo' else dev)(a.size)
