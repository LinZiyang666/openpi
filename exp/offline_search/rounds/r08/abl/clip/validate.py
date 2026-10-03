"""CPU fit/key/query parity and discovery-only candidate leave-one-demo-out sanity."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import numpy as np
import yaml

from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api, dims, store
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from .encoder import shared_encoder
from .make_arms import HERE, RUN, build
from .method import _ClipQuery
from .prepare import ROOT, DERIVED, library, write_json


def load(path):
    with open(path,'rb') as f:
        return pickle.load(f)['method']


class _Buffer:
    def __init__(self,a): self.a=a
    def view(self,lo,hi): return self.a[lo:hi]


def online_view(L,r,imgs):
    """The actual plugin image accessors and robot-state accessor, without a policy call."""
    step=int(L.step[r])
    rt=NS(api=api,opts=NS(os_tokens='on'),model='pi05')
    sess=NS(rt=rt,has_vision=[True]*(step+1),cur_obs={plugin.WIRE_IMG0:imgs[0][r],plugin.WIRE_IMG1:imgs[1][r]},
            b_rs=_Buffer(np.repeat(np.array(L.rs[r])[None],step+1,axis=0)),
            hits=[1]*step)
    sess.image=lambda k:plugin.PluginSession.image(sess,k)
    ep=NS(uid=f'library:{int(L.episode[r])}')
    return plugin.OnlineQueryView(sess,step,int(L.task_id[r]),ep)


def fitted_checks(row,m,A,L):
    for attr in ('features','k','kref','early','codes','lam','nn','state_scale','feat0','lam_c','norm_cap','hyst',
                 'serving','budget','gates','residual_threshold','lib','fit_data'):
        assert getattr(m,attr)==getattr(A,attr),attr
    for attr in ('act','sig','lib_ep','lib_step','blind_rs','blind_next','blind_event','blind_terminal'):
        assert np.array_equal(getattr(m,attr),getattr(A,attr)),attr
    key=f'{row["model"]}_{row["suite"]}'
    d=DERIVED/key/L.dir.name
    P=[np.load(d/v/'proj.npy') for v in ('v0','v1')]
    state=np.asarray(dims.valid_state(L.rs,m.model),np.float64)
    sigma=store.action_sigma(ROOT,key)
    heads=(np.asarray(L.action[:,:5,:7],np.float64)/sigma).reshape(L.L,35)
    n=0
    for task,T in m.tasks.items():
        rr=T.rows
        X=np.concatenate([P[0][rr],P[1][rr],state[rr]],axis=1).astype(np.float64)
        assert np.array_equal(T.RS,A.tasks[task].RS)
        for early in (False,True):
            selected=np.asarray(L.step)[rr]<=2 if early else np.ones(len(rr),bool)
            mean,std,W=awm.fit_metric(X[selected],heads[rr][selected],np.asarray(L.episode)[rr][selected],
                                     nn=m.nn,lam=m.lam,rank=m.codes)
            assert np.array_equal((W/std[:,None]).astype(np.float32),T.W0f if early else T.Wf)
            n+=1
    return {'stock_metric_refit_checks':n,'state_action_and_blind_arrays_bit_equal_to_A':True,'nonvisual_settings_exact':True}


def key_query_parity(row,m,L):
    d=DERIVED/f'{row["model"]}_{row["suite"]}'/L.dir.name
    imgs=[np.load(L.dir/'tok'/f'img{i}.npy',mmap_mode='r') for i in (0,1)]
    emb=[np.load(d/f'emb{i}.npy',mmap_mode='r') for i in (0,1)]
    rows=sorted(set(np.linspace(0,L.L-1,20,dtype=int).tolist()+[int(L.rows_of_task(t)[0]) for t in L.tasks()]))
    encoder=shared_encoder('cpu')
    raw_diff,proj_diff=0.,0.
    queries,tails=0,0
    for r in rows:
        q=online_view(L,r,imgs)
        keys=encoder.encode([q.img0,q.img1])
        for i,(B,muB,v) in enumerate(((m.B0T,m.muB0,'v0'),(m.B1T,m.muB1,'v1'))):
            raw_diff=max(raw_diff,float(np.max(np.abs(keys[i]-emb[i][r]))))
            saved=np.load(d/v/'proj.npy',mmap_mode='r')[r]
            proj_diff=max(proj_diff,float(np.max(np.abs(B@keys[i]-muB-saved))))
        control,_=plugin.clone_method(m,strict=True)
        control.__class__=BlindAWM
        m.reset(q.episode);control.reset(q.episode)
        # Full method query actually uses OnlineQueryView.img0/img1. The stock
        # reference is fed precisely the same CLIP features; payloads must match.
        actual=m.query(q)
        cq=_ClipQuery(q,keys,None)
        ref=control.query(cq)
        for field in ('topk','scores','action'):
            assert np.array_equal(getattr(actual,field),getattr(ref,field)),field
        assert actual.confidence==ref.confidence and actual.library==ref.library
        assert list(actual.extras)[0]=='os_clip_encode_ms' and actual.extras['os_clip_encode_ms']>0
        bq=NS(step=q.step+1,task_id=q.task_id,episode=q.episode,rs=q.rs,prev_hit=True,blind_age=0,
              hist_rs=np.array([q.rs]),executed_steps=5)
        br=m.blind_step(bq)
        assert np.array_equal(br.action[:5,:7],actual.action[5:10,:7])
        queries+=1;tails+=1
    assert raw_diff<1e-5 and proj_diff<1e-5
    return {'library_frames':len(rows),'camera_embeddings':2*len(rows),'online_library_embedding_max_abs_diff':raw_diff,
            'online_library_pca64_max_abs_diff':proj_diff,'stock_query_payloads_bit_equal':queries,
            'blind_tail_payloads_bit_equal':tails,'path':'PluginSession.image -> OnlineQueryView.img0/img1 -> shared Encoder -> A PCA GEMV',
            'server_policy_or_simulator_run':False}


def lodo(row,m,A,L):
    """Frozen deployed PCA/metric, exclude the entire query demo from candidates.

    This is retrieval sanity, not supervised cross-validation: the metric has
    been fitted on the same complete library as A. Query rows use only demos
    with recorded init 0-29. No success filter and no holdout query rows.
    """
    eps=json.loads((L.dir/'episodes.json').read_text())
    selected=[i for i,e in enumerate(eps) if e.get('orig_init_state_idx') is not None
              and 0<=int(e['orig_init_state_idx'])<30]
    unknown=sum(e.get('orig_init_state_idx') is None for e in eps)
    if not selected:
        return {'status':'blocked','reason':'Library demo init indices are all unknown; discovery-only filtering cannot be certified.',
                'unknown_init_demos':unknown,'queries':0}
    d=DERIVED/f'{row["model"]}_{row["suite"]}'/L.dir.name
    E=[np.load(d/f'emb{i}.npy',mmap_mode='r') for i in (0,1)]
    sigma=store.action_sigma(ROOT,f'{row["model"]}_{row["suite"]}')
    results=[]
    for ep in selected:
        rr=np.flatnonzero(np.asarray(L.episode)==ep)
        for r in sorted(set(rr[np.linspace(0,len(rr)-1,min(5,len(rr)),dtype=int)].tolist())):
            st=int(L.step[r])
            q=NS(step=st,task_id=int(L.task_id[r]),episode=NS(uid=f'demo:{ep}'),rs=L.rs[r],
                 prev_hit=True if st else None,key_v0=L.key_v0[r],key_v1=L.key_v1[r])
            cq=_ClipQuery(q,np.stack([E[0][r],E[1][r]]),None)
            picks,errors5,errors10=[],[],[]
            for method,view in ((m,cq),(A,q)):
                T,*_,dt=method._dist(view)
                # Drop every frame of the query demonstration, not just self.
                keep=method.lib_ep[T.rows]!=ep
                pos=np.flatnonzero(keep)
                pos=pos[np.argsort(dt[pos],kind='stable')[:method.k]]
                assert len(pos)==16 and np.all(method.lib_ep[T.rows[pos]]!=ep)
                distances=dt[pos].astype(np.float64)
                weights=awm._kernel_w(distances-distances[0],method.kref)
                _,_,action=method._mix(T.rows[pos],weights)
                picks.append(T.rows[pos])
                for size,errs in ((5,errors5),(10,errors10)):
                    delta=(action[:size,:7].astype(np.float64)-L.action[r,:size,:7])/sigma
                    errs.append(float(np.sqrt(np.mean(delta**2))))
            results.append({'row':r,'task':int(L.task_id[r]),'demo':ep,'init':int(eps[ep]['orig_init_state_idx']),
                            'top1_agree':int(picks[0][0]==picks[1][0]),'top16_overlap':len(set(picks[0])&set(picks[1]))/16,
                            'clip_sigma_rms5':errors5[0],'A_sigma_rms5':errors5[1],
                            'clip_sigma_rms10':errors10[0],'A_sigma_rms10':errors10[1]})
    assert all(0<=x['init']<30 for x in results)
    out={'status':'ready','queries':len(results),'demos':len(selected),'unknown_init_demos':unknown,
         'query_inits':sorted({x['init'] for x in results}),'query_rows':[x['row'] for x in results],
         'protocol':'candidate leave-one-demo-out; frozen complete-library PCA and supervised metric; <=5 evenly spaced rows/demo; discovery inits 0-29 only'}
    for k in ('top1_agree','top16_overlap','clip_sigma_rms5','A_sigma_rms5','clip_sigma_rms10','A_sigma_rms10'):
        out['mean_'+k]=float(np.mean([x[k] for x in results]))
    return out


def discovery_query_sanity(row,m,A):
    """Additional logged-query sanity when demo-init metadata cannot certify LODO."""
    from exp.offline_search.rounds.r04.k1_blind.checks import view
    qc=store.QueryCell(ROOT,f'{row["model"]}_{row["suite"]}_inf')
    arrays=api.QueryArrays(qc)
    class CacheQuery:
        def __init__(self,q): self.q=q
        def __getattr__(self,k): return getattr(self.q,k)
        @property
        def prev_hit(self): return True if self.q.step else None
    sigma=store.action_sigma(ROOT,f'{row["model"]}_{row["suite"]}')
    reports=[]
    for ep in qc.episodes:
        if not 0<=ep['init']<30: continue
        available=[i for i in range(ep['start'],ep['end']) if qc.tok_index[i]>=0]
        if not available: continue
        rr=np.asarray(available)
        for r in sorted(set(rr[np.linspace(0,len(rr)-1,min(3,len(rr)),dtype=int)].tolist())):
            q=CacheQuery(view(qc,arrays,r))
            m.reset(q.episode);A.reset(q.episode)
            new,old=m.query(q),A.query(q)
            err=[]
            for res in (new,old):
                diff=(res.action[:10,:7].astype(np.float64)-qc.a_inf[r,:10,:7])/sigma
                err.append(float(np.sqrt(np.mean(diff**2))))
            reports.append({'row':r,'init':ep['init'],'task':ep['task_id'],
                            'top1_agree':float(new.topk[0]==old.topk[0]),
                            'top16_overlap':len(set(new.topk)&set(old.topk))/16,
                            'clip_sigma_rms10':err[0],'A_sigma_rms10':err[1]})
    assert reports and all(0<=x['init']<30 for x in reports)
    out={'queries':len(reports),'inits':sorted({x['init'] for x in reports}),
         'protocol':'logged discovery inference queries, stale pure-cache branch, 3 token rows/episode; candidate library unchanged; not LODO',
         'query_rows':[x['row'] for x in reports]}
    for k in ('top1_agree','top16_overlap','clip_sigma_rms10','A_sigma_rms10'):
        out['mean_'+k]=float(np.mean([x[k] for x in reports]))
    return out


def validate_arms():
    import re
    import openpi.cache.config as cc
    emitted={r['arm']:r for r in json.loads((RUN/'arms.json').read_text())}
    rows,prov=build()
    for row,p in zip(rows,prov):
        restored=copy.deepcopy(row)
        restored['name'],restored['method']=p['source_row']['name'],p['source_row']['method']
        restored['plugin_args'][restored['plugin_args'].index('--os-fit-artifact')+1]=p['source_artifact']
        assert restored==p['source_row']
        e=emitted[row['name']]
        for k in ('method','kwargs','plugin_args','full_model','cost_ledger','client_overrides'):
            assert e.get(k)==row.get(k)
        cfg=yaml.safe_load(Path(e['yaml']).read_text())
        assert 'trace' not in cfg
        cc.load_cache_config(e['yaml'])
        opts,rest=plugin.parse_cli(['--os-method',e['method'],'--os-kwargs',json.dumps(e['kwargs']),
                                   '--os-cell',e['cell'],'--os-log-dir',str(RUN/'clip/parser'),
                                   *[x.replace('<RUN>',str(RUN)) for x in e['plugin_args']]])
        assert not rest and opts.os_blind and opts.os_no_shadow_native and not opts.os_debug_dir
        assert opts.os_tokens=='on' and opts.judge is None
        cls,_=plugin.load_method_class(opts.os_method)
        fit=Path(opts.os_fit_artifact)
        if fit.exists():
            with fit.open('rb') as f: blob=pickle.load(f)
            assert type(blob['method']) is cls
            assert {k:blob[k] for k in ('spec','kwargs','cell')}=={'spec':e['method'],'kwargs':e['kwargs'],'cell':e['cell']}
    expected={(t,i) for t in range(10) for i in range(50)}
    manifest=RUN/'manifests/eval500.json'
    selected=json.loads(manifest.read_text())['selected']
    assert {(x['task'],x['init']) for x in selected}==expected
    refs=[]
    for p in prov:
        for ref in p['A_references']:
            run,name=ref.split(':')
            journal=RUN.parent/run/'runs'/name/'client/journal.jsonl'
            terminal={}
            pattern=re.compile(re.escape(name)+r':eval:(\d+):(\d+)')
            with journal.open() as f:
                for line in f:
                    try: rec=json.loads(line)
                    except ValueError: continue
                    match=pattern.fullmatch(rec.get('task_uid',''))
                    if match and 'success' in rec and rec.get('accepted') and not rec.get('error') and rec.get('status') in ('done','failed'):
                        terminal[tuple(map(int,match.groups()))]=bool(rec['success'])
            assert set(terminal)==expected,(ref,len(terminal))
            refs.append({'arm':p['arm'],'reference':ref,'pairs':len(terminal),'journal':str(journal)})
    out={'PASS':True,'source_rows_exact':8,'configs_and_parser_pass':8,'official_pairs_each':500,
         'A_replicate_references':len(refs),'ready_fit_metadata_pass':4,
         'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'references':refs}
    write_json(HERE/'results/arm_validation.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='references'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arms',action='store_true');a=p.parse_args()
    if a.arms:
        validate_arms()
        return
    reports=[]
    for row,prov in zip(*build()):
        path=RUN/'fits'/f'{row["name"]}.pkl'
        if not path.exists():
            reports.append({'arm':row['name'],'status':'blocked','reason':'exact library images unavailable; no fit artifact'})
            continue
        m,A,L=load(path),load(prov['source_artifact']),library(row)
        checks=fitted_checks(row,m,A,L)
        parity=key_query_parity(row,m,L)
        retrieval=lodo(row,m,A,L)
        out={'arm':row['name'],'status':'ready','PASS':True,'fit_checks':checks,'parity':parity,'retrieval':retrieval}
        if retrieval['status']=='blocked':
            out['discovery_query_sanity']=discovery_query_sanity(row,m,A)
        write_json(HERE/'results'/f'{row["name"]}_validation.json',out)
        reports.append(out)
        print(json.dumps(out),flush=True)
    write_json(HERE/'results/validation.json',reports)


if __name__=='__main__':
    main()
