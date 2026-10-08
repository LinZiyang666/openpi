"""Frozen R12 shadow validation on authorized cache-driven B-dev recordings.

No model/rollout launches. All writes are confined by common.install().
Extract with the official hash-verifying reader, reproduce deployed full
retrieval/correction, then fit ONLY the missing selected-library half heads.
"""
from .common import *
DEV.update({'r12_devlooks_50L', 'r12_devlooks_50S', 'r12_devlooks_200', 'r12_devlooks_400', 'r12_devlooks_knob'})
DEV.update({'r12_devlooks_50S_local','r12_devlooks_200_local'})
VALID_ROOTS=('r12_devlooks_50L','r12_devlooks_400','r12_devlooks_knob','r12_devlooks_50S_local','r12_devlooks_200_local')
DEV.difference_update({'r12_devlooks_50S','r12_devlooks_200'})
install()
import argparse
import pickle
import time
from types import SimpleNamespace as NS
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from exp.offline_search.closed_loop.look_recording import read_looks
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r10.train import fit_head, anchor_weights
from exp.offline_search.rounds.r10.recipe.recipe import predict
from .retrieval import Metric, retrieve, lag
from .correction import feat, wm
from .atlas_reference import AtlasConfig, plan, half_features

DEST = HERE/'dev_validation'

def arm_info(root, arm):
    assert root in VALID_ROOTS, 'Coordinator retired non-local Spatial/200 roots'
    run = RUNS/root/'runs'/arm['arm']
    contract = json.loads((run/'pool_contract.json').read_text())
    assert contract['init_pool']=='B' and contract['dev'] is True
    args = arm['plugin_args']; fit = Path(args[args.index('--os-fit-artifact')+1])
    with fit.open('rb') as f: method = pickle.load(f)['method']
    recipe = method
    assert getattr(recipe,'knob_method','off') in ('off','periodic')
    assert recipe.fit_info['library_only']
    assert sha(fit)==contract['fit_sha256']
    cell = f"{arm['model']}_{arm['suite_short']}"; size = recipe.size
    return run,contract,fit,recipe,cell,size

def extract(root, arm):
    start=time.monotonic()
    run,contract,fit,recipe,cell,size=arm_info(root,arm)
    dest=DEST/root/f'{cell}_{size}'; dest.mkdir(parents=True,exist_ok=True)
    C,lib=load(cell,size); base=recipe.inner.base
    np.testing.assert_array_equal(C['row'],recipe.row_subset)
    for c in (0,1):
        with np.load(REPO/f'exp/offline_search/rounds/r10/pca/{cell}/{size}/v{c}.npz') as z:
            np.testing.assert_array_equal(z['basis'].T,getattr(base,f'B{c}T'))
    assert base.hyst==0 and base.norm_cap==0
    accepted={}
    for line in (run/'client/journal.jsonl').read_text().splitlines():
        a=json.loads(line)
        if a.get('accepted') and a.get('status') in ('done','failed'):
            assert a['init_pool']=='B'; accepted[a['task_uid']]=a
    dec={}
    for p in sorted(run.glob('server_*/decisions_*.jsonl')):
        for line in p.read_text().splitlines():
            a=json.loads(line); uid=a.get('uid')
            if a.get('ev')!='dec' or uid not in accepted:continue
            if a.get('attempt',1)!=accepted[uid].get('attempt',1):continue
            key=(uid,int(a['step']))
            if key in dec:assert dec[key]==a
            dec[key]=a
    ledger=json.loads((run/'summary.json').read_text())['cost_ledger']
    n,v,m=len(dec),sum(d.get('vision',True) for d in dec.values()),sum(not d['hit'] for d in dec.values())
    assert (n,v,m)==tuple(ledger[k] for k in ('decisions','vision_decisions','misses'))
    all_looks=[];manifest_evidence=[];checks=[]
    for directory in sorted(run.glob('server_*/looks')):
        for p in sorted(directory.glob('manifest_*.json')):
            manifest=json.loads(p.read_text()); c=manifest['contract']
            assert c['init_pool']=='B' and c['dev'] and c['fit_sha256']==sha(fit)
            assert not any(i in c['subset']['excluded_inits_by_task'][str(t)] for t,i in c['pairs'])
            manifest_evidence.append(dict(path=str(p),sha256=sha(p),shards=len(manifest['shards'])))
        for look in read_looks(directory):
            meta=look['meta']; uid=meta['uid']; step=int(look['step']);task=int(look['task_id'])
            if uid not in accepted or meta['attempt']!=accepted[uid].get('attempt',1):continue
            d=dec[(uid,step)]; assert d.get('vision',True) and meta['camera_mode']=='full'
            previous=dec.get((uid,step-1)); ph=None if previous is None else bool(previous['hit'])
            # The normal policy tail makes ph=True at the next real look. For
            # a direct post-MISS look, use its recorded full policy chunk.
            pa=None
            if ph is False:
                pa=next(x['_action_all'] for x in reversed(all_looks) if x['uid']==uid and x['step']==step-1)
            q=NS(key_v0=look['key_v0'],key_v1=look['key_v1'],rs=look['rs'],task_id=task,
                 step=step,prev_hit=ph,prev_a_exec=pa,episode=NS(uid=uid))
            rows,scores,aux=base.os_score_all(q)
            idx=np.argsort(-scores,kind='stable')[:16]
            ranked_rows=rows[idx];ranked_scores=scores[idx]
            recorded_rows=np.searchsorted(recipe.row_subset,look['topk'])
            np.testing.assert_array_equal(recipe.row_subset[recorded_rows],look['topk'])
            # Memory must use actual live full donors. H100 and local CPU BLAS
            # can differ slightly at very small distance gaps; retain the
            # independently recomputed ranking/score/weight error as an audit.
            rankw=np.exp(ranked_scores-ranked_scores[0]);rankw=(rankw/rankw.sum()).astype(np.float32)
            row_ok=np.array_equal(ranked_rows,recorded_rows)
            score_delta=float(np.max(np.abs(scores[np.searchsorted(rows,recorded_rows)]-look['scores'])))
            rows=recorded_rows;scores=look['scores']
            w=np.exp(scores-scores[0]);wn=(w/w.sum()).astype(np.float32)
            raw=base._mix(rows,w)[2]
            corr=base.os_synth(q,rows,w)
            P=np.r_[base.B0T@look['key_v0']-base.muB0,base.B1T@look['key_v1']-base.muB1]
            gs=look['guard_state']; guard=next(x for x in gs['method'].values() if '_noprog_span' in x)
            anchor_ok=np.array_equal(look['donor_rows'],look['topk'])
            checks.append(dict(rows=row_ok,anchor=anchor_ok,weights=float(np.max(np.abs(rankw-look['donor_weights']))) if row_ok else 1.,
                               scores=score_delta,recorded_weight_reconstruction=float(np.max(np.abs(wn-look['donor_weights']))),
                               action=float(np.max(np.abs(corr-look['action']))) if d['hit'] else 0.))
            assert anchor_ok and checks[-1]['recorded_weight_reconstruction']<1e-7
            assert not d['hit'] or checks[-1]['action']<1e-3,(uid,step,checks[-1])
            assert len(guard['_vision_progress']) and guard['_vision_progress'][-1][0]==step
            assert bool(look['hit'])==d['hit']
            periodic_due=False
            if getattr(recipe,'knob_method','off')=='periodic':
                import math
                from exp.offline_search.rounds.r11.knob.recipe import uniform
                anchors=[s for (u,s),dd in dec.items() if u==uid and s<step and dd.get('vision',True)]
                calls=[s for s in anchors if not dec[(uid,s)]['hit']]
                last=max(calls) if calls else -1;run_count=sum(s>last for s in anchors)
                setting=recipe.knob_settings['setting'];coin=uniform('R11-gap-v1',0,task,int(look['init']),last,'cap')
                cap=math.floor(setting)+int(setting%1>0 and coin<setting%1)
                periodic_due=run_count>=cap
                assert cap==d['extras']['os_r11_cap'] and run_count==d['extras']['os_r11_run']
                assert bool(d['extras']['os_r11_knob_call'])==bool(periodic_due and not d['extras']['os_r11_guard'])
            a=dict(uid=uid,step=step,task=task,init=int(look['init']),P=P,rs=look['rs'][:8],
                rows=rows,weights=look['donor_weights'],distance=float(aux['awm_d'].min()),raw=raw[:10,:7],
                full=(look['action'] if d['hit'] else corr)[:10,:7],
                served=look['action'][:10,:7],shadow=look.get('shadow_policy',np.full_like(look['action'],np.nan))[:10,:7],
                span=int(guard['_noprog_span']),hit=bool(d['hit']),look_reason=int(d['look_reason']),
                regime=int(d['extras']['regime']),threshold=guard['noprog_n']-1,
                periodic=bool(d['extras'].get('os_r11_knob_call',0)),periodic_due=periodic_due,_action_all=look['action'],
                progress=float(guard['_vision_progress'][-1][1]),prog_len=int(guard['_vision_progress'][-1][2]))
            all_looks.append(a)
            if len(all_looks)%400==0: print('extract',cell,size,len(all_looks),round(time.monotonic()-start,1),flush=True)
    all_looks.sort(key=lambda x:(x['uid'],x['step']))
    assert len(all_looks)==v and len({(x['uid'],x['step']) for x in all_looks})==v
    arrays={k:np.array([a[k] for a in all_looks]) for k in all_looks[0] if k!='_action_all'}
    np.savez_compressed(dest/'looks.npz',**arrays)
    # Keep only fields needed for the explicit execution/tail contract.
    fields=('step','hit','vision','source','src','look_reason','previous_executed_steps','exec_ok','extras','task_id','init')
    decisions={u:[{k:d[k] for k in fields if k in d} for (uu,s),d in sorted(dec.items()) if uu==u] for u in accepted}
    for uid,seq in decisions.items():assert [d['step'] for d in seq]==list(range(len(seq)))
    dump(dest/'decisions.json',decisions)
    result=dict(timestamp=now(),root=root,arm=arm['arm'],cell=cell,size=size,N=n,V=v,M=m,
        episodes=len(accepted),successes=sum(a['success'] for a in accepted.values()),
        owner_ir=((.152 if arm['model']=='pi05' else .148)*v+(.848 if arm['model']=='pi05' else .852)*m)/n,
        fit_path=str(fit),fit_sha256=sha(fit),prediction_sha256=sha(HERE/'PREDICTION.md'),manifests=manifest_evidence,
        accepted=accepted,full_parity=dict(rows=sum(c['rows'] for c in checks),
          max_weight_delta=max(c['weights'] for c in checks),max_score_delta=max(c['scores'] for c in checks),
          row_mismatches=sum(not c['rows'] for c in checks),max_recorded_weight_reconstruction_delta=max(c['recorded_weight_reconstruction'] for c in checks),
          max_served_hit_action_delta=max(c['action'] for c in checks)),
        note='Recorded Result scores are wrapper kernel scores, NOT raw AWM distances; raw distance reconstructed via base.os_score_all aux.awm_d.',
        periodic_setting=recipe.knob_settings['setting'] if getattr(recipe,'knob_method','off')=='periodic' else None,
        seconds=time.monotonic()-start)
    dump(dest/'extract.json',result);dump(dest/'extract_reads.json',sorted(READS))
    print('extracted',cell,size,v,result['full_parity'],flush=True)
    return result

def metric_from(table,mode,task):
    m=Metric.__new__(Metric)
    for field in ('W','shift','Z','z2'):setattr(m,field,table[f'{mode}_{task}_{field}'])
    return m

def fit_half(cell,size):
    """SPEC library-only heads and five-fold scale; no dev-derived setting."""
    start=time.monotonic();dest=DEST/'fits'/f'{cell}_{size}';dest.mkdir(parents=True,exist_ok=True)
    C,lib=load(cell,size);past=lag(C,2)
    path=HERE/'final_atlas'/f'{cell}_{size}'/'table.npz'
    if not path.exists():
        assert size==400, 'Only the coordinator near-dense replacement is a new fit cell'
        from .fit_selected_atlas import run as build_atlas
        build_atlas(cell,size)
    table=dict(np.load(path,allow_pickle=False));C['sigma']=table['sigma']
    np.testing.assert_array_equal(C['row'],table['row_subset'])
    H=np.c_[C['P'][past,:64],C['P'][:,64:],C['rs'],C['rs']-C['rs'][past]].astype(np.float32)
    heads={};ds=[];es=[]
    for task in range(10):
        cr=np.flatnonzero(C['task']==task);qr=cr[C['step'][cr]>=2]
        m=metric_from(table,'half',task)
        r=retrieve(m,H[qr],qr,cr,C)
        y=((C['act'][qr,:,:6]-r['action'][:,:,:6])/C['sigma'][:6]).reshape(len(qr),60)
        heads[task]=fit_head(feat(H[qr],r['action'],C['step'][qr],C['task'][qr],C['sigma']),y,anchor_weights(qr,C['ep'][qr]))
        ds.append(r['distance']);es.append(C['ep'][qr])
    loeo_scale=wm(np.concatenate(ds),np.concatenate(es))
    ds=[];es=[]
    for fold in range(5):
        tr=np.flatnonzero(C['fold']!=fold)
        Cfold=dict(C,sigma=np.maximum(C['act'][tr,:5].std((0,1)),1e-6))
        for task in range(10):
            cr=tr[C['task'][tr]==task];qr=np.flatnonzero((C['fold']==fold)&(C['task']==task))
            if not len(qr):continue
            m=Metric(H[cr],C['act'][cr],C['ep'][cr],Cfold['sigma'])
            r=retrieve(m,H[qr],qr,cr,Cfold);keep=C['step'][qr]>0
            ds.append(r['distance'][keep]);es.append(C['ep'][qr][keep])
        print('half scale',cell,size,fold,round(time.monotonic()-start,1),flush=True)
    scale=wm(np.concatenate(ds),np.concatenate(es))
    artifact=dict(heads=heads,scale=scale,loeo_scale=loeo_scale,sigma=C['sigma'],cell=cell,size=size)
    with (dest/'half.pkl').open('wb') as f:pickle.dump(artifact,f)
    result=dict(timestamp=now(),cell=cell,size=size,scale=scale,diagnostic_loeo_scale=loeo_scale,
        atlas_sha256=sha(path),half_sha256=sha(dest/'half.pkl'),fit_pool='selected B library only',
        head_rows=int((C['step']>=2).sum()),scale_rows=sum(len(x) for x in ds),
        prediction_sha256=sha(HERE/'PREDICTION.md'),seconds=time.monotonic()-start,
        note='Post-freeze completion of already specified fit, not dev calibration. Frozen metrics/atlas unchanged.')
    dump(dest/'fit.json',result);dump(dest/'reads.json',sorted(READS))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','fit']);ap.add_argument('--root',default='r12_devlooks_50L');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--model')
    args=ap.parse_args();arms=json.loads((RUNS/args.root/'arms.json').read_text())
    if args.model:arms=[a for a in arms if a['model']==args.model]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.action=='extract':fs=[pool.submit(extract,args.root,a) for a in arms]
        else:
            specs=[arm_info(args.root,a)[-2:] for a in arms]
            fs=[pool.submit(fit_half,*x) for x in specs]
        for f in fs:
            r=f.result();print(json.dumps({k:v for k,v in r.items() if k not in ('accepted','manifests')}),flush=True)

if __name__=='__main__': main()
