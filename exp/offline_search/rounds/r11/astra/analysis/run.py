#!/usr/bin/env python3
"""R11 measurement pipeline. CPU only; no fitting, launch, or run-root writes.

Run from repository root; --final requires all 94 unique test arms and writes ANALYSIS.md.
Only summary-complete arms with all accepted episodes enter the analysis.
Library ablations change donor eligibility while keeping every deployed fit fixed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent
REPO = OWNER.parents[4]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
ROOTS = ('r11_knob_1', 'r11_knob_2', 'r11_local_k3', 'r11_local_k4', 'r11_knob_4')
KNOB = OWNER.parent / 'knob'
R10 = OWNER.parents[1] / 'r10'
OWN = ('distance', 'error_hybrid', 'adaptive_error_hybrid', 'disagreement')
METHODS = ('off', 'random', 'periodic', *OWN, 'periodic_pgt1', 'random_tail2')
CV = {'pi05': .152, 'groot': .148}
VERSION = 1
RERUN = ('taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '
         'MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python '
         'exp/offline_search/rounds/r11/astra/analysis/run.py')


def fence():
    """Fail closed on unintended writes, subprocesses, or network access."""
    def check(p):
        if isinstance(p, int) or p is None:
            return
        p = Path(os.fsdecode(p)).resolve()
        if p != Path('/dev/null') and p != OWNER and OWNER not in p.parents:
            raise PermissionError(f'Analysis write outside owner directory: {p}')
    def audit(event, args):
        if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            check(args[0])
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'):
            check(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(args[0]); check(args[1])
        elif event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system'):
            raise PermissionError(f'Analysis forbids {event}')
    sys.dont_write_bytecode = True
    sys.addaudithook(audit)


fence()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE.mkdir(exist_ok=True)
(HERE / 'cache').mkdir(exist_ok=True)

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import pickle
import time
import numpy as np


def read_json(p):
    return json.loads(Path(p).read_text())


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(x) for x in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.generic): return clean(o.item())
    if isinstance(o, float) and not np.isfinite(o): return None
    if isinstance(o, Path): return str(o)
    return o


def dump(p, o):
    Path(p).write_text(json.dumps(clean(o), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(name, rows):
    p = HERE / name
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple)) else clean(v)
                        for k, v in row.items()})


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def stamp(p):
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def npz(p):
    with np.load(p, allow_pickle=False) as z: return {k: np.array(z[k]) for k in z.files}


def quant(x, prefix=''):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return {prefix + 'n': 0}
    return {prefix + 'n': len(x), prefix + 'mean': float(x.mean()),
            **{prefix + k: float(v) for k, v in zip(('p05', 'p25', 'p50', 'p75', 'p95'),
                                                        np.quantile(x, [.05, .25, .5, .75, .95]))}}


def pairkey(uid):
    parts = uid.split(':eval:')
    if len(parts) != 2: return None
    t, i = parts[1].split(':')[:2]
    return f'{int(t)}:{int(i)}'


def outcomes(path):
    out, malformed = {}, 0
    for line in path.open():
        try: r = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1; continue
        key = pairkey(r.get('task_uid', ''))
        if key is not None and 'success' in r and r.get('error') in (None, '') and r.get('accepted', True):
            out[key] = dict(success=bool(r['success']), attempt=int(r.get('attempt', 1)), uid=r['task_uid'])
    return out, malformed


def load_arm(root, spec, cohort='test', expected=500):
    p = RUNS / root / 'runs' / spec['arm']
    sp, jp = p / 'summary.json', p / 'client/journal.jsonl'
    if not sp.exists(): return None, 'no summary.json at snapshot'
    if not jp.exists(): return None, 'summary present, journal missing'
    try: summary = read_json(sp); y, malformed = outcomes(jp)
    except (ValueError, OSError) as e: return None, f'input incomplete: {e}'
    if len(y) != expected or summary.get('complete') != expected or not summary.get('weighted', {}).get('complete', True):
        return None, f'incomplete: journal={len(y)}, summary={summary.get("complete")}'
    if sum(v['success'] for v in y.values()) != summary['success']:
        return None, 'journal/summary success mismatch'
    model = spec['model']; suite = spec.get('suite_short', summary['suite'].replace('libero_', '').replace('10', 'l10'))
    size = spec.get('r10_size', 50)
    cell = f'{model}_{suite}_{size}'
    ledger = summary['cost_ledger']
    ir = CV[model] * ledger['v'] + (1-CV[model]) * ledger['m']
    a = dict(root=root, arm=spec['arm'], cohort=cohort, model=model, suite=suite, size=size,
             cell=cell, method=spec.get('r11_method') or 'off', target=spec.get('target_ir'),
             pred=spec.get('pred_IR_lib'), n=len(y), sr=np.mean([v['success'] for v in y.values()]),
             ir=ir, v=ledger['v'], m=ledger['m'], N=ledger['decisions'],
             V=ledger['vision_decisions'], M=ledger['misses'], path=p, outcomes=y,
             summary=summary, journal_malformed=malformed, input_stamps=[stamp(sp), stamp(jp)],
             summary_sha256=sha(sp), journal_sha256=sha(jp))
    return a, 'complete'


def discover():
    chosen, inventory, specs = {}, [], {}
    for cohort, roots, n in (('test', ROOTS, 500), ('dev', ('r11_devknob_50',), 50)):
        for root in roots:
            for spec in read_json(RUNS/root/'arms.json'):
                key=(cohort,spec['arm'])
                identity=(spec['model'],spec['suite_short'],spec.get('r10_size'),
                          spec.get('r11_method') or 'off',spec.get('target_ir'),spec.get('pred_IR_lib'))
                if key in specs:assert specs[key]==identity, f'Conflicting arm specs: {key}'
                specs[key]=identity
                a, status = load_arm(root, spec, cohort, n)
                inventory.append(dict(cohort=cohort, root=root, arm=spec['arm'],
                                      method=spec.get('r11_method') or 'off', status=status,selected=False))
                if a:
                    if key in chosen:raise ValueError(f'Two complete copies of {key}: {chosen[key]["root"]}, {root}')
                    chosen[key]=a
    for row in inventory:
        selected=chosen.get((row['cohort'],row['arm']))
        row['selected']=bool(selected and selected['root']==row['root'])
        row['complete_root']=selected['root'] if selected else None
        if selected and not row['selected']:
            row['status']+='; complete copy selected from '+selected['root']
    arms=[a for (cohort,_),a in chosen.items() if cohort=='test']
    dev=[a for (cohort,_),a in chosen.items() if cohort=='dev']
    return arms, dev, inventory


def parse_decisions(a):
    files = sorted(a['path'].glob('server_*/decisions_*.jsonl'))
    fp = dict(version=VERSION, files=[stamp(p) for p in files],
              summary=a['summary_sha256'], journal=a['journal_sha256'])
    cache = HERE/'cache'/f"logs_{a['root']}_{a['arm']}"
    if cache.with_suffix('.json').exists() and cache.with_suffix('.npz').exists():
        old = read_json(cache.with_suffix('.json'))
        if old['fingerprint'] == fp:
            a['log_audit'] = old['audit']; a['trace'] = npz(cache.with_suffix('.npz'))
            return
    keys = sorted(a['outcomes'], key=lambda x: tuple(map(int, x.split(':'))))
    ki = {k: i for i, k in enumerate(keys)}
    records = {}; bad = duplicate = rejected = 0
    for path in files:
        with path.open() as f:
            for line in f:
                try: r = json.loads(line)
                except json.JSONDecodeError: bad += 1; continue
                if r.get('ev') != 'dec': continue
                k = pairkey(r.get('uid', '')); y = a['outcomes'].get(k)
                if y is None or int(r.get('attempt', 1)) != y['attempt']:
                    rejected += 1; continue
                step = int(r['step']); ex = r.get('extras', {})
                key = (ki[k], step)
                if key in records: duplicate += 1
                # Compact accepted decision record. Keep last timestamp on a retry.
                g = ex.get('os_r11_guard', ex.get('r11_guard'))
                if g is None and a['method'] == 'off': g = float(not r.get('hit', True))
                knob = ex.get('os_r11_knob_call', float(bool(ex.get('r11_knob', 0)) and not g))
                val = (float(r.get('ts', 0)), bool(r.get('vision', r.get('searched', False))),
                       not bool(r.get('hit', True)), g if g is not None else np.nan,
                       knob, ex.get('os_r11_p', ex.get('r11_p', np.nan)),
                       ex.get('os_r11_score', ex.get('r11_score', np.nan)),
                       ex.get('os_r11_setting', ex.get('r11_dose', np.nan)),
                       ex.get('os_r11_nonfinite', 0), ex.get('r11_knob', np.nan),
                       ex.get('os_reason', np.nan), ex.get('top1_prog', np.nan),
                       ex.get('regime', np.nan), ex.get('os_r11_N', np.nan),
                       ex.get('os_r11_V', np.nan), ex.get('os_r11_M', np.nan))
                if key not in records or val[0] >= records[key][0]: records[key] = val
    order = sorted(records)
    mat = np.asarray([records[k] for k in order], float).reshape(-1, 16)
    tr = dict(keys=np.array(keys), success=np.array([a['outcomes'][k]['success'] for k in keys], bool),
              ep=np.array([k[0] for k in order], int), step=np.array([k[1] for k in order], int))
    for i, name in enumerate(('ts','vision','miss','guard','knob','p','score','dose','nonfinite','sampled',
                               'reason','top1_prog','regime','ledger_N','ledger_V','ledger_M')): tr[name] = mat[:, i]
    for name, weight in (('epN', np.ones(len(order))), ('epV', tr['vision']), ('epM', tr['miss'])):
        tr[name] = np.bincount(tr['ep'], weights=weight, minlength=len(keys))
    contiguous = all(np.array_equal(tr['step'][tr['ep']==i], np.arange(tr['epN'][i])) for i in range(len(keys)))
    counts = [len(order), int(tr['vision'].sum()), int(tr['miss'].sum())]
    expected = [a[k] for k in ('N','V','M')]
    audit = dict(files=len(files), malformed=bad, duplicates=duplicate, rejected_attempt_rows=rejected,
                 observed_counts=counts, ledger_counts=expected, counts_match=counts==expected,
                 episode_steps_contiguous=contiguous,
                 usable=counts==expected and contiguous and bad==0,
                 file_stats_after=[stamp(p) for p in files])
    if audit['file_stats_after'] != fp['files']: audit['usable'] = False
    np.savez_compressed(cache.with_suffix('.npz'), **tr)
    dump(cache.with_suffix('.json'), dict(fingerprint=fp, audit=audit))
    a['trace'], a['log_audit'] = tr, audit


def configs(cell):
    return read_json(KNOB/'calibration'/f'{cell}.json')['settings']


def cfg_for(a):
    return next(c for c in configs(a['cell']) if c['method']==a['method'] and c['target']==a['target'])


def ref_data(cell):
    s = npz(OWNER/'data'/cell/'signals.npz'); pack = npz(OWNER/'data'/cell/'pack.npz')
    anchor = np.zeros(len(s['ep']), bool); guard = np.zeros(len(s['ep']), bool)
    for j, length in enumerate(pack['length']):
        ii = np.arange(0, length, 2); rows = pack['idx'][ii,j]; anchor[rows] = True
        guard[rows[1:]] = np.diff(pack['prog'][ii,j]) * pack['den'][ii[1:],j] <= .5
    s['anchor'], s['guard'] = anchor, guard
    head = read_json(OWNER/'data'/cell/'predictor.json')
    z = np.clip((s['X']-np.array(head['mean']))/np.array(head['std']), -8, 8)
    s['final_error'] = np.c_[np.ones(len(z)),z] @ np.array(head['coef'])
    s['v'] = anchor.sum()/pack['length'].sum()
    return s


def score_name(method): return 'predicted_error' if 'error' in method else method


def static_p(score, cfg):
    q=cfg['dose']; beta=cfg['beta']; t=cfg['threshold']
    if q in (0.,1.): return np.full(np.shape(score),q)
    return (1-beta)*q + beta*((score>t).astype(float)+(score==t)*cfg['tie_probability'])


def frozen_score_p(score,cfg,reference):
    if cfg.get('threshold') is not None:return static_p(score,cfg)
    R=np.sort(reference);q=cfg['dose'];cutoff=1-q
    lo=np.searchsorted(R,score,'left')/len(R);hi=np.searchsorted(R,score,'right')/len(R)
    branch=np.where(hi>lo,np.clip((hi-cutoff)/np.maximum(hi-lo,1e-30),0,1),lo>cutoff)
    return .5*q+.5*branch


def diagnostics(a, ref):
    tr=a['trace']; good=a['log_audit']['usable']
    result=dict(cohort=a['cohort'],root=a['root'],arm=a['arm'],cell=a['cell'],method=a['method'],
                target=a['target'],pred_IR_lib=a['pred'],n_episodes=a['n'],sr=a['sr'],ir=a['ir'],
                error_target=None if a['target'] is None else a['ir']-a['target'],
                error_prediction=None if a['pred'] is None else a['ir']-a['pred'],v=a['v'],logs_usable=good)
    if not good: return result, [], []
    look=tr['vision'].astype(bool); g=tr['guard'][look].astype(bool)
    k=tr['knob'][look]; p=tr['p'][look]; s=tr['score'][look]; q=tr['dose'][look]
    G=float(g.mean()); K=float(k[~g].mean()) if (~g).any() else 0.
    result.update(guard_per_look=G,knob_non_guard=K,knob_per_look=float(k.mean()),
                  expected_knob_all=float(p.mean()),
                  expected_knob_non_guard=float(p[~g].mean()),
                  miss_per_look=float(tr['miss'][look].mean()),
                  guard_or_knob_discrepancies=int(np.count_nonzero((g|k.astype(bool)) != tr['miss'][look].astype(bool))),
                  nonfinite=int(tr['nonfinite'][look].sum()),
                  nonnominal_look_parity=int((tr['step'][look]%2!=0).sum()),
                  **quant(q,'dose_'))
    if a['method'] in OWN:
        cfg=cfg_for(a); anchor=ref['anchor']; gr=ref['guard'][anchor]; rs=ref[score_name(a['method'])]
        R=np.sort(rs); ranks=(np.searchsorted(R,s,side='left')+np.searchsorted(R,s,side='right'))/(2*len(R))
        result.update(dose_initial=cfg['dose'],threshold=cfg.get('threshold'),tie=cfg.get('tie_probability'),
                      **quant(s,'score_'),**quant(s[~g],'nonguard_score_'),**quant(rs,'reference_all_'),
                      **quant(rs[anchor],'reference_looks_'),**quant(ranks,'reference_rank_'),
                      below_reference_min=float(np.mean(s<R[0])),above_reference_max=float(np.mean(s>R[-1])),
                      adaptive_saturation_low=float(np.mean(q==0)),adaptive_saturation_high=float(np.mean(q==1)))
        if a['method']=='adaptive_error_hybrid':
            # Comparable cross-distribution statistic, separated from live dynamic probabilities.
            lo=np.searchsorted(R,s,side='left')/len(R);hi=np.searchsorted(R,s,side='right')/len(R)
            cutoff=1-cfg['dose'];branch=np.where(hi>lo,np.clip((hi-cutoff)/np.maximum(hi-lo,1e-30),0,1),lo>cutoff)
            result['frozen_scoring_probability_all']=float(np.mean(.5*cfg['dose']+.5*branch))
        else:result['frozen_scoring_probability_all']=float(static_p(s,cfg).mean())
        if a['method']=='adaptive_error_hybrid':
            c=next(c for c in read_json(OWNER/'data'/a['cell']/'analysis.json')['adaptive'] if c['target']==a['target'])
            sim=c['validation']; gl=sim['guard_per_look']; vl=ref['v']
            kl=(sim['miss_per_look']-gl)/(1-gl)
            result['reference_dynamic_threshold_note']='dynamic rank threshold 1-dose; no single score cutoff'
            observed_high=(p-.5*q)*2
            result['high_branch_share']=float(observed_high.mean())
            result['dose_end_mean']=float(np.mean([q[np.flatnonzero(tr['ep'][look]==i)[-1]] for i in range(a['n'])]))
            # Verify committed-ledger update and its finite-episode accounting identity.
            updates=[]; terminal=[]; clipped=[]
            for ep in range(a['n']):
                ix=np.flatnonzero(look & (tr['ep']==ep)); raw=cfg['dose']; clipcor=0.
                for prev,cur in zip(ix[:-1],ix[1:]):
                    change=(CV[a['model']]*(tr['ledger_V'][cur]-tr['ledger_V'][prev])+
                            (1-CV[a['model']])*(tr['ledger_M'][cur]-tr['ledger_M'][prev])-
                            a['target']*(tr['ledger_N'][cur]-tr['ledger_N'][prev]))
                    raw=tr['dose'][prev]-.2*change/(1-CV[a['model']])
                    updates.append(abs(np.clip(raw,0,1)-tr['dose'][cur]));clipcor+=np.clip(raw,0,1)-raw
                last=ix[-1]; b=1-CV[a['model']]
                processed=(b/.2)*(cfg['dose']-tr['dose'][last]+clipcor)
                tail=(CV[a['model']]*(tr['epV'][ep]-tr['ledger_V'][last])+
                      b*(tr['epM'][ep]-tr['ledger_M'][last])-a['target']*(tr['epN'][ep]-tr['ledger_N'][last]))
                terminal.append(tail);clipped.append(b/.2*clipcor)
                actual=CV[a['model']]*tr['epV'][ep]+b*tr['epM'][ep]-a['target']*tr['epN'][ep]
                assert abs(processed+tail-actual)<1e-8
            result.update(controller_update_max_error=max(updates,default=0),
                          adaptive_initial_to_final_ir=(1-CV[a['model']])/.2 * (a['n']*cfg['dose']-sum(q[np.flatnonzero(tr['ep'][look]==i)[-1]] for i in range(a['n'])))/a['N'],
                          adaptive_clip_ir=sum(clipped)/a['N'], adaptive_terminal_ir=sum(terminal)/a['N'])
        else:
            rp=static_p(rs,cfg); vl=ref['v'];gl=float(gr.mean());kl=float(rp[anchor][~gr].mean())
            expected=static_p(s,cfg)
            result.update(probability_max_error=float(np.max(abs(p-expected))),
                          above_threshold=float(np.mean(s>cfg['threshold'])),
                          above_threshold_nonguard=float(np.mean(s[~g]>cfg['threshold'])),
                          reference_above_threshold_all=float(np.mean(rs>cfg['threshold'])),
                          reference_above_threshold_looks=float(np.mean(rs[anchor]>cfg['threshold'])),
                          reference_above_threshold_nonguard=float(np.mean(rs[anchor][~gr]>cfg['threshold'])))
            if 'error' in a['method']:
                fp=static_p(ref['final_error'],cfg)[anchor]
                result['final_head_same_oof_features_ir']=vl*(CV[a['model']]+(1-CV[a['model']])*(gl+(1-gl)*fp[~gr].mean()))
        b=1-CV[a['model']]; cv=CV[a['model']]
        # Ordered telescoping identity: change v, then g, then conditional knob rate.
        l_ir=vl*(cv+b*(gl+(1-gl)*kl))
        dv=(a['v']-vl)*(cv+b*(gl+(1-gl)*kl))
        dg=a['v']*b*(G-gl)*(1-kl)
        dp=a['v']*b*(1-G)*(float(p[~g].mean())-kl)
        coin=a['v']*b*(1-G)*(K-float(p[~g].mean()))
        residual=a['ir']-l_ir-dv-dg-dp-coin
        result.update(lib_v=vl,lib_guard=gl,lib_knob_non_guard=kl,lib_reconstructed_ir=l_ir,
                      delta_v_ir=dv,delta_guard_ir=dg,delta_score_or_controller_ir=dp,
                      delta_coin_ir=coin,decomposition_residual=residual)
        assert abs(residual)<1e-10
    progress=[]; intervals=[]
    l_ep=tr['ep'][look]; steps=tr['step'][look]
    frac=steps/np.maximum(tr['epN'][l_ep]-1,1)
    for label,mask in [('all',np.ones(len(g),bool)),('eventual_success',tr['success'][l_ep]),
                       ('eventual_failure',~tr['success'][l_ep])]+[(f'progress_{i}',(frac>=i/5)&(frac<(i+1)/5 if i<4 else frac<=1)) for i in range(5)]:
        if not mask.any(): continue
        progress.append(dict(arm=a['arm'],cohort=a['cohort'],method=a['method'],cell=a['cell'],group=label,
                             looks=int(mask.sum()),guard=float(g[mask].mean()),knob_per_look=float(k[mask].mean()),
                             knob_non_guard=float(k[mask&~g].mean()) if (mask&~g).any() else None,
                             score_median=float(np.nanmedian(s[mask])) if np.isfinite(s[mask]).any() else None))
    near=np.zeros(len(g),bool);prevguard=np.zeros(len(g),bool);runs=[]
    for ep in range(a['n']):
        ix=np.flatnonzero(l_ep==ep); guards=ix[g[ix]]
        if len(guards):
            pos=np.flatnonzero(g[ix]);dist=np.min(abs(np.arange(len(ix))[:,None]-pos[None,:]),axis=1)
            near[ix]=dist<=2
        prevguard[ix[1:]]=g[ix[:-1]]
        miss=tr['miss'][look][ix].astype(bool); rr=0
        for m in miss:
            if m:
                if rr:runs.append(rr)
                rr=0
            else:rr+=1
        if rr:runs.append(rr)
    for label,mask in [('within_2_looks_of_guard',near),('farther_than_2_looks',~near),('after_guard',prevguard)]:
        mask=mask&~g
        intervals.append(dict(arm=a['arm'],cohort=a['cohort'],method=a['method'],cell=a['cell'],group=label,
                              eligible_looks=int(mask.sum()),knob_rate=float(k[mask].mean()) if mask.any() else None))
    result.update(near_guard_share_of_extra_calls=float(k[near].sum()/k.sum()) if k.sum() else None,
                  cache_run_units='consecutive cache-only vision anchors, including censored episode ends',
                  **quant(runs,'cache_run_'))
    return result,progress,intervals


def feature_batch(actions, w, ds, sig):
    a=actions[:,:,:10,:7].astype(np.float32);w=w.astype(np.float32)
    chunk=np.einsum('qk,qktc->qtc',w,a,optimize=False)
    var=np.einsum('qk,qk->q',w,np.mean(((a[:,:,:,:6]-chunk[:,None,:,:6])/sig[:6])**2,axis=(2,3)))
    energy=np.mean((chunk[:,:,:6]/sig[:6])**2,axis=(1,2));g=a[:,:,:,6]>=0
    trans=np.einsum('qk,qk->q',w,(g!=g[:,:,:1]).any(2));vote=np.einsum('qk,qkt->qt',w,g.astype(np.float32))
    gd=np.mean(4*vote*(1-vote),axis=1);cg=chunk[:,:,6]>=0
    return np.column_stack((np.log1p(ds[:,0]),np.log1p(var),np.log1p(energy),trans,gd,
                            (cg!=cg[:,:1]).any(1).astype(float),
                            np.log1p((ds[:,-1]-ds[:,0])/np.maximum(ds[:,0],1e-6)),1/np.sum(w*w,1)/16))


def numerical_validation():
    """Check replay feature arithmetic and distance reductions against serving code."""
    from exp.offline_search.rounds.r11.knob.recipe import score_features
    from exp.offline_search.rounds.r10.data import SubsetLibrary, STORE
    from types import SimpleNamespace
    base=pickle.load((KNOB/'artifacts/r11_pi05_l10_50_distance_ir32.pkl').open('rb'))['method'].inner.base
    lib=SubsetLibrary(STORE,'pi05_l10',50);records=[]
    for T in list(base.tasks.values())[:2]:
        a=base.act[T.rows[:16]];w=np.arange(1,17,dtype=np.float32);w/=w.sum();d=np.linspace(1,10,16)
        np.testing.assert_allclose(feature_batch(a[None],w[None],d[None],base.sig)[0],
                                   score_features(a,w,d,base.sig),atol=1e-10)
    for step in (0,2,10):
        q=SimpleNamespace(task_id=0,step=step,key_v0=lib.key_v0[step],key_v1=lib.key_v1[step],rs=lib.rs[step],prev_hit=True)
        T,_,_,_,_,xv,rs,d,*_=base._dist(q)
        X=np.r_[xv,rs][None].astype(np.float32);z=X@T.Wf-T.shift
        D=T.z2[None]-2*(z@T.Z.T)+(z*z).sum(1)[:,None]
        if step==0:
            y=X@T.W0f-T.c0;D=T.n20[None]-2*((y@T.A0.T)@T.Z.T)+(y*y).sum(1)[:,None]
        dist=np.sqrt(np.maximum(D,0))[0];nonself=T.rows!=step
        np.testing.assert_allclose(dist[nonself],d[nonself],atol=2e-4,rtol=2e-5)
        np.testing.assert_allclose(dist*dist,d*d,atol=2e-3,rtol=2e-5)
        records.append(dict(step=step,max_abs_distance_delta=float(np.max(abs(dist-d))),
                            max_nonself_delta=float(np.max(abs(dist[nonself]-d[nonself])))))
    result=dict(feature_arithmetic='passed against serving score_features',
                distance_arithmetic='passed within float32 batched reduction tolerance; exact-self near zero is roundoff sensitive',records=records)
    dump(HERE/'numerical_validation.json',result)
    return result


def fixed_retrieve(base, X, tasks, steps, donors_mask, head, query_ep=None):
    """Frozen deployment geometry; only donor eligibility changes. Never calls fit."""
    from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
    n=len(X); feat=np.zeros((n,8));top=np.zeros(n,int)
    for task,T in base.tasks.items():
        rows=np.flatnonzero(tasks==task)
        for lo in range(0,len(rows),128):
            qr=rows[lo:lo+128]; xx=X[qr]
            z=xx@T.Wf-T.shift
            D=T.z2[None]-2*(z@T.Z.T)+np.sum(z*z,1)[:,None]
            first=steps[qr]==0
            if first.any():
                yy=xx[first]@T.W0f-T.c0
                D[first]=T.n20[None]-2*((yy@T.A0.T)@T.Z.T)+np.sum(yy*yy,1)[:,None]
            D=np.sqrt(np.maximum(D,0))
            mask=donors_mask(qr,T.rows) if callable(donors_mask) else donors_mask[T.rows][None,:]
            D=np.where(mask,D,np.inf)
            if query_ep is not None: D[query_ep[qr,None]==base.lib_ep[T.rows][None,:]]=np.inf
            ids=np.argsort(D,axis=1,kind='stable')[:,:16]
            ds=np.take_along_axis(D,ids,1).astype(np.float64)
            assert np.isfinite(ds).all()
            donors=T.rows[ids];w=_kernel_w(ds-ds[:,:1],base.kref);w=(w/w.sum(1,keepdims=True)).astype(np.float32)
            feat[qr]=feature_batch(base.act[donors],w,ds,base.sig);top[qr]=donors[:,0]
    pred=np.c_[np.ones(n),np.clip((feat-np.array(head['mean']))/np.array(head['std']),-8,8)]@np.array(head['coef'])
    return dict(distance=np.expm1(feat[:,0]),disagreement=np.expm1(feat[:,1]),predicted_error=pred,top=top)


def replay_cell(cell, ref):
    """B-only reference transfer and paired donor-density diagnostics, cached by inputs."""
    model,suite,size_s=cell.split('_');size=int(size_s)
    path=KNOB/'artifacts'/f'r11_{cell}_distance_ir25.pkl'
    if not path.exists(): path=next((KNOB/'artifacts').glob(f'r11_{cell}_*.pkl'))
    cache=HERE/'cache'/f'replay_{cell}'
    libdir=Path('/home/weiland/trace_runs/offline_search_store/library')/f'{model}_{suite}'/('bpool_cs' if model=='pi05' else 'bpool_all')
    fp=dict(version=VERSION,artifact=stamp(path),artifact_sha256=sha(path),
            signals_sha256=sha(OWNER/'data'/cell/'signals.npz'),
            predictor_sha256=sha(OWNER/'data'/cell/'predictor.json'),
            library_arrays=[stamp(p) for p in sorted(libdir.glob('*.npy'))])
    if cache.with_suffix('.json').exists() and cache.with_suffix('.npz').exists():
        old=read_json(cache.with_suffix('.json'))
        if old['fingerprint']==fp:return npz(cache.with_suffix('.npz')),old
    from exp.offline_search.rounds.r10.data import SubsetLibrary,STORE
    from exp.offline_search.harness.dims import valid_state
    lib=SubsetLibrary(STORE,f'{model}_{suite}',size);base=pickle.load(path.open('rb'))['method'].inner.base
    head=read_json(OWNER/'data'/cell/'predictor.json')
    pcas=[npz(R10/'pca'/f'{model}_{suite}'/str(size)/f'v{c}.npz') for c in (0,1)]
    X=np.c_[pcas[0]['proj'],pcas[1]['proj'],valid_state(lib.rs,model)].astype(np.float32)
    anchor=ref['anchor']; qr=np.flatnonzero(anchor)
    task=np.asarray(lib.task_id)[qr];step=np.asarray(lib.step)[qr];ep=np.asarray(lib.episode)[qr]
    fold=ref['fold'];qfold=fold[qr]
    variants={}
    variants['fixed_reduced']=fixed_retrieve(base,X[qr],task,step,lambda q,d: qfold[q,None]!=fold[d][None,:],head)
    variants['fixed_leave_episode']=fixed_retrieve(base,X[qr],task,step,np.ones(lib.L,bool),head,query_ep=ep)
    # Diagnostic limit: stored selected-demo states, full donors, exact self allowed.
    variants['selected_on_demo']=fixed_retrieve(base,X[qr],task,step,np.ones(lib.L,bool),head)
    out=dict(selected_ep=ep,selected_step=step,selected_rows=qr)
    for label,r in variants.items():
        for k,v in r.items(): out[f'{label}_{k}']=v
    parent=lib.parent;parent_ep=np.asarray(parent.episode);parent_task=np.asarray(parent.task_id)
    available=np.setdiff1d(np.unique(parent_ep),np.unique(lib.episode))
    selected=[]
    # Deterministic library-only query selection: ten unused episodes per task, include failures.
    for t in range(10):
        candidates=np.intersect1d(available,np.unique(parent_ep[parent_task==t]))
        selected.extend(np.random.default_rng(20261003+t).permutation(candidates)[:10].tolist())
    if selected:
        er=np.flatnonzero(np.isin(parent_ep,selected)&(np.asarray(parent.step)%2==0))
        ext=[]
        for camera in (0,1):
            keys=getattr(parent,f'key_v{camera}');Bt=getattr(base,f'B{camera}T');mb=getattr(base,f'muB{camera}')
            pp=[]
            for lo in range(0,len(er),64):
                pp.append(np.asarray(keys[er[lo:lo+64]],np.float32)@Bt.T-mb)
            ext.append(np.concatenate(pp))
        eX=np.c_[*ext,valid_state(np.asarray(parent.rs)[er],model)].astype(np.float32)
        et=parent_task[er];es=np.asarray(parent.step)[er]
        out.update(external_ep=parent_ep[er],external_step=es,external_rows=er)
        full=fixed_retrieve(base,eX,et,es,np.ones(lib.L,bool),head)
        for k,v in full.items():out[f'external_full_{k}']=v
        for f in range(5):
            reduced=fixed_retrieve(base,eX,et,es,fold!=f,head)
            for k,v in reduced.items():out[f'external_fold{f}_{k}']=v
        assert not set(selected)&set(np.unique(lib.episode))
    meta=dict(fingerprint=fp,cell=cell,external_episodes=len(selected),external_episode_ids=selected,
              selected_episodes=len(np.unique(ep)),fitting='none; all metrics, scales, predictor, thresholds frozen',
              external_note='100 unused parent-B episodes if available; five paired donor masks with fixed geometry',
              selected_note='Full-minus-query-episode has metric-fit exposure; diagnostic only. On-demo includes exact self; limit, not performance.')
    np.savez_compressed(cache.with_suffix('.npz'),**out);dump(cache.with_suffix('.json'),meta)
    return out,meta


def replay_summary(a,ref,replay,meta):
    cfg=cfg_for(a); name=score_name(a['method']);rows=[]
    arrays={'crossfit_reference':ref[name][ref['anchor']],
            'final_head_oof_features':ref['final_error'][ref['anchor']] if 'error' in a['method'] else ref[name][ref['anchor']]}
    for label in ('fixed_reduced','fixed_leave_episode','selected_on_demo','external_full'):
        k=f'{label}_{name}'
        if k in replay:arrays[label]=replay[k]
    folds=[replay[f'external_fold{f}_{name}'] for f in range(5) if f'external_fold{f}_{name}' in replay]
    if folds:arrays['external_reduced_mean_of_5']=np.concatenate(folds)
    for label,s in arrays.items():
        row=dict(arm=a['arm'],cohort=a['cohort'],cell=a['cell'],method=a['method'],target=a['target'],variant=label,
                 **quant(s,'score_'),threshold=cfg.get('threshold'))
        if a['method']!='adaptive_error_hybrid':
            row.update(high_share=float(np.mean(s>cfg['threshold'])),mean_probability=float(static_p(s,cfg).mean()))
        else:
            # Fixed initial dose diagnostic only, NOT a dynamic adaptive IR replay.
            pp=frozen_score_p(s,cfg,ref[name])
            row.update(high_share=float(np.mean(2*pp-cfg['dose'])),
                       mean_probability=float(np.mean(pp)),
                       note='initial dose only, not realized adaptive IR')
        if label in ('fixed_reduced','fixed_leave_episode','selected_on_demo','external_full'):
            g=replay[f'{label}_guard'];v=replay['external_v' if label.startswith('external') else 'selected_v']
            pp=frozen_score_p(s,cfg,ref[name]);r_ir=v*(CV[a['model']]+(1-CV[a['model']])*np.mean(g+(1-g)*pp))
            row.update(replay_guard=float(np.mean(g)),replay_v=float(v),
                       fixed_setting_exogenous_ir=float(r_ir))
        if label=='external_reduced_mean_of_5':
            gg=np.concatenate([replay[f'external_fold{f}_guard'] for f in range(5)])
            pp=frozen_score_p(s,cfg,ref[name]);v=replay['external_v']
            row.update(replay_guard=float(np.mean(gg)),replay_v=float(v),
                       fixed_setting_exogenous_ir=float(v*(CV[a['model']]+(1-CV[a['model']])*np.mean(gg+(1-gg)*pp))))
        rows.append(row)
    if folds:
        full=arrays['external_full'];reduced=np.stack(folds)
        assert name!='distance' or np.all(full[None]<=reduced+1e-5)
        perq=(full[None]-reduced).mean(0)
        ep=replay['external_ep'];epdelta=np.array([perq[ep==e].mean() for e in np.unique(ep)])
        rng=np.random.default_rng(414);boot=epdelta[rng.integers(len(epdelta),size=(2000,len(epdelta)))].mean(1)
        rows.append(dict(arm=a['arm'],cohort=a['cohort'],cell=a['cell'],method=a['method'],target=a['target'],
                         variant='external_paired_full_minus_reduced',external_episodes=meta['external_episodes'],
                         episode_mean_score_delta=float(epdelta.mean()),delta_ci_low=float(np.quantile(boot,.025)),
                         delta_ci_high=float(np.quantile(boot,.975)),
                         probability_delta=(float(np.mean(static_p(full,cfg))-np.mean(static_p(reduced,cfg)))
                                            if a['method']!='adaptive_error_hybrid' else None)))
    return rows


def replay_accounting(replay,cell):
    from exp.offline_search.rounds.r10.data import SubsetLibrary,STORE
    model,suite,size=cell.split('_');lib=SubsetLibrary(STORE,f'{model}_{suite}',int(size))
    progress=np.asarray(lib.progress);den=np.maximum(np.asarray(lib.ep_len)-1,1)
    for cohort,querylib in [('selected',lib),('external',lib.parent)]:
        if f'{cohort}_rows' not in replay:continue
        qr=replay[f'{cohort}_rows'];ep=replay[f'{cohort}_ep'];step=replay[f'{cohort}_step']
        N=sum(np.asarray(querylib.ep_len)[qr[np.flatnonzero(ep==e)[0]]] for e in np.unique(ep))
        replay[f'{cohort}_v']=len(qr)/N
        labels=('fixed_reduced','fixed_leave_episode','selected_on_demo') if cohort=='selected' else ('external_full',*(f'external_fold{f}' for f in range(5)))
        for label in labels:
            top=replay[f'{label}_top'];g=np.zeros(len(qr),bool)
            for e in np.unique(ep):
                ix=np.flatnonzero(ep==e);ix=ix[np.argsort(step[ix])]
                assert np.array_equal(step[ix],np.arange(0,step[ix[-1]]+1,2))
                g[ix[1:]]=np.diff(progress[top[ix]])*den[top[ix[1:]]]<=.5
            replay[f'{label}_guard']=g


def references():
    out=[]
    for model in ('pi05','groot'):
        for suite in ('l10','spatial'):
            spec=dict(arm=f'r8_{model}_{suite}_P10',model=model,suite_short=suite,r11_method='pure_policy')
            a,_=load_arm('r08_main',spec,'historical')
            if a:out.append(a)
    for model in ('pi05','groot'):
        for suite,sizes in [('l10',(50,200,500)),('spatial',(50,))]:
            for size in sizes:
                arm=f'r10_{model}_{suite}_{size}_GC_dist'
                for root in (f'r10_corr3_{model}',f'r10_corr3_{model}_b'):
                    if not (RUNS/root/'runs'/arm/'summary.json').exists():continue
                    a,_=load_arm(root,dict(arm=arm,model=model,suite_short=suite,r10_size=size,
                                          r11_method='historical_off'),'historical')
                    if a:out.append(a)
                    break
    for root,arm,model in [('r11_local_idg','r11_groot_l10_50_off','groot')]:
        a,_=load_arm(root,dict(arm=arm,model=model,suite_short='l10',r10_size=50),'hardware')
        if a:out.append(a)
    return out


def bootstrap_arms(arms, reps):
    """Same episode draw for every target/size/model in a suite: preserve reuse.

    Resamples the fixed evaluation initial conditions, not tasks. It does not
    estimate policy stochasticity or deployment-population sampling uncertainty.
    """
    rng=np.random.default_rng(2026100301); groups={};out={}
    for suite in ('l10','spatial'):
        keys=sorted(set(k for a in arms if a['suite']==suite for k in a['outcomes']))
        if not keys:continue
        n=len(keys);idx=rng.integers(n,size=(reps,n))
        # Multinomial weights let every arm share exactly the same paired draw.
        W=np.zeros((reps,n),np.float64)
        for r in range(reps):W[r]=np.bincount(idx[r],minlength=n)
        groups[suite]=(keys,W)
    for a in arms:
        if not a.get('log_audit',{}).get('usable'):continue
        keys,W=groups[a['suite']];tr=a['trace'];ix={k:i for i,k in enumerate(tr['keys'])}
        if set(keys)!=set(ix):continue
        order=np.array([ix[k] for k in keys]);Y=tr['success'][order].astype(float)
        N=tr['epN'][order];cost=CV[a['model']]*tr['epV'][order]+(1-CV[a['model']])*tr['epM'][order]
        out[a['arm']]=dict(sr=W@Y/W.sum(1),ir=(W@cost)/(W@N))
    return out


def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return (float(np.quantile(x,.025)),float(np.quantile(x,.975))) if len(x) else (None,None)


def interpolation(curve,x,boot=None):
    if boot is None:
        pts=sorted((a['ir'],a['sr']) for a in curve)
    else:pts=sorted((boot[a['arm']]['ir'],boot[a['arm']]['sr']) for a in curve)
    if not pts or x<pts[0][0] or x>pts[-1][0]:return np.nan
    return float(np.interp(x,[p[0] for p in pts],[p[1] for p in pts]))


def frontier(arms,boots,reps):
    groups=defaultdict(list);off={}
    for a in arms:
        groups[(a['cell'],a['method'])].append(a)
        if a['method']=='off':off[a['cell']]=a
    rows=[];boot_rows={};eff=[];effboots={}
    # Main comparison: common support of knob-only curves, no extrapolation.
    for cell in sorted({a['cell'] for a in arms}):
        for method,baseline in [(m,b) for m in OWN for b in ('random','periodic')]+[('periodic','random')]:
            ca=groups[(cell,method)];cb=groups[(cell,baseline)]
            if not ca or not cb:
                rows.append(dict(cell=cell,method=method,baseline=baseline,status='missing method or comparator'))
                continue
            lo=max(min(a['ir'] for a in ca),min(a['ir'] for a in cb))
            hi=min(max(a['ir'] for a in ca),max(a['ir'] for a in cb))
            singleton=method=='disagreement' and len(ca)==1
            if hi<lo or (len(ca)<2 and not singleton) or len(cb)<2:
                rows.append(dict(cell=cell,method=method,baseline=baseline,n_method=len(ca),n_baseline=len(cb),
                                 status='no common interpolatable knob-only support',
                                 method_range=[min(a['ir'] for a in ca),max(a['ir'] for a in ca)],
                                 baseline_range=[min(a['ir'] for a in cb),max(a['ir'] for a in cb)]))
                continue
            x=ca[0]['ir'] if singleton else (lo+hi)/2
            d=(ca[0]['sr'] if singleton else interpolation(ca,x))-interpolation(cb,x)
            ds=np.full(reps,np.nan)
            if all(a['arm'] in boots for a in ca+cb):
                for r in range(reps):
                    b={a['arm']:{k:v[r] for k,v in boots[a['arm']].items()} for a in ca+cb}
                    if singleton:
                        ds[r]=b[ca[0]['arm']]['sr']-interpolation(cb,b[ca[0]['arm']]['ir'],b)
                    else:ds[r]=interpolation(ca,x,b)-interpolation(cb,x,b)
            low,high=ci(ds);key=f'{cell}:{method}:{baseline}'
            boot_rows[key]=ds
            rows.append(dict(cell=cell,method=method,baseline=baseline,n_method=len(ca),n_baseline=len(cb),
                             status='matched realized IR',support_low=lo,support_high=hi,matched_ir=x,
                             sr_delta_pp=100*d,ci_low_pp=None if low is None else 100*low,
                             ci_high_pp=None if high is None else 100*high,bootstrap_valid=int(np.isfinite(ds).sum()),
                             comparison_type='observed single arm vs interpolated comparator' if singleton else 'two interpolated curves',
                             paired_episodes=len(set.intersection(*(set(a['outcomes']) for a in ca+cb)))))
    # Explicit secondary comparison at each owned arm, allowing off->lowest-point interpolation.
    secondary=[]
    for a in arms:
        if a['method'] not in OWN:continue
        for baseline in ('random','periodic'):
            cb=groups[(a['cell'],baseline)]; o=off.get(a['cell']);curve=([o] if o else [])+cb
            pred=interpolation(curve,a['ir']) if len(curve)>1 else np.nan
            row=dict(arm=a['arm'],cell=a['cell'],method=a['method'],baseline=baseline,realized_ir=a['ir'],
                     status='unsupported: no bracket',off_anchor=False)
            if np.isfinite(pred):
                off_anchor=bool(o and a['ir']<min((b['ir'] for b in cb),default=np.inf))
                ds=np.full(reps,np.nan)
                if a['arm'] in boots and all(b['arm'] in boots for b in curve):
                    for r in range(reps):
                        boot={b['arm']:{k:v[r] for k,v in boots[b['arm']].items()} for b in curve}
                        ds[r]=boots[a['arm']]['sr'][r]-interpolation(curve,boots[a['arm']]['ir'][r],boot)
                low,high=ci(ds)
                row.update(status='secondary off-anchored' if off_anchor else 'arm vs comparator interpolation',
                           off_anchor=off_anchor,sr_delta_pp=100*(a['sr']-pred),ci_low_pp=None if low is None else 100*low,
                           ci_high_pp=None if high is None else 100*high,bootstrap_valid=int(np.isfinite(ds).sum()))
            secondary.append(row)
    # Frontier means weight each supported cell once. No repeated arm-pair independence assumption.
    pooled=[]
    for method,baseline in [(m,b) for m in OWN for b in ('random','periodic')]+[('periodic','random')]:
        rr=[r for r in rows if r['method']==method and r['baseline']==baseline and r['status']=='matched realized IR']
        if not rr:continue
        bd=np.stack([boot_rows[f"{r['cell']}:{method}:{baseline}"] for r in rr])
        # Only joint supported draws; avoid silently changing the cell mix.
        joint=np.all(np.isfinite(bd),axis=0);d=bd[:,joint].mean(0);lo,hi=ci(d)
        pooled.append(dict(method=method,baseline=baseline,n_cells=len(rr),cells=[r['cell'] for r in rr],
                           sr_delta_pp=np.mean([r['sr_delta_pp'] for r in rr]),ci_low_pp=None if lo is None else 100*lo,
                           ci_high_pp=None if hi is None else 100*hi,bootstrap_valid=int(joint.sum())))
    for method in METHODS[1:]:
        pairs=[(a,off[a['cell']]) for a in arms if a['method']==method and a['cell'] in off]
        if not pairs:continue
        gain=sum(a['sr']-o['sr'] for a,o in pairs);spend=sum(a['ir']-o['ir'] for a,o in pairs)
        bs=np.full(reps,np.nan)
        if all(a['arm'] in boots and o['arm'] in boots for a,o in pairs):
            gg=sum(boots[a['arm']]['sr']-boots[o['arm']]['sr'] for a,o in pairs)
            ii=sum(boots[a['arm']]['ir']-boots[o['arm']]['ir'] for a,o in pairs)
            bs=np.where(ii>0,gg/ii,np.nan)
        lo,hi=ci(bs)
        eff.append(dict(method=method,n_arms=len(pairs),n_cells=len(set(a['cell'] for a,o in pairs)),
                        sr_gain_sum=gain,ir_spend_sum=spend,efficiency=gain/spend if spend>0 else None,
                        ci_low=lo,ci_high=hi,baseline='same-batch off only'))
    return rows,secondary,pooled,eff


def hardware(arms,refs,reps):
    results=[]
    for model,local,remote in [
        ('pi05',next((a for a in arms if a['cell']=='pi05_l10_50' and a['method']=='off'),None),
         next((a for a in refs if a['arm']=='r10_pi05_l10_50_GC_dist'),None)),
        ('groot',next((a for a in refs if a['root']=='r11_local_idg'),None),
         next((a for a in arms if a['cell']=='groot_l10_50' and a['method']=='off'),None))]:
        if local is None or remote is None:
            results.append(dict(model=model,status='pending; one required arm missing'));continue
        rng=np.random.default_rng(123);keys=sorted(set(local['outcomes'])&set(remote['outcomes']))
        d=np.array([int(local['outcomes'][k]['success'])-int(remote['outcomes'][k]['success']) for k in keys])
        b=d[rng.integers(len(d),size=(reps,len(d)))].mean(1);lo,hi=ci(b)
        results.append(dict(model=model,status='available',local=local['arm'],remote=remote['arm'],n=len(keys),
                            local_root=local['root'],remote_root=remote['root'],
                            local_sr=local['sr'],remote_sr=remote['sr'],delta_sr_pp=100*d.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi,
                            local_ir=local['ir'],remote_ir=remote['ir'],delta_ir=local['ir']-remote['ir'],
                            wins=int((d>0).sum()),losses=int((d<0).sum())))
    return results


def final_comparisons(arms,refs,boots,placement,reps):
    """Paired secondary summaries, preserving episode reuse across grid points."""
    by={(a['cell'],a['method'],a['target']):a for a in arms}
    off={a['cell']:a for a in arms if a['method']=='off'}
    targets=[];effdiff=[];baseline=[]
    for method,control in [('periodic','random'),('error_hybrid','random'),('distance','random'),
                            ('error_hybrid','periodic'),('distance','periodic')]:
        pairs=[(a,by[(a['cell'],control,a['target'])]) for a in arms if a['method']==method
               and (a['cell'],control,a['target']) in by]
        gains=[];wins=losses=0
        for a,b in pairs:
            keys=set(a['outcomes'])&set(b['outcomes']);assert len(keys)==500
            diffs=[int(a['outcomes'][k]['success'])-int(b['outcomes'][k]['success']) for k in keys]
            wins+=sum(d>0 for d in diffs);losses+=sum(d<0 for d in diffs)
            gains.append(a['sr']-b['sr'])
        ds=np.mean([boots[a['arm']]['sr']-boots[b['arm']]['sr'] for a,b in pairs],axis=0)
        lo,hi=ci(ds)
        targets.append(dict(method=method,control=control,n_arms=len(pairs),arm_episode_pairs=500*len(pairs),
                            wins=wins,losses=losses,sr_delta_pp=100*np.mean(gains),ci_low_pp=100*lo,ci_high_pp=100*hi,
                            mean_ir_difference=np.mean([a['ir']-b['ir'] for a,b in pairs]),
                            estimand='same target, not same realized IR; bootstrap preserves reused episode identities'))
        def ratio(side,boot=False):
            group=[p[side] for p in pairs]
            if boot:
                g=sum(boots[a['arm']]['sr']-boots[off[a['cell']]['arm']]['sr'] for a in group)
                s=sum(boots[a['arm']]['ir']-boots[off[a['cell']]['arm']]['ir'] for a in group)
                return np.where(s>0,g/s,np.nan)
            return sum(a['sr']-off[a['cell']]['sr'] for a in group)/sum(a['ir']-off[a['cell']]['ir'] for a in group)
        diff=ratio(0,True)-ratio(1,True);lo,hi=ci(diff)
        effdiff.append(dict(method=method,control=control,n_arms=len(pairs),n_cells=len({a['cell'] for a,b in pairs}),
                            method_efficiency=ratio(0),control_efficiency=ratio(1),difference=ratio(0)-ratio(1),
                            ci_low=lo,ci_high=hi,estimand='difference of average gain/spend ratios; not matched-IR marginal utility'))
    for a in off.values():
        for label,b in [('R10 off',next(r for r in refs if r['cell']==a['cell'] and r['method']=='historical_off')),
                         ('R8 pure',next(r for r in refs if (r['model'],r['suite'])==(a['model'],a['suite']) and r['method']=='pure_policy'))]:
            keys=sorted(set(a['outcomes'])&set(b['outcomes']));assert len(keys)==500
            d=np.array([int(a['outcomes'][k]['success'])-int(b['outcomes'][k]['success']) for k in keys])
            rng=np.random.default_rng(529);draws=d[rng.integers(len(d),size=(reps,len(d)))].mean(1);lo,hi=ci(draws)
            baseline.append(dict(cell=a['cell'],reference=label,n=len(keys),current_root=a['root'],reference_root=b['root'],
                                 current_sr=a['sr'],reference_sr=b['sr'],delta_sr_pp=100*d.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi,
                                 current_ir=a['ir'],reference_ir=b['ir'],delta_ir=a['ir']-b['ir']))
    placement_pooled=[]
    for method in OWN:
        for group in ['eventual_success','eventual_failure',*(f'progress_{i}' for i in range(5))]:
            rr=[r for r in placement if r['cohort']=='test' and r['method']==method and r['group']==group]
            looks=sum(r['looks'] for r in rr);guards=sum(r['looks']*r['guard'] for r in rr)
            calls=sum(r['looks']*r['knob_per_look'] for r in rr)
            placement_pooled.append(dict(method=method,group=group,looks=looks,guard_per_look=guards/looks,
                                         knob_per_look=calls/looks,knob_non_guard=calls/(looks-guards)))
    extras=dict(target_crosscheck=targets,efficiency_comparisons=effdiff,baseline_comparisons=baseline,placement_pooled=placement_pooled)
    for name,rows in extras.items():csvout(name+'.csv',rows)
    return extras


def aggregate(diags,inventory):
    rows=[]
    for cohort in ('test','dev'):
        for method in METHODS:
            rr=[r for r in diags if r['cohort']==cohort and r['method']==method]
            planned=len({r['arm'] for r in inventory if r['cohort']==cohort and r['method']==method})
            if not planned:continue
            row=dict(cohort=cohort,method=method,complete=len(rr),planned=planned,
                     logged=sum(r['logs_usable'] for r in rr),episodes=sum(r['n_episodes'] for r in rr))
            if rr and method!='off':
                row.update(mean_error_target=np.mean([r['error_target'] for r in rr]),
                           mean_error_prediction=np.mean([r['error_prediction'] for r in rr]),
                           max_abs_error=max(abs(r['error_target']) for r in rr),
                           within_02=sum(abs(r['error_target'])<=.02 for r in rr))
                for name in ('delta_v_ir','delta_guard_ir','delta_score_or_controller_ir','delta_coin_ir'):
                    values=[r[name] for r in rr if name in r]
                    if values:row[name]=np.mean(values)
            rows.append(row)
    return rows


def plots(arms,refs,diags,replays,final=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(METHODS,('#222222','#777777','#ad7900','#c53b33','#16749b','#6d48a7','#269256','#b77265','#8c849d')))
    cells=sorted({a['cell'] for a in arms});fig,axs=plt.subplots(2,4,figsize=(18,8),squeeze=False)
    suite_limits={}
    for suite in ('l10','spatial'):
        ys=[100*a['sr'] for a in arms+refs if a['suite']==suite]
        suite_limits[suite]=(5*np.floor(min(ys)/5)-1,min(101,5*np.ceil(max(ys)/5)+1))
    for ax,cell in zip(axs.flat,cells):
        for method in METHODS:
            rr=sorted([a for a in arms if a['cell']==cell and a['method']==method],key=lambda a:a['ir'])
            if not rr:continue
            ax.plot([a['ir'] for a in rr],[100*a['sr'] for a in rr],marker='o',label=method,color=colors[method],lw=1.3)
        m,s,_=cell.split('_');pure=next((a for a in refs if a['model']==m and a['suite']==s and a['method']=='pure_policy'),None)
        if pure:ax.scatter(pure['ir'],100*pure['sr'],marker='*',s=110,color='black',label='R8 pure policy')
        old=next((a for a in refs if a['cell']==cell and a['method']=='historical_off'),None)
        if old:ax.scatter(old['ir'],100*old['sr'],marker='D',s=45,facecolors='none',edgecolors='black',label='R10 historical off')
        ax.set(title=cell,xlabel='Realized owner IR',ylabel='Success (%)',xlim=(.07,.53),ylim=suite_limits[s]);ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(cells):]:ax.axis('off')
    handles={}
    for ax in axs.flat:
        hh,ll=ax.get_legend_handles_labels();handles.update(zip(ll,hh))
    fig.legend(handles.values(),handles.keys(),loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('R11 '+('FINAL: all 94 arms' if final else 'INTERIM: completed arms only')+'; lines interpolate observed points')
    fig.tight_layout(rect=(0,.10,1,.95));fig.savefig(HERE/'frontier.png',dpi=160);plt.close(fig)
    own=[r for r in diags if r['cohort']=='test' and r['method'] in OWN and r['logs_usable']]
    fig,axs=plt.subplots(max(1,(len(own)+3)//4),4,figsize=(17,3*max(1,(len(own)+3)//4)),squeeze=False)
    lookup={a['arm']:a for a in arms}
    for ax,r in zip(axs.flat,own):
        a=lookup[r['arm']];ref=ref_data(a['cell']);name=score_name(a['method']);tr=a['trace']
        ss=tr['score'][tr['vision'].astype(bool)]
        for label,x in [('B crossfit looks',ref[name][ref['anchor']]),('Closed loop',ss)]:
            x=np.sort(x[np.isfinite(x)]);ax.plot(x,np.arange(1,len(x)+1)/len(x),label=label)
        rep=replays.get(a['cell'])
        if rep and f'external_full_{name}' in rep[0]:
            x=np.sort(rep[0][f'external_full_{name}']);ax.plot(x,np.arange(1,len(x)+1)/len(x),label='External B/full donors',ls='--')
        if r.get('threshold') is not None:ax.axvline(r['threshold'],color='black',ls=':',label='Frozen threshold')
        allx=np.r_[ref[name][ref['anchor']],ss];ax.set_xlim(np.quantile(allx,.005),np.quantile(allx,.99))
        ax.set_title(f"{a['cell']}\n{a['method']} target {a['target']:.2f}",fontsize=9);ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(own):]:ax.axis('off')
    if own:axs.flat[0].legend(fontsize=7)
    fig.suptitle(('FINAL' if final else 'INTERIM')+' score CDFs (x-axis trimmed at pooled 0.5–99 percentiles; adaptive cutoff changes)')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(HERE/'score_cdfs.png',dpi=150);plt.close(fig)


def fmt(x,n=4,sign=False):
    if x is None or isinstance(x,float) and not np.isfinite(x):return '—'
    return format(x,('+' if sign else '')+f'.{n}f')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                     ['| '+' | '.join(str(c) for c in row)+' |' for row in rows])


def report(arms,dev,inventory,diags,agg,replayrows,metas,matched,secondary,pooled,eff,hw,refs,provenance,extras=None):
    # Report rendering lives with the entry script so the single rerun refreshes all numbers.
    final=provenance.get('final',False)
    extras=extras or {}
    pooled_by={(r['method'],r['baseline']):r for r in pooled}
    def contrast(method,baseline,language='en'):
        r=pooled_by.get((method,baseline))
        if r is None:return '缺少可比较区间' if language=='cn' else 'No comparable support.'
        values=f"{r['sr_delta_pp']:+.2f} [{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]"
        return (f"{values} 个百分点（{r['n_cells']} 个单元）" if language=='cn' else
                f"{values} pp across {r['n_cells']} cells, {r['bootstrap_valid']} supported bootstrap draws")
    owner_frontier=('按实际调用成本匹配后，定期调用相对随机调用的成功率差为 '+contrast('periodic','random','cn')+'，支持定期调用优于随机调用。'
                    '误差混合相对随机为 '+contrast('error_hybrid','random','cn')+'，相对定期为 '+contrast('error_hybrid','periodic','cn')+'；'
                    '这两项区间都跨过零，尚不能认定误差混合在同样花费下更好。方括号表示配对抽样区间。')
    counts={r['method']:r for r in agg if r['cohort']=='test'}
    cn={'distance':'距离触发','error_hybrid':'误差评分与随机混合','adaptive_error_hybrid':'自适应误差混合',
        'disagreement':'邻居分歧触发','random':'随机调用','periodic':'定期调用','off':'关闭额外调用',
        'periodic_pgt1':'定期调用并追加一段','random_tail2':'随机调用并追加一段'}
    own=[r for r in diags if r['cohort']=='test' and r['method'] in OWN]
    dd=[r for r in replayrows if r['cohort']=='test' and r['method']=='distance' and r['variant']=='external_paired_full_minus_reduced']
    donor_drop=[-100*r['probability_delta'] for r in dd]
    donor_find=(f"独立示范回放中，仅增加可检索示范造成的距离阈值越线率下降为 {min(donor_drop):.2f}–{max(donor_drop):.2f} 个百分点，"
                '不足以单独解释目前较大的欠支。最终检索参数与留出校准参数也不同；现有数据支持分布转移，'
                '但不能证明剩余差距都由闭环跟随示范造成。') if donor_drop else '独立供体数量对照尚未覆盖已完成的距离实验，不能归因。'
    dsummary=counts.get('distance',{})
    score_find=(f"Distance: mean IR error {fmt(dsummary.get('mean_error_target'),sign=True)}; "
                f"the logged score-probability component is {fmt(dsummary.get('delta_score_or_controller_ir'),sign=True)}, "
                f"guard component {fmt(dsummary.get('delta_guard_ir'),sign=True)}, and look-rate component {fmt(dsummary.get('delta_v_ir'),sign=True)}. "
                'Zero coin contribution for deterministic distance thresholds and the probability reconstruction checks rule out an incorrectly '
                'implemented threshold as the observed explanation. This decomposition identifies changed score selection, not its causal origin.')
    lines=['# R11 最终分析 / FINAL ANALYSIS' if final else '# R11 分析中期稿 / ANALYSIS INTERIM',
           '在仓库根目录运行以下唯一命令，即可重新扫描已完成实验、更新全部表格、图和本报告：',
           '```bash\n'+RERUN+'\n```',
           '## 给负责人的说明（最终）' if final else '## 给负责人的说明（中期）',
           (f"已纳入全部 {len(arms)}/{provenance['test_planned']} 个实验，每个实验 500 个回合，共 47,000 次测试。"
            '补跑的五个实验使用远端完成版本，中止副本不计入；每个实验只计算一次。'
            '另有二十个开发集实验用于检查偏差是否重复出现，不混入测试结果。' if final else
            f"本次纳入 {len(arms)}/{provenance['test_planned']} 个已完成实验。不同方法完成比例不同，不能当作最终排名。"),
           table(['方法','已完成／计划','实际调用成本减目标','误差不超过零点零二'],[
               [cn[m],f"{counts[m]['complete']}/{counts[m]['planned']}",fmt(counts[m].get('mean_error_target')),
                f"{counts[m].get('within_02',0)}/{counts[m]['complete']}" if m!='off' else '不适用'] for m in METHODS if m in counts]),
           '距离触发的主要问题是运行中的分数低于原校准分数，因而很少越过固定阈值。误差评分加入随机部分后，少调用的幅度减小；'
           '自适应方法会提高调用概率，但每回合从原始设定重新开始，短回合可能结束于尚未补足预算的状态。'
           '这些是冻结方法的测量结果，没有据此改阈值或重新训练。',
           '这里把两种解释分开检查：一是可检索示范从原来的五分之四增加到全库，会让检索距离变小；二是闭环执行缓存动作后，'
           '状态可能更贴近已有示范。新增回放在未进入检索库的另一批示范上固定全部检索参数，只改变可检索示范的数量，'
           '因此第一项可以独立测量。直接用库里的示范本身查询全库只是“非常贴近示范”的极端参考，不能当作第二项的因果证明。',
           donor_find,
           owner_frontier if final else
           '成功率必须在实际调用成本相同处比较。只在已测曲线重叠范围内插值；从关闭状态连接到最低调用点的结果另列为次要分析。',
           ('邻居分歧触发相对随机的差为 '+contrast('disagreement','random','cn')+'，相对定期为 '+contrast('disagreement','periodic','cn')+'。'
            '它提示了一点有用信号，但每个单元只测了一个设置，且这些探索性区间没有校正多项比较，仍需复现。' if final else ''),
           ('若现在需要一个能准确控制花费的开关，优先采用定期调用。误差混合的平均成本效率最高，值得保留为候选，'
            '但它花得更少，不能把这个比值当作在相同花费下优于定期调用的证据。距离触发的预算偏差过大，不建议作为通用预算开关。'
            '自适应方案改善了预算跟踪，仍有八个实验中的三个欠支超过零点零二。' if final else ''),
           '下一轮的建议是用不进入检索库的示范、完整检索库和最终预测器校准分数，再让一个全局、与任务无关的成本反馈器控制实际预算。'
           '状态分数负责决定把预算花在哪里，随机部分负责保留基本调用机会。只靠离线阈值无法保证闭环成本精确达标；'
           '原有强制保护成本超过目标时，额外调用只能关闭。这套重新校准与反馈方案仅为未来假设，未在本轮重新拟合或作为测试成绩。',
           ('是否默认关闭，应看本轮对照的成功率，不能只看示范库大小。原先一组大库的历史对照成功率为九成以上，本轮却降到约八成六；'
            '因此已撤回把它归为接近纯策略、默认关闭的建议。两项硬件对照没有显示明确的系统差异，但区间较宽，也不能解释这次基线变化。' if final else ''),
           '## Technical scope and provenance',
           f"Snapshot started **{provenance['snapshot_utc']}**, report generated **{provenance['finished_utc']}**. "
           f"Completed: {len(arms)} test arms and {len(dev)} separate dev arms. "
           '`r11_knob_3` is never read. `r11_knob_4` contributes the five H100 completion runs; the other nineteen listed specs '
           'are completed in `r11_local_k4`. Its aborted `r11_pi05_l10_200_random_ir25` copy has no summary and is excluded. '
           '`r11_local_idg` is used only for the requested hardware check. '
           'All numbers are reconstructed directly from journals, summaries and accepted decision logs; no coordinator aggregate is needed.',
           table(['Root','Selected complete arms','Listed specs (overlap across migrated roots)'],[[root,sum(a['root']==root for a in arms),sum(r['root']==root for r in inventory)] for root in ROOTS]),
           f"Unique planned arms: **{provenance['test_planned']}**. Counts are deduplicated by arm name, not summed over root manifests. "
           'More than one complete copy of any arm fails discovery instead of silently picking an outcome.',
           'Completion requires summary and journal to agree on 500 accepted, error-free episode outcomes (50 in dev). '
           'Decision logs are filtered to the accepted attempt and deduplicated by episode and step. '
           'Log counts must equal the cost ledger and episode steps must be contiguous; diagnostics are suppressed when they fail. '
           'The manifest records source hashes for summaries/journals, decision-file sizes/mtimes before and after reading, replay hashes, '
           'and exclusions. Caches are invalidated by source metadata changes. Original calibration artifacts and run roots are read-only.',
           f"Logs reconciled: {sum(r['logs_usable'] for r in diags)}/{len(diags)} analyzed test+dev arms. "
           'All owner costs use `IR = c_v V/N + (1-c_v) M/N`, with `c_v=.152` for pi05 and `.148` for groot. '
           'This is neither measured latency nor the newer stage-pricing field in summary.json. N counts requested five-control slots; terminal slots may be partial.',
           '## Realized IR and exact accounting decomposition',
           table(['Method','n / planned','IR − target','IR − library','within ±.02','max abs error'],[
               [r['method'],f"{r['complete']}/{r['planned']}",fmt(r.get('mean_error_target')),fmt(r.get('mean_error_prediction')),
                f"{r.get('within_02',0)}/{r['complete']}",fmt(r.get('max_abs_error'))]
               for r in agg if r['cohort']=='test' and r['method']!='off']),
           'For each arm, let `v=V/N`, `g=G/V`, and `k=K/(V-G)`, with K counting knob-only calls. '
           'Then `IR=v[c_v+c_m(g+(1-g)k)]`. The table changes library v to live v, then library g to live g, '
           'then library conditional knob probability to the logged live probability, then probability to sampled calls. '
           'This ordered telescoping decomposition is exact; the attribution depends on this stated order. '
           'For adaptive arms the reference is the frozen 128-seed replay, and the score/controller term includes dynamic dose changes. '
           '`r11_knob` is the pre-OR sample and can overlap a guard; `os_r11_knob_call` is the additional call. They are not interchangeable.',
           table(['Method','n','look-rate contribution','guard contribution','score/controller contribution','coin contribution'],[
               [r['method'],r['complete'],fmt(r.get('delta_v_ir'),sign=True),fmt(r.get('delta_guard_ir'),sign=True),
                fmt(r.get('delta_score_or_controller_ir'),sign=True),fmt(r.get('delta_coin_ir'),sign=True)]
               for r in agg if r['cohort']=='test' and r['method'] in OWN]),
           score_find,
           table(['Arm','SR','IR','target','pred','v live/lib','g live/lib','k live/lib','Δv','Δg','Δscore/control','Δcoin'],[
               [r['arm'],fmt(r['sr'],3),fmt(r['ir']),fmt(r['target'],2),fmt(r['pred_IR_lib']),
                f"{fmt(r['v'])}/{fmt(r.get('lib_v'))}",f"{fmt(r.get('guard_per_look'))}/{fmt(r.get('lib_guard'))}",
                f"{fmt(r.get('knob_non_guard'))}/{fmt(r.get('lib_knob_non_guard'))}",fmt(r.get('delta_v_ir')),
                fmt(r.get('delta_guard_ir')),fmt(r.get('delta_score_or_controller_ir')),fmt(r.get('delta_coin_ir'))] for r in own]),
           'Complete per-arm values for every comparator and the separate dev cohort are in [arms.csv](analysis/arms.csv). '
           'No calibration parameter is changed by this script.',
           '## Per-arm score distributions and frozen thresholds',
           'Reference quantiles below use actual even-step calibration looks; the original threshold/CDF was fitted on **all rows**. '
           'The CSV also includes the all-row quantiles, non-guard quantiles, CDF ranks, support tails, dose saturation and threshold exceedance. '
           'Static probability reconstruction is checked against `os_r11_p`. Dynamic arms have a changing rank cutoff `1-q`, so no fixed threshold is invented.',
           table(['Arm','live q05/q50/q95','B-look q05/q50/q95','threshold','live >t','B looks >t','live median rank'],[
               [r['arm'],' / '.join(fmt(r.get('score_'+q),3) for q in ('p05','p50','p95')),
                ' / '.join(fmt(r.get('reference_looks_'+q),3) for q in ('p05','p50','p95')),fmt(r.get('threshold'),3),
                fmt(r.get('above_threshold'),3),fmt(r.get('reference_above_threshold_looks'),3),fmt(r.get('reference_rank_p50'),3)] for r in own]),
           '![Score CDFs](analysis/score_cdfs.png)',
           'Dev remains a separate, non-test cohort:',
           table(['Method','n','dev IR − target'],[[r['method'],r['complete'],fmt(r.get('mean_error_target'))] for r in agg if r['cohort']=='dev' and r['method'] in OWN]),
           '## Separating donor count, deployment transfer, and closed-loop state shift',
           'No model is fitted in these replays. The serialized deployment metric, action scale, pooled predictor and each frozen knob setting remain fixed. '
           'Selected-library queries are compared in three forms: exclusion of the whole calibration fold; exclusion of only their entire own episode; '
           'and full donors including the query itself. The first two isolate donor eligibility under the same metric, but the metric has seen selected queries. '
           'The last is an on-demo limit with exact-self leakage, explicitly not independent value evidence.',
           'For size 50, whole-fold exclusion and whole-episode exclusion leave the same four episodes per task. '
           'They therefore cannot identify a clean “4/5 to full donor” effect for a novel query. '
           'Instead, use ten deterministically sampled **unused parent-B episodes per task** (100 per eligible cell, failures retained), '
           'projected through the frozen selected-library PCA. Each exact query is replayed against all five reduced donor masks and against all donors. '
           'Queries are never donors or fitting data for these deployment fits. The metric, action scale, predictor and query path are held fixed. '
           'At size 500 no unused B episodes exist; only the disclosed selected-library ablations are available. '
           'No task-specific layer-4 settings are introduced; task identity is used only for inherited lower-layer retrieval and episode matching.',
           table(['Cell','independent external B episodes','selected episodes'],[[m['cell'],m['external_episodes'],m['selected_episodes']] for m in metas]),
           (f"**Donor-count result:** with independent B queries and the deployment metric fixed, expanding donors lowers distance "
            f"threshold exceedance by only {min(donor_drop):.2f}–{max(donor_drop):.2f} pp across these completed distance arms. "
            'This is much smaller than the largest sparse-library crossfit-reference to live selection gaps. The simple donor-count explanation '
            'does not account for the major undershoots on the tested independent B states; its relative importance can be larger in arms with small errors. '
            'This does not identify the remaining gap as demo tracking: fold metric/scale '
            'fitting, query distribution and guard overlap are separate changes.' if donor_drop else 'Donor-count comparison pending.'),
           table(['Arm','external 4/5 p','external full p','Δp full−4/5','live fixed-setting p','live median / external full median'],
                 replay_table_rows(own,replayrows)),
           'Here p is a **score-selection** probability, unconditioned on guard. For adaptive rows it is evaluated at the frozen **initial dose only**, '
           'not a simulated adaptive IR. The paired full-minus-reduced score estimates and episode-bootstrap intervals are in '
           '[library_replays.csv](analysis/library_replays.csv). Five donor masks are averaged per query and are not treated as independent episodes. '
           'Distances under a fixed geometry cannot increase when donors are added; this monotonicity is asserted. '
           'Disagreement and predicted error need not be monotone.',
           table(['Arm','external 4/5 guard','external full guard','external 4/5 IR','external full IR','ΔIR (fixed B paths)'],
                 replay_ir_rows(own,replayrows)),
           'The preceding IRs include reconstructed no-progress guards and terminal cadence on these **fixed B episode paths**, '
           'with the unchanged thresholds. They are additional measurement diagnostics, not test results or closed-loop predictions. '
           'Adaptive rows keep q fixed at q0 for this contrast; they must not be read as a replay of the adaptive controller.',
           'For disagreement, donor expansion has a more visible effect than for distance: the per-arm probability and fixed-path IR changes '
           'above should be considered alongside the live guard-rate contribution. For error_hybrid, the random half limits the cost effect '
           'of a shifted threshold distribution. The final-head-only transfer table below is small and often positive, so it does not by itself '
           'explain the systematic negative hybrid cost error.',
           'The selected-query/full-metric ablations in the CSV also show large metric/scale-transfer shifts even with the reduced donor mask. '
           'For size 50 the reduced and leave-one-episode arrays are identical. Because the full fitted metric has seen these selected queries, '
           'their low scores cannot serve as an honest calibration replacement. The independent external-B experiment avoids that exposure. '
           'Batched float32 distance reductions agree with serving arithmetic away from exact-self zero within the checked tolerance; '
           'near-zero self distances are roundoff-sensitive (see numerical_validation.json).',
           'The full-minus-reduced external-B comparison identifies the donor-eligibility component on those fixed B states. '
           'The difference between external full-library B and closed loop is consistent with state-distribution shift, but also includes B-vs-A '
           'episode composition, termination weighting and feedback. Decision logs lack the complete visual query keys needed for a full donor '
           'ablation on the **same live state**. Therefore these data do not identify a causal percentage of the total undershoot due to “following demos”. '
           'Lower live scores alone do not prove that mechanism.',
           'Changing cross-fitted error heads to the final deployed head is a third transfer, not a donor-count effect:',
           table(['Arm','frozen IR','final head on same OOF features IR'],[[r['arm'],fmt(r['pred_IR_lib']),fmt(r['final_head_same_oof_features_ir'])] for r in own if 'final_head_same_oof_features_ir' in r]),
           '## Why adaptation can still undershoot',
           'The script verifies every committed-ledger update. With gain η=.2, the cost error processed before the final proposal is '
           '`(c_m/η)(q0−q_last+Σ clipping_adjustment)`. Add the cost error of the final unprocessed segment for the exact whole-episode error. '
           'An upward change from q0 to q_last is therefore a finite-episode underspend term; resetting at every episode prevents carrying that deficit forward. '
           'This is an accounting explanation, not a proposed changed controller evaluated on test.',
           table(['Arm','q0','mean final q','IR initial→final','IR clipping','IR final segment','max update error'],[
               [r['arm'],fmt(r.get('dose_initial')),fmt(r.get('dose_end_mean')),fmt(r.get('adaptive_initial_to_final_ir')),
                fmt(r.get('adaptive_clip_ir')),fmt(r.get('adaptive_terminal_ir')),fmt(r.get('controller_update_max_error'),10)]
               for r in own if r['method']=='adaptive_error_hybrid']),
           'The remaining adaptive delivery problem is concentrated in Spatial: all four L10 adaptive arms are within ±.02, '
           'but only one of four Spatial arms is. The shorter Spatial episodes show larger startup-to-final-dose underspend terms; '
           'the exact terminal/clipping decomposition above is more informative than attributing this to a broken update rule.',
           '## Success versus realized IR',
           '![Realized-IR frontier](analysis/frontier.png)',
           'Every cell-size plot includes measured methods, same-batch knob-off, and the historical R8 pure-policy point. '
           'Hollow diamonds show the historical R10 knob-off configuration, with common y-axis limits within each suite. '
           'Historical points do not replace the same-batch controls in comparisons. Curves are piecewise linear through measured IR points, '
           'ordered by realized IR rather than target labels. Some dense cells have only one or two planned targets.',
           'Main matched comparisons use the midpoint of each pair of observed common-support intervals. Both curves need at least two measured points, '
           'except that each single-point disagreement arm is compared at its own measured IR against the interpolated comparator curve. '
           'For that exception the arm IR is recalculated in each bootstrap draw. No extrapolation is performed. '
           f"Intervals use {provenance['bootstrap_reps']} paired episode bootstrap draws: the same evaluation episode draw is reused across "
           'all targets, nested sizes and models in a suite. Realized IR and interpolation weights are recalculated in each draw; '
           'unsupported draws are excluded and counted. These are percentile intervals conditional on supported draws, not unconditional '
           'coverage guarantees; distance has particularly narrow overlaps and substantial excluded draws. These exploratory intervals describe finite-episode variation conditional on '
           'the observed rollout per arm; they do not capture repeated-policy-run variance or adjust for multiple comparisons.',
           table(['Cell','method − comparator','IR','ΔSR pp','95% interval pp','valid bootstrap draws'],[
               [r['cell'],r['method']+' − '+r['baseline'],fmt(r['matched_ir']),fmt(r['sr_delta_pp'],2,True),
                f"[{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]",r['bootstrap_valid']]
               for r in matched if r['status']=='matched realized IR']),
           table(['Method − comparator','cells','mean ΔSR pp','paired 95% interval pp','valid draws'],[
               [r['method']+' − '+r['baseline'],r['n_cells'],fmt(r['sr_delta_pp'],2,True),
                f"[{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]",r['bootstrap_valid']] for r in pooled]),
           ('**Final interpretation:** periodic−random is '+contrast('periodic','random')+'. Its interval excludes zero and the original +0.6 pp '
            'forecast has the correct positive direction. The matched-IR point estimate is larger, but the different weighting/support makes this '
            'a different estimand from the original pooled forecast, not a formal test of its numeric accuracy. Error_hybrid−random is '+contrast('error_hybrid','random')+
            ', while error_hybrid−periodic is '+contrast('error_hybrid','periodic')+'. Both hybrid intervals include zero. '
            'Distance is not shown to improve SR over periodic despite favorable portions of its low-IR curve; it fails budget calibration '
            'and has no knob-only overlap for groot Spatial-50. Adaptive error hybrid likewise does not establish superiority over periodic. '
            'Disagreement−random is '+contrast('disagreement','random')+' and disagreement−periodic is '+contrast('disagreement','periodic')+
            ': a promising unadjusted single-setting comparison against random, not evidence that it beats periodic. '
            'Each pooled contrast weights its supported cells equally, and different methods can have different supports.' if final else ''),
           f"Unsupported main comparisons: {sum(r['status']!='matched realized IR' for r in matched)}/{len(matched)}. "
           'See [matched_frontier.csv](analysis/matched_frontier.csv). In the final grid, unsupported comparisons reflect single-point designs, '
           'methods not planned for that cell, or nonoverlapping realized IR—not unfinished arms.',
           'Per-arm interpolation against completed random/periodic points, with explicit labeling when a same-batch off anchor is needed:',
           table(['Arm','comparator','type','ΔSR pp','95% interval pp','valid draws'],[
               [r['arm'],r['baseline'],r['status'],fmt(r.get('sr_delta_pp'),2,True),
                f"[{fmt(r.get('ci_low_pp'),2)}, {fmt(r.get('ci_high_pp'),2)}]",r.get('bootstrap_valid',0)]
               for r in secondary if 'sr_delta_pp' in r]),
           'Pooled efficiency over **same-batch off** controls is `Σ(SR−SR_off)/Σ(IR−IR_off)`. '
           'The shared bootstrap preserves reuse of the off episodes across targets. Random, periodic, distance and error_hybrid cover '
           'the same 17 planned cell-target points; adaptive and disagreement cover smaller designs. Realized spending still differs:',
           table(['Method','arms / cells','ΔSR sum','ΔIR sum','SR gain per IR','paired 95% interval'],[
               [r['method'],f"{r['n_arms']}/{r['n_cells']}",fmt(r['sr_gain_sum']),fmt(r['ir_spend_sum']),fmt(r['efficiency'],3),
                f"[{fmt(r['ci_low'],3)}, {fmt(r['ci_high'],3)}]"] for r in eff]),
           'Error_hybrid has the highest observed gain/spend ratio on the common 17-arm design (0.340), versus periodic 0.293, '
           'distance 0.215 and random 0.188. Paired differences of these ratios are reported below. They compare average returns at '
           'the actual budgets each method spent, and do not override the matched-realized-IR frontier:',
           table(['Method − control','common arms','efficiency difference','paired interval'],[
               [r['method']+' − '+r['control'],r['n_arms'],fmt(r['difference'],3,True),f"[{fmt(r['ci_low'],3)}, {fmt(r['ci_high'],3)}]"]
               for r in extras.get('efficiency_comparisons',[])]),
           'The error_hybrid−periodic efficiency difference is +0.046 SR per IR with a paired interval [−0.024, +0.119], '
           'so even the higher observed ratio does not establish a clear efficiency advantage over periodic. Both outperform random '
           'on this average-return metric. Matched-target error_hybrid−random below is positive while its matched-realized-IR interval '
           'includes zero because the estimand, supported cells and sampled budgets differ.',
           'For reconciliation only, the next table reproduces the coordinator’s **matched-target** effects from the raw accepted outcomes. '
           'It is a different estimand from the realized-IR frontier. Our intervals resample the same episode identities jointly across '
           'all targets and cells, avoiding the assumption that repeated uses of one initial condition are independent:',
           table(['Method − control','arm-episode comparisons','ΔSR pp','paired interval pp','mean ΔIR'],[
               [r['method']+' − '+r['control'],r['arm_episode_pairs'],fmt(r['sr_delta_pp'],2,True),
                f"[{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]",fmt(r['mean_ir_difference'],4,True)]
               for r in extras.get('target_crosscheck',[])]),
           '## Where the extra calls landed',
           'Episode progress is observed decision step divided by the final episode length minus one. It is retrospective measurement, '
           'never an input to layer 4. Successful/failed labels are eventual outcomes, so conditional call rates are associations, not recovery effects. '
           '[placement.csv](analysis/placement.csv) gives per-arm progress quintiles and eventual-outcome groups. '
           '[guard_neighborhood.csv](analysis/guard_neighborhood.csv) compares eligible looks within two looks of a guard, farther away, and immediately after a guard.',
           table(['Arm','extra calls near guard','cache-only run median','cache-only run p95'],[
               [r['arm'],fmt(r.get('near_guard_share_of_extra_calls'),3),fmt(r.get('cache_run_p50'),1),fmt(r.get('cache_run_p95'),1)] for r in own]),
           'Cache-only stretches count consecutive vision anchors with no policy call, including censored start/end runs. '
           'At nominal cadence one anchor spans ten controls (two five-control slots); a policy tail is not counted as cache-only.',
           'Pooled placement below weights actual looks, rather than treating a short and long episode as equal exposure. '
           'The outcome association is descriptive: eventually failed episodes can be longer and more stalled. It cannot identify '
           'whether a particular extra call caused recovery.',
           table(['Method','group','looks','guard / look','extra call / eligible look'],[
               [r['method'],r['group'],r['looks'],fmt(r['guard_per_look'],3),fmt(r['knob_non_guard'],3)]
               for r in extras.get('placement_pooled',[])]),
           'Placement interpretation: distance concentrates additional calls on eventually failed episodes (34.8% of eligible looks '
           'versus 15.2% on successes), while error_hybrid is much flatter (33.9% versus 32.5%). Adaptive makes fewer additional calls '
           'on failed episodes (44.9% versus 55.3%) while their guards consume much more of the budget (48.2% versus 14.0% of looks). '
           'This is consistent with guard-aware budget feedback, not evidence that reducing extra calls causes failure. Adaptive and disagreement '
           'increase their conditional call rates from early to late episode progress; the CSV retains every cell/arm so these pooled '
           'associations are not mistaken for task-dependent settings.',
           '## Hardware identity checks',
           table(['Model','status','local SR','remote SR','ΔSR pp [95%]','local IR / remote IR'],[
               [r['model'],r['status'],fmt(r.get('local_sr'),3),fmt(r.get('remote_sr'),3),
                f"{fmt(r.get('delta_sr_pp'),2,True)} [{fmt(r.get('ci_low_pp'),2)}, {fmt(r.get('ci_high_pp'),2)}]",
                f"{fmt(r.get('local_ir'))} / {fmt(r.get('remote_ir'))}"] for r in hw]),
           'The pi05 comparison is local R11 knob-off versus historical H100 R10 distance-corrected control; it mixes hardware and batch. '
           'The groot duplicate compares the explicitly requested local identity run with the H100 R11 off run. '
           'Both point differences are small and both paired intervals include zero: these two checks show no clear systematic hardware shift. '
           'They are not equivalence tests; the groot SR interval still spans several percentage points. Hardware cannot be assumed '
           'irrelevant to every other cell, and root allocation is not randomized replication.',
           'Current knob-off versus historical knob-off, paired on the same initial-condition identities:',
           table(['Cell','current off SR','R10 off SR','ΔSR pp [paired interval]','ΔIR'],[
               [r['cell'],fmt(r['current_sr'],3),fmt(r['reference_sr'],3),
                f"{fmt(r['delta_sr_pp'],2,True)} [{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]",fmt(r['delta_ir'],4,True)]
               for r in extras.get('baseline_comparisons',[]) if r['reference']=='R10 off']),
           'The largest negative historical-baseline change is groot L10-500: R10 off 0.906 versus current off 0.858 (−4.8 pp). '
           'The current run is an H100 completion in r11_knob_4, so this difference cannot be attributed to a 4090-versus-H100 switch. '
           'Batch/rollout variation or another historical-to-current difference remains unresolved; use current off for every R11 gain estimate. '
           'This revises the interim report’s historical near-pure/default-off classification.',
           '## Future-round hypothesis: a library-only mapping that aims to hit target',
           '1. Reserve whole B-pool episodes **outside** the deployed donor subset (or acquire additional B-only episodes at size 500). '
           'Use the complete deployment donor bank, exact deployed metric/action scale, and final pooled score predictor to generate '
           'calibration references. Preserve whole-episode cadence and mandatory guard logic. Never self-query a donor episode to claim an honest target match.',
           '2. Fit one cell-wide score CDF and an initial dose from these external B sequences only; validate on a separate B split. '
           'Use `k_target=clip((target/v−c_v−c_m*g)/(c_m*(1−g)),0,1)` as an accounting constraint, '
           'not as a task-specific rule. Check the guard floor and all-call ceiling. Correct head/CDF transfer by calibrating the actual final head.',
           '3. Retain a random floor and drive total cost with one global committed-ledger feedback state. '
           'For a future design, consider a persistent deficit budget across episodes or an analytically faster startup so repeated resets '
           'do not systematically lose the first part of each episode. Use bounded probabilities and anti-windup when guards force overspend. '
           'Tune any gain/floor/reset convention only on B sequences, freeze it, and measure a new untouched test round. '
           'Adaptation changes placement and feasibility: no library-only calculation guarantees exact cost on unknown short closed-loop trajectories.',
           '4. **Operational recommendation from this round:** prefer periodic when an accurate budget knob is required: it is within ±.02 '
           'on all 17 arms and improves success over random on common realized-IR support. Keep error_hybrid as the most promising '
           'state-dependent cost-efficiency candidate, but do not advertise its target label as delivered IR or claim it beats periodic '
           'at equal actual cost. Distance is unsuitable as a general budget knob in this calibration; adaptive improves delivery but '
           'does not establish a success advantage over periodic. Disagreement has only four single-point cells and needs a wider frozen '
           'curve before a general adoption claim. These recommendations use the unchanged measured methods.',
           'Same-batch off versus historical pure policy provides the default-off context:',
           table(['Cell','current off SR','R8 pure SR','off − pure pp [paired interval]','current off IR'],[
               [r['cell'],fmt(r['current_sr'],3),fmt(r['reference_sr'],3),
                f"{fmt(r['delta_sr_pp'],2,True)} [{fmt(r['ci_low_pp'],2)}, {fmt(r['ci_high_pp'],2)}]",fmt(r['current_ir'])]
               for r in extras.get('baseline_comparisons',[]) if r['reference']=='R8 pure']),
           '**Default off:** pi05 L10-500 is the clearest current near-pure case (0.906 versus 0.908 pure; IR 0.148). '
           'Pi05 L10-200 is a more cautious cost-first candidate (0.890 versus 0.908); extra spend does raise its observed SR, '
           'so it should remain optional rather than assumed useless. Do not blanket-disable groot L10-500: current off is 0.858 '
           'versus 0.898 pure, and periodic/error_hybrid reach 0.900/0.896 at IR about 0.239/0.231. '
           'Near-pure is a current cell-level judgment, not an equivalence claim or a task-indexed serving parameter.',
           'No item above is a refitted test result. No thresholds, fits, arm settings, serving code, run roots or running processes were changed.',
           '## Executed script and artifacts',
           '[Single entry script](analysis/run.py) produces [provenance.json](analysis/provenance.json), '
           '[inventory.csv](analysis/inventory.csv), [arms.csv](analysis/arms.csv), [method_summary.csv](analysis/method_summary.csv), '
           '[library_replays.csv](analysis/library_replays.csv), [matched_frontier.csv](analysis/matched_frontier.csv), '
           '[matched_arms.csv](analysis/matched_arms.csv), [pooled_frontier.csv](analysis/pooled_frontier.csv), '
           '[efficiency.csv](analysis/efficiency.csv), [hardware.json](analysis/hardware.json), placement tables and the two PNG figures. '
           'Final paired checks are in [efficiency_comparisons.csv](analysis/efficiency_comparisons.csv), '
           '[target_crosscheck.csv](analysis/target_crosscheck.csv), [baseline_comparisons.csv](analysis/baseline_comparisons.csv), '
           'and [placement_pooled.csv](analysis/placement_pooled.csv). '
           '[Numerical validation](analysis/numerical_validation.json) checks serving feature and distance arithmetic. '
           'The command at the top was executed. The `--final` flag requires exactly 94 unique completed test arms and fully reconciled '
           'decision logs before producing **ANALYSIS.md**. Without that flag the pipeline writes ANALYSIS_INTERIM.md. '
           'No serving process is started; the write fence confines outputs to this owner directory.']
    (OWNER/('ANALYSIS.md' if final else 'ANALYSIS_INTERIM.md')).write_text('\n\n'.join(lines)+'\n')


def replay_table_rows(own,rows):
    out=[]
    for r in own:
        by={x['variant']:x for x in rows if x['arm']==r['arm'] and x['cohort']=='test'}
        a=by.get('external_reduced_mean_of_5',{});b=by.get('external_full',{})
        d=b.get('mean_probability',np.nan)-a.get('mean_probability',np.nan)
        out.append([r['arm'],fmt(a.get('mean_probability')),fmt(b.get('mean_probability')),fmt(d,sign=True),
                    fmt(r.get('frozen_scoring_probability_all')),
                    f"{fmt(r.get('score_p50'),3)} / {fmt(b.get('score_p50'),3)}"])
    return out


def replay_ir_rows(own,rows):
    out=[]
    for r in own:
        by={x['variant']:x for x in rows if x['arm']==r['arm'] and x['cohort']=='test'}
        a=by.get('external_reduced_mean_of_5',{});b=by.get('external_full',{})
        d=b.get('fixed_setting_exogenous_ir',np.nan)-a.get('fixed_setting_exogenous_ir',np.nan)
        out.append([r['arm'],fmt(a.get('replay_guard')),fmt(b.get('replay_guard')),
                    fmt(a.get('fixed_setting_exogenous_ir')),fmt(b.get('fixed_setting_exogenous_ir')),fmt(d,sign=True)])
    return out


def main():
    global RERUN
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--final',action='store_true',help='Require all 94 unique arms and write ANALYSIS.md')
    args=parser.parse_args();started=datetime.now(timezone.utc).isoformat()
    if args.final:RERUN+=' --final'
    allowed=set(range(10,22))|set(range(54,66))
    if not set(os.sched_getaffinity(0))<=allowed:raise RuntimeError('Use the documented taskset command')
    validation=numerical_validation()
    arms,dev,inventory=discover();refs=references();print(f'SNAPSHOT test={len(arms)}/94 dev={len(dev)}/20',flush=True)
    planned=len({r['arm'] for r in inventory if r['cohort']=='test'})
    if args.final:
        assert len(arms)==planned==94, (len(arms),planned)
        assert Counter(a['method'] for a in arms)==dict(off=8,random=17,periodic=17,distance=17,error_hybrid=17,
                                                      adaptive_error_hybrid=8,disagreement=4,periodic_pgt1=4,random_tail2=2)
        assert next(a for a in arms if a['arm']=='r11_pi05_l10_200_random_ir25')['root']=='r11_knob_4'
    csvout('inventory.csv',inventory)
    refcache={};diags=[];placement=[];neighborhood=[]
    for i,a in enumerate(arms+dev):
        parse_decisions(a)
        if a['cell'] not in refcache:refcache[a['cell']]=ref_data(a['cell'])
        d,p,n=diagnostics(a,refcache[a['cell']]);diags.append(d);placement.extend(p);neighborhood.extend(n)
        print(f"LOG {i+1}/{len(arms)+len(dev)} {a['arm']} IR={a['ir']:.4f} SR={a['sr']:.3f} reconciled={a['log_audit']['usable']}",flush=True)
    agg=aggregate(diags,inventory)
    for name,rows in [('arms.csv',diags),('method_summary.csv',agg),('placement.csv',placement),('guard_neighborhood.csv',neighborhood)]:csvout(name,rows)
    replays={};metas=[];replayrows=[]
    for cell in sorted({a['cell'] for a in arms+dev if a['method'] in OWN}):
        print('REPLAY',cell,flush=True);r,m=replay_cell(cell,refcache[cell]);replay_accounting(r,cell);replays[cell]=(r,m);metas.append(m)
        for a in arms+dev:
            if a['cell']==cell and a['method'] in OWN:replayrows.extend(replay_summary(a,refcache[cell],r,m))
        print('REPLAY DONE',cell,'external episodes',m['external_episodes'],flush=True)
    csvout('library_replays.csv',replayrows)
    boots=bootstrap_arms(arms,args.bootstrap)
    matched,secondary,pooled,eff=frontier(arms,boots,args.bootstrap)
    hw=hardware(arms,refs,args.bootstrap)
    extras=final_comparisons(arms,refs,boots,placement,args.bootstrap) if args.final else {}
    if args.final:
        expected_within=dict(periodic=17,random=16,adaptive_error_hybrid=5,error_hybrid=6,disagreement=1,distance=2)
        assert all(r['within_02']==expected_within[r['method']] for r in agg if r['cohort']=='test' and r['method'] in expected_within)
    for name,rows in [('matched_frontier.csv',matched),('matched_arms.csv',secondary),('pooled_frontier.csv',pooled),('efficiency.csv',eff)]:csvout(name,rows)
    dump(HERE/'hardware.json',hw)
    if args.final:assert all(a['log_audit']['usable'] for a in arms+dev), 'Incomplete logs cannot support a final analysis'
    print('PLOTS',flush=True);plots(arms,refs,diags,replays,args.final)
    provenance=dict(snapshot_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),entry_sha256=sha(__file__),
        bootstrap_reps=args.bootstrap,command=RERUN,affinity=sorted(os.sched_getaffinity(0)),
        roots=list(ROOTS),test_complete=len(arms),test_planned=planned,final=args.final,
        measurement_only=True,refits=0,network=False,gpu=False,
        numerical_validation=validation,
        calibration_sources=[dict(path=str(p),sha256=sha(p)) for cell in sorted(refcache) for p in
                             (KNOB/'calibration'/f'{cell}.json',OWNER/'data'/cell/'signals.npz',
                              OWNER/'data'/cell/'pack.npz',OWNER/'data'/cell/'predictor.json')],
        arms=[dict(arm=a['arm'],root=a['root'],cohort=a['cohort'],summary_sha256=a['summary_sha256'],journal_sha256=a['journal_sha256'],
                   log_audit=a.get('log_audit'),input_stamps=a['input_stamps']) for a in arms+dev+refs],
        replay_metadata=metas,inventory=inventory,
        caveats=['All 94 unique test arms complete; migrated copies deduplicated.' if args.final else 'Unbalanced interim completion; no missing-arm imputation.',
                 'One rollout per episode/arm; paired bootstrap is not stochastic-policy replication.',
                 'Library replays are exogenous score diagnostics, not closed-loop SR or retuned test outcomes.'])
    dump(HERE/'provenance.json',provenance)
    report(arms,dev,inventory,diags,agg,replayrows,metas,matched,secondary,pooled,eff,hw,refs,provenance,extras)
    print('DONE',OWNER/('ANALYSIS.md' if args.final else 'ANALYSIS_INTERIM.md'),flush=True)


if __name__=='__main__':main()
