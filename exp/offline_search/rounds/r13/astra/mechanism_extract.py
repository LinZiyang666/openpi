"""B-only, CPU-only inputs for the descriptive call-placement study.

Run from the repository root under taskset -c 10-21,54-65, with
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1, using .venv/bin/python.
Only mechanism_* and MECHANISM.md in this directory may be written.
No rollout, fitting of controllers, test-A results, or network access.
"""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
CELLS = [f'{m}_{s}' for m in ('groot', 'pi05') for s in ('l10_50', 'l10_200', 'spatial_50')]
ROOTS = tuple('r13_pilot_' + c for c in CELLS) + ('r13_devrec_50', 'r13_devrec_pi05_spatial_50')
INPUTS = {}

def fence():
    def check(path, write=False):
        if isinstance(path, int) or path is None: return
        p = Path(os.fsdecode(path)).resolve()
        if str(p) == '/dev/null': return
        if write:
            rel = p.relative_to(HERE) if HERE in p.parents else None
            if rel is None or not (rel.parts[0].startswith('mechanism_') or rel.parts[0] == 'MECHANISM.md'):
                raise PermissionError(f'write outside new mechanism files: {p}')
        elif RUNS in p.parents and p.relative_to(RUNS).parts[0] not in ROOTS:
            raise PermissionError(f'non-approved rollout input: {p}')
        elif 'offline_search_store' in p.parts and 'library' not in p.parts:
            raise PermissionError(f'non-library store input: {p}')
    def audit(event, args):
        if event == 'open': check(args[0], bool(args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod'): check(args[0], True)
        elif event in ('os.rename', 'os.link', 'os.symlink'): check(args[0], True); check(args[1], True)
        elif event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system'):
            raise PermissionError(f'offline analysis refuses {event}')
    sys.addaudithook(audit)

fence()
import numpy as np
import pandas as pd

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def record(p):
    p = Path(p)
    INPUTS[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    return p

def read(p): return json.loads(record(p).read_text())
def rows(p):
    with record(p).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def dump(p,x): Path(p).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def frozen():
    results=[]
    for line in (HERE/'MANIFEST.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        p=HERE/name.lstrip('*')
        actual=sha(p)
        results.append(dict(path=str(p), expected=expected, actual=actual, ok=actual==expected))
    assert all(x['ok'] for x in results), [x for x in results if not x['ok']]
    return results

def extract_arm(root, a):
    run=root/'runs'/a['arm']; pilot=root.name.startswith('r13_pilot_')
    method=a.get('r13_method',a.get('r11_method','off'))
    cell=f"{a['model']}_{a['suite_short']}_{a['r10_size']}"
    base_meta=dict(root=root.name, arm=a['arm'], cell=cell, model=a['model'], method=method,
                   target=a.get('target_ir'), rep=int(a.get('r13_grid',{}).get('replicate',0)),
                   source='pilot' if pilot else 'devrec')
    if not (run/'summary.json').exists(): return None, dict(base_meta,status='no_complete_summary')
    summary=read(run/'summary.json'); contract=read(run/'pool_contract.json')
    assert summary['init_pool']==contract['init_pool']=='B' and a['dev'] and a['init_pool']=='B'
    accepted={(j['task_uid'],j['attempt']):j for j in rows(run/'client/journal.jsonl')
              if j.get('accepted') and j.get('status') in ('done','failed') and not j.get('error')}
    assert len(accepted)==summary['complete']==100
    assert all(j['init_pool']=='B' for j in accepted.values())
    decs=collections.defaultdict(list); startups=[]; rejected=0
    for p in sorted(run.glob('server_*/decisions*.jsonl')):
        for r in rows(p):
            if r['ev']=='startup': startups.append(r)
            if r['ev']!='dec': continue
            if (r['uid'],r['attempt']) not in accepted: rejected+=1;continue
            decs[(r['uid'],r['attempt'])].append(r)
    entries={}; shard_bytes=0; options=[]
    for p in sorted(run.glob('server_*/looks/manifest*.json')):
        m=read(p);assert m['init_pool']==m['contract']['init_pool']=='B' and m['dev']
        options.append(m['options'])
        for e in m['shards']:
            k=(e['uid'],e['attempt'],e['step'])
            if k[:2] not in accepted: continue
            assert k not in entries
            entries[k]=(p.parent/e['file'],e)
    need_labels=any(o['shadow'] for o in options)
    args=a['plugin_args']; fit=Path(args[args.index('--os-fit-artifact')+1])
    assert sha(record(fit))==contract['fit_sha256']
    # Frozen model is read only for sigma, and old recordings' missing geometry.
    obj=pickle.loads(fit.read_bytes())['method']; base=obj.inner.base
    sig=np.asarray(base.sig).copy()
    data=[]; max_cache_error=0.; distance_checks=[]; n_rebuilt=0; valid_labels=0
    for (uid,attempt),rr in sorted(decs.items()):
        rr.sort(key=lambda r:r['step']);N=len(rr)
        assert [r['step'] for r in rr]==list(range(N))
        task,init=map(int,uid.split(':')[-2:]);assert (task,init)==(rr[0]['task_id'],rr[0]['init'])
        ep=f"{root.name}/{a['arm']}/{task}:{init}"
        last_step=last_look=last_knob_step=last_knob_look=last_guard_look=None
        last_kind='startup'; lookidx=0; previous_distance=previous_call_distance=np.nan
        V=M=0
        for r in rr:
            step=r['step'];call=not r['hit'];ex=r.get('extras',{})
            if not r['vision']:
                assert not call
                continue
            pref='os_r13_' if method.startswith('distance_') else 'os_r11_'
            guard=bool(ex.get(pref+'guard',call if method=='off' else False))
            knob=bool(ex.get(pref+'knob_call',False))
            assert call==(guard or knob) and not (guard and knob)
            p,ent=entries.pop((uid,attempt,step));shard_bytes+=p.stat().st_size
            with np.load(p,allow_pickle=False) as z:
                assert int(z['step'])==step and int(z['task_id'])==task and int(z['init'])==init
                assert bool(z['hit'])==r['hit']
                if 'raw_distance' in z:
                    d=float(z['raw_distance'])
                else:
                    q=SimpleNamespace(step=step,task_id=task,key_v0=z['key_v0'],key_v1=z['key_v1'],rs=z['rs'],
                                      prev_hit=bool(rr[step-1]['hit']) if step else None,
                                      prev_a_exec=np.zeros((10,32),np.float32),episode=uid)
                    d=float(np.min(base._dist(q)[7]));n_rebuilt+=1
                logged=ex.get('os_r13_raw_distance',ex.get('os_r11_score') if method=='distance' else None)
                if logged is not None and np.isfinite(d): distance_checks.append(abs(d-logged))
                risk5=risk10=grip5=grip10=np.nan
                if need_labels:
                    policy=z['action'] if call else z['shadow_policy']
                    if 'cache_proposal' in z: cache=z['cache_proposal']
                    else:
                        donor=np.asarray(z['donor_rows'],int);pos=np.searchsorted(obj.row_subset,donor)
                        assert np.array_equal(obj.row_subset[pos],donor)
                        base._anchor=None
                        cache=base.os_synth(q,pos,z['donor_weights'])
                    if not call:
                        err=float(np.max(np.abs(cache[:10,:7]-z['action'][:10,:7])))
                        max_cache_error=max(max_cache_error,err)
                    risk5=float(np.mean(((policy[:5,:6]-cache[:5,:6])/sig[:6])**2))
                    risk10=float(np.mean(((policy[:10,:6]-cache[:10,:6])/sig[:6])**2))
                    grip5=float(np.mean((policy[:5,6]>=0)!=(cache[:5,6]>=0)))
                    grip10=float(np.mean((policy[:10,6]>=0)!=(cache[:10,6]>=0)))
                    valid_labels+=1
            gap=lookidx-last_look if last_look is not None else np.nan
            row=dict(base_meta,ep=ep,task=task,init=init,attempt=attempt,N=N,step=step,controls=5*step,
                fraction=step/N,progress=ex.get('top1_prog',np.nan),look=lookidx,call=call,guard=guard,knob=knob,
                distance=d,gap=gap,gap_controls=5*(step-last_step) if last_step is not None else np.nan,
                gap_knob=lookidx-last_knob_look if last_knob_look is not None else np.nan,
                gap_knob_controls=5*(step-last_knob_step) if last_knob_step is not None else np.nan,
                gap_guard=lookidx-last_guard_look if last_guard_look is not None else np.nan,
                last_call_kind=last_kind,startup=last_look is None,run=lookidx-last_look-1 if last_look is not None else lookidx,
                V=V,M=M,theta=ex.get('os_r13_theta',np.nan),u=ex.get('os_r13_u',np.nan),
                setting=ex.get('os_r11_setting',np.nan),logged_cap=ex.get('os_r11_cap',np.nan),
                reason=ex.get('os_reason',np.nan),risk5=risk5,risk10=risk10,grip5=grip5,grip10=grip10,
                distance_delta=d-previous_distance,delta_from_last_call=d-previous_call_distance)
            data.append(row)
            if last_look is not None and gap==1: data[-1]['first_after_call']=True
            else: data[-1]['first_after_call']=False
            previous_distance=d
            if call:
                last_step=step;last_look=lookidx;last_kind='guard' if guard else 'knob';previous_call_distance=d
            if knob: last_knob_step=step;last_knob_look=lookidx
            if guard:last_guard_look=lookidx
            lookidx+=1;V+=1;M+=call
        assert M==sum(not r['hit'] for r in rr)
    assert not entries
    df=pd.DataFrame(data)
    led=summary['cost_ledger']
    assert [sum(len(v) for v in decs.values()),len(df),int(df.call.sum())]==[led[k] for k in ('decisions','vision_decisions','misses')]
    assert max_cache_error<2e-4
    assert not distance_checks or max(distance_checks)<1e-7
    meta=dict(base_meta,status='complete',episodes=len(accepted),N=led['decisions'],looks=len(df),calls=int(df.call.sum()),
              guards=int(df.guard.sum()),knobs=int(df.knob.sum()),rejected_decisions=rejected,labels=valid_labels,
              recorded_options=options,sigma=sig,max_cache_hit_error=max_cache_error,
              raw_distance_rebuilt=n_rebuilt,max_raw_vs_logged_error=max(distance_checks) if distance_checks else None,
              recorded_shard_bytes=shard_bytes,host=sorted(set(s['host'] for s in startups)),
              shard_hash_policy='Manifest SHA recorded; shard identity/fields checked; individual compressed shard SHA not rehashed here (pilot GATES audit already did so).')
    print(root.name,a['arm'],len(df),'looks',valid_labels,'labels',flush=True)
    return df,meta

def main():
    before=frozen();all_data=[];inventory=[]
    for name in ROOTS:
        root=RUNS/name
        for a in read(root/'arms.json'):
            df,meta=extract_arm(root,a);inventory.append(meta)
            if df is not None:all_data.append(df)
    df=pd.concat(all_data,ignore_index=True)
    df.to_pickle(HERE/'mechanism_looks.pkl')
    after=frozen();assert before==after
    dump(HERE/'mechanism_inputs.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          inventory=inventory,inputs=INPUTS,frozen_before=before,frozen_after=after,script_sha=sha(__file__),
          rows=len(df),boundary='Only the eight named B roots; no test-A result access.'))

if __name__=='__main__':main()
