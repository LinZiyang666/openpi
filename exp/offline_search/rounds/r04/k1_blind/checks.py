"""CPU verification; store labels are evaluator-only, never passed to methods."""
from __future__ import annotations
import argparse, copy, csv, itertools, json, pathlib, pickle, time
from types import SimpleNamespace as NS
import numpy as np
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r02.g1_awm.awm import AWM
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM, BlindAWM3, BlindResult, LookReason
from exp.offline_search.rounds.r04.k1_blind.control_step import ControlStepLibrary
from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge, MemoResetMixedJudge
from exp.offline_search.rounds.r04.ideation_A.measure_blind import get_method

ROOT = pathlib.Path('/home/weiland/trace_runs/offline_search_store')
OUT = pathlib.Path(__file__).resolve().parent / 'results'
IDEA = OUT.parent.parent / 'ideation_A'


def ev_for(qc, i):
    e = qc.episodes[int(qc.ep[i])]
    ev = api.EpisodeView(e['uid'], e['task'], e['task_id'], e['init'], int(qc.ep[i]), 0)
    return ev, e


def view(qc, A, i):
    ev, e = ev_for(qc, i)
    return api.QueryView(A, int(i), e['start'], int(qc.step[i]), e['task_id'], ev)


def blind_view(q, age=0, **updates):
    vals = {n:getattr(q,n) for n in ('step','task_id','episode','rs','raw_state','prev_hit','prev_a_exec',
                                   'hist_a_exec','hist_hit','hist_rs')}
    vals.update(hist_has_vision=np.ones(q.step, bool), blind_age=age)
    vals.update(updates)
    return NS(**vals)


def equal(a, b, extras=True, confidence=True):
    for field in ('topk','scores','action'):
        assert np.array_equal(getattr(a,field),getattr(b,field)), field
    assert a.library == b.library
    if confidence:
        assert a.confidence == b.confidence, (a.confidence,b.confidence)
    if extras:
        assert a.extras == b.extras, (a.extras,b.extras)


def adapt(base, lib, **kw):
    method = BlindAWM(**kw)
    config = {k:v for k,v in vars(method).items() if k in ('serving','budget','gates','residual_threshold','name')}
    method.__dict__.update(base.__dict__)
    method.__dict__.update(config)
    method._fit_blind(lib)
    method._anchor = None
    return method


def parity(key, scale):
    old = get_method(key, scale)
    lib = store.LibraryView(ROOT, key, old.cand_name)
    method = adapt(old,lib)
    variants = list(itertools.product(('phase_particles','kernel_clock','top1_clock','anchor_tail'),
                                     range(5),('all','budget_only'),(.25,.5,1.)))
    count = 0
    for arm in ('inf','cache'):
        qc = store.QueryCell(ROOT, f'{key}_{arm}')
        arrays = api.QueryArrays(qc)
        indexes = sorted(set(np.linspace(0,qc.N-1,12,dtype=int).tolist()+[qc.episodes[j]['start'] for j in (0,250,499)]))
        for i in indexes:
            q = view(qc,arrays,i)
            expected = old.query(q)
            for serving,budget,gates,residual in variants:
                method.serving,method.budget,method.gates,method.residual_threshold = serving,budget,gates,residual
                equal(expected,method.query(q))
                assert len(method._anchor['rows']) == 16
                count += 1
    result = dict(key=key,scale=scale,variants=len(variants),comparisons=count,bit_equal=True)
    (OUT/f'parity_{key}_{scale}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def replay(key, scale):
    base=get_method(key,scale)
    lib=store.LibraryView(ROOT,key,base.cand_name)
    m=adapt(base,lib,budget=2,gates='budget_only')
    expected={}
    for p in IDEA.glob(f'metrics_{key}_{scale}.csv'):
        for row in csv.DictReader(p.open()):
            if row['split']=='all' and row['method'] in ('phase_abs','kernel_clock') and int(row['h'])<=2:
                expected[(row['cell'],row['method'],int(row['h']))]=float(row['err_mean'])
    assert expected, (key,scale)
    reports=[]
    for arm in ('inf','cache'):
        cell=f'{key}_{arm}'
        qc=store.QueryCell(ROOT,cell)
        data=np.load(IDEA/f'anchors_{cell}_{scale}.npz')
        rows,weights,actions=data['rows'],data['weights'],data['action']
        sigma=store.action_sigma(str(ROOT),key)
        totals={(serv,h):[0.,0] for serv in ('phase_particles','kernel_clock') for h in (1,2)}
        for e in qc.episodes:
            ep=api.EpisodeView(e['uid'],e['task'],e['task_id'],e['init'],0,0)
            for i in range(e['start'],e['end']-1):
                anchor=NS(step=i-e['start'],task_id=e['task_id'],episode=ep,rs=qc.rs[i])
                action=np.zeros((m.H,32),np.float32); action[:,:7]=actions[i]
                for serv in ('phase_particles','kernel_clock'):
                    m.serving=serv
                    m._remember_anchor(anchor,rows[i],weights[i],action)
                    for h in (1,2):
                        if i+h>=e['end']:break
                        j=i+h
                        q=NS(step=j-e['start'],task_id=e['task_id'],episode=ep,rs=qc.rs[j],prev_hit=True,
                             blind_age=h-1,hist_rs=qc.rs[e['start']:j])
                        result=m.blind_step(q)
                        assert isinstance(result,BlindResult), result
                        err=float(np.sqrt(np.mean(((result.action[:5,:7]-qc.a_inf[j,:5,:7])/sigma)**2)))
                        totals[serv,h][0]+=err;totals[serv,h][1]+=1
                        if serv=='phase_particles':
                            assert np.array_equal(lib.episode[result.rows],lib.episode[rows[i]])
                # Inf replay uses a fixed hypothetical HIT between anchor and target, as ideation A.
        for (serv,h),(total,n) in totals.items():
            ref=expected[cell,'phase_abs' if serv=='phase_particles' else serv,h]
            measured=total/n
            assert abs(measured-ref)<2e-6,(cell,scale,serv,h,measured,ref)
            reports.append(dict(cell=cell,scale=scale,serving=serv,h=h,n=n,error=measured,
                                reference=ref,absolute_delta=abs(measured-ref)))
    (OUT/f'replay_{key}_{scale}.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','replay']);p.add_argument('--key',required=True)
    p.add_argument('--scale',type=int,required=True);a=p.parse_args()
    globals()[a.mode](a.key,a.scale)

if __name__=='__main__':main()
