"""Planted/null Monte Carlo, exact effects, sparse bins and raw-log/CLI checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from exp.offline_search.rounds.r04.k5_rand import estimate as E
from exp.offline_search.rounds.r04.k5_rand.overlay import assignment,RandomizedLandmark
BASE=Path(__file__).resolve().parent
PREFIX=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]


def sample(seed,tau=(.12,-3,-1),n=500):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        task,init=divmod(i,50)
        exposed=rng.random()<.8
        context=dict(progress=('<0.5','>=0.5')[rng.integers(2)],gripper=('open','closed')[rng.integers(2)],
                     confidence=('>-0.3','<=-0.3')[rng.integers(2)],stall_age=('before','0-9','>=10')[rng.integers(3)])
        difficulty=rng.uniform(-.15,.15)
        for rep in (1,2):
            a=assignment(20260927,task,init,rep); tr=a['assigned_treatment']=='CALL'
            y=int(rng.random()<.55+difficulty+tau[0]*tr*exposed)
            requests=int(rng.poisson(50+tau[1]*tr*exposed)); misses=min(requests,int(rng.poisson(12+tau[2]*tr*exposed)))
            rows.append(dict(a,arm=f'rep{rep}',Y=y,N=requests,M=misses,exposed=bool(exposed),
                             context=context if exposed else None,actual_treatment=a['assigned_treatment'] if exposed else None))
    return rows


def monte_carlo(tau):
    cover=np.zeros(5); effects=[]; widths=[]
    truth=np.array([*.8*np.array(tau),*[.848*.8*tau[2]+(.152-rho)*.8*tau[1] for rho in E.RHOS.values()]])
    for seed in range(200):
        rows=sample(10000+seed,tau)
        cid,w=E.cluster_draws(rows,499,seed)
        out,_=E.effect(rows,np.ones(len(rows),bool),cid,w)
        point=np.array(list(out['delta'].values())); ci=np.array(list(out['ci95'].values()))
        cover+=(ci[:,0]<=truth)&(ci[:,1]>=truth); effects.append(point); widths.append(ci[:,1]-ci[:,0])
    coverage=cover/200
    assert np.all((coverage>=.88)&(coverage<=.995)),coverage
    means=np.mean(effects,axis=0)
    assert np.all(np.abs(means-truth)<np.array([.01,.12,.08,.08,.08])),(means,truth)
    return dict(experiments=200,clusters=500,replicates=2,bootstrap_draws=499,truth=dict(zip(E.METRICS,truth)),
                mean_estimate=dict(zip(E.METRICS,means)),coverage=dict(zip(E.METRICS,coverage)),mean_interval_width=dict(zip(E.METRICS,np.mean(widths,axis=0))))


def raw_fixture(root):
    arms=[]
    for rep in (1,2):
        arm=f'rep{rep}'; arms.append(dict(arm=arm)); d=root/'runs'/arm
        (d/'client').mkdir(parents=True); (d/'server_1').mkdir()
        journals=[]; log=[]
        for init in range(8):
            uid=f'{arm}:eval:0:{init}'; ov=RandomizedLandmark(20260927,0,init,rep)
            previous=None; misses=0; n=2 if init==0 else 8
            for step in range(n):
                ex=dict(os_force_miss=int(step%2==0),os_reason=(step//2%4+1) if step%2==0 else 0,top1_prog=.25,stuck_n=2,noprog_n=0)
                hit,record=ov.apply(step,not ex['os_force_miss'],-.2,ex,previous)
                head=np.full((5,7),1. if hit else 2.)
                row=dict(record,ev='dec',uid=uid,attempt=2,step=step,hit=hit,extras=ex,conf=-.2,src='cache' if hit else 'policy',served_head=head.tolist(),exec_ok=True if hit else None,ok=True)
                log.append(row); previous=head; misses+=not hit
            # An identical duplicate and a rejected retry must not add requests.
            log.append(dict(log[-1])); log.append(dict(log[-1],attempt=1,hit=not hit))
            log.append(dict(ev='episode',reason='episode_end',uid=uid,attempt=2,n_decisions=n,n_exec=n,n_miss=misses,success=bool(init%2),exposed=ov.exposed))
            journals.extend([dict(task_uid=uid,accepted=True,status='done',success=False,error='retry',attempt=1),dict(task_uid=uid,accepted=True,status='done',success=bool(init%2),attempt=2)])
        (d/'client/journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in journals))
        (d/'server_1/decisions_test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in log))
        (d/'summary.json').write_text(json.dumps(dict(arm=arm,complete=8)))
    (root/'arms.json').write_text(json.dumps(arms))
    (root/'kpi.json').write_text(json.dumps(dict(args=dict(run_roots=[str(root)]),arms=arms)))
    return root


def main():
    result={}
    rows=sample(7)
    out=E.estimate(rows,499,7)
    assert len(out['contexts'])==48
    for c in out['contexts']:
        if c['supported']:
            parent=out['exposed_by_landmark'][str(c['landmark_class'])]
            for m in E.METRICS:
                assert np.isclose(c['shrunk_delta'][m],c['shrink_weight']*c['delta'][m]+(1-c['shrink_weight'])*parent['delta'][m])
    try: E.estimate(rows[:-1],100)
    except ValueError as e: assert 'pair' in str(e)
    else: raise AssertionError('incomplete pair accepted')
    exact=sample(1,n=100)
    for r in exact:
        t=r['assigned_treatment']=='CALL'; r.update(Y=int(t),N=40+4*t,M=10+2*t,exposed=True,actual_treatment=r['assigned_treatment'])
        r['context']=dict(progress='<0.5',gripper='open',confidence='>-0.3',stall_age='before')
    exact_out=E.estimate(exact,499,1)
    assert [exact_out['ITT']['delta'][m] for m in ['Y','N','M']]==[1,4,2]
    assert all(c['shrunk_delta'] is None for c in exact_out['contexts'] if not c['n'])
    baseline=[dict(r,Y=0,N=40,M=10) for r in exact if r['replicate']==1]
    comparison=E.estimate(exact,199,1,{'baseline':baseline})['baseline_comparisons']
    assert comparison[0]['delta']['Y']==1 and comparison[1]['delta']['Y']==0
    result['exact_effect']={m:exact_out['ITT']['delta'][m] for m in E.METRICS}
    result['planted']=monte_carlo((.12,-3,-1)); result['null']=monte_carlo((0,0,0))
    root=raw_fixture(Path(tempfile.mkdtemp(prefix='k10_estimator_')))
    rr=[]
    for arm in ('rep1','rep2'):
        rows,audit=E.load_arm(root,arm); assert len(rows)==8 and audit['duplicate_decisions']==8; rr+=rows
    expected=E.estimate(rr,199,17)
    for kind,extra in [('raw',['--run-root',str(root)]),('collect',['--input',str(root/'runs/rep1/summary.json')]),('kpi',['--input',str(root/'kpi.json')])]:
        output=root/f'{kind}.json'; export=root/f'{kind}_episodes.json'
        subprocess.run(PREFIX+[str(Path('/home/weiland/projects/openpi/exp/offline_search/rounds/r04/k5_rand/estimate.py')),*extra,'--arms','rep1','rep2','--boot','199','--seed','17','--out',str(output),'--episodes-out',str(export)],check=True)
        got=json.loads(output.read_text()); assert got['ITT']==expected['ITT']
    subprocess.run(PREFIX+[str(Path('/home/weiland/projects/openpi/exp/offline_search/rounds/r04/k5_rand/estimate.py')),'--input',str(root/'raw_episodes.json'),'--arms','rep1','rep2','--boot','199','--seed','17','--out',str(root/'exported.json')],check=True)
    assert json.loads((root/'exported.json').read_text())['ITT']==expected['ITT']
    # Corrupted duplicate, missing accepted rows and aggregate-only input all fail clearly.
    logfile=root/'runs/rep1/server_1/decisions_test.jsonl'
    original=logfile.read_text(); parsed=[json.loads(l) for l in original.splitlines()]
    bad=dict(parsed[0],hit=not parsed[0]['hit']); logfile.write_text(original+json.dumps(bad)+'\n')
    try: E.load_arm(root,'rep1')
    except ValueError as e: assert 'duplicate' in str(e)
    else: raise AssertionError('conflict accepted')
    logfile.write_text('\n'.join(json.dumps(r) for r in parsed if r.get('step')!=0)+'\n')
    try: E.load_arm(root,'rep1')
    except ValueError as e: assert 'non-contiguous' in str(e)
    else: raise AssertionError('gap accepted')
    logfile.write_text(original)
    result['raw_cli']=dict(episodes=16,duplicates_ignored=16,rejected_attempts_ignored=True,input_modes=['raw','collect','kpi','episode export'],fixture_root=str(root),conflict_rejected=True,gap_rejected=True)
    result['PASS']=True
    (BASE/'results/installed/estimator_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
