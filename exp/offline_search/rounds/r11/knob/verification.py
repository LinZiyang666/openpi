"""CPU-only integration, equivalence, ledger and deployment checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import numpy as np
from exp.offline_search.rounds.r10.data import sha, write_json
from .build import HERE, ROOTS, SPEC, R10, run_cpu
from .controller import COSTS
from .recipe import REASONS, uniform


def audit_trace(decs,row):
    grouped={}
    for d in decs: grouped.setdefault(d['uid'],[]).append(d)
    knob_calls=guards=tails=periodic_checks=fallbacks=0
    for seq in grouped.values():
        seq.sort(key=lambda d:d['step'])
        for i,d in enumerate(seq):
            assert d['step']==i and bool(d['vision'])==(d['src'] in ('cache','policy'))
            x=d.get('extras',{})
            if d['vision'] and row.get('r11_method','off')!='off':
                past=seq[:i]
                N=len(past);V=sum(t['vision'] for t in past);M=sum(t['vision'] and not t['hit'] for t in past)
                assert (x['os_r11_N'],x['os_r11_V'],x['os_r11_M'])==(N,V,M)
                g=bool(x['os_r11_guard']);k=bool(x['os_r11_knob_call'])
                assert not (g and k)
                assert (not d['hit'])==bool(g or k)
                if g: assert x['os_reason']==4 and x['os_flags']==8
                if k: assert x['os_reason']==REASONS[row['r11_method']]
                guards+=g;knob_calls+=k
                method=row['r11_method']
                if method in ('distance','disagreement','error_hybrid','adaptive_error_hybrid'):
                    assert {'r11_knob','r11_guard','r11_p','r11_score','r11_dose'} <= x.keys()
                    assert x['r11_guard']==x['os_r11_guard']
                    assert x['r11_p']==x['os_r11_p'] and x['r11_dose']==x['os_r11_setting']
                if method.startswith('periodic'):
                    assert x['os_r11_run']<=x['os_r11_cap']
                    if method=='periodic' and not g:
                        assert k==(x['os_r11_run']>=x['os_r11_cap'])
                    periodic_checks+=1
                if method=='random':
                    u=uniform('R11-random-v1',0,d['task_id'],d['init'],i,'knob-anchor')
                    assert u==x['os_r11_coin']
                    assert k==((not g) and u<x['os_r11_setting'])
            if d['vision'] and not d['hit']:
                if i+1<len(seq):
                    if seq[i+1].get('previous_executed_steps',5) != 5:
                        assert seq[i+1]['vision'] and seq[i+1]['look_reason']==6
                        fallbacks+=1
                    else:
                        assert seq[i+1]['src']=='policy_tail' and not seq[i+1]['vision'] and seq[i+1]['hit']
                        tails+=1
                if i+2<len(seq) and seq[i+1].get('previous_executed_steps',5)==5:
                    assert seq[i+2]['vision'] and seq[i+2]['src'] in ('cache','policy')
    N=len(decs);V=sum(d['vision'] for d in decs);M=sum(d['vision'] and not d['hit'] for d in decs)
    a,b=COSTS[row['model']]
    return dict(N=N,V=V,M=M,owner_IR=(a*V+b*M)/N,knob_calls=knob_calls,guard_anchors=guards,
                checked_policy_tails=tails,lifecycle_fallbacks=fallbacks,periodic_checks=periodic_checks,ledger_pass=True,lifecycle_pass=True)


def trace(rowfile,label,size):
    from .replay import replay
    row=json.loads(Path(rowfile).read_text())
    output=HERE/'replays'/label
    result=replay(row,label,size)
    with gzip.open(output/'trace.pkl.gz','wb') as f: pickle.dump(result,f,protocol=4)


def isolated(row,label):
    file=HERE/'replays'/label/'row.json';write_json(file,row)
    run_cpu(['-m','exp.offline_search.rounds.r11.knob.verification','trace','--row',str(file),
             '--label',label,'--size',str(row['r10_size'])],HERE/'validation'/f'{label}.log')
    with gzip.open(file.parent/'trace.pkl.gz','rb') as f: return pickle.load(f)


def all_tests(workers):
    rows=[(root,row) for root in ROOTS for row in json.loads((root/'arms.json').read_text())]
    def one(item):
        root,row=item
        t,report=isolated(row,row['arm'])
        report=dict(report,arm=row['arm'],root=str(root),fit_sha256=sha(row['plugin_args'][-1]))
        if row['r11_method']=='off' and not row.get('dev'):
            old=dict(row,method='exp.offline_search.rounds.r10.recipe.recipe:R10Recipe')
            old['kwargs']={k:v for k,v in row['kwargs'].items() if k not in ('method','target')}
            old['plugin_args']=list(row['plugin_args'])
            old['plugin_args'][-1]=str(R10/'recipe/artifacts'/f'r10_recipe_{row["model"]}_{row["suite_short"]}_{row["r10_size"]}.pkl')
            ref,ref_report=isolated(old,row['arm']+'_r10')
            assert t==ref, f'off differs from R10: {row["arm"]}'
            report['equivalence_decisions']=len(t['decisions']);report['equivalence_forced_guards']=ref_report['forced_guard_triggers']
        print('SELFTEST_PASS',row['arm'],report['decisions'],flush=True)
        return report
    with ThreadPoolExecutor(workers) as pool: reports=list(pool.map(one,rows))
    eq=[r for r in reports if 'equivalence_decisions' in r]
    assert len(eq)==8
    summary=dict(PASS=True,test_arms=sum(not r.get('root','').endswith('r11_devknob_50') for r in reports),
                 dev_arms=20,decisions=sum(r['decisions'] for r in reports),equivalence_cell_sizes=len(eq),
                 equivalence_decisions=sum(r['equivalence_decisions'] for r in eq),
                 equivalence_forced_guards=sum(r['equivalence_forced_guards'] for r in eq),differing_decisions=0,records=reports)
    write_json(HERE/'selftests.json',summary)
    print('ALL_SELFTESTS',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('trace','all'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--row');p.add_argument('--label');p.add_argument('--size',type=int)
    a=p.parse_args()
    if a.action=='trace': trace(a.row,a.label,a.size)
    else: all_tests(a.workers)

if __name__=='__main__': main()
