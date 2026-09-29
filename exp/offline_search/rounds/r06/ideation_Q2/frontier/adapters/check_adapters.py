"""Audit completed CPU replays, keyed lotteries, fit/spec and zero-dose contracts."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import pickle
import random
import re
import sys
import numpy as np

from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop import plugin
from .budget import DOSES, digest, solve
from .methods import uniform, RiskLottery, CacheDose, ExtraDosePi05, ExtraDoseGroot
from .prefit import load_source

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/offline_search_store')


def load_fit(case):
    cls,_ = plugin.load_method_class(case['spec']['method'])
    cls(**case['spec']['kwargs'])
    with Path(case['artifact']).open('rb') as f:
        return pickle.load(f)['method']


def assert_same(a,b):
    if isinstance(a,dict):
        assert list(a) == list(b)
        for k in a:
            assert_same(a[k],b[k])
    elif a is None or isinstance(a,str):
        assert a == b
    else:
        x,y = np.asarray(a),np.asarray(b)
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def result_equal(a,b):
    for attr in ('topk','scores','confidence','action','library','extras'):
        assert_same(getattr(a,attr),getattr(b,attr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replays',type=Path,required=True)
    args = ap.parse_args()
    cases = json.loads((HERE/'replay_cases.json').read_text())
    by = {(c['cell'],c['kind']):c for c in cases}
    reports = []
    for c in cases:
        p = args.replays/c['name']
        r = json.loads((p/'report.json').read_text())
        assert r['assertions']=='PASS' and r['stage1_calls']==r['anchors']
        if r['rate_z'] is not None:
            assert abs(r['rate_z']) < 6  # predetermined wide sanity limit, not an SR claim
        reports.append(r)
    comparisons = []
    def compare(cell,left,right):
        ca,cb = by[cell,left],by[cell,right]
        xa = np.load(args.replays/ca['name']/'served.npz')
        xb = np.load(args.replays/cb['name']/'served.npz')
        ia,ib = np.argsort(xa['row']),np.argsort(xb['row'])
        for key in xa.files:
            assert_same(xa[key][ia],xb[key][ib])
        ra = [json.loads(s) for s in (args.replays/ca['name']/'decisions_replay.jsonl').read_text().splitlines() if '"ev": "dec"' in s]
        rb = [json.loads(s) for s in (args.replays/cb['name']/'decisions_replay.jsonl').read_text().splitlines() if '"ev": "dec"' in s]
        ra.sort(key=lambda r:(r['task_id'],r['init'],r['step']))
        rb.sort(key=lambda r:(r['task_id'],r['init'],r['step']))
        keys = ['task_id','init','step','lib','top1','topk','scores','conf','synth','nfix','vision','hit','src','served_head']
        for x,y in zip(ra,rb):
            for key in keys:
                assert x.get(key) == y.get(key),(cell,left,right,key)
            # A baseline's pure-cache logger has a smaller scalar cap. Every
            # field it emits must remain bit-identical; full Results checked below.
            for key,value in x.get('extras',{}).items():
                assert y.get('extras',{}).get(key) == value,(cell,left,right,key)
        comparisons.append(dict(cell=cell,left=left,right=right,decisions=len(ra),status='BIT_IDENTICAL'))
    cells = sorted({c['cell'] for c in cases})
    for cell in cells:
        compare(cell,'A','A0')
        compare(cell,'B','B0')
        for kind in ('Risk','Bextra'):
            if (cell,kind+'_reverse') in by:
                compare(cell,kind,kind+'_reverse')
    # Numeric schema and force flags survive plugin's 40-scalar mixed log cap.
    episodes_logged = 0
    for c in cases:
        if c['kind'] not in ('Risk','Uniform'):
            continue
        decs = [json.loads(s) for s in (args.replays/c['name']/'decisions_replay.jsonl').read_text().splitlines() if '"ev": "dec"' in s]
        assigned = {}
        method = load_fit(c)
        for r in decs:
            if not r['vision']:
                continue
            ex = r['extras']
            key = (r['task_id'],r['init'])
            dose,prop,u = (ex[k] for k in ('os_q2_episode_dose','os_q2_episode_propensity','os_q2_episode_coin'))
            if key in assigned:
                assert assigned[key] == (dose,prop,u)
            else:
                assert r['step']==0
                assigned[key] = (dose,prop,u)
            weights = method.q2_budget['tasks'][r['task_id']]['weights']
            assert prop == weights[list(DOSES).index(dose)]
            assert np.isclose(np.dot(DOSES,weights),ex['os_q2_task_p'])
        assert len(assigned)==20
        episodes_logged += len(assigned)
    # Exact public Result identity on recorded all-vision queries, including
    # GR00T's sign-correction branch that a normal 10-control schedule rarely uses.
    method_rows, terminal_closed, terminal_open = 0,0,0
    for cell in cells:
        model,suite,_ = cell.split('_')
        qc = store.QueryCell(ROOT,f'{model}_{suite}_cache')
        arr = api.QueryArrays(qc)
        methods = {k:load_fit(by[cell,k]) for k in ('A','A0','B','B0')}
        selected = [i for i,e in enumerate(qc.episodes) if int(e['init']) in (0,49)]
        for ei in selected:
            e = qc.episodes[ei]
            ep = api.EpisodeView(e['uid'],e['task'],e['task_id'],e['init'],ei,26092901)
            for method in methods.values():
                method.reset(ep)
            for step,row in enumerate(range(e['start'],min(e['end'],e['start']+30))):
                q = api.QueryView(arr,row,e['start'],step,e['task_id'],ep)
                result = {k:m.query(q) for k,m in methods.items()}
                result_equal(result['A'],result['A0'])
                result_equal(result['B'],result['B0'])
                if model=='groot' and result['B'].extras.get('term1') == 1.:
                    ex = result['B'].extras
                    closed = ex['gexec'] < 0
                    assert bool(int(ex['os_flags']) & 2) == closed
                    terminal_closed += closed
                    terminal_open += not closed
                method_rows += 1
    assert terminal_closed > 0 and terminal_open > 0
    # RNGs: coins do not mutate the policy/global RNG, and arm uid/arrival order
    # is absent from the key. All endpoint and bad-input behavior is explicit.
    ns = np.random.get_state(); ps = random.getstate()
    z = [uniform('unit',7,3,11,i,'extra-anchor') for i in range(10000)]
    assert z == [uniform('unit',7,3,11,i,'extra-anchor') for i in range(10000)]
    assert min(z)>=0 and max(z)<1 and len(set(z))==len(z)
    assert z != [uniform('unit',8,3,11,i,'extra-anchor') for i in range(10000)]
    assert z != [uniform('unit',7,3,11,i,'episode-dose') for i in range(10000)]
    assert_same(ns[1],np.random.get_state()[1]); assert ns[2:] == np.random.get_state()[2:]
    assert ps == random.getstate()
    refusals=0
    for cls,kw in [(ExtraDosePi05,{'dose':-.1}),(ExtraDoseGroot,{'dose':1.1}),
                   (CacheDose,{'budget':2}),(CacheDose,{'random_seed':True}),
                   (RiskLottery,{'rho':-.1}),(RiskLottery,{'allocation':'task_lookup'})]:
        try:
            cls(**kw)
        except ValueError:
            refusals+=1
        else:
            raise AssertionError('invalid config accepted')
    cal = json.loads((HERE/'calibration.json').read_text())
    allocation_checks, lottery_checks = 0,0
    sys.path.insert(0,str(HERE.parents[1]))
    import pilot_q2
    quality,_ = pilot_q2.quality_rows()
    for bank,calrow in cal['libraries'].items():
        model_suite,lib = bank.split('/')
        qcell = model_suite.replace('_spatial','_sp')+('_50' if lib=='current' else '_500')
        for mode in ('risk','uniform'):
            for rho in (0.,.065,.10,.15,.20,.25,.35,.45,1.):
                rule = solve(calrow['tasks'],rho,calrow['c1'],mode)
                prior = pilot_q2.allocation(quality,qcell,rho,mode=='uniform')
                assert np.isclose(rule['predicted_IR'],prior['predicted_IR'],rtol=0,atol=1e-15)
                for t,r in rule['tasks'].items():
                    assert np.allclose(r['weights'],list(prior['weights'][t].values()),rtol=0,atol=1e-14)
                    assert np.isclose(np.dot(DOSES,r['weights']),r['p'])
                    nz = np.flatnonzero(r['weights'])
                    assert len(nz)<=2 and (len(nz)==1 or nz[1]-nz[0]==1)
                allocation_checks+=1
        rule = solve(calrow['tasks'],.2,calrow['c1'])
        for t,r in rule['tasks'].items():
            u = [uniform(bank,26092901,t,i,-1,'episode-dose') for i in range(2048)]
            hist = np.bincount(np.searchsorted(np.cumsum(r['weights']),u,side='right'),minlength=5)
            expected = 2048*np.array(r['weights'])
            variance = expected*(1-np.array(r['weights']))
            assert np.all(np.abs(hist-expected) <= 6*np.sqrt(variance)+1)
            lottery_checks += 2048
    # All 43 emitter specs parse with explicit non-/dev/shm store root; fit blobs
    # exactly match method/kwargs/cell, including reused serving variants.
    specs = json.loads((HERE/'emit_arms_all43.json').read_text())
    notes = []
    for s in specs:
        argv = ['--os-method',s['method'],'--os-kwargs',json.dumps(s['kwargs']),
                '--os-cell',f'{s["model"]}_{s["suite"]}_cache','--os-log-dir',str(HERE/'never_created'),*s['plugin_args']]
        opts,rest = plugin.parse_cli(argv)
        assert not rest and opts.os_root == str(ROOT) and opts.os_no_shadow_native
        assert s['plugin_args'].count('--os-root')==1 and s['full_model'] and s['cost_ledger']
        assert not any('shadow' in v for v in s['plugin_args'] if v!='--os-no-shadow-native')
        assert re.fullmatch(r'[A-Za-z0-9_]+',s['name'])
        path = Path('/tmp/q2_adapter_fits')/(s['name']+'.pkl')
        note = json.loads(path.with_suffix('.json').read_text())
        assert digest(path)==note['sha256']
        assert note['identity']==dict(spec=s['method'],kwargs=s['kwargs'],cell=opts.os_cell)
        cls,_ = plugin.load_method_class(s['method'])
        cls(**s['kwargs'])
        with path.open('rb') as f:
            blob = pickle.load(f)
        method = blob['method']
        for field in ('dose','rho','allocation'):
            if field in s['kwargs']:
                assert getattr(method,{'dose':'q2_dose','rho':'q2_rho','allocation':'q2_allocation'}[field])==s['kwargs'][field]
        if 'budget' in s['kwargs']:
            assert method.budget==s['kwargs']['budget']
        for field in ('cycle_k','tail_blocks'):
            if field in s['kwargs']:
                assert getattr(method,field)==s['kwargs'][field]
        clone,mode = plugin.clone_method(method,strict=True)
        assert clone is not method and mode=='deepcopy_shared_arrays'
        if isinstance(method,RiskLottery):
            assert clone.q2_budget is not method.q2_budget
        notes.append(note)
    (HERE/'prefit_manifest.json').write_text(json.dumps(notes,indent=2)+'\n')
    with (HERE/'replay_rates.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(reports[0]));writer.writeheader();writer.writerows(reports)
    result = dict(status='PASS',replay_cases=len(cases),recorded_stream_replays=sum(r['episodes'] for r in reports),
        replay_decisions=sum(r['decisions'] for r in reports),policy_tails=sum(r['policy_tails'] for r in reports),
        identity_comparisons=comparisons,full_result_query_pairs=2*method_rows,
        groot_terminal_closed=terminal_closed,groot_terminal_open=terminal_open,
        invalid_config_refusals=refusals,allocation_checks=allocation_checks,
        episode_lottery_draws=lottery_checks,existing_fit_specs_verified=len(notes),
        total_deployment_fit_bytes=sum(n['bytes'] for n in notes),
        scope='CPU fixed-observation replays only; no model inference, server, rollout, or SR estimate',
        replay_root=str(args.replays),spec_sha256=digest(HERE/'emit_arms_all43.json'),
        calibration_sha256=digest(HERE/'calibration.json'))
    (HERE/'test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
