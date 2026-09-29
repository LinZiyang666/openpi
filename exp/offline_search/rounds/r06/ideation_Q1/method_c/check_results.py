"""Replay, seed, calibration-rate, exclusion, and stub integration checks.

R6-C-v2 + SELECTION §7b (TAG v2b). Superseded expectations: the pre-7b
synthetic checks asserted that an always-ambiguous path never calls at p=1 and
enumerated ambiguous anchors with p=0 on a fixed cadence; both encoded the old
ambiguous=p0 rule and are replaced by the §7b expectations below.
"""
import argparse
import copy
import itertools
import json
from collections import deque, Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .common import HERE,OUT,STORE,sources,load_base,read_bank,write_json,sha
from .methods import CalibratedRescue
from .budget import expected_cost,episode_expectation,replay_cadence,simulate_path
from .stall_bridge import load_stall,AMBIGUOUS_RULE
from .recording import ExcludedRecordingA
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods import RiskLottery,uniform
from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.budget import solve as q2_solve

CASES=['A','floor','uniform','R','stall','R45','stall30','R_reverse']


def oracle(rows,tracker,j,last_extra,cooled,prob,acc,param,placement,scope,b=5,h=10):
    """Independent §7b expectation: explicit branch enumeration with a sequential tracker copy."""
    if j>=len(rows) or prob==0.:return
    t=copy.deepcopy(tracker);t.observe(rows[j]['key'],rows[j]['control_index']);st=t.status()
    ctrl=rows[j]['control_index']
    eligible=st['state']=='slow_ambiguous' and (last_extra is None or ctrl-last_extra>=st['W']*h)
    nominal=param if placement=='uniform' else min(1.,param*rows[j]['Ehat'])
    p=0. if cooled else 1. if st['state']=='slow_confirmed' else nominal  # SELECTION 7b text
    acc['anchors']+=prob;acc['calls']+=prob*p
    if p>0:oracle(rows,t,j+h//b,last_extra,scope=='all' or st['state']=='slow_confirmed',prob*p,acc,param,placement,scope)
    if p<1:
        if eligible:acc['looks']+=prob*(1-p)
        oracle(rows,t,j+1 if eligible else j+h//b,ctrl if eligible else last_extra,False,prob*(1-p),acc,param,placement,scope)


class WindowTracker:
    """Synthetic history-dependent tracker exposing a bounded window, like Q3's."""
    STATES=['ok','slow_confirmed','ok','slow_ambiguous','slow_ambiguous']
    def __init__(self,model=None,task=None):
        self._window=deque(maxlen=3);self._last=None
    def observe(self,key,control_index):
        if self._last is not None and control_index<=self._last:raise ValueError('order')
        self._last=control_index;self._window.append(key)
    def status(self):
        if len(self._window)<3:return dict(state='inactive',W=2)
        return dict(state=self.STATES[(sum(self._window)+3*self._window[-1])%5],W=2)


class HistoryTracker(WindowTracker):
    """Same statuses without an exposed window: the DAG must key on full history."""
    def __init__(self,model=None,task=None):
        self._hist=[];self._last=None
    def observe(self,key,control_index):
        if self._last is not None and control_index<=self._last:raise ValueError('order')
        self._last=control_index;self._hist.append(key)
    def status(self):
        w=self._hist[-3:]
        if len(w)<3:return dict(state='inactive',W=2)
        return dict(state=self.STATES[(sum(w)+3*w[-1])%5],W=2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v2b');ap.add_argument('--seeds',type=int,default=1000)
    a=ap.parse_args();dry='dryrun_'+a.tag
    report={'status':'PASS','controller_version':'R6-C-v2','ambiguous_rule':AMBIGUOUS_RULE,'tag':a.tag,'tests':{},'rates':[],'stall_rates':[]}
    reports=[];amb=Counter()
    for cell in ['pi05_l10_50','groot_l10_50']:
        def product(case):return np.load(OUT/f'replays_{a.tag}'/f'{cell}_{case}'/'served.npz')
        pa,pf=product('A'),product('floor')
        for field in pa.files:assert np.array_equal(pa[field],pf[field]),(cell,field)
        r,rev=product('R'),product('R_reverse')
        for field in r.files:assert np.array_equal(r[field][np.argsort(r['row'])],rev[field][np.argsort(rev['row'])]),(cell,field)
        for case in CASES:
            rr=json.loads((OUT/f'replays_{a.tag}'/f'{cell}_{case}'/'report.json').read_text());assert rr['status']=='PASS';reports.append(rr)
            if case!='A':
                assert rr['ambiguous_rule']==AMBIGUOUS_RULE and rr['dag_path_agreement']['anchors']==rr['anchors']
            for k,v in rr.get('ambiguous',{}).items():amb[k]+=v
    assert amb['lottery_calls']>0 and amb['extra_LOOKs']>0,amb  # both 7b branches exercised by the real tracker
    # 7b touches only slow_ambiguous anchors: every no-stall replay is bit-identical to the pre-7b v2 replay.
    identical=[]
    if (OUT/'replays_v2').exists() and a.tag!='v2':
        for cell in ['pi05_l10_50','groot_l10_50']:
            for case in ['A','floor','uniform','R','R45','R_reverse']:
                new_,old_=(np.load(OUT/t/f'{cell}_{case}'/'served.npz') for t in (f'replays_{a.tag}','replays_v2'))
                assert new_.files==old_.files and all(np.array_equal(new_[k],old_[k]) for k in new_.files),(cell,case)
                identical.append(f'{cell}_{case}')
    report['plugin_replays']=reports
    report['tests']['A_floor_bit_identity']='all action columns, histories/sources, vision/HIT schedules; both models'
    report['tests']['seed_reproducibility']='same complete action/source sequence with reversed episode arrival and renamed UIDs'
    report['tests']['commit_gripper']='every nonfinal call has one exact shift-by-five tail, then fresh vision; all 32 columns preserved for both models'
    report['tests']['selection_7b_replays']=dict(note='every fresh anchor: logged p/call/extra LOOK/reason re-derived from logged state and nominal p; '
        'each realized episode is one path of the budget DAG built from the same recorded keys (states, cooldowns, LOOK eligibility)',
        ambiguous_totals=dict(amb),dag_checked_anchors=sum(r.get('dag_path_agreement',{}).get('anchors',0) for r in reports),
        bit_identical_to_pre7b_v2=identical)
    # Explicit absent-Q3 stub path as well as the no-stall configuration.
    name='exp.offline_search.rounds.r06.ideation_Q3.stall.stall'
    with patch('importlib.import_module',side_effect=ModuleNotFoundError(name=name)):
        model,cls,fingerprint=load_stall('not-installed-yet')
        assert model is None and fingerprint=='inactive_stub_missing_module'
        tracker=cls(model,0);tracker.observe(None,0);tracker.observe(None,5);assert tracker.status()['state']=='inactive'
    # Extra LOOK cap with repeated ambiguity; §7b: at p=1 every ambiguous anchor calls and no LOOK is taken.
    class Ambiguous:
        def __init__(self,*args):pass
        def observe(self,*args):pass
        def status(self):return dict(state='slow_ambiguous',W=2)
    fake=dict(task=0,uid='synthetic',init=0,weight=1.,block_controls=5,commit_controls=10,
        decisions=[dict(step=j,control_index=j*5,key=None,Ehat=1.) for j in range(20)])
    cad=replay_cadence(fake,None,Ambiguous)
    path=simulate_path(cad,0.,'uniform',lambda n:.5)
    look_times=[n*5 for n,_,_,look in path['path'] if look]
    assert path['calls']==0 and len(look_times)>1 and np.all(np.diff(look_times)>=20)
    m0=expected_cost([cad],0.,'uniform',.152);assert m0['calls']==0 and m0['extra_LOOKs']==len(look_times)
    m1=expected_cost([cad],1.,'uniform',.152);assert m1['calls']==m1['anchors']==10 and m1['extra_LOOKs']==0
    # Exhaustive §7b oracle (sequential tracker copies, explicit branching) versus the DAG.
    rng=np.random.default_rng(7);worst=0.;compared=0
    for n_rows in (13,18):
        rows=[dict(step=j,control_index=5*j,key=int(k),Ehat=float(e)) for j,(k,e) in
              enumerate(zip(rng.integers(0,50,n_rows),rng.uniform(.1,1.2,n_rows)))]
        ep=dict(task=0,uid='exhaustive',init=0,weight=1.,block_controls=5,commit_controls=10,decisions=rows)
        for tracker_class in (WindowTracker,HistoryTracker):
            for scope in ('stall','all'):
                tree=replay_cadence(ep,None,tracker_class,scope)
                for placement,params in (('uniform',(0.,.3,.7,1.)),('R',(0.,.5,1.3,20.))):
                    for param in params:
                        acc=Counter();oracle(rows,tracker_class(),0,None,False,1.,acc,param,placement,scope)
                        got=episode_expectation(tree,param,placement)
                        for x,y in zip(got,(acc['anchors'],acc['calls'],acc['looks'])):
                            worst=max(worst,abs(x-y));compared+=1
    assert worst<1e-12,worst
    report['tests']['stub_cooldown_LOOK']=dict(note='missing-module and unconfigured stub; ambiguity LOOK cap at p=0; SELECTION 7b: at p=1 all ambiguous anchors call, zero LOOKs; '
        'DAG expectation equals an independent exhaustive branch enumeration (windowed and full-history trackers, stall/all cooldown scopes, uniform/R)',
        exhaustive_values_compared=compared,max_abs_diff=worst)
    for cell,source in sources().items():
        calpath=OUT/cell/dry/'calibration.json';cal=json.loads(calpath.read_text())
        assert cal['ambiguous_rule']==AMBIGUOUS_RULE and cal['controller_version']=='R6-C-v2'
        replay=json.loads((calpath.parent/'cost_replay.json').read_text())
        bank,arr=read_bank(OUT/cell/'r_bank.json');c1=cal['c1']
        obj=CalibratedRescue(.18,calibration_path=calpath)
        try:obj._load(cell.rsplit('_',1)[0]+'_cache')
        except ValueError as e:assert 'DRYRUN_TEST_INITS' in str(e)
        else:raise AssertionError('test init deployment accepted')
        # No-stall cadence is a chain: one node per anchor row.
        episodes=replay['modes']['no_stall']
        for e in episodes:
            assert len({n['j'] for n in e['nodes']})==len(e['nodes'])
            assert all(n['call']==n['nocall'] for n in e['nodes'])
        # Actual RiskLottery's episode assignment, including its dose mixture.
        lottery=RiskLottery(rho=.18,allocation='uniform',calibration_path='unused',calibration_sha256='unused')
        lottery.q2_budget=q2_solve(bank['library_lengths'],.18,c1,'uniform')
        repetitions=a.seeds; draws={k:[] for k in ['uniform','R','RiskLottery']}
        for seed in range(repetitions):
            total={k:0. for k in draws}
            for e in episodes:
                task,init=e['task'],e['init'];weights=e['weight']/e['nominal_blocks']
                coins={n['step']:uniform('rate-test',seed,task,init,n['step'],'dose-anchor') for n in e['nodes']}
                for placement in ['uniform','R']:
                    param=cal['solutions']['no_stall'][placement]['0.18']['parameter']
                    s=simulate_path(e,param,placement,lambda n:coins[n['step']])
                    total[placement]+=weights*(c1*s['anchors']+(1-c1)*s['calls'])
                lottery.q2_seed=seed;lottery.q2_key='rate-test'
                lottery._reset_assignment(SimpleNamespace(task_id=task,init=init))
                n=sum(u<lottery.q2_episode_dose for u in coins.values())
                total['RiskLottery']+=weights*(c1*len(coins)+(1-c1)*n)
            for k in draws:draws[k].append(total[k])
        expected_rl=sum(e['weight']*len(e['nodes'])/e['nominal_blocks']*(c1+(1-c1)*lottery.q2_budget['tasks'][e['task']]['p']) for e in episodes)
        stats={}
        for k,d in draws.items():
            mean,se=float(np.mean(d)),float(np.std(d,ddof=1)/np.sqrt(repetitions))
            expected=expected_rl if k=='RiskLottery' else .18
            assert abs(mean-expected)<5*se,(cell,k,mean,se,expected)
            stats[k]=dict(mean_IR=mean,MC_SE=se,expected_IR=expected)
        delta=np.asarray(draws['uniform'])-draws['RiskLottery']
        observed=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(repetitions))
        # Same feasible nominal budget; library-vs-recording endpoint weighting
        # explains its explicit deterministic discrepancy, not random failure.
        assert abs(observed-(.18-expected_rl))<5*se
        assert abs(observed)<.01,(cell,observed)
        stats['uniform_minus_RiskLottery']=dict(mean_IR=observed,MC_SE=se,recording_length_shift=.18-expected_rl)
        report['rates'].append(dict(cell=cell,repetitions=repetitions,results=stats,
            caveat='Expected realized cost matched at feasible rho; RiskLottery uses an episode mixture, C uniform uses independent anchors; neither cools lottery calls.'))
        # Stall mode (real Q3 statuses, §7b branching): keyed-coin paths through the DAG with the
        # deployed rule versus the exact DAG expectation, at the .30 target (feasible in every cell).
        stall_eps=replay['modes']['stall'];sres={}
        for placement in ['uniform','R']:
            sol=cal['solutions']['stall'][placement]['0.3'];assert sol['feasible']
            ir,looks,calls=[],[],[]
            for seed in range(repetitions):
                t_ir=t_look=t_call=0.
                for e in stall_eps:
                    s=simulate_path(e,sol['parameter'],placement,
                        lambda n,e=e:uniform('stall-rate-test',seed,e['task'],e['init'],n['step'],'dose-anchor'))
                    t_ir+=e['weight']*(c1*s['anchors']+(1-c1)*s['calls'])/e['nominal_blocks']
                    t_look+=e['weight']*s['extra_LOOKs'];t_call+=e['weight']*s['calls']
                ir.append(t_ir);looks.append(t_look);calls.append(t_call)
            out={}
            for k,d,want in [('IR',ir,sol['predicted_IR']),('extra_LOOKs',looks,sol['modeled']['extra_LOOKs']),('calls',calls,sol['modeled']['calls'])]:
                mean,se=float(np.mean(d)),float(np.std(d,ddof=1)/np.sqrt(repetitions))
                assert abs(mean-want)<5*se+1e-12,(cell,placement,k,mean,se,want)
                out[k]=dict(mean=mean,MC_SE=se,modeled=want)
            sres[placement]=out
        report['stall_rates'].append(dict(cell=cell,rho=.30,repetitions=repetitions,results=sres))
        # Validate recording exclusion rows using real bank queries. No rollout.
        rec=ExcludedRecordingA(OUT/cell/'r_bank.json',OUT/cell/'recording_exclusions.json',sha(OUT/cell/'recording_exclusions.json'))
        rec.fit(None,SimpleNamespace(cell=cell.rsplit('_',1)[0]+'_cache'))
        exc=json.loads((OUT/cell/'recording_exclusions.json').read_text())
        libdir=Path(bank['library_directory']);keys=[np.load(libdir/(k+'.npy'),mmap_mode='r') for k in ['key_v0','key_v1','rs']]
        for task,r in exc['tasks'].items():
            row=int(np.flatnonzero(arr['task_id']==int(task))[0])
            ep=SimpleNamespace(uid='Bval',task_id=int(task),init=r['init'])
            q=SimpleNamespace(key_v0=keys[0][row],key_v1=keys[1][row],rs=keys[2][row],task_id=int(task),step=0,prev_hit=None,episode=ep)
            result=rec._dist(q);mask=np.isin(rec.lib_ep[result[0].rows],r['source_episodes'])
            assert np.isinf(result[-1][mask]).all() and np.isfinite(result[-1][~mask]).all()
    # Loader gates: a pre-7b calibration and an infeasible target are refused even in test mode.
    gates={}
    old=OUT/'pi05_l10_50'/'dryrun_v2'/'calibration.json'
    if old.exists():
        try:CalibratedRescue(.18,calibration_path=old)._load('pi05_l10_cache',allow_dryrun=True)
        except ValueError as e:assert '7b' in str(e);gates['pre_7b_calibration']='rejected'
        else:raise AssertionError('pre-7b calibration accepted')
    for cell in sources():
        sol=json.loads((OUT/cell/dry/'calibration.json').read_text())['solutions']['stall']['R'].get('0.45')
        if sol is not None and not sol['feasible']:
            obj=CalibratedRescue(.45,placement='R',stall_model_path=str(Path('/tmp/q3_stall_fits')/cell.replace('spatial','sp')),
                                 calibration_path=OUT/cell/dry/'calibration.json')
            try:obj._load(cell.rsplit('_',1)[0]+'_cache',allow_dryrun=True)
            except ValueError as e:assert 'infeasible target' in str(e);gates[f'{cell}_C45']='infeasible target refused'
            else:raise AssertionError('infeasible C.45 accepted')
            # 7b C.max: the stored calibrated-ceiling pseudo-target loads and saturates the parameter.
            key=format(sol['ceiling'],'.12g')
            cmax=CalibratedRescue(float(key),placement='R',stall_model_path=obj.stall_model_path,calibration_path=OUT/cell/dry/'calibration.json')
            cmax._load(cell.rsplit('_',1)[0]+'_cache',allow_dryrun=True)
            assert cmax.budget['feasible'] and cmax.parameter==cmax.budget['ceiling_parameter']
            gates[f'{cell}_Cmax']=dict(rho=float(key),parameter=cmax.parameter,status='loads at calibrated ceiling')
    report['tests']['loader_gates']=gates
    report['tests']['budget_rates']=f'{a.seeds} independent keyed seeds/cell; no-stall R/uniform owner IR within 5 MC SE of .18; uniform versus actual RiskLottery assignment within .01'
    report['tests']['stall_budget_rates']=f'{a.seeds} keyed seeds/cell/placement at rho .30 with real Q3 statuses and 7b branching; realized IR, calls and extra LOOKs within 5 MC SE of the exact DAG model'
    report['tests']['Bval_exclusions']='all 80 task/cell candidate masks checked on real bank observations; whole matching source episode excluded'
    report['tests']['deployment_firewall']='all eight DRYRUN_TEST_INITS calibrations rejected by production loader'
    write_json(HERE/f'test_results_{a.tag}.json',report)
    print(json.dumps(dict(status='PASS',tag=a.tag,plugin_replays=len(reports),rate_cells=8,rate_seeds=a.seeds,ambiguous=dict(amb),
        exhaustive_values=compared,gates=gates)))

if __name__=='__main__':main()
