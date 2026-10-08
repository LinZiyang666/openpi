"""Read-only, B-only analysis of the frozen R15 query-key pilot.

Run from the repository with taskset -c 10-21,54-65 and one BLAS thread.
Writes only PILOT_RESULTS.{md,json} and pilot_results_episodes.csv beside this script.
No model execution, fitting, rollout, network operation, or mutable run-root access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
GATES_SHA = '188d3cc1d3beb807174c3bff84696542be72f00bcb66ac0cdc66cbc913a2d937'
SOURCES = {}


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def js(path):
    return json.loads(read(path))


def lines(path):
    return (json.loads(line) for line in read(path).splitlines() if line.strip())


def pair(uid):
    return tuple(map(int, uid.rsplit(':', 2)[-2:]))


def bound_low(k, n, alpha=.025):
    return 0. if not k else float(beta.ppf(alpha, k, n-k+1))


def bound_high(k, n, alpha=.025):
    return 1. if k == n else float(beta.ppf(1-alpha, k+1, n-k))


def paired(current, light):
    assert set(current) == set(light)
    n = len(current)
    g = sum(light[k] and not current[k] for k in current)
    l = sum(current[k] and not light[k] for k in current)
    return dict(n=n, current=sum(current.values()), light=sum(light.values()), gains=g, losses=l,
                delta=(g-l)/n,
                lower95=bound_low(g,n)-bound_high(l,n),
                upper95=bound_high(g,n)-bound_low(l,n),
                # Additional two-sided interval, without changing the frozen lower-bound test.
                simultaneous_two_sided95=[bound_low(g,n,.0125)-bound_high(l,n,.0125),
                                          bound_high(g,n,.0125)-bound_low(l,n,.0125)])


def arm_path(size, kind, level):
    suffix = 'off' if level == 'off' else 'periodic_ir25'
    if size == 200 and kind == 'current':
        return RUNS/'r15_descent_pi05_l10_200'/'runs'/f'r15d_pi05_l10_200_{suffix}'
    return RUNS/f'r15_qk_pilot_pi05_l10_{size}'/'runs'/f'r15qk_pi05_l10_{size}_{kind}_{suffix}'


def analyze_arm(size, kind, level, expected, frozen, default_wait):
    run = arm_path(size,kind,level)
    summary = js(run/'summary.json');manifest = js(run/'manifest.json')
    arm_spec = next(a for a in js(run.parent.parent/'arms.json') if a['arm']==run.name)
    assert arm_spec['dev'] and arm_spec['init_pool']=='B'
    assert arm_spec['client_overrides']['replan_steps']==5
    warmup = int(arm_spec['client_overrides'].get('num_steps_wait',default_wait))
    assert summary['init_pool'] == manifest['init_pool'] == 'B' and manifest['dev']
    assert manifest['seed'] == 2026101401 and manifest['per_task'] == 30
    assert set(map(tuple,manifest['pairs'])) == expected
    assert not manifest['independent_validation']
    accepted = {};journal = list(lines(run/'client/journal.jsonl'))
    errors = sum(bool(x.get('error')) or x.get('status') not in ('done','failed') for x in journal)
    for row in journal:
        assert row['init_pool'] == 'B'
        if row.get('accepted') and row.get('status') in ('done','failed') and not row.get('error'):
            assert row['task_uid'] not in accepted
            accepted[row['task_uid']] = row
    assert len(accepted) == len(journal) == summary['complete'] == 300
    assert errors == summary['error_rows'] == 0 and summary['retried'] == 0
    assert set(map(pair,accepted)) == expected
    assert sum(x['success'] for x in accepted.values()) == summary['success']
    light = kind == 'light';epsilon = frozen['epsilon'] if light else 0.
    if light:
        spec = summary['kwargs']['encoder_spec']
        assert spec['kwargs']['model_sha256'] == frozen['kwargs']['model_sha256']
        assert spec['epsilon'] == epsilon and spec['completed_epochs'] == frozen['completed_epochs']
    dec = {};ends = {};rejected = 0;startups = 0
    for file in sorted(run.glob('server_*/decisions_*.jsonl')):
        for row in lines(file):
            ev = row.get('ev')
            if ev == 'startup':
                startups += 1
                assert row['model'] == 'pi05' and row['suite'] == 'l10'
                assert row['rollout_seed']['seed'] == 2026101401 and row['rollout_seed']['knob_coin_seed'] == 0
                assert row['miss_steps'] == 10 and row['policy_tail_blocks'] == 1
                if light:
                    assert row['light_key']['epsilon'] == epsilon
                    assert row['kwargs']['encoder_spec']['kwargs']['model_sha256'] == frozen['kwargs']['model_sha256']
                continue
            if ev not in ('dec','episode'):
                continue
            uid = row['uid']
            if uid not in accepted or int(row.get('attempt',1)) != int(accepted[uid]['attempt']):
                rejected += 1;continue
            assert (int(row['task_id']),int(row['init'])) == pair(uid)
            if ev == 'episode':
                assert uid not in ends
                ends[uid] = row;continue
            key = (uid,int(row['step']));assert key not in dec
            assert row['ok'] and type(row['hit']) is bool and type(row['vision']) is bool
            miss = not row['hit'];judge = row['judge'];src = row['src']
            guard = judge == 'force:4';knob = judge == 'force:62'
            assert miss == (guard or knob), (run.name,judge)
            assert level != 'off' or not knob
            if miss:assert row['miss_k'] == 10
            ex = row.get('extras',{})
            if 'os_r11_setting' in ex:
                assert math.isclose(ex['os_r11_setting'],2.6125 if size == 50 else 2.)
                assert bool(ex['os_r11_guard']) == guard and bool(ex['os_r11_knob_call']) == knob
            if light:
                work = row['qk_work'];assert work['N'] == 1
                q,t,m = (int(work[k]) for k in ('Q','T','M'))
                assert all(x in (0,1) for x in (q,t,m))
                assert q == int(row['light_encoder_ran']) == int(row['look'])
                assert t == int(row['teacher_tower_ran']) == int(row['vision'])
                assert m == int(row['policy_full_call']) == int(miss) == t
                assert row['epsilon'] == epsilon
                assert math.isclose(row['decision_cost'],epsilon*q+m,abs_tol=1e-10)
                if row['hit']:assert t == m == 0
                if src in ('cache_blind','policy_tail'):assert q == t == m == 0
                cum = {k:int(row[k]) for k in ('Q','T','M','N')}
            else:
                q,t,m = 0,int(row['vision']),int(miss);cum = None
                assert not m or t
            dec[key] = dict(uid=uid,step=int(row['step']),Q=q,T=t,M=m,N=1,guard=int(guard),knob=int(knob),
                            src=src,hit=row['hit'],cum=cum,stage1_calls=row['stage1_calls'],
                            encoder_ms=row.get('light_encoder_ms'),s1_ms=row.get('s1_ms'))
    assert startups == 4 and len(ends) == 300 and rejected == 0
    client = {};terminal = {}
    for row in lines(run/'client/per_step.jsonl'):
        if row['task_uid'] not in accepted or row['attempt'] != accepted[row['task_uid']]['attempt']:continue
        assert row['run_id'] == accepted[row['task_uid']]['run_id']
        if row.get('_kind') == 'client_timing':
            assert row['accepted'] and row['termination_reason'] in ('success','step_cap')
            assert row['success'] == accepted[row['task_uid']]['success']
            assert row['task_uid'] not in terminal
            terminal[row['task_uid']] = row
            continue
        assert row['accepted'] and row['phase'] == 'eval'
        assert (row['task_id'],row['orig_init_state_idx']) == pair(row['task_uid'])
        assert row['step_idx'] % 5 == 0
        key = (row['task_uid'],row['step_idx']//5);assert key not in client
        assert row['hit_type'] in ('FULL_HIT','MISS')
        assert (row['hit_type']=='MISS') == bool(dec[key]['M'])
        assert row['success'] == accepted[row['task_uid']]['success']
        client[key] = True
    assert set(client) == set(dec) and set(terminal) == set(accepted)
    groups = defaultdict(list)
    for (uid,step),d in dec.items():groups[uid].append(d)
    episodes = []
    for uid,j in accepted.items():
        ds = sorted(groups[uid],key=lambda x:x['step'])
        assert [x['step'] for x in ds] == list(range(len(ds)))
        cumulative = Counter()
        # Physical tower counter belongs to a reused connection, not the episode.
        tower_before_episode = ds[0]['stage1_calls']-ds[0]['T']
        assert tower_before_episode >= 0
        for d in ds:
            cumulative.update({k:d[k] for k in ('Q','T','M','N')})
            assert d['stage1_calls'] == tower_before_episode+cumulative['T']
            if light:assert d['cum'] == dict(cumulative)
        e = ends[uid]
        assert e['success'] == j['success'] and e['n_decisions'] == e['n_exec'] == len(ds)
        assert e['n_miss'] == cumulative['M'] and e['n_hit'] == len(ds)-cumulative['M']
        if light:
            assert all(e[k] == cumulative[k] for k in ('Q','T','M','N'))
            assert math.isclose(e['IR_new'],(epsilon*cumulative['Q']+cumulative['M'])/len(ds))
        task,init = pair(uid)
        controls = int(terminal[uid]['steps'])-warmup
        assert int(terminal[uid]['infers']) == len(ds)
        assert 5*(len(ds)-1) < controls <= 5*len(ds)
        episodes.append(dict(size=size,level=level,kind=kind,task=task,init=init,success=bool(j['success']),
            **dict(cumulative),guard=sum(d['guard'] for d in ds),knob=sum(d['knob'] for d in ds),
            duration_s=j['duration_s'],nominal_controls=5*len(ds),
            executed_controls=controls,warmup_controls=warmup,unused_final_chunk_controls=5*len(ds)-controls))
    total = {k:sum(e[k] for e in episodes) for k in ('N','Q','T','M','guard','knob')}
    ledger = summary['cost_ledger']
    assert [total[k] for k in ('N','T','M')] == [ledger[k] for k in ('decisions','vision_decisions','misses')]
    if light:assert all(total[k] == ledger[k] for k in ('N','Q','T','M'))
    assert total['N'] == summary['client_decisions'] == summary['server']['decisions_logged']
    cost = epsilon*total['Q']+total['M'] if light else .152*total['T']+.848*total['M']
    ir = cost/total['N']
    if light:
        assert math.isclose(ir,ledger['ir_per_request'],abs_tol=1e-10)
    else:
        # Generic KPI files use the eager stage-cost ratio, not owner .152/.848.
        stages=ledger['full_stage_costs'];full=sum(stages[k] for k in ('s1','s2','s3'))
        generic=(stages['s1']*total['T']+(stages['s2']+stages['s3'])*total['M'])/(full*total['N'])
        assert math.isclose(generic,ledger['ir_per_request'],abs_tol=1e-10)
    bytask = {}
    for t in range(10):
        es = [e for e in episodes if e['task']==t];assert len(es)==30
        bytask[str(t)] = dict(n=30,success=sum(e['success'] for e in es),**{k:sum(e[k] for e in es) for k in total})
        assert bytask[str(t)]['success'] == summary['per_task'][str(t)]['success']
    durations = [e['N'] for e in episodes]
    timings = [d['encoder_ms'] for d in dec.values() if d['encoder_ms'] is not None]
    if light:assert len(timings)==total['Q']
    out = dict(size=size,kind=kind,level=level,run=str(run),episodes=episodes,by_task=bytask,totals=total,
        successes=sum(e['success'] for e in episodes),epsilon=epsilon,ir=ir,cost_per_episode=cost/300,
        stored_summary_ir=ledger['ir_per_request'],summary_pricing='frozen light epsilon' if light else 'generic eager stage cost, not owner coefficients',
        m_over_n=total['M']/total['N'],guard_over_n=total['guard']/total['N'],knob_over_n=total['knob']/total['N'],
        decisions_per_episode=total['N']/300,nominal_controls_per_episode=5*total['N']/300,
        executed_controls=sum(e['executed_controls'] for e in episodes),
        executed_controls_per_episode=sum(e['executed_controls'] for e in episodes)/300,
        episode_decisions_median=float(np.median(durations)),episode_decisions_p95=float(np.quantile(durations,.95)),
        miss_per_episode=total['M']/300,guard_per_episode=total['guard']/300,
        sources=dict(Counter(d['src'] for d in dec.values())),
        actual_terminal_control_rows=len(terminal),warmup_controls=warmup,
        encoder_ms=dict(n=len(timings),median=float(np.median(timings)),p95=float(np.quantile(timings,.95))) if timings else None,
        audit=dict(accepted=300,error_rows=errors,retries=summary['retried'],rejected_attempt_rows=rejected,
                   contiguous_steps=True,client_server_parity=True,summary_reconciled=True,paired_identities=True,
                   light_tower_on_hit_violations=0 if light else None,light_t_ne_m_violations=0 if light else None,
                   blind_or_policy_tail_work_violations=0 if light else None,cumulative_ledger_verified=True))
    print(run.name,total,'SR',out['successes']/300,'IR',ir,flush=True)
    return out


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.sched_getaffinity(0) <= set(range(10,22))|set(range(54,66))
    gates = read(HERE/'GATES.md');assert hashlib.sha256(gates).hexdigest()==GATES_SHA
    read(HERE/'PILOT.md');freeze = js(HERE/'ROUND2_MANIFEST.json')
    assert SOURCES[str(HERE/'PILOT.md')] == freeze['files'][str(HERE/'PILOT.md')]['sha256']
    selection = js(REPO/'exp/offline_search/rounds/r15/descent/selection.json')
    expected = set(map(tuple,selection['model_suites']['pi05_l10']['pairs']));assert len(expected)==300
    equivalence = js(HERE.parent/'impl/baseline_equivalence.json');assert equivalence['PASS']
    for record in equivalence['records']:
        root=Path(record['reused_root'])
        assert root==RUNS/'r15_descent_pi05_l10_200'
        read(root/'dev_selection.json');read(record['fit_path'])
        assert SOURCES[str(root/'dev_selection.json')]==record['manifest_sha256']
        assert SOURCES[record['fit_path']]==record['fit_sha256']
    decisions = read(REPO/'exp/offline_search/rounds/r15/COORDINATOR_DECISIONS.md').decode()
    assert '## D5' in decisions
    read(REPO/'exp/offline_search/rounds/r04/k1_blind/judge.py')
    read(REPO/'exp/offline_search/rounds/r11/knob/recipe.py')
    client_source = ast.parse(read(REPO/'examples/libero/main.py').decode())
    waits = [x for x in ast.walk(client_source) if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id=='num_steps_wait']
    assert len(waits)==1
    default_wait = int(ast.literal_eval(waits[0].value))
    arms = {};comparisons = {};offline = []
    for size in (50,200):
        cell = f'pi05_l10_{size}';config = js(HERE/'round2/pilot'/cell/'encoder_config.json')
        assert SOURCES[str(HERE/'round2/pilot'/cell/'encoder_config.json')] == freeze['files'][str(HERE/'round2/pilot'/cell/'encoder_config.json')]['sha256']
        for name,round_ in (('r1_action',1),('state_drop_aug',2)):
            a = js(HERE/'round2'/cell/name/'assessment.json')
            assert (a['episodes'],a['looks']) == ((100,3198) if size==50 else (73,2289))
            if round_ == 1:
                lat = js(HERE/f'latency_cpu_extra_{size}.json')
                timing = next(x['end_to_end'] for x in lat['results'] if x['name']=='symmetric_cnn_action_cpu' and x['mode']=='torchscript' and x['cpu_threads']==1)
            else:timing = js(HERE/'round2/pilot'/cell/'latency.json')['timings'][0]
            offline.append(dict(size=size,round=round_,name=name,library_ratio=a['library']['balanced_ratio'],
                bdev_ratio=a['b_dev']['balanced_ratio'],shuffle_rise=a['blindness']['shuffle_mean_relative_rise'],
                zero_rise=a['blindness']['zero']['relative_rise'],blindness_pass=a['blindness']['pass'],
                median_ms=timing['median_ms'],p95_ms=timing['p95_ms']))
        for level in ('off','ir25'):
            current = analyze_arm(size,'current',level,expected,config,default_wait)
            light = analyze_arm(size,'light',level,expected,config,default_wait)
            key = f'{size}_{level}';arms[key] = dict(current=current,light=light)
            maps = [{(e['task'],e['init']):e['success'] for e in a['episodes']} for a in (current,light)]
            result = paired(*maps)
            result['tasks'] = {str(t):paired(*[{k:v for k,v in m.items() if k[0]==t} for m in maps]) for t in range(10)}
            result['saving'] = 1-light['ir']/current['ir']
            result['ir_absolute_saving'] = current['ir']-light['ir']
            result['cost_per_episode_saving'] = 1-light['cost_per_episode']/current['cost_per_episode']
            result['checks'] = dict(cell_delta_ge_minus_2pp=result['delta']>=-.02-1e-12,
                every_task_delta_ge_minus_5pp=all(t['delta']>=-.05-1e-12 for t in result['tasks'].values()),
                paired_lower95_ge_minus_5pp=result['lower95']>=-.05-1e-12,
                saving_ge_frozen_target=result['saving'] >= (.25 if level=='off' else .15))
            result['pass'] = all(result['checks'].values())
            counterfactual = (config['epsilon']*current['totals']['T']+current['totals']['M'])/current['totals']['N']
            result['fixed_current_counts_ir'] = counterfactual
            result['fixed_current_counts_saving'] = 1-counterfactual/current['ir']
            result['ir_penalty_from_m_over_n'] = light['m_over_n']-current['m_over_n']
            result['ir_penalty_from_look_rate'] = config['epsilon']*(light['totals']['Q']/light['totals']['N']-current['totals']['T']/current['totals']['N'])
            assert math.isclose(light['ir']-counterfactual,result['ir_penalty_from_m_over_n']+result['ir_penalty_from_look_rate'],abs_tol=1e-12)
            comparisons[key] = result
    original=js(HERE/'FINAL_MANIFEST.json')
    immutable={**original['files'],**freeze['files']}
    for path,digest in SOURCES.items():
        if path in immutable:assert digest==immutable[path]['sha256'],path
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(),cpu_only=True,run_roots_read_only=True,
        gates_sha256=GATES_SHA,gate_changes=False,owner_disposition='D5: appendix/ablation, not main design',
        paired_bound_method='L=CP_low(g,n,.025)-CP_high(l,n,.025); U=CP_high(g,n,.025)-CP_low(l,n,.025); each endpoint is a conservative one-sided95 bound; gate uses L',
        arms=arms,comparisons=comparisons,offline=offline,input_sha256=SOURCES)
    payload['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    render(payload)
    (HERE/'PILOT_RESULTS.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    rows = [e for pair_ in arms.values() for a in pair_.values() for e in a['episodes']]
    with (HERE/'pilot_results_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('COMPARISONS',json.dumps({k:{f:x[f] for f in ('delta','lower95','saving','checks','pass')} for k,x in comparisons.items()}),flush=True)


def render(data):
    def bounds(x):return f"[{100*x['lower95']:+.1f}, {100*x['upper95']:+.1f}]"
    def yn(x):return 'PASS' if x else 'FAIL'
    p = ['# R15-B appendix: replacing the vision tower with a light image key', '',
        'The light key remains an ablation/appendix comparison, following [owner decision D5](../../COORDINATOR_DECISIONS.md). It uses images and removes tower work from cache hits, but loses closed-loop success and misses the frozen joint accuracy/savings criteria. The vision-tower key remains the main design. No fitting, model changes, new rollouts or GPU work were performed for this analysis.', '',
        'Both students use current agentview/wrist images, normalized robot state and task identity; training uses only the exact served library. Both library and query use the student features, followed by the same per-task action-supervised AWM, early metric and 16-NN kernel. Round 2 selected the augmented state-dropout recipe using two within-library episode hold-outs (40 epochs at size 50; 120 at size 200).', '',
        '## Offline progression', '',
        '| Library | Student round | Library donor-LOEO error ratio | Unseen B-dev error excess | Shuffle / black-image error rise | CPU key median / p95 (ms) |',
        '|---|---|---:|---:|---:|---:|']
    for a in data['offline']:
        p.append(f"| {a['size']} | {a['round']} | {a['library_ratio']:.4f} | {100*(a['bdev_ratio']-1):+.2f}% | {100*a['shuffle_rise']:.1f}% / {100*a['zero_rise']:.1f}% | {a['median_ms']:.3f} / {a['p95_ms']:.3f} |")
    p += ['', 'Error is equal-episode/equal-task RMS10 relative to the deployed tower key. B-dev contains 100 unseen episodes / 3,198 looks at size 50 and 73 / 2,289 at size 200, from existing size-50 trajectories. Both rounds pass the image-use audit (state held fixed); state-only is rejected. Image sensitivity does not imply adequate retrieval accuracy: both final students fail the unchanged B-dev gate. Local one-thread timing includes image conversion and AWM code, excludes common retrieval, and is not H100 serving latency. The frozen graph-basis median budget is 1.026 ms: round-2 size 50 narrowly fails; size 200 passes locally. [Full offline analysis](PROTOTYPE2.md).', '',
        '## Paired closed-loop success', '',
        'All eight arms contain 300 accepted, error-free episodes, 30/task, with identical task/B-init pairs from the R15 list outside the 200-episode subset. Six arms were newly collected; the two size-200 current arms reuse descent runs under the [baseline equivalence proof](../impl/baseline_equivalence.json). Accepted attempts, contiguous server decisions, client verdicts, episode-end counts and summaries were independently reconciled. This is reused B development data, not independent validation.', '',
        '| Library | Level | Current successes | Light successes | Δ SR (pp) | Paired gains / losses | Conservative paired bounds (pp) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,x in data['comparisons'].items():
        size,level=key.split('_');p.append(f"| {size} | {level} | {x['current']}/300 ({100*x['current']/300:.2f}%) | {x['light']}/300 ({100*x['light']/300:.2f}%) | {100*x['delta']:+.2f} | {x['gains']} / {x['losses']} | {bounds(x)} |")
    p += ['', 'Δ is light minus current. The frozen lower bound is `L = CP_low(g, n, .025) − CP_high(l, n, .025)`; the displayed upper companion reverses the bounds. Each endpoint is a conservative one-sided 95% bound; the displayed pair is not claimed as a simultaneous two-sided 95% interval. The gate uses L. The machine-readable results also include a simultaneous two-sided 95% Bonferroni interval using .0125 tails. Pairing controls task/init identity; process-seeded stochastic rollout order is not an identical-noise guarantee.', '',
        'Per-task Δ SR and the same conservative paired bounds (pp; 30 paired episodes/task):', '',
        '| Task | 50 off | 50 ir25 | 200 off | 200 ir25 |',
        '|---|---:|---:|---:|---:|']
    for t in range(10):
        cells=[]
        for key in ('50_off','50_ir25','200_off','200_ir25'):
            x=data['comparisons'][key]['tasks'][str(t)];cells.append(f"{100*x['delta']:+.1f} {bounds(x)}")
        p.append(f"| {t} | "+' | '.join(cells)+' |')
    p += ['', '## Actual work and modeled cost', '',
        'Compute from accepted decision logs, without repricing: `IR_current=(.152 V+.848 M)/N`; `IR_light=(ε Q+M)/N`. V is current-system tower calls, Q student calls, M full-policy calls, N five-control requests. The frozen ε values are '+', '.join(f"{data['arms'][f'{s}_off']['light']['epsilon']:.8f} (size {s})" for s in (50,200))+'. IR uses actual work counts with frozen cost coefficients; it is not measured wall time.', '',
        '| Library | Level | IR current → light | Relative IR saving | M/N current → light | Executed controls/episode current → light | Cost/episode current → light |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,a in data['arms'].items():
        c,l=a['current'],a['light'];x=data['comparisons'][key]
        p.append(f"| {c['size']} | {c['level']} | {c['ir']:.5f} → {l['ir']:.5f} | {100*x['saving']:.2f}% | {100*c['m_over_n']:.2f}% → {100*l['m_over_n']:.2f}% | {c['executed_controls_per_episode']:.2f} → {l['executed_controls_per_episode']:.2f} | {c['cost_per_episode']:.3f} → {l['cost_per_episode']:.3f} |")
    p += ['', 'Current-arm summary IR fields use the generic eager stage-cost ratio, not the owner coefficients; the table recomputes the required .152/.848 formula from audited V/M/N. Light summaries reproduce the frozen-epsilon formula exactly. Cost/episode is in full-policy-call equivalents. Executed controls come from accepted `client_timing.steps`, subtracting the configured warm-up; every episode satisfies `5(N−1) < controls ≤ 5N`, retaining its final partial chunk. Request counts remain the frozen IR denominator. Longer light-key episodes weaken savings per episode further; failure/termination is not credited as useful efficiency.', '',
        '| Library | Level | No-progress guard calls current → light | Guard calls / 100 requests current → light | Scheduled calls current → light | Saving with current counts fixed → actual |',
        '|---|---|---:|---:|---:|---:|']
    for key,a in data['arms'].items():
        c,l=a['current'],a['light'];x=data['comparisons'][key]
        p.append(f"| {c['size']} | {c['level']} | {c['totals']['guard']} → {l['totals']['guard']} | {100*c['guard_over_n']:.2f} → {100*l['guard_over_n']:.2f} | {c['totals']['knob']} → {l['totals']['knob']} | {100*x['fixed_current_counts_saving']:.2f}% → {100*x['saving']:.2f}% |")
    p += ['', 'Every guard MISS is logged as `force:4` (no progress); every additional scheduled MISS is `force:62` (periodic knob). Guard calls take priority, so more guard calls partially replace scheduled calls at ir25. The observed mechanism is consistent with the offline retrieval deficit: less reliable retrieved actions/progress tracking → more no-progress fallbacks and longer episodes → higher M/N, consuming the tower-skipping benefit. The decomposition is exact: relative to holding current counts fixed, the IR penalty is the increase in M/N plus ε times the change in look rate. It does not identify whether each individual guard was caused by a wrong progress estimate or a physically poor action; the trajectories differ.', '',
        '## Tower-skipping audit and frozen verdict', '',
        '| Library | Level | Current N / V / M | Light N / Q / T / M | Light tower-on-HIT violations |',
        '|---|---|---:|---:|---:|']
    for key,a in data['arms'].items():
        c,l=a['current'],a['light'];ct,lt=c['totals'],l['totals']
        p.append(f"| {c['size']} | {c['level']} | {ct['N']} / {ct['T']} / {ct['M']} | {lt['N']} / {lt['Q']} / {lt['T']} / {lt['M']} | {l['audit']['light_tower_on_hit_violations']} |")
    p += ['', 'Every light decision satisfies logged T=M, with exactly one tower/full-policy call on each MISS and none on any HIT. Legal blind continuations and policy tails have Q=T=M=0. Per-request work, cumulative episode counters, terminal totals and summary ledgers agree; model hashes and fixed ir25 settings match the frozen artifacts. The implementation achieves tower skipping; the retrieval/SR trade-off causes the negative design result.', '',
        '| Library | Level | Δ SR ≥−2 pp | Every task ≥−5 pp | Paired L ≥−5 pp | IR saving ≥25% off / ≥15% ir25 | Joint result |',
        '|---|---|---|---|---|---|---|']
    for key,x in data['comparisons'].items():
        size,level=key.split('_');p.append(f"| {size} | {level} | "+' | '.join(yn(v) for v in x['checks'].values())+f" | {yn(x['pass'])} |")
    runtime=[a['light']['encoder_ms'] for a in data['arms'].values()]
    p += ['', f"Logged H100-host encoder-only medians are {min(x['median'] for x in runtime):.2f}–{max(x['median'] for x in runtime):.2f} ms (p95 {min(x['p95'] for x in runtime):.2f}–{max(x['p95'] for x in runtime):.2f} ms). These runtime timings exclude the subsequent AWM projection and do not certify the frozen end-to-end deployment cost gate; all exceed its historical graph-basis sensitivity budgets. The primary IR comparison retains the frozen epsilon rather than substituting new timing prices after the outcomes."]
    p += ['', 'The boundary Δ=−2 pp is a pass for that individual criterion; it does not override task or paired-bound failures. No cell/setting is promoted by pooling away another failure. The separate frozen R14 task-table saving criterion remains a labeled projection in PROTOTYPE2.md, not an outcome of these off/ir25 arms. GATES.md and PILOT.md are hash-verified unchanged. The conclusion supports D5: retain the vision tower in the main design and report this light, image-sensitive student as the unsuccessful comparison.', '',
        'Reproduce from the repository root:', '',
        '```bash',
        'taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r15/qk/astra/pilot_results.py',
        '```', '',
        'Sources and exact results: [PILOT_RESULTS.json](PILOT_RESULTS.json) (all input paths/SHA256, checks, offline values, counts and intervals); [paired episode rows](pilot_results_episodes.csv); [analysis script](pilot_results.py). Only the eight named B-dev arms were read. No test-A data or run-root writes.']
    (HERE/'PILOT_RESULTS.md').write_text('\n'.join(p)+'\n')


if __name__ == '__main__':
    main()
