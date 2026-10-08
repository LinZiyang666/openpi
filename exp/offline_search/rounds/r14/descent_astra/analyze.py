"""Frozen B-only descent analysis. Default is protocol verification, never outcomes.

Only the explicit `apply` subcommand opens descent roots. No launches or test-A IO.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, kendalltau, rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R14 = HERE.parent
ORDER = ('c10', 'ir45', 'ir40', 'ir32', 'ir25', 'off')
ORDINAL = {level: 5-i for i, level in enumerate(ORDER)}
ARMS = (*ORDER, 'task_ir32', 'task_ir40')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Reader:
    """Hash the bytes actually used and recheck before any result write."""
    def __init__(self):
        self.inputs = {}
        self.shards = {}

    def read(self, path):
        path = Path(path).resolve()
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f'Input changed during read: {path}')
        record = dict(sha256=sha(data), bytes=len(data), mtime_ns=after.st_mtime_ns)
        require(str(path) not in self.inputs or record == self.inputs[str(path)],
                f'Input changed between reads: {path}')
        self.inputs[str(path)] = record
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def jsonl(self, path):
        for number, line in enumerate(self.read(path).splitlines(), 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except (ValueError, UnicodeDecodeError) as e:
                    raise ValueError(f'Malformed JSONL {path}:{number}') from e

    def stable(self):
        for path, record in list(self.inputs.items()):
            self.read(path)
        for pattern, expected in self.shards.items():
            require([str(x) for x in sorted(Path(pattern).glob('server_*/decisions_*.jsonl'))] == expected,
                    f'Decision shard set changed: {pattern}')


def verify_protocol(reader):
    seal = reader.read(HERE / 'PROTOCOL.sha256').decode()
    require(len(seal.splitlines()) == 2, 'Invalid protocol seal')
    require({line.split()[1] for line in seal.splitlines()} == {'protocol.json','PROTOCOL.md'}, 'Bad sealed names')
    for line in seal.splitlines():
        digest, name = line.split()
        require(sha(reader.read(HERE/name)) == digest, f'Frozen protocol changed: {name}')
    p = reader.json(HERE/'protocol.json')
    require(p['pre_outcome'] and p['outcome_reads_before_freeze'] == 0, 'Missing pre-outcome registration')
    require(p['selection_rule'] == 'point_estimate_first_failure' and tuple(p['descent_order']) == ORDER,
            'Unsupported registered rule')
    for path, digest in p['input_sha256'].items():
        require(sha(reader.read(REPO/path)) == digest, f'Frozen input changed: {path}')
    return p


def cp_low(k, n, tail):
    return 0. if k == 0 else float(beta.ppf(tail, k, n-k+1))


def cp_high(k, n, tail):
    return 1. if k == n else float(beta.ppf(1-tail, k+1, n-k))


def paired(y, reference, family=200, identical=False):
    y, reference = np.asarray(y), np.asarray(reference)
    require(y.shape == reference.shape and y.size > 0, 'Unpaired outcomes')
    require(np.isin(y,[0,1]).all() and np.isin(reference,[0,1]).all(), 'Nonbinary SR')
    n = y.size
    gains = int(((y == 1) & (reference == 0)).sum())
    losses = int(((y == 0) & (reference == 1)).sum())
    def lower(tail):
        return max(-1., cp_low(gains,n,tail)-cp_high(losses,n,tail))
    lo = lower(.05/4)
    hi = min(1.,cp_high(gains,n,.05/4)-cp_low(losses,n,.05/4))
    return dict(n=n, successes=int(y.sum()), pure_successes=int(reference.sum()),
                sr=float(y.mean()), pure_sr=float(reference.mean()), gains=gains, losses=losses,
                delta=(gains-losses)/n, lower95=0. if identical else lower(.05/2),
                simultaneous_lower95=0. if identical else lower(.05/(2*family)),
                ci95=[0.,0.] if identical else [lo,hi])


def cell_pair(y, reference, family=200, identical=False):
    require(np.shape(y) == (10,30) and np.shape(reference) == (10,30), 'Expected balanced 10 x 30')
    strata = [paired(a,b,family,identical) for a,b in zip(y,reference,strict=True)]
    out = {k:sum(s[k] for s in strata) for k in ('n','successes','pure_successes','gains','losses')}
    out.update({k:float(np.mean([s[k] for s in strata])) for k in ('sr','pure_sr','delta','lower95','simultaneous_lower95')})
    # An average of marginal intervals is descriptive, not a cell 95% interval.
    # Use Bonferroni over the ten task strata for the cell's marginal 95% bound/CI.
    def combined(tail, upper=False):
        return float(np.mean([(cp_high(s['gains'],30,tail)-cp_low(s['losses'],30,tail)) if upper else
                              (cp_low(s['gains'],30,tail)-cp_high(s['losses'],30,tail)) for s in strata]))
    out['lower95'] = 0. if identical else combined(.05/(2*10))
    out['ci95'] = [0.,0.] if identical else [combined(.05/(4*10)),combined(.05/(4*10),True)]
    return out


def descend(stats, tolerance, bound=False):
    selected, stopped = 'c10', None
    passing = {}
    for level in ORDER:
        row = stats[level]
        passing[level] = (row['simultaneous_lower95'] >= -tolerance/100 - 1e-12) if bound else (
            100 * (row['successes']-row['pure_successes']) >= -tolerance*row['n'])
    for level in ORDER[1:]:
        if not passing[level]:
            stopped = level
            break
        selected = level
    below = ORDER[ORDER.index(stopped)+1:] if stopped else ()
    return dict(level=selected, first_failure=stopped, passes=passing,
                passing_below_first_failure=[l for l in below if passing[l]],
                sr_increases_on_descent=[b for a,b in zip(ORDER,ORDER[1:]) if stats[b]['sr']>stats[a]['sr']])


def arm_name(cell, level):
    suffix = ('task_periodic_'+level[5:] if level.startswith('task_') else
              'periodic_'+level if level.startswith('ir') else 'pure_c10' if level=='c10' else level)
    return f'r14d_{cell}_{suffix}'


def uid_pair(uid):
    parts = uid.split(':')
    require(len(parts) >= 3, f'Invalid UID: {uid}')
    return tuple(map(int,parts[-2:]))


def accepted_rows(rows, expected):
    by_uid = {}
    errors = 0
    for row in rows:
        require(row.get('init_pool') == 'B', 'Non-B or unlabelled journal row')
        errors += bool(row.get('error'))
        if row.get('task_uid'):
            require(uid_pair(row['task_uid']) in expected, 'Unexpected journal identity')
        if row.get('accepted') and row.get('status') in ('done','failed') and not row.get('error'):
            require(type(row.get('success')) is bool, 'Success must be explicitly boolean')
            by_uid[row['task_uid']] = row
    pairs = [uid_pair(u) for u in by_uid]
    require(len(set(pairs)) == len(pairs), 'Multiple UIDs for one identity')
    require(set(pairs) == set(expected), f'Incomplete/extra accepted identities: {len(pairs)} / {len(expected)}')
    return by_uid, errors


@dataclass
class Episodes:
    y: np.ndarray
    n: np.ndarray
    v: np.ndarray
    m: np.ndarray
    cost: np.ndarray
    audit: dict

    def stats(self):
        return dict(sr=float(self.y.mean()),successes=int(self.y.sum()),n_episodes=self.y.size,
                    ir=float(self.cost.sum()/self.n.sum()),cost_per_episode=float(self.cost.mean()),
                    decisions=int(self.n.sum()),vision=int(self.v.sum()),misses=int(self.m.sum()))


def load_cell(root, cell, p, selection, caps, reader):
    """Read only a caller-specified B root; production caller uses frozen exact paths."""
    suite = cell.split('_')[1]
    expected = sorted(map(tuple,selection['suites'][suite]['pairs']))
    require(len(expected)==300 and len(set(expected))==300, 'Selection must contain exactly 300 identities')
    require(all(sum(t==j for t,i in expected)==30 for j in range(10)), 'Unbalanced selection')
    seal = reader.json(root/'ROOT.json')
    require(seal['init_pool']=='B' and seal['cell']==cell and seal['episodes_per_arm']==300 and seal['arm_count']==8,
            'Wrong sealed cell/pool/design')
    for name,digest in seal['definitions_sha256'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()), 'Definition outside root')
        require(sha(reader.read(path))==digest, f'Changed root definition: {path}')
    manifest = reader.json(root/'dev_selection.json')
    require(manifest['init_pool']=='B' and manifest['dev'] is True, 'Not a B manifest')
    require(sorted(map(tuple,manifest['pairs']))==expected, 'Root selection differs from freeze')
    require(manifest['selection_sha256']==p['input_sha256'][str((R14/'descent/selection.json').relative_to(REPO))],
            'Root uses another selection registration')
    # devset.arm_contract binds load_manifest's canonical selected-identity hash,
    # not the file-byte hash (which ROOT.json binds separately above).
    manifest_digest = sha(json.dumps(expected,separators=(',',':')).encode())
    rows = reader.json(root/'arms.json')
    contracts = reader.json(root/'dev_contracts.json')
    require({r['arm'] for r in rows}=={arm_name(cell,l) for l in ARMS} and len(rows)==8, 'Wrong arm set')
    result = {}
    for level in ARMS:
        name = arm_name(cell,level)
        row = next(r for r in rows if r['arm']==name)
        require(row.get('dev') is True and row.get('init_pool')=='B' and row.get('r14_descent_level')==level,
                'Wrong arm pool/level')
        require(row['episodes']==300 and row['model']==cell.split('_')[0], 'Wrong arm size/model')
        run = root/'runs'/name
        contract = reader.json(run/'pool_contract.json')
        require(contract==contracts[name] and contract['init_pool']=='B' and contract['dev'] is True, 'Changed pool contract')
        require(contract['manifest_sha256']==manifest_digest, 'Wrong contract manifest')
        require(Path(row['manifest']).resolve()==(root/'dev_selection.json').resolve(), 'Wrong arm manifest path')
        selected = reader.json(run/'selection.json')
        require(selected['init_pool']=='B' and
                Path(selected['manifest']).resolve()==(run/'manifest.json').resolve(), 'Wrong run selection binding')
        require(reader.read(run/'manifest.json')==reader.read(root/'dev_selection.json'),
                'Run manifest differs from sealed root selection')
        fit_digest = caps['cells'][cell]['fit_sha256_by_level'][level]
        args = row['plugin_args']
        fit_path = Path(args[args.index('--os-fit-artifact')+1])
        require(contract['fit_sha256']==fit_digest and seal['fit_sha256'][str(fit_path)]==fit_digest, 'Wrong frozen fit')
        require(sha(reader.read(fit_path))==fit_digest, 'Fit bytes changed')
        excluded = contract['subset']['excluded_inits_by_task']
        require(all(i not in excluded[str(t)] for t,i in expected), 'Selected init in served subset')
        require(contract['subset']==selection['suites'][suite]['subsets'][row['model']], 'Subset differs from selection')
        accepted, errors = accepted_rows(reader.jsonl(run/'client/journal.jsonl'),expected)
        require(all(uid.split(':')[0]==name for uid in accepted), 'Journal UID belongs to another arm')
        summary = reader.json(run/'summary.json')
        require(summary['init_pool']=='B' and summary['arm']==name, 'Wrong summary pool/arm')
        require(summary['complete']==300 and summary['success']==sum(r['success'] for r in accepted.values()), 'Summary/journal mismatch')
        by_pair = {uid_pair(u):r for u,r in accepted.items()}
        positions = {pair:i for i,pair in enumerate(expected)}
        decs, duplicates, rejected = {},0,0
        shards = sorted(run.glob('server_*/decisions_*.jsonl'))
        require(shards, 'Missing decision logs')
        reader.shards[str(run)] = list(map(str,shards))
        for path in shards:
            for d in reader.jsonl(path):
                if d.get('ev')!='dec':
                    continue
                require(d.get('init_pool','B')=='B', 'Non-B decision')
                uid = d['uid']
                require(uid_pair(uid) in positions, 'Unexpected decision identity')
                if uid not in accepted or int(d.get('attempt',1) or 1)!=int(accepted[uid].get('attempt',1) or 1):
                    rejected += 1
                    continue
                require(type(d['step']) is int and d['step']>=0, 'Invalid decision step')
                key = (uid,d['step'])
                duplicates += key in decs
                decs[key] = d
        n,v,m = (np.zeros(300,dtype=np.int64) for _ in range(3))
        steps = {u:[] for u in accepted}
        ledger = summary['cost_ledger']
        full = ledger['full_stage_costs']
        require(full['k']>0, 'Invalid full-policy length')
        for (uid,step),d in decs.items():
            j = positions[uid_pair(uid)]
            require(type(d.get('hit')) is bool and type(d.get('vision')) is bool, 'Need explicit boolean hit and vision')
            require(d.get('stage1_mode','full')=='full' and d.get('replan_steps',5)==5, 'Unsupported cost mode or request length')
            require(d['hit'] or d['vision'], 'Policy miss without vision')
            require(d['hit'] or float(d['miss_k'])==float(full['k']), 'Partial-policy call cannot use registered IR')
            require(d.get('ok',True), 'Invalid accepted decision')
            n[j] += 1
            v[j] += d['vision']
            m[j] += not d['hit']
            steps[uid].append(step)
        require(all(sorted(s)==list(range(len(s))) and s for s in steps.values()), 'Missing/noncontiguous decision steps')
        require([int(n.sum()),int(v.sum()),int(m.sum())]==[ledger[k] for k in ('decisions','vision_decisions','misses')],
                'Decision counts differ from summary ledger')
        require(ledger['controls']==int(n.sum())*5, 'Wrong nominal controls')
        measured = (full['s1']*v.sum()+(full['s2']+full['s3'])*m.sum())/(sum(full[k] for k in ('s1','s2','s3'))*n.sum())
        require(math.isclose(measured,ledger['ir_per_five_controls'],abs_tol=1e-8), 'Stage IR does not reconcile')
        y = np.array([by_pair[k]['success'] for k in expected],dtype=np.int64).reshape(10,30)
        if 'per_task' in summary:
            require(all(summary['per_task'][str(t)]=={'n':30,'success':int(y[t].sum())} for t in range(10)), 'Task summary mismatch')
        cv,cm = p['cost']['coefficients'][row['model']]
        result[level] = Episodes(y,n.reshape(10,30),v.reshape(10,30),m.reshape(10,30),
            (cv*v+cm*m).reshape(10,30),
            dict(arm=name,summary_stage_ir=ledger['ir_per_five_controls'],full_stage_costs=full,
                 duplicate_decisions=duplicates,rejected_attempt_decisions=rejected,error_rows=errors,
                 retried_episodes=sum(int(r.get('attempt',1) or 1)>1 for r in accepted.values())))
    return result


def stitch(arms, levels):
    arrays = [np.stack([getattr(arms[levels[str(t)]],field)[t] for t in range(10)]) for field in ('y','n','v','m','cost')]
    return Episodes(*arrays,dict(stitched=True,levels=levels))


def bootstrap_indices(p):
    return np.random.default_rng(p['bootstrap']['seed']).integers(0,30,size=(p['bootstrap']['replicates'],10,30))


def sampled_sum(a, indices):
    return a[np.arange(10)[None,:,None],indices].sum(axis=(1,2))


def interval(a):
    return [float(x) for x in np.quantile(a,[.025,.975])]


def savings(task, uniform, pure, indices):
    ts,us = task.stats(),uniform.stats()
    ti = sampled_sum(task.cost,indices)/sampled_sum(task.n,indices)
    ui = sampled_sum(uniform.cost,indices)/sampled_sum(uniform.n,indices)
    delta = sampled_sum(task.y-uniform.y,indices)/300
    task_comparison=cell_pair(task.y,pure.y,identical=all(l=='c10' for l in task.audit['levels'].values()))
    # All task-level comparisons share the registered simultaneous family, including selection.
    task_comparison['simultaneous_lower95']=float(np.mean([
        paired(task.y[t],pure.y[t],identical=task.audit['levels'][str(t)]=='c10')['simultaneous_lower95'] for t in range(10)]))
    return dict(task=ts,uniform=us,task_vs_pure=task_comparison,uniform_vs_pure=cell_pair(uniform.y,pure.y,identical=uniform is pure),
                ir_saved=us['ir']-ts['ir'],relative_ir_saved=(us['ir']-ts['ir'])/us['ir'],
                ir_saved_conditional_ci95=interval(ui-ti),relative_saved_conditional_ci95=interval((ui-ti)/ui),
                task_minus_uniform_sr=ts['sr']-us['sr'],sr_delta_conditional_ci95=interval(delta))


def nearest_level(cap, uniform_caps):
    require(math.isfinite(cap) and cap>=0, 'Invalid allocation cap')
    distances={level:abs(math.log1p(cap)-math.log1p(uniform_caps[level])) for level in ORDER[:-1]}
    closest=min(distances.values())
    return max((l for l,d in distances.items() if math.isclose(d,closest,rel_tol=0,abs_tol=1e-12)),key=ORDINAL.get)


def ranks(x,y):
    if len(set(x))<2 or len(set(y))<2:
        return dict(spearman=None,kendall_tau_b=None,reason='constant priority or selected levels')
    return dict(spearman=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]),
                kendall_tau_b=float(kendalltau(x,y,variant='b').statistic),reason=None)


def library_agreement(cell, levels, caps, fits, csv_rows):
    out = {}
    actual = [ORDINAL[levels[str(t)]] for t in range(10)]
    c = caps['cells'][cell]
    for dose in ('ir32','ir40'):
        fit = next(f for f in fits if f['cell']==cell and f['target']==int(dose[-2:])/100)
        alloc = c['frozen_library_allocation'][dose]
        require(fit['K']==alloc['K'] and fit['priority']['z']==alloc['priority'] and fit['reference_K']==alloc['reference_K'],
                'Allocation/cap snapshot mismatch')
        rows = [r for r in csv_rows if r['cell']==cell and float(r['target'])==fit['target']]
        require(len(rows)==10 and {int(r['task']) for r in rows}==set(range(10)), 'Missing library task rows')
        for row in rows:
            t=int(row['task'])
            require(float(row['K'])==fit['K'][t] and float(row['priority'])==fit['priority']['z'][t], 'CSV/fit disagreement')
        predicted = [nearest_level(k,c['periodic_cap_by_level']) for k in fit['K']]
        error = np.array([ORDINAL[l] for l in predicted])-actual
        confusion = {a:{b:0 for b in reversed(ORDER)} for a in reversed(ORDER)}
        for t,l in enumerate(predicted):
            confusion[l][levels[str(t)]]+=1
        out[dose]=dict(**ranks(fit['priority']['z'],actual),predicted_levels={str(t):l for t,l in enumerate(predicted)},
                       priority=fit['priority']['z'],caps=fit['K'],exact_fraction=float(np.mean(error==0)),
                       within_one_fraction=float(np.mean(np.abs(error)<=1)),mean_absolute_ordinal_error=float(np.mean(np.abs(error))),
                       mean_signed_ordinal_error=float(np.mean(error)),confusion_predicted_rows_actual_columns=confusion)
    return out


def matched_ir(arms, dose, p, indices):
    allocation,uniform = arms['task_'+dose],arms[dose]
    a,u = allocation.stats(),uniform.stats()
    gap = a['ir']-u['ir']
    out = dict(target=dose,allocation=a,uniform=u,actual_ir_gap=gap,
               approximately_matched=abs(gap)<=p['matched_ir']['max_abs_ir_gap'],
               paired_sr=cell_pair(allocation.y,uniform.y),
               per_task=[paired(allocation.y[t],uniform.y[t],family=1) for t in range(10)])
    # The registered simultaneous family covers descent levels, not these separate diagnostics.
    for row in [out['paired_sr'],*out['per_task']]:
        row.pop('simultaneous_lower95')
    # Duplicate achieved IRs retain the higher inference level, predeclared tie break.
    periodic = sorted(('ir25','ir32','ir40','ir45'),key=lambda l:(arms[l].stats()['ir'],-ORDINAL[l]))
    unique=[]
    for level in periodic:
        if not unique or arms[level].stats()['ir']!=arms[unique[-1]].stats()['ir']:
            unique.append(level)
    target = a['ir']
    equal = next((l for l in unique if arms[l].stats()['ir']==target),None)
    if equal:
        low=high=equal
        weight=0.
    else:
        brackets=[(l,h) for l,h in zip(unique,unique[1:]) if arms[l].stats()['ir']<target<arms[h].stats()['ir']]
        if not brackets:
            out['exact_ir_expected_mixture']=dict(available=False,reason='No periodic IR bracket; no extrapolation')
            return out
        low,high=brackets[0]
        lo,hi=arms[low],arms[high]
        weight=float((target*lo.n.sum()-lo.cost.sum())/((hi.cost.sum()-lo.cost.sum())-target*(hi.n.sum()-lo.n.sum())))
    lo,hi=arms[low],arms[high]
    y=(1-weight)*lo.y+weight*hi.y
    cost=(1-weight)*lo.cost+weight*hi.cost
    n=(1-weight)*lo.n+weight*hi.n
    require(0<=weight<=1 and math.isclose(float(cost.sum()/n.sum()),target,abs_tol=1e-12), 'Mixture IR mismatch')
    out['exact_ir_expected_mixture']=dict(available=True,low=low,high=high,high_episode_weight=weight,
        ir=float(cost.sum()/n.sum()),expected_uniform_sr=float(y.mean()),allocation_sr=a['sr'],
        allocation_minus_expected_uniform_sr=float((allocation.y-y).mean()),
        paired_conditional_ci95=interval(sampled_sum(allocation.y-y,indices)/300),
        caveat='Expected randomized episode mixture; not an observed deterministic uniform arm; fixed weights in bootstrap')
    return out


def analyze_cells(all_arms,p,caps,fits,csv_rows):
    indices=bootstrap_indices(p)
    result=dict(protocol_frozen_at_utc=p['frozen_at_utc'],primary_tolerance_pp=p['primary_tolerance_pp'],cells={})
    for cell in p['cells']:
        arms=all_arms[cell]
        pure=arms['c10']
        tasks={str(t):{level:paired(arms[level].y[t],pure.y[t],identical=level=='c10') for level in ORDER} for t in range(10)}
        pooled={level:cell_pair(arms[level].y,pure.y,identical=level=='c10') for level in ORDER}
        c=dict(arm_stats={l:dict(**a.stats(),**a.audit) for l,a in arms.items()},task_stats=tasks,cell_stats=pooled,
               tolerances={},matched_ir={dose:matched_ir(arms,dose,p,indices) for dose in ('ir32','ir40')})
        for tol in p['tolerances_pp']:
            decisions={str(t):descend(tasks[str(t)],tol) for t in range(10)}
            levels={t:d['level'] for t,d in decisions.items()}
            uniform=descend(pooled,tol)
            selected=stitch(arms,levels)
            c['tolerances'][str(tol)]=dict(levels=levels,uniform=uniform['level'],task_decisions=decisions,
                uniform_decision=uniform,savings=savings(selected,arms[uniform['level']],pure,indices),
                library=library_agreement(cell,levels,caps,fits,csv_rows),
                confidence_sensitivity=dict(levels={t:descend(s,tol,True)['level'] for t,s in tasks.items()},
                                            uniform=descend(pooled,tol,True)['level']))
        result['cells'][cell]=c
    return result


def number(x,scale=1,digits=3):
    return 'NA' if x is None else f'{x*scale:.{digits}f}'


def cell_name(cell):
    model,suite,_=cell.split('_')
    return ('π0.5' if model=='pi05' else 'GR00T')+' / '+('长程任务' if suite=='l10' else '空间任务')


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def report(result,p):
    tol=str(p['primary_tolerance_pp'])
    lines=['# 给负责人的结果说明','',
        '主规则：每个任务从纯推理逐级降低，在第一次低于纯推理成功率减 5 个百分点时停止，保留上一档。每任务使用相同的 30 个初始状态。以下是开发集上的选档结果。', '',
        '档位从低到高为 off、ir25、ir32、ir40、ir45、c10；c10 表示纯推理。ir 数字是档位名称，实际算力比例见后表。','']
    rows=[]
    for cell,c in result['cells'].items():
        r=c['tolerances'][tol]
        rows.append([cell_name(cell)]+[r['levels'][str(t)] for t in range(10)]+[r['uniform']])
    lines += [markdown_table(['模型 / 任务组']+[f'任务 {t}' for t in range(10)]+['统一档位'],rows),
        '统一档位也按同一容差逐级下降，但它检查整个任务组的平均成功率，允许任务间相互抵消。因此两种选法不保证相同的实测成功率。','']
    rows=[]
    for cell,c in result['cells'].items():
        s=c['tolerances'][tol]['savings']
        rows.append([cell_name(cell),number(c['arm_stats']['c10']['sr'],100,1),number(s['task']['sr'],100,1),number(s['uniform']['sr'],100,1),
                     number(s['task']['ir']),number(s['uniform']['ir']),number(s['ir_saved'],100,2),number(s['relative_ir_saved'],100,1)])
    lines += [markdown_table(['模型 / 任务组','纯推理成功率 %','逐任务成功率 %','统一成功率 %','逐任务 IR','统一 IR','节省 IR 百分点','相对节省 %'],rows),
        'IR 按冻结的原始成本系数计算，表示归一化计算开销。正数表示节省，负数表示开销增加。逐任务结果由所选档位的完整任务轨迹拼合，并非已经独立运行的混合策略。选档与评估使用同一开发集，节省和成功率可能偏乐观。','',
        '库估计是否吻合：下表同时给出优先级与所需档位的排序相关性，以及原有预算分配对应档位的吻合比例。相关性为正表示方向一致；NA 表示档位全相同等情况，无法计算相关性。原有分配不是成功率边界预测，因此不能只凭其中一项宣布“预测正确”。','']
    rows=[]
    for cell,c in result['cells'].items():
        for dose,l in c['tolerances'][tol]['library'].items():
            rows.append([cell_name(cell),dose,number(l['spearman']),number(l['kendall_tau_b']),number(l['exact_fraction'],100,0),number(l['within_one_fraction'],100,0),number(l['mean_absolute_ordinal_error'],1,2)])
    lines += [markdown_table(['模型 / 任务组','原预算','排序相关 ρ','排序相关 τ','档位相同 %','相差至多一档 %','平均差几档'],rows),
        '优先级来自库内状态；预算偏置曾使用较早的开发集状态轨迹校准。有限上限映射无法预测完全关闭档位。','',
        '原有逐任务分配与统一周期策略的比较如下。相同名义预算不保证实际开销相同，实际 IR 差在 1 个百分点以内才标为近似匹配。精确匹配一栏是相邻统一周期档位的期望混合，仅作估算。','']
    rows=[]
    for cell,c in result['cells'].items():
        for dose,m in c['matched_ir'].items():
            mix=m['exact_ir_expected_mixture']
            rows.append([cell_name(cell),dose,number(m['allocation']['ir']),number(m['uniform']['ir']),number(m['actual_ir_gap'],100,2),
                         '是' if m['approximately_matched'] else '否',number(m['paired_sr']['delta'],100,2),
                         number(mix.get('allocation_minus_expected_uniform_sr'),100,2)])
    lines += [markdown_table(['模型 / 任务组','原预算','逐任务 IR','统一 IR','IR 差 百分点','近似匹配','成功率差 百分点','精确 IR 期望混合成功率差 百分点'],rows),
        '每任务 30 次试验的区间很宽。0 与 2 个百分点容差的任务表相同；5 个百分点允许净少成功一次。点估计达标不等于已统计证明不劣。配对下界、首次失败后出现的回升、所有容差及置信规则敏感性见下文。开发集存在历史复用，本报告不使用测试集。','',
        '## Audit and alternate tolerances','',f"Frozen: `{p['frozen_at_utc']}`. Outcome inputs and code hashes are in `provenance.json`. All values are computed from accepted B episodes; no test-A evaluation was read or executed.",'']
    for tolerance in p['tolerances_pp']:
        lines += [f'### Tolerance {tolerance} pp','']
        rows=[]
        for cell,c in result['cells'].items():
            r=c['tolerances'][str(tolerance)]
            for t,d in r['task_decisions'].items():
                s=c['task_stats'][t][d['level']]
                rows.append([cell,t,d['level'],d['first_failure'] or 'none',','.join(d['passing_below_first_failure']) or 'none',
                             f"{s['gains']}/{s['losses']}",number(s['delta'],100,2),number(s['lower95'],100,2),number(s['simultaneous_lower95'],100,2),
                             r['confidence_sensitivity']['levels'][t]])
        lines += [markdown_table(['Cell','Task','Selected','First fail','Lower recovery ignored','Gain/loss','ΔSR pp','Lower95 pp','Simultaneous lower95 pp','Bound sensitivity'],rows)]
        rows=[]
        for cell,c in result['cells'].items():
            r=c['tolerances'][str(tolerance)];s=r['savings']
            ci=s['ir_saved_conditional_ci95']
            rows.append([cell,r['uniform'],r['confidence_sensitivity']['uniform'],number(s['ir_saved'],100,2),
                         f'[{number(ci[0],100,2)}, {number(ci[1],100,2)}]',number(s['task']['cost_per_episode']),number(s['uniform']['cost_per_episode'])])
        lines += [markdown_table(['Cell','Uniform','Bound uniform','IR saved pp','Conditional 95% interval pp','Task cost/episode','Uniform cost/episode'],rows)]
    lines += ['## Interpretation limits','',
        'The exact paired task intervals use discordant gains/losses and conservative Clopper–Pearson tails. Cell intervals are stratified with Bonferroni correction over tasks. The simultaneous lower band covers all 200 non-reference task/level comparisons under within-task sampling assumptions. The bound-based choices are diagnostics only. The full numerical record includes every tested level, confusion matrix, original stage-ledger IR and nominal allocation comparison interval.', '',
        'Savings and mixture bootstrap intervals condition on the selected levels and mixture weights; they omit selection uncertainty. The expected mixture is randomized across whole episodes, with the weight solved from total cost and total decision counts, not a linear interpolation of nominal IR labels. No mixture is extrapolated beyond observed periodic IRs. These development data do not establish independent deployment SR or rank predictiveness.','']
    return '\n'.join(lines)


def csv_text(rows):
    f=io.StringIO()
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    return f.getvalue()


def outputs(result,p):
    files={'DESCENT.md':report(result,p),'analysis.json':result}
    task_rows=[]; stat_rows=[]; library_rows=[]; matched_rows=[]
    for tolerance in p['tolerances_pp']:
        levels={cell:c['tolerances'][str(tolerance)]['levels'] for cell,c in result['cells'].items()}
        uniform={cell:c['tolerances'][str(tolerance)]['uniform'] for cell,c in result['cells'].items()}
        require(set(levels)==set(p['cells'])==set(uniform), 'Emitter cell mismatch')
        require(all(set(v)==set(map(str,range(10))) and all(l in ORDER for l in v.values()) for v in levels.values()), 'Bad emitter levels')
        files[f'levels_tol{tolerance}pp.json']=levels
        files[f'uniform_tol{tolerance}pp.json']=uniform
        if tolerance==p['primary_tolerance_pp']:
            files['levels.json']=levels;files['uniform.json']=uniform
        for cell,c in result['cells'].items():
            r=c['tolerances'][str(tolerance)]
            for task,d in r['task_decisions'].items():
                task_rows.append(dict(cell=cell,tolerance_pp=tolerance,task=task,level=d['level'],uniform=r['uniform'],
                    first_failure=d['first_failure'],recoveries_ignored=';'.join(d['passing_below_first_failure']),
                    confidence_level=r['confidence_sensitivity']['levels'][task]))
            for target,l in r['library'].items():
                library_rows.append(dict(cell=cell,tolerance_pp=tolerance,target=target,**{k:l[k] for k in
                    ('spearman','kendall_tau_b','exact_fraction','within_one_fraction','mean_absolute_ordinal_error','mean_signed_ordinal_error')}))
    for cell,c in result['cells'].items():
        for task,levels in {**c['task_stats'],'cell':c['cell_stats']}.items():
            for level,s in levels.items():
                stat_rows.append(dict(cell=cell,task=task,level=level,**{k:v for k,v in s.items() if k!='ci95'},ci95_low=s['ci95'][0],ci95_high=s['ci95'][1]))
        for dose,m in c['matched_ir'].items():
            mix=m['exact_ir_expected_mixture']
            matched_rows.append(dict(cell=cell,target=dose,allocation_ir=m['allocation']['ir'],uniform_ir=m['uniform']['ir'],
                ir_gap=m['actual_ir_gap'],approximately_matched=m['approximately_matched'],delta_sr=m['paired_sr']['delta'],
                delta_ci95_low=m['paired_sr']['ci95'][0],delta_ci95_high=m['paired_sr']['ci95'][1],
                mixture_available=mix['available'],mixture_low=mix.get('low'),mixture_high=mix.get('high'),
                mixture_weight=mix.get('high_episode_weight'),mixture_delta_sr=mix.get('allocation_minus_expected_uniform_sr'),
                mixture_ci95_low=mix.get('paired_conditional_ci95',[None,None])[0],mixture_ci95_high=mix.get('paired_conditional_ci95',[None,None])[1]))
    for name,rows in [('task_levels.csv',task_rows),('level_stats.csv',stat_rows),('library_agreement.csv',library_rows),('matched_ir.csv',matched_rows)]:
        files[name]=csv_text(rows)
    return files


def write_outputs(files,directory):
    directory=Path(directory).resolve()
    require(directory.is_relative_to(HERE), 'Writes must stay inside descent_astra')
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():
        require('/' not in name and name not in ('PROTOCOL.md','protocol.json','PROTOCOL.sha256'), 'Protected output path')
        path=directory/name
        require(not path.is_symlink(), 'Refuse symlink output')
        text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        tmp=directory/(name+'.tmp')
        require(not tmp.is_symlink(), 'Refuse symlink temporary file')
        tmp.write_text(text)
        tmp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify-protocol','apply'),nargs='?',default='verify-protocol')
    args=parser.parse_args(argv)
    reader=Reader();p=verify_protocol(reader)
    if args.action=='verify-protocol':
        print(json.dumps(dict(protocol='verified',frozen_at_utc=p['frozen_at_utc'],outcome_reads=0)))
        return
    allowed=set(range(10,22))|set(range(54,66))
    require(set(os.sched_getaffinity(0))<=allowed, 'Use taskset -c 10-21,54-65')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Hide GPUs before apply')
    selection=reader.json(R14/'descent/selection.json')
    caps=reader.json(R14/'descent/cap_tables.json')
    fits=reader.json(R14/'astra/allocation_fits.json')
    csv_rows=list(csv.DictReader(io.StringIO(reader.read(R14/'astra/allocation_tasks.csv').decode())))
    require(caps['source_allocation_sha256']==sha(reader.read(R14/'astra/allocation_fits.json')), 'Cap provenance mismatch')
    all_arms={}
    for cell in p['cells']:
        print(f'Reading complete B cell: {cell}',flush=True)
        root=Path(p['trace_parent'])/(p['root_prefix']+cell)
        require(root.name=='r14_descent_'+cell and root.parent==Path('/home/weiland/trace_runs/os_closed_loop'), 'Disallowed root')
        all_arms[cell]=load_cell(root,cell,p,selection,caps,reader)
    result=analyze_cells(all_arms,p,caps,fits,csv_rows)
    files=outputs(result,p)
    reader.read(Path(__file__))
    reader.stable()
    files['provenance.json']=dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),inputs=reader.inputs,
        protocol_sha256=sha((HERE/'protocol.json').read_bytes()),outcome_pool='B',test_a_read=False,
        command=sys.argv,python=sys.version,numpy=np.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
        selected_output_sha256={name:sha((value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()) for name,value in files.items()})
    write_outputs(files,HERE)
    print(f'Wrote {len(files)} outputs in {HERE}',flush=True)


if __name__=='__main__':
    main()
