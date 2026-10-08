"""Frozen B-only descent analysis. Default is protocol verification, never outcomes.

Only the explicit `apply` subcommand opens descent roots. No launches or test-A IO.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import beta, kendalltau, rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R15 = HERE.parent
R14 = R15.parent / 'r14'
ORDER = ('c10', 'ir45', 'ir40', 'ir32', 'ir25', 'off')
ORDINAL = {level: 5-i for i, level in enumerate(ORDER)}
ARMS = ORDER
MODELS = ('pi05', 'groot')
UNCHANGED_FUNCTIONS = ('cp_low', 'cp_high', 'paired', 'cell_pair', 'descend',
    'accepted_rows', 'stitch', 'bootstrap_indices', 'sampled_sum',
    'interval', 'savings', 'nearest_level', 'ranks', 'library_agreement', 'matched_ir')


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


def verify_source_adapter(reader, path, frozen_digest, current_bytes, protocol_bytes):
    """Bind the authorized layout adapter without editing the frozen registration."""
    source_path = str((HERE/'analyze.py').relative_to(REPO))
    coordinator_path = str((R15/'COORDINATOR_DECISIONS.md').relative_to(REPO))
    require(path in (source_path, coordinator_path), f'Frozen input changed: {path}')
    adapter = reader.json(HERE/'partial_apply_adapter.json')
    require(adapter['schema'] == 'r15.descent.partial_apply_adapter.v1' and
            adapter['protocol_sha256'] == sha(protocol_bytes), 'Wrong partial-apply registration')
    if path == coordinator_path:
        # The coordinator appended D5 about track B after the freeze. Bind the
        # exact frozen D1-D4 bytes; later appended notes are provenance only.
        record = adapter['coordinator_log_append']
        original = reader.read(HERE/'partial_apply_original_coordinator_decisions.md')
        require(record['source'] == path and record['frozen_sha256'] == frozen_digest and
                sha(original) == frozen_digest and current_bytes.startswith(original),
                'Frozen coordinator decisions changed (only appended notes allowed)')
        return
    require(adapter['source'] == path and adapter['frozen_sha256'] == frozen_digest and
            adapter['adapted_sha256'] == sha(current_bytes), 'Unregistered source change')
    original = reader.read(HERE/'partial_apply_original_analyze.py.txt')
    require(sha(original) == frozen_digest, 'Changed original analysis source')
    def functions(source):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)}
    before, after = functions(original), functions(current_bytes)
    for name in UNCHANGED_FUNCTIONS:
        require(before[name] == after[name], f'Frozen numerical/loader function changed: {name}')
        require(sha(before[name].encode()) == adapter['unchanged_function_ast_sha256'][name],
                f'Invalid unchanged-function proof: {name}')
    fix = adapter['served_fit_loader_fix']
    loader_before = reader.read(HERE/'analyze.before_served_hash_fix.py.txt')
    require(sha(loader_before) == fix['before_source_sha256'] and
            sha(functions(loader_before)['load_cell'].encode()) ==
                adapter['unchanged_function_ast_sha256']['load_cell'] and
            sha(after['load_cell'].encode()) == fix['load_cell_ast_sha256'],
            'Unregistered served-fit loader change')
    require(fix['relocation_source'] == 'exp/offline_search/closed_loop/ops/h100/assets.py' and
            sha(reader.read(REPO/fix['relocation_source'])) == fix['relocation_source_sha256'],
            'H100 relocation implementation changed')


def verify_protocol(reader):
    seal = reader.read(HERE / 'PROTOCOL.sha256').decode()
    require(len(seal.splitlines()) == 2, 'Invalid protocol seal')
    require({line.split()[1] for line in seal.splitlines()} == {'protocol.json','PROTOCOL.md'}, 'Bad sealed names')
    for line in seal.splitlines():
        digest, name = line.split()
        require(sha(reader.read(HERE/name)) == digest, f'Frozen protocol changed: {name}')
    protocol_bytes = reader.read(HERE/'protocol.json')
    p = json.loads(protocol_bytes)
    require(p['pre_outcome'] and p['outcome_reads_before_freeze'] == 0, 'Missing pre-outcome registration')
    require(p['selection_rule'] == 'point_estimate_first_failure' and tuple(p['descent_order']) == ORDER,
            'Unsupported registered rule')
    for path, digest in p['input_sha256'].items():
        data = reader.read(REPO/path)
        if sha(data) != digest:
            verify_source_adapter(reader, path, digest, data, protocol_bytes)
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
    return f'r15d_{cell}_{suffix}' if level != 'c10' else f'r15d_{cell.rsplit("_",1)[0]}_pure_c10'


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


def h100_relocated_sha256(source, root, store, base):
    """Match per-arm sync in a fresh interpreter: NumPy pickle state is shared."""
    code = '''import hashlib, pickle, sys
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100.assets import relocated_fit
source = sys.stdin.buffer.read()
blob, changed = relocated_fit(pickle.loads(source), *(Path(p) for p in sys.argv[1:]))
data = pickle.dumps(blob, protocol=4) if changed else source
print(hashlib.sha256(data).hexdigest())
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(REPO)+':'+str(REPO/'src'))
    result = subprocess.run(['taskset','-c','2-9,46-53',sys.executable,'-c',code,
                             str(root),str(store),str(base)],input=source,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=True)
    digest = result.stdout.decode().strip()
    require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),
            'Invalid H100 relocation hash result')
    return digest


class ServedFitVerifier:
    """Accept only original bytes or the exact path relocation used by H100 sync."""
    def __init__(self, root, row, source_sha256, reader):
        self.root, self.row = Path(root).resolve(), row
        self.source_sha256, self.reader = source_sha256, reader
        self.relocated_sha256 = None
        self.remote_run = None
        self.observed = {}

    def relocated_hash(self):
        from exp.offline_search.closed_loop.ops.h100.assets import flag
        require(self.row.get('intended_host') == 'h100', 'Wrong served fit hash: not an H100 arm')
        plan = self.reader.json(self.root/'h100_sync/plan.json')
        require(Path(plan['run']).resolve() == self.root, 'H100 plan belongs to another root')
        remote = Path(plan['remote_run'])
        require(remote.is_absolute() and '..' not in remote.parts and
                remote.name == self.root.name and remote.parent.name == 'runs' and
                remote != self.root and plan.get('server_host') != 'local', 'Invalid H100 remote run path')
        store = flag(self.row['plugin_args'], '--os-root')
        require(store and Path(store).is_absolute(), 'Missing frozen source store path')
        source = self.reader.read(flag(self.row['plugin_args'], '--os-fit-artifact'))
        require(sha(source) == self.source_sha256, 'Fit bytes changed')
        # Match assets.build_plan: reserialize only when relocation changed paths.
        self.relocated_sha256 = h100_relocated_sha256(source,self.root,Path(store),remote.parent.parent)
        self.remote_run = str(remote)
        return self.relocated_sha256

    def check(self, observed):
        if observed == self.source_sha256:
            branch = 'frozen_source'
        else:
            require(isinstance(observed,str) and len(observed)==64 and
                    all(c in '0123456789abcdef' for c in observed), 'Wrong served fit hash')
            expected = self.relocated_sha256 if self.relocated_sha256 is not None else self.relocated_hash()
            require(observed == expected, 'Wrong served fit hash: neither frozen source nor deterministic H100 relocation')
            branch = 'h100_relocated'
        self.observed[observed] = branch
        return branch

    def audit(self):
        return dict(source_sha256=self.source_sha256,relocated_sha256=self.relocated_sha256,
                    remote_run=self.remote_run,observed_sha256_branches=dict(sorted(self.observed.items())))


def load_cell(root, cell, p, selection, caps, reader):
    """Same accepted-attempt loader, adapting five cell arms + a shared C10 root."""
    primary_root=Path(root)
    model,suite,size=cell.split('_')
    selected_design=selection['model_suites'][f'{model}_{suite}']
    expected=sorted(map(tuple,selected_design['pairs']))
    require(len(expected)==len(set(expected))==300, 'Expected exact 300 identities')
    require(all(sum(t==j for t,i in expected)==30 for j in range(10)), 'Unbalanced selection')
    result={}
    for level in ARMS:
        root=Path(p['trace_parent'])/f'r15_descent_c10_{model}_{suite}' if level=='c10' else primary_root
        source_cell=f'{model}_{suite}_100' if level=='c10' else cell
        seal=reader.json(root/'ROOT.json')
        count=1 if level=='c10' else 5
        require(seal['init_pool']=='B' and seal['cell']==source_cell and
                seal['episodes_per_arm']==300 and seal['arm_count']==count, 'Wrong B root design')
        for filename,digest in seal['definitions_sha256'].items():
            path=(root/filename).resolve()
            require(path.is_relative_to(root.resolve()) and sha(reader.read(path))==digest, 'Changed root definition')
        manifest=reader.json(root/'dev_selection.json')
        require(manifest['init_pool']=='B' and manifest['dev'] is True and
                sorted(map(tuple,manifest['pairs']))==expected, 'Wrong manifest')
        require(manifest['selection_sha256']==p['input_sha256'][str((HERE/'selection.json').relative_to(REPO))],
                'Wrong frozen selection binding')
        manifest_digest=sha(json.dumps(expected,separators=(',',':')).encode())
        rows=reader.json(root/'arms.json');contracts=reader.json(root/'dev_contracts.json')
        allowed={arm_name(cell,'c10')} if level=='c10' else {arm_name(cell,l) for l in ORDER[1:]}
        require(len(rows)==count and {r['arm'] for r in rows}==allowed, 'Wrong B arm set')
        name=arm_name(cell,level);row=next(r for r in rows if r['arm']==name)
        require(row.get('dev') is True and row.get('init_pool')=='B' and
                row['r14_descent_level']==level and row['episodes']==300 and row['model']==model,
                'Wrong B arm metadata')
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
        fit_digest = caps['cells'][source_cell]['fit_sha256_by_level'][level]
        args = row['plugin_args']
        fit_path = Path(args[args.index('--os-fit-artifact')+1])
        require(contract['fit_sha256']==fit_digest and seal['fit_sha256'][str(fit_path)]==fit_digest, 'Wrong frozen fit')
        require(sha(reader.read(fit_path))==fit_digest, 'Fit bytes changed')
        fit_verifier = ServedFitVerifier(root,row,fit_digest,reader)
        excluded = contract['subset']['excluded_inits_by_task']
        loeo = level != 'c10' and cell.endswith('_500')
        require(contract.get('leave_own_episode_out',False) == loeo, 'Wrong leave-own-out contract')
        if not loeo:
            require(all(i not in excluded[str(t)] for t,i in expected), 'Selected init in served subset')
        require(contract['subset']==selected_design['served_subsets'][source_cell.split('_')[-1]],
                'Subset differs from selected actual library')
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
            diag=d.get('r14',{})
            fit_verifier.check(diag.get('fit_sha256'))
            require(int(d['task_id'])==uid_pair(uid)[0] and int(d['init'])==uid_pair(uid)[1], 'Wrong served identity')
            if loeo and d['vision']:
                require(diag.get('leave_own_episode_out') is True and diag.get('query_pool')=='B',
                        'Missing leave-own-out decision audit')
                own=diag['excluded_episode']
                require(own>=0 and diag['query_init']==uid_pair(uid)[1], 'Wrong exclusion identity')
                require(own not in diag['candidate_episode_ids'] and own not in diag['donor_episode_ids'],
                        'Own episode leaked into a library lookup')
                require(len(diag['donor_episode_ids'])==len(diag['parent_donor_rows'])>0,
                        'Malformed donor audit')
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
            dict(arm=name,root=str(root),fit_hash_verification=fit_verifier.audit(),leave_own_episode_out=loeo,summary_stage_ir=ledger['ir_per_five_controls'],full_stage_costs=full,
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


def analysis_scope(p, models=None):
    requested = set(MODELS if models is None else models)
    require(requested and requested <= set(MODELS), 'Choose pi05 and/or groot')
    cells = [cell for cell in p['cells'] if cell.split('_')[0] in requested]
    require(cells and (models is None or requested == {c.split('_')[0] for c in cells}),
            'Requested model has no registered cells')
    partial = cells != list(p['cells'])
    return dict(partial=partial, models=[m for m in MODELS if any(c.startswith(m+'_') for c in cells)],
                cells=cells, registered_cells=list(p['cells']),
                simultaneous_family=200,
                cross_cell_reporting='partial; pooled summaries deferred' if partial else 'full',
                selection_dependencies='this cell arms and its model/suite shared C10 only')


def analyze_cell(cell, arms, p, indices):
    """No other cell or cross-cell summary is available to the decision code."""
    require(set(arms) == set(ORDER), f'Incomplete descent arms: {cell}')
    pure=arms['c10']
    tasks={str(t):{level:paired(arms[level].y[t],pure.y[t],identical=level=='c10') for level in ORDER} for t in range(10)}
    pooled={level:cell_pair(arms[level].y,pure.y,identical=level=='c10') for level in ORDER}
    c=dict(arm_stats={l:dict(**a.stats(),**a.audit) for l,a in arms.items()},task_stats=tasks,cell_stats=pooled,
           tolerances={},matched_ir={},leave_own_episode_out=cell.endswith('_500'))
    for tol in p['tolerances_pp']:
        decisions={str(t):descend(tasks[str(t)],tol) for t in range(10)}
        levels={t:d['level'] for t,d in decisions.items()}
        uniform=descend(pooled,tol)
        selected=stitch(arms,levels)
        c['tolerances'][str(tol)]=dict(levels=levels,uniform=uniform['level'],task_decisions=decisions,
            uniform_decision=uniform,savings=savings(selected,arms[uniform['level']],pure,indices),
            library={},
            confidence_sensitivity=dict(levels={t:descend(s,tol,True)['level'] for t,s in tasks.items()},
                                        uniform=descend(pooled,tol,True)['level']))
    return c


def analyze_cells(all_arms,p,caps,fits,csv_rows,*,models=None):
    scope=analysis_scope(p,models)
    require(set(all_arms)==set(scope['cells']), 'Missing/extra requested cells')
    indices=bootstrap_indices(p)
    result=dict(protocol_frozen_at_utc=p['frozen_at_utc'],primary_tolerance_pp=p['primary_tolerance_pp'],
                analysis_scope=scope,cells={})
    for cell in scope['cells']:
        result['cells'][cell]=analyze_cell(cell,all_arms[cell],p,indices)
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
    lines=['# 给负责人的结果说明','',
      '冻结规则不变：每任务 30 个配对 B 初始状态，以纯 C10 为参照按 ir45、ir40、ir32、ir25、off 下降，首次失败即停止。主容差 5 个百分点；0 和 2 个百分点保留为备选。','',
      '500 集库在服务时排除当前 B 初始状态的自身库轨迹；拟合仍包含该轨迹，因此存在轻微乐观偏差。这里只使用开发集。','']
    if result['analysis_scope']['partial']:
        lines += ['部分模型结果：'+', '.join(result['analysis_scope']['models'])+
                  '。每个所选单元的全部 B 臂与共享 C10 必须完整；未读取其他模型结果。跨单元汇总延后至完整分析。', '',
                  'PARTIAL reporting: the fixed 200-comparison family is retained within each library-size group; '
                  'only the selected model cells are reported. Confidence sensitivity is reporting only and never changes selection.', '']
    for tol in p['tolerances_pp']:
        rows=[]
        for cell,c in result['cells'].items():
            r=c['tolerances'][str(tol)];v=r['savings']
            rows.append([cell]+[r['levels'][str(t)] for t in range(10)]+[r['uniform'],
                         number(v['task']['sr'],100),number(v['uniform']['sr'],100),
                         number(v['task']['ir']),number(v['uniform']['ir']),number(v['ir_saved'],100),
                         'yes' if c['leave_own_episode_out'] else 'no'])
        lines += [f'## Tolerance {tol} pp','',markdown_table(['Cell']+[str(t) for t in range(10)]+
            ['Uniform','Task SR %','Uniform SR %','Task IR','Uniform IR','IR saved pp','Leave own out'],rows),'']
    lines += ['R14 point-estimate selection, paired CP intervals, conditional paired bootstrap (5,000, seed 20261005), and fixed owner IR coefficients are unchanged. The registered 200-comparison simultaneous band applies separately to each four-cell library-size group, not jointly across all 600 R15 comparisons. C10 is shared within model/suite. R14 task_periodic allocation arms are not repeated in R15, so allocation agreement and matched-allocation comparisons are not available. B-selected stitched estimates remain optimistic and are not independent mixed-policy measurements.','']
    return '\n'.join(lines)


def csv_text(rows):
    f=io.StringIO()
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
    return f.getvalue()


def output_suffix(scope):
    return '_'+ '_'.join(scope['models']) if scope['partial'] else ''


def output_name(name, scope):
    path=Path(name)
    return path.stem+output_suffix(scope)+path.suffix


def output_text(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'


def outputs(result,p):
    scope=result['analysis_scope']
    require(scope==analysis_scope(p,scope['models']), 'Invalid analysis scope')
    files={'DESCENT.md':report(result,p),'analysis.json':result}
    task_rows=[]; stat_rows=[]; library_rows=[]; matched_rows=[]
    for tolerance in p['tolerances_pp']:
        levels={cell:c['tolerances'][str(tolerance)]['levels'] for cell,c in result['cells'].items()}
        uniform={cell:c['tolerances'][str(tolerance)]['uniform'] for cell,c in result['cells'].items()}
        require(set(levels)==set(scope['cells'])==set(uniform), 'Emitter cell mismatch')
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
        if rows: files[name]=csv_text(rows)
    return {output_name(name,scope):value for name,value in files.items()}


def check_partial_equivalence(result,p,reader,directory=None):
    """Reject any changed frozen model table before a later apply writes outputs."""
    directory=HERE if directory is None else Path(directory)
    checks=[]
    for model in MODELS:
        scope=analysis_scope(p,[model]) if any(c.startswith(model+'_') for c in p['cells']) else None
        if scope is None or not scope['partial'] or not set(scope['cells'])<=set(result['cells']):
            continue
        tables={}
        for tol in p['tolerances_pp']:
            for kind in ('levels','uniform'):
                value={cell:result['cells'][cell]['tolerances'][str(tol)][kind] for cell in scope['cells']}
                tables[output_name(f'{kind}_tol{tol}pp.json',scope)]=value
                if tol==p['primary_tolerance_pp']:
                    tables[output_name(kind+'.json',scope)]=value
        existing=[name for name in tables if (Path(directory)/name).exists()]
        if not existing:
            continue
        require(len(existing)==len(tables), f'Incomplete frozen partial tables: {model}')
        for name,value in tables.items():
            actual=reader.read(Path(directory)/name)
            require(actual==output_text(value).encode(), f'Partial/full table differs byte-for-byte: {name}')
            checks.append(dict(model=model,file=name,sha256=sha(actual),byte_identical=True))
    return dict(PASS=True,checks=checks,checked_models=[m for m in MODELS if any(c['model']==m for c in checks)])


def write_outputs(files,directory):
    directory=Path(directory).resolve()
    require(directory.is_relative_to(HERE), 'Writes must stay inside rounds/r15/descent')
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in files.items():
        require('/' not in name and name not in ('PROTOCOL.md','protocol.json','PROTOCOL.sha256'), 'Protected output path')
        path=directory/name
        require(not path.is_symlink(), 'Refuse symlink output')
        text=output_text(value)
        tmp=directory/(name+'.tmp')
        require(not tmp.is_symlink(), 'Refuse symlink temporary file')
        tmp.write_text(text)
        tmp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify-protocol','apply'),nargs='?',default='verify-protocol')
    parser.add_argument('--models',nargs='+',choices=MODELS,
                        help='Apply only complete registered cells of these models; separate partial outputs')
    args=parser.parse_args(argv)
    if args.models is not None and args.action!='apply':
        parser.error('--models requires apply')
    reader=Reader();p=verify_protocol(reader)
    if args.action=='verify-protocol':
        print(json.dumps(dict(protocol='verified',frozen_at_utc=p['frozen_at_utc'],outcome_reads=0)))
        return
    allowed=set(range(2,10))|set(range(46,54))
    require(set(os.sched_getaffinity(0))<=allowed, 'Use taskset -c 2-9,46-53')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Hide GPUs before apply')
    selection=reader.json(HERE/'selection.json')
    caps=reader.json(HERE/'cap_tables.json')
    fits=[];csv_rows=[]
    all_arms={}
    scope=analysis_scope(p,args.models)
    for cell in scope['cells']:
        print(f'Reading complete B cell: {cell}',flush=True)
        root=Path(p['trace_parent'])/(p['root_prefix']+cell)
        require(root.name=='r15_descent_'+cell and root.parent==Path('/home/weiland/trace_runs/os_closed_loop'), 'Disallowed root')
        all_arms[cell]=load_cell(root,cell,p,selection,caps,reader)
    result=analyze_cells(all_arms,p,caps,fits,csv_rows,models=args.models)
    files=outputs(result,p)
    files[output_name('partial_equivalence.json',scope)]=check_partial_equivalence(result,p,reader)
    reader.read(Path(__file__))
    reader.stable()
    files[output_name('provenance.json',scope)]=dict(analyzed_at_utc=datetime.now(timezone.utc).isoformat(),inputs=reader.inputs,
        protocol_sha256=sha((HERE/'protocol.json').read_bytes()),outcome_pool='B',test_a_read=False,
        analysis_scope=scope,
        command=sys.argv,python=sys.version,numpy=np.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
        selected_output_sha256={name:sha(output_text(value).encode()) for name,value in files.items()})
    write_outputs(files,HERE)
    print(f'Wrote {len(files)} outputs in {HERE}',flush=True)


if __name__=='__main__':
    main()
