"""R16 strict paired analyses. Only explicit apply commands read outcomes.

R14/R15 statistical helpers and collector accepted-attempt semantics are reused.
The c10 descent sentinel executes k-step pure inference; all lower levels are
selected against a separate full-step C10 reference, never against that sentinel.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import yaml

from exp.offline_search.rounds.r15.descent.analyze import (
    Reader, Episodes, require, sha, uid_pair, paired, cell_pair, descend,
    stitch, sampled_sum, interval, ServedFitVerifier, nearest_level, ranks, ORDINAL)
from exp.offline_search.rounds.r15.descent.analyze_test_a import AServedFitVerifier
from .prepare import HERE, REPO, CELLS, ORDER, parts, atomic, write_json
from .cost import pricing

LEVEL_ORDER = ('c10',*ORDER)


def choose(stats,tolerance,bound=False):
    """Full C10 is the construction-pass sentinel; runtime fallback is k-C10."""
    selection=dict(stats)
    top=dict(stats['c10'])
    top.update(successes=top['pure_successes'],sr=top['pure_sr'],delta=0.,
               simultaneous_lower95=0.)
    selection['c10']=top
    return descend(selection,tolerance,bound)


def verify(reader):
    design = reader.json(HERE/'design.json')
    lock = reader.json(HERE/'DESIGN_LOCK.json')
    require(lock['design_sha256']==sha(reader.read(HERE/'design.json')),'Changed design')
    for path,digest in lock['source_sha256'].items():
        current=reader.read(path)
        if sha(current)!=digest:
            from .conductor_analysis import verify_amendment
            verify_amendment(reader,path,digest,current)
    require(set(design['cells'])==set(CELLS),'Wrong cell design')
    for path,digest in design['source_catalog_sha256'].items():
        require(sha(reader.read(path))==digest,'Changed original frozen fit/reference catalog')
    return design


def checked_root(record,reader):
    root = Path(record['root'])
    seal = reader.json(root/'ROOT.json')
    if 'seal' in record:
        require(seal==record['seal'] and sha(reader.read(root/'ROOT.json'))==record['seal_sha256'],
                'Changed reference seal')
    else:
        require(seal==record,'Changed R16 root index')
    for name,digest in seal['definitions_sha256'].items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(reader.read(path))==digest,'Changed sealed definition')
    return root,seal


def load(record,name,expected,pool,k,reader,*,freeze=None):
    """Last accepted terminal error-free journal row, then its last uid/step log."""
    root,seal = checked_root(record,reader)
    rows = reader.json(root/'arms.json')
    row = next(r for r in rows if r['arm']==name)
    require(row['init_pool']==pool and row['dev']==(pool=='B'),'Wrong pool')
    expected = sorted(map(tuple,expected)); positions = {pair:i for i,pair in enumerate(expected)}
    width = 30 if pool=='B' else 50
    require(len(expected)==len(positions)==10*width,'Wrong identity count')
    require(all(sum(t==j for t,i in expected)==width for j in range(10)),'Unbalanced identities')
    require(row['episodes']==10*width and row['replan_steps']==5 and row['full_model'] and not row['recording'],
            'Changed horizon or model settings')
    args = row['plugin_args']
    # The long execution segment (--os-blind --os-policy-tail) is a per-batch setting and the stall judge is
    # Q3 stall_only (guard_only was removed, owner 10-20); only the rollout seed is part of the protocol.
    require(args[args.index('--os-rollout-seed')+1]=='2026101401','Changed seed')
    manifest = reader.read(row['manifest']); md = json.loads(manifest)
    require(sorted(map(tuple,md['pairs'] if isinstance(md,dict) else md))==expected,'Wrong manifest identities')
    fit = Path(args[args.index('--os-fit-artifact')+1]); digest = seal['fit_sha256'][str(fit)]
    require(sha(reader.read(fit))==digest,'Changed source fit')
    # Retain R15 deterministic relocated-path/hash support for the full-step references.
    verifier = AServedFitVerifier(root,row,digest,reader)
    cfg = yaml.safe_load(reader.read(row['yaml']))
    default = 10 if row['model']=='pi05' else 8
    require((cfg.get('miss') or {}).get('num_steps',default)==k,'YAML K mismatch')
    run = root/'runs'/name
    if 'r16_k' in row:
        launch=reader.json(run/'h100_launch.json')
        require(launch.get('server_host')=='local' and launch.get('worker_host')=='timan107',
                'R16 must use local servers and timan107 workers')
        require(launch.get('servers') and all(s['env']['STAGE1_ONLY']=='0' for s in launch['servers']),
                'R16 requires full-model servers')
    require(reader.read(run/'manifest.json')==manifest,'Changed run manifest')
    selected = reader.json(run/'selection.json')
    require(selected.get('init_pool','A')==pool and Path(selected['manifest'])==run/'manifest.json',
            'Changed selection binding')
    loeo = bool(row.get('r15_leave_own_episode_out',False))
    if pool=='B':
        contracts = reader.json(root/'dev_contracts.json')
        contract = reader.json(run/'pool_contract.json')
        require(contract==contracts[name] and contract['fit_sha256']==digest and
                contract['manifest_sha256']==sha(json.dumps(expected,separators=(',',':')).encode()),'Changed B contract')
        require(contract.get('leave_own_episode_out',False)==loeo,'Wrong LOEO contract')
        require(contract['init_pool']=='B' and contract['dev'] is True,'Wrong contract pool')
        if not loeo:
            require(all(i not in contract['subset']['excluded_inits_by_task'][str(t)] for t,i in expected),
                    'B init leaked into served subset')
    accepted,errors,journal_count,duplicate_accepted = {},0,0,0
    for r in reader.jsonl(run/'client/journal.jsonl'):
        journal_count += 1; errors += bool(r.get('error'))
        require(r.get('init_pool')==pool,'Wrong journal pool')
        uid = r.get('task_uid')
        if uid:
            require(uid.startswith(name+':') and uid_pair(uid) in positions,'Unexpected journal UID')
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
            require(type(r.get('success')) is bool,'Nonbinary success')
            duplicate_accepted += uid in accepted; accepted[uid] = r
    by_pair = {uid_pair(u):r for u,r in accepted.items()}
    require(len(accepted)==len(by_pair)==len(expected) and set(by_pair)==set(expected),'Incomplete accepted episodes')
    summary = reader.json(run/'summary.json')
    y = np.array([by_pair[p]['success'] for p in expected],dtype=np.int64).reshape(10,width)
    require(summary['init_pool']==pool and summary['arm']==name and summary['model']==row['model'],'Wrong summary')
    require(summary['complete']==len(expected) and summary['success']==int(y.sum()) and
            (summary['sr']==round(float(y.mean()),4) or
             math.isclose(summary['sr'],float(y.mean()),abs_tol=1e-12)),'Summary SR mismatch')
    for key,value in [('journal_rows',journal_count),('error_rows',errors),
                      ('retried',sum(int(r.get('attempt',1) or 1)>1 for r in accepted.values()))]:
        require(summary[key]==value,f'Summary {key} mismatch')
    require(all(summary['per_task'][str(t)]==dict(n=width,success=int(y[t].sum())) for t in range(10)),
            'Task summary mismatch')
    decisions,duplicates,rejected,starts = {},0,0,0
    shards = sorted(run.glob('server_*/decisions_*.jsonl'))
    require(shards,'Missing decision logs'); reader.shards[str(run)] = list(map(str,shards))
    for shard in shards:
        for d in reader.jsonl(shard):
            if d.get('ev')=='startup':
                starts += 1
                require(d['kwargs']==row['kwargs'] and d['method_spec']==row['method'],'Wrong served method')
                argv = d['argv']; observed_path = argv[argv.index('--os-fit-artifact')+1]
                verifier.check_path(observed_path)
                require(d.get('miss_steps',default)==k,'Wrong live startup K')
                if freeze:
                    require(float(d['ts'])>=datetime.fromisoformat(freeze['frozen_at_utc']).timestamp(),
                            'Test A started before table freeze')
                continue
            if d.get('ev')!='dec': continue
            require(d.get('init_pool',pool)==pool,'Wrong decision pool')
            uid = d['uid']
            require(uid.startswith(name+':') and uid_pair(uid) in positions,'Unexpected decision identity')
            if uid not in accepted or int(d.get('attempt',1) or 1)!=int(accepted[uid].get('attempt',1) or 1):
                rejected += 1; continue
            require(type(d['step']) is int and d['step']>=0,'Invalid step')
            key = uid,d['step']; duplicates += key in decisions; decisions[key] = d
    require(starts>0,'Missing startup audit')
    n,v,m = (np.zeros(len(expected),dtype=np.int64) for _ in range(3))
    steps = {u:[] for u in accepted}
    for (uid,step),d in decisions.items():
        require(type(d.get('hit')) is bool and type(d.get('vision')) is bool,'Need explicit hit/vision booleans')
        require(d.get('stage1_mode','full')=='full' and d.get('replan_steps',5)==5,'Unsupported cost mode')
        require(d.get('ok',True) and (d['hit'] or d['vision']),'Invalid decision')
        require(d['hit'] or (type(d.get('miss_k')) in (int,float) and d['miss_k']==k),'Policy call used wrong K')
        require(int(d['task_id'])==uid_pair(uid)[0] and int(d['init'])==uid_pair(uid)[1],'Wrong served task/init')
        verifier.check(d.get('r14',{}).get('fit_sha256'))
        if d.get('source',d.get('src'))=='policy_tail':
            require(d['hit'] and not d['vision'] and d.get('miss_k') is None,'Policy-tail made a policy call')
        if loeo and d['vision']:
            diag = d['r14']; own = diag['excluded_episode']
            require(diag.get('leave_own_episode_out') is True and diag.get('query_pool')=='B' and
                    own>=0 and diag['query_init']==uid_pair(uid)[1],'Missing/wrong LOEO audit')
            require(own not in diag['candidate_episode_ids'] and own not in diag['donor_episode_ids'],
                    'Own episode leaked into lookup')
            require(len(diag['donor_episode_ids'])==len(diag['parent_donor_rows'])>0,'Malformed donor audit')
        if row['r14_descent_level']=='c10':
            require(not d['vision'] or not d['hit'],'Pure C10 used a vision cache HIT')
        j = positions[uid_pair(uid)]; n[j] += 1; v[j] += d['vision']; m[j] += not d['hit']; steps[uid].append(step)
    require(all(s and sorted(s)==list(range(len(s))) for s in steps.values()),'Noncontiguous decision steps')
    ledger = summary['cost_ledger']; full = ledger['full_stage_costs']
    require(full['k']==default and all(float(full[x])>=0 for x in ('s1','s2','s3')),'Invalid full stage costs')
    require([int(n.sum()),int(v.sum()),int(m.sum())]==[ledger[x] for x in ('decisions','vision_decisions','misses')],
            'Decision/ledger count mismatch')
    require(ledger['controls']==int(n.sum())*5,'Wrong nominal controls')
    stage_ir = (full['s1']*v.sum()+(full['s2']+full['s3']*k/default)*m.sum())/(sum(full[x] for x in ('s1','s2','s3'))*n.sum())
    require(math.isclose(stage_ir,ledger['ir_per_five_controls'],rel_tol=0,abs_tol=1e-8),'Stage IR mismatch')
    require(not m.sum() or ledger['k_per_miss']['mean']==k,'Ledger K mismatch')
    prices = pricing(row['model'],k)
    audit = dict(root=str(root),arm=name,k=k,pricing=prices,fit_hash_verification=verifier.audit(),
                 duplicate_decisions=duplicates,rejected_attempt_decisions=rejected,error_rows=errors,
                 duplicate_accepted=duplicate_accepted,summary_stage_ir=stage_ir,leave_own_episode_out=loeo)
    return Episodes(y,n.reshape(10,width),v.reshape(10,width),m.reshape(10,width),
                    (prices['vision']*v+prices['miss']*m).reshape(10,width),audit)


def dev_cell(cell,arms,reference,*,draws=5000):
    require(set(arms)==set(LEVEL_ORDER),'Incomplete descent arms')
    # Frozen R14/R15 family stays 200 separately within each size x k group.
    family = 200
    tasks = {str(t):{l:paired(arms[l].y[t],reference.y[t],family=family) for l in LEVEL_ORDER} for t in range(10)}
    pooled = {l:cell_pair(arms[l].y,reference.y,family=family) for l in LEVEL_ORDER}
    indices = np.random.default_rng(20261005).integers(0,30,size=(draws,10,30))
    result = dict(reference=reference.stats(),arm_stats={l:a.stats() for l,a in arms.items()},
                  task_stats=tasks,cell_stats=pooled,tolerances={},fallback='c10 executes k-step pure; sentinel always available')
    for tol in (0,2,5):
        decisions = {t:choose(stats,tol) for t,stats in tasks.items()}
        levels = {t:d['level'] for t,d in decisions.items()}
        uniform_decision = choose(pooled,tol); uniform = uniform_decision['level']
        selected = stitch(arms,levels); u = arms[uniform]
        ti = sampled_sum(selected.cost,indices)/sampled_sum(selected.n,indices)
        ui = sampled_sum(u.cost,indices)/sampled_sum(u.n,indices)
        signed = u.stats()['ir']-selected.stats()['ir']
        result['tolerances'][str(tol)] = dict(levels=levels,uniform=uniform,task_decisions=decisions,
            uniform_decision=uniform_decision,selected=selected.stats(),uniform_stats=u.stats(),
            selected_vs_full=cell_pair(selected.y,reference.y,family=family),
            absolute_ir_savings=signed,relative_ir_savings=signed/u.stats()['ir'],
            absolute_ir_savings_ci95=interval(ui-ti),relative_ir_savings_ci95=interval((ui-ti)/ui),
            selected_vs_uniform_bootstrap_ci95=interval(sampled_sum(selected.y-u.y,indices)/300),
            confidence_sensitivity=dict(levels={t:choose(s,tol,True)['level'] for t,s in tasks.items()},
                                        uniform=choose(pooled,tol,True)['level']))
    return result


def sampled(array,index):
    return array[np.arange(10)[None,:,None],index].sum(axis=(1,2))


def compare(a,b,delta_boot):
    return dict(**paired(a.y,b.y,family=40),bootstrap_ci95=interval(delta_boot))


def calculate_test(all_arms,*,draws=5000):
    """Pool separately for each (size,k); share original suite/task/init draws."""
    rng = np.random.default_rng(20261005)
    indices = {s:rng.integers(0,50,size=(draws,10,50)) for s in ('l10','spatial')}
    cells,boots = {},{}
    for cell,arms in all_arms.items():
        m,s,n,k = parts(cell); idx = indices[s]
        boots[cell] = {a:dict(sr=sampled(e.y,idx)/500,cost=sampled(e.cost,idx),n=sampled(e.n,idx)) for a,e in arms.items()}
        cells[cell] = dict(arms={a:e.stats() for a,e in arms.items()},comparisons={})
        for ref in ('full_c10','full_descent'):
            delta = boots[cell]['task']['sr']-boots[cell][ref]['sr']
            cells[cell]['comparisons'][f'task_vs_{ref}'] = compare(arms['task'],arms[ref],delta)
            cells[cell]['comparisons'][f'task_vs_{ref}']['ir_delta'] = arms['task'].stats()['ir']-arms[ref].stats()['ir']
    pooled = {}
    for n in (50,100,200,500):
        for k in (1,2):
            group = [c for c in CELLS if parts(c)[2:]==(n,k) and c in all_arms]
            if not group: continue
            require(len(group)==4,'Pooling requires both models and both suites')
            stats,samples = {},{}
            for a in ('task','full_c10','full_descent'):
                arrays = {f:np.stack([getattr(all_arms[c][a],f) for c in group]) for f in ('y','n','v','m','cost')}
                stats[a] = Episodes(**arrays,audit={}).stats()
                samples[a] = dict(sr=sum(boots[c][a]['sr'] for c in group)/4,
                                  ir=sum(boots[c][a]['cost'] for c in group)/sum(boots[c][a]['n'] for c in group))
            contrasts = {}
            for ref in ('full_c10','full_descent'):
                y = np.stack([all_arms[c]['task'].y for c in group]); r = np.stack([all_arms[c][ref].y for c in group])
                contrasts[ref] = dict(**paired(y,r,family=40),
                    bootstrap_ci95=interval(samples['task']['sr']-samples[ref]['sr']),
                    ir_delta=stats['task']['ir']-stats[ref]['ir'],
                    ir_delta_ci95=interval(samples['task']['ir']-samples[ref]['ir']))
            pooled[f'size{n}_k{k}'] = dict(arms=stats,comparisons=contrasts,episode_outcomes=2000,
                                         original_suite_task_init_clusters=1000)
    by_k={}
    if set(all_arms)==set(CELLS):
        for k in (1,2):
            group=[c for c in CELLS if parts(c)[3]==k]
            stats,samples={},{}
            for arm in ('task','full_c10','full_descent'):
                fields={f:np.stack([getattr(all_arms[c][arm],f) for c in group]) for f in ('y','n','v','m','cost')}
                stats[arm]=Episodes(**fields,audit={}).stats()
                samples[arm]=dict(sr=sum(boots[c][arm]['sr'] for c in group)/16,
                    ir=sum(boots[c][arm]['cost'] for c in group)/sum(boots[c][arm]['n'] for c in group))
            comparisons={}
            for ref in ('full_c10','full_descent'):
                ci=interval(samples['task']['sr']-samples[ref]['sr'])
                comparisons[ref]=dict(delta=stats['task']['sr']-stats[ref]['sr'],ci95=ci,bootstrap_ci95=ci,
                    ir_delta=stats['task']['ir']-stats[ref]['ir'],
                    ir_delta_ci95=interval(samples['task']['ir']-samples[ref]['ir']))
            by_k[f'k{k}_across_sizes']=dict(arms=stats,comparisons=comparisons,weighted_episode_outcomes=8000,
                original_suite_task_init_clusters=1000,reference_reuse='Repeated full references are weights, never additional independent trials',
                interval='Paired suite/task/init cluster bootstrap, shared across sizes and models')
    return dict(measurement_only=True,cells=cells,pooled=pooled,pooled_by_k=by_k,bootstrap=dict(draws=draws,seed=20261005,
                shared_across_models_k_sizes=True,fixed_task_mixture=True,between_run_uncertainty=False))


def markdown(result,kind):
    lines = ['# R16 '+kind,'','Measure only. IR uses D5 source-derived pricing. Delta is percentage points.', '',
             '| cell | arm | SR | IR | contrast | Δ pp | paired 95% CI pp |',
             '|---|---|---:|---:|---|---:|---|']
    groups = {**result.get('cells',{}),**result.get('pooled',{}),**result.get('pooled_by_k',{})}
    for cell,r in groups.items():
        for arm,stats in r.get('arms',{}).items():
            lines.append(f'| {cell} | {arm} | {stats["sr"]:.4f} | {stats["ir"]:.6f} | | | |')
        for contrast,c in r.get('comparisons',{}).items():
            ci = c.get('bootstrap_ci95',c['ci95'])
            lines.append(f'| {cell} | | | | {contrast} | {100*c["delta"]:.3f} | [{100*ci[0]:.3f}, {100*ci[1]:.3f}] |')
    return '\n'.join(lines)+'\n'


def apply_phase1(output):
    reader = Reader(); design = verify(reader); result = dict(measurement_only=True,cells={})
    expected = [(t,i) for t in range(10) for i in range(50)]
    for rec in design['phase1']:
        m,s,n,k = parts(rec['cell']); ref = design['full_test_refs'][f'{m}_{s}']
        a = load(rec,rec['arms'][0],expected,'A',k,reader)
        b = load(ref,ref['arm'],expected,'A',10 if m=='pi05' else 8,reader)
        result['cells'][f'{m}_{s}_k{k}'] = dict(arms=dict(reduced=a.stats(),full_c10=b.stats()),
            comparisons=dict(reduced_vs_full=paired(a.y,b.y,family=8)),audit=dict(reduced=a.audit,full=b.audit))
    reader.stable(); result['inputs'] = reader.inputs
    write_json(output/'phase1.json',result); atomic(output/'PHASE1.md',markdown(result,'phase 1').encode())


def apply_dev(output):
    require(not (output/'TABLE_FREEZE.json').exists(),'Tables are frozen; use a fresh output directory')
    reader = Reader(); design = verify(reader); cells = {}
    for cell in CELLS:
        rec = design['cells'][cell]; m,s,n,k = parts(cell)
        arms = {l:load(rec['dev'],arm,rec['pairs'],'B',k,reader) for l,arm in zip(ORDER,rec['dev']['arms'],strict=True)}
        arms['c10'] = load(rec['fallback'],rec['fallback']['arms'][0],rec['pairs'],'B',k,reader)
        ref = rec['full_dev']; full = load(ref,ref['arm'],rec['pairs'],'B',10 if m=='pi05' else 8,reader)
        cells[cell] = dev_cell(cell,arms,full)
        cells[cell]['audits'] = {l:a.audit for l,a in arms.items()}; cells[cell]['reference_audit'] = full.audit
    result = dict(cells=cells,primary_tolerance_pp=5,bootstrap_draws=5000,
                  stitched_development_is_optimistic=True,fits_include_own_episode_at_500=True,
                  confidence_family=200,confidence_scope='Separately within each size x k; no joint cross-group claim',
                  selection_reference='full-step pure C10',fallback='k-step pure C10',
                  allocation_diagnostics='R15 adapter: allocation arms not repeated; no allocation agreement or matched-IR claim')
    reader.stable(); result['inputs'] = reader.inputs
    write_json(output/'dev.json',result)
    for tol in (0,2,5):
        levels = {c:r['tolerances'][str(tol)]['levels'] for c,r in cells.items()}
        uniform = {c:r['tolerances'][str(tol)]['uniform'] for c,r in cells.items()}
        write_json(output/f'levels_tol{tol}pp.json',levels); write_json(output/f'uniform_tol{tol}pp.json',uniform)
        if tol==5:
            write_json(output/'levels.json',levels); write_json(output/'uniform.json',uniform)
    lines = ['# R16 development descent','','容忍度以同初始状态的完整步数纯推理为参考。主规则为 5 个百分点、第一次失败即停止。',
             'c10 回退执行 k 步纯推理；开发拼接估计存在选择乐观偏差，500 库的拟合仍包含本 episode。','',
             '| cell | uniform | tasks 0..9 | selected SR | selected IR | IR savings vs uniform |','|---|---|---|---:|---:|---:|']
    for c,r in cells.items():
        a = r['tolerances']['5']; levels = ' / '.join(a['levels'][str(t)] for t in range(10))
        lines.append(f'| {c} | {a["uniform"]} | {levels} | {a["selected"]["sr"]:.4f} | {a["selected"]["ir"]:.6f} | {a["absolute_ir_savings"]:+.6f} |')
    atomic(output/'DESCENT.md',('\n'.join(lines)+'\n').encode())


def freeze_tables(output):
    require(not (output/'TABLE_FREEZE.json').exists(),'Already frozen')
    reader = Reader(); design = verify(reader); result = reader.json(output/'dev.json')
    require(set(result['cells'])==set(CELLS),'Need all 32 cells before freeze')
    for name in ('levels.json','uniform.json'):
        table = reader.json(output/name)
        require(table=={c:r['tolerances']['5']['levels' if name=='levels.json' else 'uniform'] for c,r in result['cells'].items()},
                'Table differs from registered dev selection')
    for path,record in result['inputs'].items():
        require(sha(reader.read(path))==record['sha256'],'Dev input changed since analysis')
    reader.stable()
    names = ['dev.json','DESCENT.md','levels.json','uniform.json']+[f'{t}_tol{n}pp.json' for n in (0,2,5) for t in ('levels','uniform')]
    write_json(output/'TABLE_FREEZE.json',dict(schema='r16.tables.v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=sha(reader.read(HERE/'design.json')),sha256={n:sha(reader.read(output/n)) for n in names},
        cells=list(CELLS),primary_tolerance_pp=5,test_a_outcome_reads=0))


def apply_test(output,tables):
    from .prepare import frozen_tables
    levels,uniform,freeze = frozen_tables(tables)
    reader = Reader(); design = verify(reader); all_arms = {}
    records = reader.json(HERE/'test_roots.json')
    require({r['cell'] for r in records}==set(CELLS),'Incomplete emitted test roots')
    expected = [(t,i) for t in range(10) for i in range(50)]
    for rec in records:
        cell = rec['cell']; m,s,n,k = parts(cell); source = design['cells'][cell]
        table = reader.json(Path(rec['root'])/'descent_table.json')
        require(table['levels']==levels[cell] and table['uniform']==uniform[cell] and table['freeze']==freeze,'Changed test table binding')
        capvec=[source['periodic_caps'][levels[cell][str(t)]] for t in range(10)]
        require(table['caps']==capvec and table['k']==k and
                table['off_tasks']==[t for t in range(10) if levels[cell][str(t)]=='off'],'Changed caps/off mask')
        row=reader.json(Path(rec['root'])/'arms.json')[0]
        require(row['kwargs']['task_levels']==levels[cell] and row['kwargs']['task_caps']==capvec,
                'Served adapter differs from frozen task table')
        require(table['fit']['path']==row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1] and
                rec['fit_sha256'][table['fit']['path']]==table['fit']['sha256'],'Wrong adapter fit binding')
        require(datetime.fromisoformat(table['emitted_at_utc'])>=datetime.fromisoformat(freeze['frozen_at_utc']),
                'Test root emitted before freeze')
        all_arms[cell] = dict(task=load(rec,rec['arms'][0],expected,'A',k,reader,freeze=freeze))
        for key,record in [('full_c10',source['full_test_c10']),('full_descent',source['full_test_descent'])]:
            all_arms[cell][key] = load(record,record['arm'],expected,'A',10 if m=='pi05' else 8,reader)
    result = calculate_test(all_arms)
    result['audit'] = {c:{a:e.audit for a,e in arms.items()} for c,arms in all_arms.items()}
    reader.stable(); frozen_tables(tables); result['inputs'] = reader.inputs
    write_json(output/'test_a.json',result); atomic(output/'TEST_A.md',markdown(result,'test A').encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='action',required=True)
    sub.add_parser('verify')
    for action,default in [('phase1','phase1_results'),('dev','dev_results'),('freeze','dev_results'),('test','test_results')]:
        parser = sub.add_parser(action); parser.add_argument('--out',type=Path,default=HERE/default)
        if action=='test': parser.add_argument('--tables',type=Path,default=HERE/'dev_results')
        if action in ('dev','test'):
            parser.add_argument('--conductor',type=Path,help='Resident group index (adaptive dev or measure-only test)')
    args = ap.parse_args()
    if args.action=='verify':
        r=Reader(); verify(r); r.stable(); print('DESIGN_VERIFIED; no outcome reads'); return
    out = args.out.resolve(); require(out.is_relative_to(HERE) and out!=HERE,'Output must be below R16')
    if args.action=='phase1': apply_phase1(out)
    elif args.action=='dev':
        if args.conductor:
            from .conductor_analysis import apply_dev as apply_conductor_dev
            apply_conductor_dev(out,args.conductor)
        else: apply_dev(out)
    elif args.action=='freeze':
        if json.loads((out/'dev.json').read_bytes()).get('adaptive'):
            from .conductor_analysis import freeze
            freeze(out)
        else: freeze_tables(out)
    else:
        if args.conductor:
            from .conductor_analysis import apply_test as apply_conductor_test
            apply_conductor_test(out,args.conductor,args.tables)
        else: apply_test(out,args.tables)


if __name__=='__main__': main()
