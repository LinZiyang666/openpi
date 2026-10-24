"""Strict partial-cohort adapter. Numerical pairing/bootstrap remain R14/R15's."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import yaml

from . import analyze as a
from .conductor import read_group, verify_pack, unpack, make_task_uid
from .prepare import HERE, CELLS, ORDER, parts, write_json, atomic, frozen_tables
from .cost import pricing


def verify_amendment(reader, path, original_digest, current):
    lock=reader.json(HERE/'CONDUCTOR_LOCK.json')
    a.require(lock['schema']=='r16.conductor.amendment.v1' and
              lock['original_design_lock_sha256']==a.sha(reader.read(HERE/'DESIGN_LOCK.json')) and
              lock['design_sha256']==a.sha(reader.read(HERE/'design.json')) and
              lock['protocol_sha256']==a.sha(reader.read(HERE/'PROTOCOL.md')),'Changed amendment anchor')
    record=lock['adapted_sources'].get(path)
    a.require(record and record['original_sha256']==original_digest and
              record['adapted_sha256']==a.sha(current),'Unregistered conductor source change')
    a.require(a.sha(reader.read(record['original_copy']))==original_digest,'Changed original source copy')
    for source,digest in lock['new_sources'].items():
        a.require(a.sha(reader.read(source))==digest,'Changed conductor implementation')
    a.require(a.sha(reader.read(HERE/'CONDUCTOR_AMENDMENT.json'))==lock['amendment_sha256'],'Changed amendment')


def load_entry(group,entry,reader,*,freeze=None):
    """An accepted failure without a single decision (a worker failure: the server never served it) is no outcome:
    -1, listed in audit['decisionless']."""
    row=entry['row']; name=row['arm']; k=group['k']; pool=row['init_pool']
    width=30 if pool=='B' else 50
    pairs=sorted(map(tuple,entry['pairs'])); pos={p:j for j,p in enumerate(pairs)}
    a.require(len(pairs)==len(pos)==10*width and all(sum(t==j for t,i in pairs)==width for j in range(10)),
              'Wrong balanced manifest')
    fit=row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1]
    a.require(a.sha(reader.read(fit))==entry['fit_sha256'],'Changed resident fit')
    args=row['plugin_args']
    # Long execution segment on/off and the Q3 stall_only judge are per-batch settings (guard_only removed 10-20).
    a.require(row['replan_steps']==5 and row['full_model'] and row['recording'] is False and
              args[args.index('--os-rollout-seed')+1]=='2026101401','Changed client/policy protocol')
    cfg=yaml.safe_load(reader.read(row['yaml']))
    a.require(cfg['miss']['num_steps']==k,'Wrong resident K')
    manifest=json.loads(reader.read(row['manifest']))
    actual_pairs=manifest if isinstance(manifest,list) else manifest['pairs']
    a.require(sorted(map(tuple,actual_pairs))==pairs and row['dev']==(pool=='B'),'Wrong manifest/pool')
    if pool=='B':
        contract=entry['pool_contract']
        a.require(contract['fit_sha256']==entry['fit_sha256'] and contract['init_pool']=='B' and contract['dev'] is True and
                  contract['manifest_sha256']==a.sha(json.dumps(pairs,separators=(',',':')).encode()),'Wrong B contract')
        if not row.get('r15_leave_own_episode_out',False):
            a.require(all(i not in contract['subset']['excluded_inits_by_task'][str(t)] for t,i in pairs),'B subset leak')
    root=Path(group['root']); launch=reader.json(root/'launch.json')
    a.require(launch['server_host']=='local' and launch['worker_host']=='timan107' and launch['STAGE1_ONLY']==0 and
              launch['group_sha256']==group['group_sha256'] and launch['smoke_episodes'] is None,'Wrong fleet/full model/smoke')
    accepted={}
    for r in reader.jsonl(root/'journal.jsonl'):
        if r['yaml_id']!=name: continue
        uid=r['task_uid']
        a.require(uid==make_task_uid(name,'eval',*a.uid_pair(uid)) and a.uid_pair(uid) in pos,'Wrong original UID')
        if r.get('accepted') and not r.get('error') and r['status'] in ('done','failed'):
            a.require(type(r['success']) is bool,'Nonbinary success'); accepted[uid]=r
    y=np.full(len(pairs),-1,dtype=np.int64); n=np.zeros(len(pairs),dtype=np.int64);v=n.copy();m=n.copy()
    reused=None
    if 'reuse' in entry:
        record=reader.json(entry['reuse']);verify_pack(record,reader);reused=unpack(record)
        a.require(reused.audit['arm']==name and reused.audit['k']==k,'Wrong reused arm')
    decisions={};starts=0;void=[]
    run=root/'runs'/name; shards=sorted(run.glob('server_*/decisions_*.jsonl'))
    reader.shards[str(run)]=list(map(str,shards))
    for path in shards:
        for d in reader.jsonl(path):
            if d.get('ev')=='startup':
                starts+=1
                a.require(d['method_spec']==row['method'] and d['kwargs']==row['kwargs'] and d['miss_steps']==k,
                          'Wrong resident startup method/K')
                argv=d['argv'];a.require(argv[argv.index('--os-fit-artifact')+1]==fit,'Wrong resident source fit')
                if freeze:
                    a.require(d['ts']>=datetime.fromisoformat(freeze['frozen_at_utc']).timestamp(),'A started before freeze')
            if d.get('ev')!='dec':continue
            uid=d['uid'];a.require(d['arm']==name and d['yaml_id']==name and d['bundle']==name,'Cross-arm decision contamination')
            a.require(a.uid_pair(uid) in pos and uid.startswith(name+':eval:'),'Wrong decision identity')
            if uid in accepted and int(d.get('attempt',1))==accepted[uid]['attempt']:
                key=(uid,d['step'])
                if key in decisions:
                    a.require({k:v for k,v in decisions[key].items() if k not in ('ts','conn')}==
                              {k:v for k,v in d.items() if k not in ('ts','conn')},'Conflicting accepted decision duplicate')
                decisions[key]=d
    for uid,r in accepted.items():
        j=pos[a.uid_pair(uid)];t=a.uid_pair(uid)[0]
        y[j]=r['success']
        if r['attempt']==0:
            a.require(reused is not None and r['run_id'].startswith('reuse_'),'Unproved reused outcome')
            offset=j%width
            a.require(y[j]==reused.y[t,offset],'Reused journal outcome changed')
            n[j],v[j],m[j]=reused.n[t,offset],reused.v[t,offset],reused.m[t,offset]
            continue
        rows=sorted((d for (u,s),d in decisions.items() if u==uid),key=lambda d:d['step'])
        if not rows and r['success'] is False:
            y[j]=-1;void.append(a.uid_pair(uid));continue
        a.require(starts>0,'Missing startup audit')
        a.require(rows and [d['step'] for d in rows]==list(range(len(rows))),'Missing/noncontiguous decisions')
        for d in rows:
            a.require(type(d['hit']) is bool and type(d['vision']) is bool and d.get('ok',True) and
                      (d['hit'] or d['vision']) and d.get('stage1_mode','full')=='full','Invalid decision')
            a.require(d['hit'] or d['miss_k']==k,'Wrong actual policy K')
            a.require(d['task_id']==t and d['init']==a.uid_pair(uid)[1] and
                      d['r14']['fit_sha256']==entry['fit_sha256'],'Wrong task/fit audit')
            if d.get('source',d.get('src'))=='policy_tail':
                a.require(d['hit'] and not d['vision'] and d.get('miss_k') is None,'Policy tail made a call')
            if row.get('r15_leave_own_episode_out') and d['vision']:
                diag=d['r14'];own=diag['excluded_episode']
                a.require(diag['leave_own_episode_out'] is True and diag['query_pool']=='B' and own>=0 and
                          diag['query_init']==a.uid_pair(uid)[1] and own not in diag['candidate_episode_ids'] and
                          own not in diag['donor_episode_ids'] and
                          len(diag['donor_episode_ids'])==len(diag['parent_donor_rows'])>0,'Wrong LOEO evidence')
            if row['r14_descent_level']=='c10': a.require(not d['vision'] or not d['hit'],'Pure C10 vision HIT')
            n[j]+=1;v[j]+=d['vision'];m[j]+=not d['hit']
    held=np.zeros(len(pairs),dtype=bool)
    held[[pos[p] for p in map(tuple,entry.get('dispatch_pairs',pairs)) if p in pos and p not in void]]=True
    n,v,m,y,held=(x.reshape(10,width) for x in (n,v,m,y,held))
    # Each scheduled task-level (its dispatch pairs less decision-less ones) must be a whole balanced cohort; skipped
    # tasks are explicitly unknown (-1), never manufactured successes or SR failures.
    a.require(all(np.all(y[t][held[t]]>=0) or np.all(y[t][held[t]]==-1) for t in range(10)),'Partial cohort')
    price=pricing(row['model'],k)
    return a.Episodes(y,n,v,m,price['vision']*v+price['miss']*m,dict(
        arm=name,group_sha256=group['group_sha256'],k=k,pricing=price,
        completed_tasks=[t for t in range(10) if np.all(y[t]>=0)],reuse=entry.get('reuse'),
        decisions=int(n.sum()),vision=int(v.sum()),misses=int(m.sum()),controls=int(5*n.sum()),
        **({'decisionless':[list(p) for p in sorted(void)]} if void else {})))


def partial_choose(stats,tolerance,bound=False):
    selected='c10';failure=None;passes={'c10':True}
    for level in ORDER:
        a.require(level in stats,'Missing level before first failure')
        row=stats[level]
        passed=(row['simultaneous_lower95']>=-tolerance/100-1e-12) if bound else (
            100*(row['successes']-row['pure_successes'])>=-tolerance*row['n'])
        passes[level]=passed
        if not passed:failure=level;break
        selected=level
    return dict(level=selected,first_failure=failure,passes=passes,
                diagnostics_below_first_failure='unobserved under adaptive dispatch')


def adaptive_cell(cell,arms,reference,*,draws=5000):
    tasks={str(t):{l:a.paired(ep.y[t],reference.y[t],family=200) for l,ep in arms.items()
                   if np.all(ep.y[t]>=0)} for t in range(10)}
    result=dict(reference=reference.stats(),task_stats=tasks,tolerances={},fallback='k-step pure C10',
                uniform='dropped by owner speed amendment',unobserved_tasks={l:[t for t in range(10) if np.all(ep.y[t]<0)] for l,ep in arms.items()})
    indices=np.random.default_rng(20261005).integers(0,30,size=(draws,10,30))
    for tol in (0,2,5):
        decisions={t:partial_choose(stats,tol) for t,stats in tasks.items()}
        levels={t:d['level'] for t,d in decisions.items()}
        record=dict(levels=levels,task_decisions=decisions,
                    confidence_sensitivity=dict(levels={t:partial_choose(stats,tol,True)['level'] for t,stats in tasks.items()}))
        # Sensitivity tables do not justify running extra fallback cohorts or
        # interpreting unknown outcomes. Primary stitched SR/IR is measurable.
        if tol==5:
            a.require(all(np.all(arms[l].y[int(t)]>=0) for t,l in levels.items()),'Missing selected fallback/level')
            selected=a.stitch(arms,levels)
            record.update(selected=selected.stats(),selected_vs_full=a.cell_pair(selected.y,reference.y,family=200),
                selected_vs_full_bootstrap_ci95=a.interval(a.sampled_sum(selected.y-reference.y,indices)/300),
                selected_ir_ci95=a.interval(a.sampled_sum(selected.cost,indices)/a.sampled_sum(selected.n,indices)))
        result['tolerances'][str(tol)]=record
    return result


def group_index(index,reader):
    spec=reader.json(index)
    groups=[read_group(p,reader) for p in spec['groups']]
    a.require(len(groups)==4 and {(g['model'],g['k']) for g in groups}=={(m,k) for m in ('pi05','groot') for k in (1,2)},'Incomplete group index')
    for g in groups:
        a.require(g['design_sha256']==a.sha(reader.read(HERE/'design.json')) and
                  g['amendment_sha256']==a.sha(reader.read(HERE/'CONDUCTOR_AMENDMENT.json')),'Changed group design binding')
        done=reader.json(Path(g['root'])/'DONE.json')
        a.require(done['group_sha256']==g['group_sha256'],'Group not complete')
    return groups


def apply_dev(output,index):
    a.require(not (output/'TABLE_FREEZE.json').exists(),'Frozen tables cannot be overwritten')
    reader=a.Reader();design=a.verify(reader);groups=group_index(index,reader)
    cells={}
    for group in groups:
        a.require(group['phase']=='dev','Wrong dispatch phase')
        references=reader.json(group['references'])
        for record in references.values():verify_pack(record,reader)
        loaded={e['row']['arm']:load_entry(group,e,reader) for e in group['entries']}
        for cell in CELLS:
            if (parts(cell)[0],parts(cell)[3])!=(group['model'],group['k']):continue
            entries={e['row']['r14_descent_level']:e for e in group['entries'] if e['row']['r16_cell']==cell}
            arms={l:loaded[e['row']['arm']] for l,e in entries.items() if l in ORDER}
            m,s,n,k=parts(cell)
            fallback=next(e for e in group['entries'] if e['row']['r14_descent_level']=='c10' and
                          parts(e['row']['r16_cell'])[1]==s and ((parts(e['row']['r16_cell'])[2]==50)==(n==50)))
            arms['c10']=loaded[fallback['row']['arm']]
            cells[cell]=adaptive_cell(cell,arms,unpack(references[cell]))
            cells[cell]['audits']={l:e.audit for l,e in arms.items()}
    a.require(set(cells)==set(CELLS),'Incomplete 32 cells')
    result=dict(cells=cells,primary_tolerance_pp=5,adaptive=True,uniform_tables=False,
                amendment_sha256=a.sha(reader.read(HERE/'CONDUCTOR_AMENDMENT.json')),
                stitched_development_is_optimistic=True,fits_include_own_episode_at_500=True,
                confidence_family=200,selection_reference='full-step pure C10')
    reader.stable();result['inputs']=reader.inputs
    write_json(output/'dev.json',result)
    for tol in (0,2,5):
        table={c:r['tolerances'][str(tol)]['levels'] for c,r in cells.items()}
        write_json(output/f'levels_tol{tol}pp.json',table)
        if tol==5:write_json(output/'levels.json',table)
    lines=['# R16 adaptive development','', 'Primary 5 pp first failure. Uniform tables dropped by owner speed amendment.',
           'Skipped task-levels are unobserved. Test A has not been read.','',
           '| cell | tasks 0..9 | selected SR | selected IR |','|---|---|---:|---:|']
    for c,r in cells.items():
        d=r['tolerances']['5'];lines.append(f'| {c} | '+ ' / '.join(d['levels'][str(t)] for t in range(10))+
            f' | {d["selected"]["sr"]:.4f} | {d["selected"]["ir"]:.6f} |')
    atomic(output/'DESCENT.md',('\n'.join(lines)+'\n').encode())


def freeze(output):
    a.require(not (output/'TABLE_FREEZE.json').exists(),'Already frozen')
    reader=a.Reader();a.verify(reader);result=reader.json(output/'dev.json')
    a.require(result.get('adaptive') is True and result['uniform_tables'] is False and set(result['cells'])==set(CELLS),'Wrong adaptive result')
    for tol in (0,2,5):
        table=reader.json(output/f'levels_tol{tol}pp.json')
        a.require(table=={c:r['tolerances'][str(tol)]['levels'] for c,r in result['cells'].items()},'Changed selection')
    a.require(reader.json(output/'levels.json')==reader.json(output/'levels_tol5pp.json'),'Wrong primary table')
    for path,info in result['inputs'].items():a.require(a.sha(reader.read(path))==info['sha256'],'Input changed since selection')
    reader.stable();names=['dev.json','DESCENT.md','levels.json']+[f'levels_tol{tol}pp.json' for tol in (0,2,5)]
    write_json(output/'TABLE_FREEZE.json',dict(schema='r16.tables.conductor.v1',adaptive=True,uniform_tables=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),design_sha256=a.sha(reader.read(HERE/'design.json')),
        amendment_sha256=a.sha(reader.read(HERE/'CONDUCTOR_AMENDMENT.json')),
        conductor_lock_sha256=a.sha(reader.read(HERE/'CONDUCTOR_LOCK.json')),
        protocol_sha256=a.sha(reader.read(HERE/'PROTOCOL.md')),
        sha256={n:a.sha(reader.read(output/n)) for n in names},cells=list(CELLS),primary_tolerance_pp=5,test_a_outcome_reads=0))


def apply_test(output,index,tables):
    levels,uniform,freeze_record=frozen_tables(tables)
    reader=a.Reader();design=a.verify(reader);groups=group_index(index,reader);all_arms={}
    for group in groups:
        a.require(group['phase']=='test','Wrong dispatch phase')
        for e in group['entries']:
            row=e['row'];cell=row['r16_cell'];m,s,n,k=parts(cell)
            a.require(e['freeze']==freeze_record,'Wrong test freeze')
            table=reader.json(Path(e['source_root'])/'descent_table.json')
            capvec=[design['cells'][cell]['periodic_caps'][levels[cell][str(t)]] for t in range(10)]
            a.require(table['levels']==levels[cell] and table['caps']==capvec and table['uniform']==uniform[cell] and
                      table['off_tasks']==[t for t in range(10) if levels[cell][str(t)]=='off'] and
                      row['kwargs']['task_levels']==levels[cell] and row['kwargs']['task_caps']==capvec,'Wrong frozen serving table')
            task=load_entry(group,e,reader,freeze=freeze_record)
            a.require(np.all(task.y>=0),'Test arm needs all 500 episodes')
            all_arms[cell]=dict(task=task)
            for key,source in (('full_c10','full_test_c10'),('full_descent','full_test_descent')):
                ref=design['cells'][cell][source]
                all_arms[cell][key]=a.load(ref,ref['arm'],[(t,i) for t in range(10) for i in range(50)],'A',10 if m=='pi05' else 8,reader)
    result=a.calculate_test(all_arms);result['audit']={c:{n:e.audit for n,e in arms.items()} for c,arms in all_arms.items()}
    reader.stable();frozen_tables(tables);result['inputs']=reader.inputs
    write_json(output/'test_a.json',result);atomic(output/'TEST_A.md',a.markdown(result,'test A').encode())
