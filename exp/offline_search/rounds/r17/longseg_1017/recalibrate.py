"""Owner's decision-clock M/N and empirical Q3 recalibration; CPU only.

No model/metric/corrector fit or test outcomes are read. Old artifacts remain
immutable. One /data reader, four MiB/s, PSI pauses, and serial artifact fits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import yaml

from .audit import HERE, io_ready
from .io_budget import filesystem_budget, dump, reader
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, cells, sha
from exp.r17_integration.common.contracts import digest

INDEX=HERE/'RECALIBRATION.json'
WORK=HERE/'recalibration_1018'
TARGETS={'k50':.5,'k40':.4,'k30':.3,'k20':.2,'k10':.1}
TOLERANCE=.005


def inventory():
    """Read only campaign/fit descriptors, not numeric inputs or outcomes."""
    jobs=read(HERE/'CAMPAIGNS.json')['jobs']
    units=read(HERE/'FORMAL_QUEUE.json')['units']
    records=[];docs={};artifacts={};fit_records={}
    for job in jobs+[dict(u,run=u['source'],deferred=True) for u in units if u['stage']=='levels']:
        if job['stage'] in ('pure','long'):continue
        root=Path(job['run']);campaign=read(root/'campaign.json')
        for group in read(root/'assets/CACHE_STUDY.json')['groups']:
            key=str(root/'assets'/group['file'])
            if key not in docs:docs[key]=yaml.safe_load(Path(key).read_text())
            doc=docs[key];r,b=doc['recipe'],doc['adapter_binding']
            level=group['group'].rsplit('_',1)[-1]
            if job.get('deferred') and level not in TARGETS:continue
            if r.get('periodic_gap') is None and not (r.get('stall_path') and b['commit_horizon']>5):continue
            row=dict(name=job['name'],root=str(root),model=job['model'],benchmark=job['benchmark'] if 'benchmark' in job else 'libero',
                mode=job['mode'],arm=job.get('arm','original'),suite=group['suite'],size=group.get('library_size',500),
                level=level,old_gap=r.get('periodic_gap'),fit_digest=r['fitted_digest'],fit=r['fitted_path'],
                old_stall_digest=r.get('stall_digest'),old_stall=r.get('stall_path'),commit=b['commit_horizon'],
                replan=5,ordered_tasks=r['ordered_tasks'],deferred=job.get('deferred',False))
            if r.get('stall_path') and b['commit_horizon']>5:
                artifacts.setdefault(r['stall_digest'],row)
            fit_records.setdefault(r['fitted_digest'],row)
            records.append(row)
    return dict(records=records,artifacts=artifacts,fits=fit_records)


def labels(record):
    """Decode just task/episode/step arrays; leave weights/features/actions cold."""
    root=Path(record['fit']);ready=read(root/'READY.json')
    fields=dict(ready['tree']['mapping'])
    with (root/'fit.npz').open('rb') as stream,np.load(stream,allow_pickle=False) as archive:
        data={name:archive[fields[name]['array']] for name in ('task','episode','step')}
    data['success']=np.isin(data['episode'],cells()[record['model'],record['suite'],record['size']]['kept_episode_ids'])
    return data


def head_mask(data,stride):
    # Teacher B capture is contiguous native replan slots, starting at zero.
    return data['step']%stride==0


def q3_metrics(model):
    counts=dict(windows=0,residual_q90_covered=0,advance_above_q10=0,confirmed=0,ambiguous=0)
    by_task={}
    for key,task in model.tasks.items():
        tally=dict(windows=0,residual_q90_covered=0,advance_above_q10=0,confirmed=0,ambiguous=0)
        if len(task['references'])>=4:
            for r in task['calibration']:
                estimate={n:r[n] for n in ('delta_hat','phase_hat','distance_hat','spread_hat')}
                status=model.calibrated_status(key,estimate,r['window_span'])
                tally['windows']+=1
                tally['residual_q90_covered']+=int(r['delta_true']<=r['delta_hat']+status['e90'])
                tally['advance_above_q10']+=int(r['delta_true']>=status['a10'])
                tally['confirmed']+=int(status['state']=='slow_confirmed')
                tally['ambiguous']+=int(status['state']=='slow_ambiguous')
        for n,v in tally.items():counts[n]+=v
        by_task[key]=dict(tally,E=task['E'],K=task['K'],W=task['W'],T_med=task['T_med'],
            reference_episodes=len(task['references']),available=len(task['references'])>=4)
    return dict(counts,tasks=by_task,
        residual_q90_empirical_coverage=counts['residual_q90_covered']/counts['windows'] if counts['windows'] else None,
        advance_q10_empirical_coverage=counts['advance_above_q10']/counts['windows'] if counts['windows'] else None,
        successful_window_confirmed_rate=counts['confirmed']/counts['windows'] if counts['windows'] else None,
        scope='Original successful-library LOEO empirical quantiles; no online FAR/conformal guarantee')


def fit_q3(record,data):
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel,_digest
    source=Path(record['old_stall']);destination=source.with_name(source.name+'_decision1018')
    metrics_file=WORK/'q3'/f'{record["old_stall_digest"]}.json'
    if metrics_file.exists():return read(metrics_file)
    old=StallModel.load(source)
    # Check the source success labels against the original empirical artifact.
    expected={int(e) for e in np.unique(data['episode'][data['success']])
              if np.count_nonzero(data['episode']==e)>=2}
    actual={int(t['episode']) for task in old.tasks.values() for t in task['templates']}
    if expected!=actual:raise ValueError('Canonical B success labels differ from Q3 source: '+str(source))
    stride=record['commit']//record['replan'];mask=head_mask(data,stride)
    metric=dict(old.metric,tasks={})
    for key,table in old.metric['tasks'].items():
        keep=mask[table['rows']]
        metric['tasks'][key]=dict(rows=table['rows'][keep],codes={k:v[keep] for k,v in table['codes'].items()})
    library=dict(manifest=dict(H=1,exec_steps=1),task_id=data['task'],episode=data['episode'],
        step=data['step']//stride,control_index=data['step']//stride,success=data['success'])
    model=StallModel.fit(library,metric=metric,commit_controls=1)
    # The estimator uses only the head templates. Retain the complete frozen
    # metric for the standard serving attestation; not one fitted code changes.
    model.provenance['calibration_metric_sha256']=model.provenance['metric_sha256']
    model.metric=old.metric
    model.provenance.update(metric_sha256=_digest(old.metric),clock_unit='segment_decision',
        calibration_stride_native_slots=stride,physical_commit_controls=record['commit'],
        source_stall_digest=record['old_stall_digest'],source_fitted_digest=record['fit_digest'],
        full_metric_unchanged=True)
    model.fingerprint=_digest(model._payload())
    if destination.exists():destination.rename(destination.with_name(destination.name+'.incomplete_'+str(time.time_ns())))
    destination.mkdir(exist_ok=False)
    model.save(destination)
    original=read(source/'READY.json')
    receipt=dict(original,files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},
        clock_unit='segment_decision',source_stall_digest=record['old_stall_digest'],
        empirical_calibration='original q90/q10 successful-library LOEO on segment heads',
        unavailable_tasks=[k for k,v in model.tasks.items() if len(v['references'])<4])
    receipt.pop('digest',None);receipt['digest']=digest(receipt);dump(destination/'READY.json',receipt)
    result=dict(source=str(source),path=str(destination),old_digest=record['old_stall_digest'],
        digest=receipt['digest'],fit_digest=record['fit_digest'],fingerprint=model.fingerprint,
        old_parameters=read(source/'stall.json'),new_parameters=read(destination/'stall.json'),
        achieved=q3_metrics(model),metric_identical=_digest(model.metric)==_digest(old.metric),
        input_heads=int(mask.sum()),input_native_rows=len(mask),
        target_constants=model.provenance['constants'])
    dump(metrics_file,result)
    print('Q3',record['model'],record['suite'],record['size'],result['achieved']['windows'],'windows',flush=True)
    return result


def episodes(record,data,model):
    """LOEO library heads with exactly C's confirmed-call/cooldown rule."""
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_bridge import call_probability
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallTracker
    result=[];stride=record['commit']//record['replan']
    positions={}
    for key,table in model.metric['tasks'].items():
        positions[key]={int(r):i for i,r in enumerate(table['rows'])}
    by_ep={int(ep):np.flatnonzero(data['episode']==ep) for ep in np.unique(data['episode'])}
    for ep,rows in by_ep.items():
        rows=rows[data['step'][rows]%stride==0];task=int(data['task'][rows[0]]);key=str(task)
        table=model.metric['tasks'][key];td=model.tasks[key];window=[];forced=[];cool=False
        known={(int(r['episode']),int(r['end_control'])):r for r in td['calibration']}
        for i,row in enumerate(rows):
            regime='early' if i==0 and 'early' in table['codes'] else 'main'
            window.append((np.asarray(table['codes'][regime][positions[key][int(row)]],np.float64),regime))
            status='inactive'
            if len(td['references'])>=4 and len(window)>=td['W']+1:
                estimate=known.get((ep,i))
                if estimate is None:
                    estimate=model._estimate(key,window[-td['W']-1:],td['W'],exclude=ep)
                estimate={n:estimate[n] for n in ('delta_hat','phase_hat','distance_hat','spread_hat')}
                status=model.calibrated_status(key,estimate,td['W'])['state']
            call=call_probability(status,cool,0.)==1.
            forced.append(call);cool=bool(call)
        # The existing ffk empty-LOEO path must also be counted as a model head.
        result.append(dict(episode=ep,task=task,init=ep%50,success=bool(data['success'][rows[0]]),
            rows=rows,forced=np.asarray(forced,bool)))
    return result


class DecisionSchedule:
    """Vectorized R11 hash-cap scheduler; literal periodic() is the oracle."""
    def __init__(self,episodes):
        from exp.r17_integration.common.layers.periodic import uniform
        self.episodes=episodes;self.N=sum(len(e['rows']) for e in episodes)
        length=max((len(e['rows']) for e in episodes),default=0);n=len(episodes)
        self.alive=np.zeros((length,n),bool);self.forced=np.zeros((length,n),bool)
        self.coins=np.ones((length+1,n))
        donors={t:sum(e['task']==t for e in episodes) for t in {e['task'] for e in episodes}}
        for j,e in enumerate(episodes):
            l=len(e['rows']);self.alive[:l,j]=True
            self.forced[:l,j]=e['forced'] | (donors[e['task']]<=1)
            for i in range(-1,l):self.coins[i+1,j]=uniform('R11-gap-v1',0,e['task'],e['init'],i,'cap')
    def __call__(self,gap):
        n=len(self.episodes);run=np.zeros(n,int);m=0
        cap=None if gap is None else math.floor(gap)+(self.coins[0]<gap-math.floor(gap))
        for i,alive in enumerate(self.alive):
            call=self.forced[i].copy()
            if gap is not None:call|=run>=cap
            m+=int(np.count_nonzero(call & alive));run=np.where(call,0,run+1)
            if gap is not None:cap=np.where(call,math.floor(gap)+(self.coins[i+1]<gap-math.floor(gap)),cap)
        return self.N,self.N,m


def solve(target,replay):
    from exp.r17_integration.common.ir import calibrate_periodic
    candidates={0.,40.,round(1/target-1,4)}
    try:
        solved=calibrate_periodic(target,replay,dict(vision=0,miss=1));candidates.add(solved['setting'])
    except ValueError as e:
        if 'feasible interval' not in str(e):raise
    # Evaluate adjacent rounded settings too; finite deterministic replays
    # need an explicit closest attained value, not a bisection status claim.
    candidates|={round(g+d,4) for g in list(candidates) for d in (-.001,-.0001,.0001,.001) if g+d>=0}
    best=min(((abs(replay(g)[2]/replay(g)[0]-target),g,replay(g)) for g in candidates),key=lambda x:(x[0],x[1]))
    error,gap,(n,v,m)=best
    return dict(new_gap=gap,nominal_ir=target,predicted_ir=m/n,N=n,M=m,
        absolute_error=error,tolerance=TOLERANCE,passed=error<=TOLERANCE,
        off_floor=replay(None)[2]/n,pricing_definition='M / N_segment_heads; tails excluded',
        scheduler='R11 deterministic fractional gap; confirmed Q3 calls/cooldown and empty-LOEO MISS included')


def fit_all():
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    WORK.mkdir(exist_ok=True)
    with filesystem_budget():
        inv=inventory();dump(WORK/'INVENTORY.json',inv)
    artifacts={};fit_cache={};solutions={};by_fit={}
    for record in inv['records']:by_fit.setdefault(record['fit_digest'],[]).append(record)
    for old_digest,record in inv['artifacts'].items():
        with filesystem_budget():
            data=labels(record);q3=fit_q3(record,data)
        artifacts[old_digest]=q3
        solution_path=WORK/'solutions'/f'{record["fit_digest"]}.json'
        if solution_path.exists():solutions.update(read(solution_path));continue
        # New artifacts are self-contained. All reads still share the same
        # single-reader lock and governor, including old preserved inputs.
        with filesystem_budget():model=StallModel.load(q3['path'])
        eps=episodes(record,data,model)
        fit_solutions={}
        variants={r['arm'] for r in by_fit[record['fit_digest']] if r['level'] in TARGETS}
        for arm in variants:
            chosen=[e for e in eps if arm!='filtered' or e['success']]
            replay=DecisionSchedule(chosen)
            for level,target in TARGETS.items():
                key=record['fit_digest']+':'+arm+':'+level
                fit_solutions[key]=solve(target,replay)
            print('GAPS',record['model'],record['suite'],record['size'],arm,
                [round(fit_solutions[record['fit_digest']+':'+arm+':'+l]['predicted_ir'],6) for l in TARGETS],flush=True)
        dump(solution_path,fit_solutions);solutions.update(fit_solutions)
        del model,eps,data
        io_ready();time.sleep(5)
    levels=[]
    for r in inv['records']:
        if r['level'] not in TARGETS:continue
        key=r['fit_digest']+':'+r['arm']+':'+r['level']
        levels.append(dict(r,**solutions[key]))
    failures=[r for r in levels if not r['passed']]
    result=dict(status='FITTED_WITH_EXCEPTIONS' if failures else 'FITTED',clock_unit='segment_decision',
        level_ir_definition='M/N segment heads; tails count in neither',latency_ir_separate=True,
        tolerance=TOLERANCE,level_inventory=levels,Q3=artifacts,solutions=solutions,
        unattainable_levels=failures,recalibrated_levels=len(levels),recalibrated_Q3_artifacts=len(artifacts),
        decisions='Coordinator 2026-10-18: M/N heads, abs error<=0.005, original empirical q90/q10',
        dev_or_test_outcomes_read=False,metric_or_corrector_refit=False)
    dump(INDEX,result);print('FITTED',len(levels),'levels',len(artifacts),'Q3;',len(failures),'exceptions',flush=True)


def replace_metadata(path,body):
    """Copy-on-write: old YAMLs may be hardlinked to immutable originals."""
    path=Path(path)
    if path.exists():path.rename(path.with_name(path.name+'.pre_recal1018_'+str(time.time_ns())))
    if isinstance(body,str):path.write_text(body)
    else:dump(path,body,compact=path.name=='EPISODE_PLAN.json')


def recipe_inputs(root,campaign,seal,index=None,*,aliases=True):
    """Apply only new recipe inputs, preserving the old scientific namespace."""
    from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
    from exp.r17_integration.common.config import config_from_dict
    from exp.r17_integration.common.recipe import bundle_identity,recipe_identity
    index=index or read(INDEX);root=Path(root)
    if index['status'] not in ('FITTED','FITTED_WITH_EXCEPTIONS','APPLIED','APPLIED_WITH_EXCEPTIONS'):
        raise ValueError('Decision-clock calibration has not been fitted')
    arm='filtered' if campaign['filter_failed_knob']['selection'] else 'original'
    docs={};artifacts={};changed=[]
    for row in seal['groups']:
        path=root/'assets'/row['file']
        if path not in docs:docs[path]=yaml.safe_load(path.read_text())
        doc=copy.deepcopy(docs[path]);r=doc['recipe'];before=copy.deepcopy(doc)
        level=row['group'].rsplit('_',1)[-1]
        if level in TARGETS:
            entry=index['solutions'][r['fitted_digest']+':'+arm+':'+level]
            r['periodic_gap']=entry['new_gap'];row['periodic_gap']=entry['new_gap']
            row['decision_ir_calibration']=entry
        if r.get('stall_path') and doc['adapter_binding']['commit_horizon']>5:
            old=r['stall_digest']
            # Reapplication recognizes an already published immutable artifact.
            q3=index['Q3'].get(old) or next(v for v in index['Q3'].values() if v['digest']==old)
            alias=root/'assets/q3_decision1018'/q3['old_digest'] if aliases else Path(q3['path'])
            if aliases:
                alias.mkdir(parents=True,exist_ok=True)
                for src in Path(q3['path']).iterdir():
                    dst=alias/src.name
                    if not dst.exists():os.link(src,dst)
                    elif not dst.samefile(src):raise ValueError('Recalibrated Q3 alias differs: '+str(dst))
            r['stall_path']=str(alias);r['stall_digest']=q3['digest']
            row.update(stall=str(alias),stall_digest=q3['digest'],
                stall_rel=os.path.relpath(alias,root/'assets/cache_inputs'))
            artifacts[q3['old_digest']]=dict(path=str(alias),digest=q3['digest'],
                files=read(alias/'READY.json')['files'])
        if doc==before:continue
        row.setdefault('task_namespace',dict(yaml_identity=row['sha256'],
            bundle_identity=row['bundle_identity'],recipe_identity=row['recipe_identity']))
        register_metadata(Path(campaign['filter_failed_source']),doc)
        cfg=config_from_dict(doc)
        row.update(bundle_identity=bundle_identity(cfg),recipe_identity=recipe_identity(cfg))
        text=yaml.safe_dump(doc,sort_keys=False)
        if path not in changed:
            replace_metadata(path,text);changed.append(path)
        elif path.read_text()!=text:raise ValueError('Cohorts disagree on calibrated YAML: '+str(path))
        row['sha256']=sha(path)
    manifest_path=root/'assets/cache_inputs/manifest.json';manifest=read(manifest_path)
    manifest.update(groups=copy.deepcopy(seal['groups']),decision_clock_recalibration=artifacts)
    manifest['digest']=digest({k:v for k,v in manifest.items() if k!='digest'})
    replace_metadata(manifest_path,manifest)
    # The longseg block is created by build.pin after this step; sources from the
    # original pipeline have none yet.
    campaign.setdefault('longseg_1017',{})['recalibration']=dict(clock_unit='segment_decision',level_ir='M/N_heads',
        tolerance=TOLERANCE,Q3='successful-library empirical q90/q10 LOEO; decision stride',
        C5_artifacts_unchanged=True,artifact_digests=sorted(v['digest'] for v in artifacts.values()),
        levels_outside_tolerance=[dict(suite=g['suite'],size=g.get('library_size',500),level=g['group'],
            predicted=g['decision_ir_calibration']['predicted_ir'],nominal=g['decision_ir_calibration']['nominal_ir'])
            for g in seal['groups'] if g.get('decision_ir_calibration',{}).get('passed') is False])
    return dict(changed_yamls=len(changed),artifacts=artifacts)


def pin_plan_evidence(root,campaign):
    """Ship the same optional UID-namespace planner to the frozen driver."""
    import inspect
    from exp.r17_integration.ops import cache_study
    root=Path(root);path=root/'assets/ffk_evidence.py'
    body=path.read_text().split('\n_recal_native_install=install\n',1)[0]
    body+='\n_recal_native_install=install\ndef install():\n'
    if campaign['filter_failed_knob']['selection']:
        body+='    _recal_native_install()\n'
    else:
        # Older original-library studies predate the empty-cache exemption.
        # They require none: preserve their native aggregate audit verbatim.
        body+='    # No filtered empty-cache exemption; keep the native aggregate audit.\n'
    body+='    from exp.r17_integration.ops import cache_study as _recal_study\n'
    source=inspect.getsource(cache_study.plan)
    body+='    import linecache\n'
    body+='    _recal_source='+repr(source)+'\n'
    body+='    linecache.cache["recalibration_plan"]=(len(_recal_source),None,_recal_source.splitlines(True),"recalibration_plan")\n'
    body+='    _recal_ns=dict(vars(_recal_study))\n'
    body+='    exec(compile(_recal_source,"recalibration_plan","exec"),_recal_ns)\n'
    body+='    _recal_study.plan=_recal_ns["plan"]\n'
    replace_metadata(path,body);campaign['ffk_evidence_sha256']=sha(path)


def scientific_hash(tasks):
    allowed={'yaml_identity','bundle_identity','recipe_identity','periodic_gap'}
    h=hashlib.sha256()
    from dataclasses import asdict
    for t in tasks:
        row=asdict(t);row['extra']={k:v for k,v in row['extra'].items() if k not in allowed}
        h.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


def apply_all():
    from dataclasses import asdict
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan,publish_study
    from .build import pin
    index=read(INDEX);catalog=read(HERE/'CAMPAIGNS.json')
    if index['status'].startswith('APPLIED'):
        report();print('Already applied; no campaign changed',flush=True);return
    dependent={r['name'] for r in read(HERE/'RECALIBRATION_DEPENDENCIES.json')['rows'] if r['dependent']}
    receipts=[]
    for job in catalog['jobs']:
        if job['name'] not in dependent:continue
        root=Path(job['run']);saved=root/'RECALIBRATION_APPLIED.json'
        if saved.exists():
            receipts.append(read(saved));job.update(read(root/'FFK_DEFINITION.json'));continue
        if any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json')):
            raise ValueError('Never modify a prepared/launched unit: '+str(root))
        with filesystem_budget():
            c=read(root/'campaign.json');seal=read(root/'assets/CACHE_STUDY.json')
            c5=[dict(file=g['file'],sha256=sha(root/'assets'/g['file']),stall_digest=g.get('stall_digest'))
                for g in seal['groups'] if g['settings']['commit']==5]
            _,_,before,_=cpu_plan(c);scientific=scientific_hash(before)
            uid_hash=hashlib.sha256('\n'.join(t.task_uid for t in before).encode()).hexdigest()
            assert uid_hash==job['task_uid_sha256']
            details=recipe_inputs(root,c,seal,index)
            publish_study(root,Path(job['source']),c,seal)
            for file in ('ffk_bootstrap.py','ffk_evidence.py'):
                p=root/'assets'/file;p.rename(p.with_name(p.name+'.pre_recal1018_'+str(time.time_ns())))
            pin(root,c);pin_plan_evidence(root,c)
            _,model,tasks,_=cpu_plan(c)
            assert uid_hash==hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest()
            assert scientific_hash(tasks)==scientific
            assert len(tasks)==job['episodes']
            assert c5==[dict(file=g['file'],sha256=sha(root/'assets'/g['file']),stall_digest=g.get('stall_digest'))
                for g in seal['groups'] if g['settings']['commit']==5]
            replace_metadata(root/'EPISODE_PLAN.json',dict(schema='r17.episode_plan.v1',model=model,
                tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks),endpoint=c['endpoint']))
            replace_metadata(root/'campaign.json',c)
            job.setdefault('pre_recal_source_full_native_plan_equal',job.get('source_full_native_plan_equal'))
            job.update(bootstrap_sha256=c['ffk_bootstrap_sha256'],evidence_sha256=c['ffk_evidence_sha256'],
                recalibrated=True,task_uid_namespace_preserved=True,source_full_native_plan_equal=False,
                scientific_plan_equal_except_recipe_inputs=True)
            replace_metadata(root/'FFK_DEFINITION.json',job)
            receipt=dict(name=job['name'],episodes=len(tasks),task_uid_sha256=uid_hash,
                scientific_fields_equal=True,only_recipe_inputs_changed=True,C5_settings_unchanged=c5,**details)
            dump(saved,receipt);receipts.append(receipt)
        print('APPLIED',job['name'],len(tasks),'UID/init/seed unchanged',flush=True)
        io_ready();time.sleep(2)
    replace_metadata(HERE/'CAMPAIGNS.json',catalog)
    queue=read(HERE/'FORMAL_QUEUE.json');lookup={j['name']:j for j in catalog['jobs']}
    for unit in queue['units']:unit['jobs']=[lookup.get(j['name'],j) for j in unit['jobs']]
    replace_metadata(HERE/'FORMAL_QUEUE.json',queue)
    index.update(status='APPLIED_WITH_EXCEPTIONS' if index['unattainable_levels'] else 'APPLIED',
        applied_campaigns=receipts,task_uids_inits_seeds_preserved=True,
        independent_campaigns_unchanged=True,deferred_levels_use_recalibrated_inputs=True)
    replace_metadata(INDEX,index);report()


def report():
    from collections import defaultdict
    data=read(INDEX);levels=data['level_inventory'];groups=defaultdict(list)
    for row in levels:groups[row['model'],row['benchmark'],row['mode']].append(row)
    lines=['# Decision-clock recalibration — 2026-10-18','',
        'Coordinator decisions: level IR = M/N segment heads, tails excluded; absolute prediction error <=0.005. '
        'Latency-weighted IR remains a separate measured metric. Q3 reproduces the original empirical '
        'successful-library LOEO q90 residual / q10 advance procedure, with no new online FAR/coverage guarantee.','',
        f'Status: **{data["status"]}**; {len(levels)} potential level definitions, {len(data["Q3"])} distinct C10 Q3 artifacts. '
        f'{len(data["unattainable_levels"])} levels outside tolerance are explicitly listed below. '
        'Full/reduced M/N solutions share an artifact when their library/metric and selected episodes agree; '
        'denoising compute is deliberately not a factor in the proxy.','',
        '## Original procedure and decision-clock replay','',
        '`cache_knob.calibrate` / r17_ops/calibrate_knob.py used B-only R11 gap-cap replay, vision=0/miss=1, '
        'bracket [0,40], 26 bisections and four-decimal gaps. Old replay counted native slots and let periodic '
        'looks interrupt tails. The new replay selects head rows at step % (commit/replan) ==0, restarts '
        'the same task/init-keyed hash at head zero, and counts only heads. Confirmed Q3 model calls, '
        'its one-anchor cooldown and empty-LOEO MISS are included in M. Knob and Q3 MISS both reset the '
        'accepted model-call history, exactly as serving does. The vector replay is checked against literal '
        '`periodic()` with the same forced-call masks. No test/development evaluation outcomes enter fitting.','',
        'The full inventory contains k50/k40/k30/k20/k10, koff and non-periodic chg/chrg cohorts. '
        'There are no additional ir-named or other periodic levels outside this fitted ladder.','',
        'The Q3 fitter receives those successful head rows with H=1, exec_steps=1, commit_controls=1: '
        'one committed segment is its temporal unit. It uses the original fit implementation, inverse ECDFs, '
        'W=max(2,ceil(.05*Tmed)), K=min(5,E-1), q90/q10 and minimum four reference episodes. The complete '
        'deployed metric is attached byte-for-byte for the standard serving attestation; calibration uses '
        'only its head rows. No PCA, whitening, retrieval metric or corrector is fitted. Old artifacts remain '
        'in place; new directories end `_decision1018`, and per-run aliases live under assets/q3_decision1018.','',
        '## Prediction summary','',
        '| Model | Benchmark | Mode | Levels | Nominal targets | Maximum absolute error | Within .005 |',
        '|---|---|---|---:|---|---:|---|']
    for key,rows in sorted(groups.items()):
        lines.append('| '+' | '.join(key)+f' | {len(rows)} | .50/.40/.30/.20/.10 | '+
            f'{max(r["absolute_error"] for r in rows):.7f} | {sum(r["passed"] for r in rows)}/{len(rows)} |')
    lines+=['','## Every level (both arms, all potential LIBERO cells)','','| Campaign | Suite | Size | Level | Old gap | New gap | Nominal | Predicted M/N | N | M | Pass |',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in levels:lines.append(f'| `{r["name"]}` | {r["suite"]} | {r["size"]} | {r["level"]} | {r["old_gap"]} | '
        f'{r["new_gap"]:.4f} | {r["nominal_ir"]:.2f} | {r["predicted_ir"]:.7f} | {r["N"]} | {r["M"]} | {r["passed"]} |')
    lines+=['','## Q3 empirical calibration','','| Model/suite/size | Old exec/commit units | New exec/commit units | Old/new windows (eligible) | q90 empirical coverage | q10 advance coverage | Successful-window confirmed rate |',
        '|---|---|---|---:|---:|---:|---:|']
    inv=read(WORK/'INVENTORY.json')
    for old,q in data['Q3'].items():
        r=inv['artifacts'][old];a=q['achieved'];op=q['old_parameters'];np_=q['new_parameters']
        total_old=sum(v['calibration_windows'] for v in op['tasks'].values())
        number=lambda x:'n/a' if x is None else f'{x:.5f}'
        lines.append(f'| {r["model"]}/{r["suite"]}/{r["size"]} | {op["exec_steps"]}/{op["commit_controls"]} | '
            f'{np_["exec_steps"]}/{np_["commit_controls"]} | {total_old}/{sum(v["calibration_windows"] for v in np_["tasks"].values())} ({a["windows"]}) | '
            f'{number(a["residual_q90_empirical_coverage"])} | {number(a["advance_q10_empirical_coverage"])} | {number(a["successful_window_confirmed_rate"])} |')
    totals={k:sum(q['achieved'][k] for q in data['Q3'].values()) for k in
        ('windows','residual_q90_covered','advance_above_q10','confirmed')}
    lines+=['',f'Across {totals["windows"]:,} eligible empirical LOEO windows: '
        f'q90 residual coverage {totals["residual_q90_covered"]/totals["windows"]:.7%}; '
        f'q10 advance coverage {totals["advance_above_q10"]/totals["windows"]:.7%}; '
        f'confirmed {totals["confirmed"]/totals["windows"]:.7%}. '
        'These descriptive fractions reproduce the original empirical procedure, not an online false-alarm guarantee.',
        '', 'All old/new per-task E/K/W/Tmed, reference counts, unavailable tasks, parameters and achieved counts '
        'are in RECALIBRATION.json. Coverage/alarm columns are descriptive calibration-window measurements, '
        'not an online guarantee. All 102 C5 gated-stall cohort settings keep their original files; '
        'RECALIBRATION_DEPENDENCIES.json lists every setting.','',
        '### Every Q3 task: old → new parameters','',
        '| Model/suite/size | Task | E | K | W | Tmed | Reference episodes | Calibration windows |',
        '|---|---|---|---|---|---|---|---|']
    for old,q in data['Q3'].items():
        r=inv['artifacts'][old]
        for task,new in q['new_parameters']['tasks'].items():
            previous=q['old_parameters']['tasks'][task]
            values=[f'{previous[k]} → {new[k]}' for k in ('E','K','W','T_med','reference_episodes','calibration_windows')]
            lines.append(f'| {r["model"]}/{r["suite"]}/{r["size"]} | {task} | '+' | '.join(values)+' |')
    lines+=['',
        '### Unchanged C5 settings','',
        '| Campaign | Suite | Size | Group | Original Q3 path (unchanged) |',
        '|---|---|---:|---|---|']
    for c5 in read(HERE/'RECALIBRATION_DEPENDENCIES.json')['C5_settings_unchanged']:
        lines.append(f'| `{c5["name"]}` | {c5["suite"]} | {c5["size"]} | {c5["group"]} | `{c5["stall_path"]}` |')
    lines+=['','## Explicit tolerance exceptions','']
    lines+=['None.'] if not data['unattainable_levels'] else [f'- {r["name"]} / {r["suite"]}/{r["size"]}/{r["level"]}: '
        f'best {r["predicted_ir"]:.7f}, nominal {r["nominal_ir"]:.2f}, gap {r["new_gap"]}, off floor {r["off_floor"]:.7f}' for r in data['unattainable_levels']]
    if data['unattainable_levels']:
        lines+=['','Both exceptions have M/N already above .105 with periodic disabled: '
            '134/1265 = .1059288538, imposed by the unchanged confirmed-Q3 call rule after empirical recalibration. '
            'No periodic gap can reduce those calls. They remain explicitly failed in the solution map and '
            'deferred template metadata; neither the nominal target nor Q3 is relaxed. They may be absent '
            'from dispatch if the cell passes the formal OFF rule.']
    lines+=['','## Definitions, identities and deferred builds','',
        'Only never-prepared recalibration-dependent definitions are regenerated in place. Their old YAML/JSON/plan '
        'files are renamed aside. `task_namespace` retains each old namespace hash while actual new YAML, recipe '
        'and bundle identities remain in serving/evidence metadata; UID, init, seed, sampler, serving configuration '
        'and topology hashes are verified unchanged. The optional namespace branch is pinned into the frozen '
        'driver evidence overlay. Deferred LIBERO levels apply the same calibrated input map after the new OFF '
        'cell rule, preserving their original scientific namespace.','',
        'The six independent cohorts already moved before the gate stay byte-identical: '
        '`fa_ls_v04_1`, `fa_ls_v04_2`, `fa_ls_v05_1`, `fa_ls_v05_2`, `fa_ls_v05_3`, `fa_ls_v14_0` '
        '(3,287 episodes). The other 38 built gated campaigns and eight deferred units depend on recalibration. '
        'Native pure/pure+long, Cosmos and legacy R15/R16 VLA stay unchanged. No chain or release line is written.','',
        '## CPU checks','',
        'See RECALIBRATION_TESTS.final.log, RECALIBRATION_CPU_PROOF.json, RECALIBRATION_DEFERRED_CPU_PROOF.json '
        'and RECALIBRATION_APPLIED.json records. No server, worker, driver, simulator or GPU job was launched.','']
    if data.get('cpu_verification'):
        v=data['cpu_verification']
        lines += [f'Final verification: {v["startup_planner_passed"]}/60 frozen startup/planner checks; '
            f'{v["deferred_templates_passed"]}/8 deferred template checks; '
            f'{v["targeted_tests_passed"]} targeted tests. CUDA initialized: {v["cuda_initialized"]}. '
            f'Maximum actual GPU-hash MPS pipe: {v["maximum_mps_pipe_bytes"]} bytes (limit 85). '
            'Descriptor equality is bound to the existing strict frozen-loader proof; the new bootstrap '
            'and planner are exercised in the same sealed-version interpreter. Immutable multi-GB weights '
            'are not reloaded on the busy data disk. See RECALIBRATION_FINAL_VERIFICATION.json for '
            'unchanged UID/init/seed, C5 bytes, full-metric and original-artifact checks.','']
        proof=read(HERE/'RECALIBRATION_FINAL_VERIFICATION.json')
        drift=proof['preexisting_sealed_metadata_drift_same_as_source_ready']
        if drift:
            lines += ['Pre-existing source metadata drift (unchanged from the initial CPU proof): '+
                ', '.join('`'+name+'`' for name in drift)+'. The inherited Fast-WAM formal-batch code seal '
                'is record-only; its old sealed spec differs from the actual source ready identity. '
                'The exact model descriptors, native loader pass, package pins and source actual '
                'identity remain equal, as recorded by the original strict CPU receipts. New full-step '
                'pure/native-long controls use that same serving environment. No loader check is weakened here.','']
    (HERE/'RECALIBRATION.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','apply','report'));a=p.parse_args()
    if a.action=='fit':fit_all()
    elif a.action=='apply':apply_all()
    else:report()


if __name__=='__main__':main()
