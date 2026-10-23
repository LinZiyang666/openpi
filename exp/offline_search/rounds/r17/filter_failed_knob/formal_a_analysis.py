"""Read only the new formal grids and explicitly reused off/pure test references."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

from . import formal_a as f
from .common import HERE,REPO,read,dump,sha
from exp.offline_search.rounds.r17.knob import common as c,descent


def choose(arms,pure,order):
    """The MT50 integer rule, including its incomplete-cohort semantics."""
    off=order[-1];expected=set(pure);tasks=sorted({k[1] for k in pure})
    no_knob=(len(expected)==500 and set(arms.get(off,{}))==expected and
             sum(e['success'] for e in arms[off].values())>=sum(e['success'] for e in pure.values()))
    selected={};decisions={}
    for task in tasks:
        ref={k:v for k,v in pure.items() if k[1]==task}
        baseline=sum(e['success'] for e in ref.values());levels={'pure':dict(n=len(ref),successes=baseline,passes={'4':True})}
        for level in order:
            data={k:v for k,v in arms.get(level,{}).items() if k in ref}
            levels[level]=dict(n=len(data),successes=sum(e['success'] for e in data.values()),
                missing_inits=sorted(k[2] for k in ref.keys()-data.keys()),
                passes={'4':baseline-sum(e['success'] for e in data.values())<=2 if set(data)==set(ref) and len(ref)==50 else None})
        with patch.object(c,'LEVELS',('pure',*order)):
            decision=(dict(selected=off,first_failure=None,incomplete=False,non_monotone_recoveries=[])
                      if no_knob else descent.descend(levels,4,count=50))
        # MT50's helper names its terminal level koff; VLA calls it off.
        if decision['selected']=='koff':decision['selected']=off
        if len(ref)!=50:decision.update(selected=None,incomplete=True)
        level=decision['selected']
        if level:selected.update(ref if level=='pure' else {k:arms[level][k] for k in ref})
        decisions[task]=dict(levels=levels,**decision)
    return dict(no_knob=no_knob,tuned=not no_knob,tasks=decisions,data=selected,complete=len(selected)==500,
        grid_complete=len(expected)==500 and all(set(arms.get(level,{}))==expected for level in order))


def legacy_data(ep,suite):
    return {(suite,str(t),i):dict(success=bool(ep.y[t,i]),n=int(ep.n[t,i]),cost_sum=float(ep.cost[t,i]))
            for t in range(10) for i in range(50) if ep.y[t,i]>=0}


def vla_references(model,mode):
    from exp.offline_search.rounds.r16 import analyze as a
    refs={}
    if mode=='reduced':
        design=read(REPO/'exp/offline_search/rounds/r16/design.json')
        records=[(r,r['arms'][0]) for r in design['phase1'] if r['cell'].startswith(model+'_') and r['cell'].endswith('_k1')]
    elif model=='groot':
        proof=read(f.vf.SOURCE/'c10_proof.json')
        records=[(read(Path(r['root'])/'ROOT.json'),r['arm']) for k,r in proof['test_references'].items() if k.startswith(model+'_')]
    else:
        # Operator may name an already existing H100 pure-A reference. Never
        # substitute the withdrawn pipeline's 4090 reference or define a rerun.
        records=[]
        path=f.ROOT/'PURE_A_OVERRIDES.json'
        overrides=read(path) if path.exists() else {}
        for suite in ('libero_10','libero_spatial'):
            override=overrides.get(f'{model}:{mode}:{suite}')
            if override is None:continue
            from .formal_a_names import resolve_reference
            root,name=resolve_reference(override['root'],override['arm']);root=Path(root)
            if not (root/'runs'/name/'h100_launch.json').exists():continue
            launch=read(root/'runs'/name/'h100_launch.json')
            assert launch.get('server_host','h100')=='h100'
            records.append((read(root/'ROOT.json'),name))
    for record,name in records:
        root=Path(record['root'])
        if not (root/'runs'/name/'summary.json').exists():continue
        row=next(r for r in read(root/'arms.json') if r['arm']==name)
        ep=a.load(record,name,[(t,i) for t in range(10) for i in range(50)],'A',1 if mode=='reduced' else 10 if model=='pi05' else 8,a.Reader())
        refs[row['suite']]=legacy_data(ep,row['suite'])
    return refs


def load_r17(job):
    from exp.offline_search.rounds.r17.filter_failed_knob.common import continued_lanes
    cells,_=c.load_knob(continued_lanes([Path(job['run'])]),'A') if not job.get('reuse_only') else ({},{})
    # (suite, size), or (suite, size, tag) for a seeds_1023 library (tag = seed and segment).
    result={cell[1:]:arms for cell,arms in cells.items()}
    if job.get('off_reuse'):
        for lane in continued_lanes([Path(job['off_reuse'])]):
            sources=c.cache.Sources()
            loaded=c.cache.load_lane(lane,c.cache.receipt_snapshot(lane),sources,c.SIZES,suites=c.cache.SUITES)
            base=c.study.full_group(c.campaign_bundle(read(lane/'campaign.json')))
            for (m,n,group,s),row in loaded.items():
                if group in (base,base+'_koff'):c.merge(result.setdefault((s,n),{}).setdefault('koff',{}),row['data'])
    return result


def load_vla(job):
    from exp.offline_search.rounds.r16 import analyze as a,conductor_analysis as ca
    result={};root=Path(job['run'])
    if job['pipeline']=='R15':
        record=read(root/'ROOT.json')
        for row in read(root/'arms.json'):
            if not (root/'runs'/row['arm']/'summary.json').exists():continue
            ep=a.load(record,row['arm'],[(t,i) for t in range(10) for i in range(50)],'A',10 if job['model']=='pi05' else 8,a.Reader())
            result.setdefault((row['suite'],row['r10_size']),{})[row['r14_descent_level']]=legacy_data(ep,row['suite'])
    else:
        group=read(root/'group.json')
        if not job.get('reuse_only') and (root/'DONE.json').exists():
            for entry in group['entries']:
                view=dict(entry,pairs=[(t,i) for t in range(10) for i in range(50)])
                ep=ca.load_entry(group,view,a.Reader());row=entry['row']
                c.merge(result.setdefault((row['suite'],row['r10_size']),{}).setdefault(entry['formal_grid_level'],{}),legacy_data(ep,row['suite']))
        # These are standalone OFF definitions, never a mixed dev inventory.
        off=Path(job['off_reuse']);offgroup=read(off)
        if (Path(offgroup['root'])/'DONE.json').exists():
            from exp.offline_search.rounds.r16.k1ref import workflow as w
            for entry in offgroup['entries']:
                ep=w.load_sparse(offgroup,entry,a.Reader());row=entry['row']
                data=legacy_data(ep,row['suite'])
                result.setdefault((row['suite'],row['r10_size']),{}).setdefault('off',{}).update(data)
    return result


def stats(data,model,mode,suite):
    if not data:return dict(n=0,sr=None,ir=None)
    sr=sum(e['success'] for e in data.values())/len(data)
    if model in f.TEST_SOURCE:
        from exp.offline_search.rounds.r17.knob.ir_formal import coefficients
        coef,pure_coef=coefficients(f.LABELS[model],suite);num=den=0
        for e in data.values():
            if 'cost' not in e:return dict(n=len(data),sr=sr,ir=None)
            price=pure_coef if e.get('pure') else coef;cost=e['cost']
            num+=(price['vision']*(cost.get('policy',0)+cost.get('library',0))+price['reduced_miss' if mode=='reduced' else 'full_miss']*cost.get('policy',0))*c.C0[f.LABELS[model]]
            den+=cost.get('controls',0)
        measured=num/den if den else None
    else:
        n=sum(e['n'] for e in data.values());measured=sum(e['cost_sum'] for e in data.values())/n if n else None
    return dict(n=len(data),sr=sr,ir=measured)


def model_identity(job,rows):
    """Compare native model startup argv to source-A definitions, never logs."""
    from exp.offline_search.closed_loop.ops.h100 import control,node,local_server
    from exp.offline_search.rounds.r16 import conductor_runner as runner
    root=Path(job['run'])
    if job['pipeline']=='R15':
        top=job['topology'];source=Path(top['source_root']);old=read(source/'runs'/top['source_arm']/'h100_launch.json')['servers'][0]
        oldrow=next(r for r in read(source/'arms.json') if r['arm']==top['source_arm'])
        old=dict(old,yaml=oldrow['yaml']);env=top['env']
        with patch.dict(__import__('os').environ,env):new=control.server_spec(root,rows[0],int(env['PORTS'].split(',')[0]))
        def command(spec):
            spec=dict(spec,out=str(f.ROOT/'cpu/model_argv'/job['name']))
            if top['gpu']=='RTX 4090':return local_server.command(spec)[0]
            return node.command(spec['model'],spec['suite'],spec['port'],spec['yaml'],spec['out'],spec['tag'],spec['plugin'],dict(spec['env']))[0]
        pairs=[(command(old),command(new))]
    else:
        oldgroup=read(Path(job['source'])/'group.json');newgroup=read(root/'group.json')
        ports=list(range(23150,23150+job['replicas']))
        old=runner.server_specs(oldgroup,ports,'cpu_identity');new=runner.server_specs(newgroup,ports,'cpu_identity')
        old=[dict(s,out=str(f.ROOT/'cpu/model_argv'/job['name']/'source'/str(i))) for i,s in enumerate(old)]
        new=[dict(s,out=str(f.ROOT/'cpu/model_argv'/job['name']/'new'/str(i))) for i,s in enumerate(new)]
        pairs=[(local_server.command(a)[0],local_server.command(b)[0]) for a,b in zip(old,new)]
    flags=('--policy.config','--policy.dir') if job['model']=='pi05' else ('--checkpoint','--denoising-steps')
    def extract(argv):return {k:argv[argv.index(k)+1] for k in flags}
    for left,right in pairs:assert extract(left)==extract(right),(extract(left),extract(right))
    return dict(source_model_argv_equal=True,model_flags=[extract(right) for left,right in pairs],
                server_python=next(v for v in pairs[0][1] if str(v).endswith('/bin/python')))


def active_jobs(include_reuse=True):
    from . import formal_a_queue as queue
    return queue.analysis_jobs(include_reuse) if queue.INDEX.exists() else read(f.ROOT/'CAMPAIGNS.json')['jobs']


def analyze(unused=None):
    jobs=active_jobs();all_data=defaultdict(dict);references={};rows=[]
    for job in jobs:
        key=job['model'],job['mode'],job['arm']
        loaded=load_r17(job) if job['pipeline']=='R17' else load_vla(job)
        for cell,arms in loaded.items():
            for level,data in arms.items():all_data[key].setdefault(cell,{}).setdefault(level,{}).update(data)
    for model in f.MODELS:
        for mode in ('reduced','full'):
            if model in f.TEST_SOURCE:
                from exp.offline_search.rounds.r17.knob.test import load_controls
                refs=load_controls(f.EVAL,{f.LABELS[model]},mode,[f.LABELS[model]+'='+f.PURE[model][mode]])
                refs={s:data for (m,s),data in refs.items()}
            else:refs=vla_references(model,mode)
            sizes=(50,100,200,300,400,500) if model in f.TEST_SOURCE else (50,100,200,500)
            for suite in ('libero_10','libero_spatial'):
                pure=refs.get(suite,{});order=f.R17_LEVELS if model in f.TEST_SOURCE else f.VLA_LEVELS
                for size in sizes:
                    outcomes={arm:choose(all_data[model,mode,arm].get((suite,size),{}),pure,order) for arm in f.ARMS}
                    pair=None
                    if all(v['complete'] for v in outcomes.values()):
                        a=outcomes['original']['data'];b=outcomes['filtered']['data'];keys=sorted(set(a)&set(b))
                        delta={k:int(b[k]['success'])-int(a[k]['success']) for k in keys}
                        pair=dict(n=len(keys),difference=sum(delta.values())/len(keys),
                            by_init={str(i):sum(d for k,d in delta.items() if k[2]==i)/10 for i in range(50)},
                            by_task_init=[dict(task=k[1],init=k[2],difference=d) for k,d in delta.items()])
                    row=dict(model=model,suite=suite,size=size,step_mode=mode,reference=stats(pure,model,mode,suite),
                        reference_status='complete' if len(pure)==500 else 'pending_same_step_same_GPU_pure_A',
                        paired_filtered_minus_original=pair)
                    if model in f.TEST_SOURCE:
                        info=next(r for r in f.read(f.HERE.parent/'filter_failed/CELLS.json')['cells']
                                  if (r['model'],r['suite'],r['size'])==(model,suite,size))
                    else:
                        info=next(r for r in read(f.ROOT/'LIBRARIES.json')['cells']
                                  if (r['model'],r['suite'],r['size'],r['mode'])==(model,suite,size,mode))
                    row.update(filtered_episodes_kept=info['kept'],original_episodes=info['collected'],
                               empty_tasks=info['empty_tasks'])
                    for arm,decision in outcomes.items():
                        row[arm]=dict(stats(decision['data'],model,mode,suite),no_knob=decision['no_knob'],tuned=decision['tuned'],
                            per_task_levels={t:d['selected'] for t,d in decision['tasks'].items()},tasks=decision['tasks'],
                            complete=decision['complete'],grid_complete=decision['grid_complete'],
                            measured_levels={level:stats(data,model,mode,suite)
                                             for level,data in all_data[model,mode,arm].get((suite,size),{}).items()})
                    rows.append(row)
    dump(f.ROOT/'RESULTS.json',dict(design='test_A_only_first_failure_two_successes',rows=rows,dev_outcomes_used=False))
    print('Wrote',len(rows),'formal paired cells')


def check(selected=None):
    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.rounds.r16 import conductor as rc
    from openpi.conductor.task import ServerEndpoint
    from . import startup,build as b
    from .formal_a_names import resolve_name
    selected=resolve_name(selected) if selected else selected
    jobs=active_jobs(False);results=[]
    selected_names={selected} if selected else None
    if selected:
        from . import formal_a_queue as queue
        if queue.INDEX.exists():
            unit=next((u for u in read(queue.INDEX)['units'] if u['name']==selected),None)
            if unit is not None:selected_names={j['name'] for j in unit['jobs']}
    for job in jobs:
        if selected_names is not None and job['name'] not in selected_names:continue
        root=Path(job['run'])
        if job['pipeline']=='R17':
            campaign=read(root/'campaign.json');_,_,tasks,_=b.cpu_plan(campaign)
            assert [t.task_uid for t in tasks]==[t['task_uid'] for t in read(root/'EPISODE_PLAN.json')['tasks']]
            # FW's final A definition was never installed. Its preceding A
            # release is installed; startup.check proves its bytes/topology
            # equal before using it for this read-only remote CPU probe.
            installed=f.EVAL/'knobtestfw2' if Path(job['source']).name=='knobtestfw3' else None
            result=startup.check(job,virtual_assets=True,installed_source=installed)
            assert result['status']=='PASS',result.get('full_system_startup_error') or result.get('error')
            inputs=read(root/'assets/cache_inputs/manifest.json');source_inputs=read(Path(job['source'])/'assets/cache_inputs/manifest.json')
            for name,pin in inputs['files'].items():
                assert source_inputs['files'][name]==pin and sha(root/'assets/cache_inputs'/name)==pin
            if job['arm']=='original':assert not campaign['filter_failed_knob']['selection']
            results.append(dict(name=job['name'],startup='PASS',source_identity_equal=result['equals_source_actual'],
                sealed_descriptor_equal=result['spec_equal'],pinned_packages_equal=result['pinned_packages_equal'],
                original_serving_tables_equal=True if job['arm']=='original' else None,planner_episodes=len(tasks)))
        else:
            if job['pipeline']=='R16':
                group=rc.read_group(root/'group.json');entries=group['entries'];rows=[e['row'] for e in entries]
                endpoint=ServerEndpoint('localhost',23150)
                graph=rc.R16Strategy(group).plan({}, {'cpu':endpoint})
                planned=[e for s in graph.stages.values() for e in s.episodes]
                assert len(planned)==job['episodes'] and len({e.task_uid for e in planned})==len(planned)
            else:
                from exp.offline_search.closed_loop.ops.h100 import control
                from exp.offline_search.rounds.r15.descent import verify
                rows=read(root/'arms.json')
                for row in rows:
                    assert control.selection(root,row)[0]==500
                    assert read(row['manifest'])==[[t,i] for t in range(10) for i in range(50)]
                f.vf.p.check_seal(root)
                # All cells exercise native selection/seal and plugin identity.
                # Packing the full code/library is the same path for all sizes;
                # exercise it once per model/suite/arm, at the size-50 edge.
                if rows[0]['r10_size']==50:
                    verify.plans([read(root/'ROOT.json')],f.ROOT/'plans'/job['name'])
            for row in rows:
                args=['--os-method',row['method'],'--os-cell',row['cell'],*row['plugin_args'],'--os-kwargs',json.dumps(row['kwargs']),
                    '--os-log-dir',str(f.ROOT/'cpu'/job['name']/row['arm'])]
                opts,rest=plugin.parse_cli(args);assert not rest
                if job['pipeline']=='R16':opts.os_miss_steps=1
                runtime=plugin.PluginRuntime(opts,job['model']);assert runtime.method is not None
                assert runtime.r14_fit_sha256==sha(opts.os_fit_artifact);runtime.flush_all()
            identity=model_identity(job,rows)
            serving=[]
            if job['pipeline']=='R16':
                from exp.offline_search.rounds.r16 import retest as rt
                source={e['row']['r16_cell']:e['row'] for e in read(Path(job['source'])/'group.json')['entries']}
                for row in rows:
                    before=pickle.loads(rt.fit_path(source[row['r16_cell']]).read_bytes())
                    after=pickle.loads(rt.fit_path(row).read_bytes())
                    f.library_info(row['r16_cell'],job['mode'],before)
                    if job['arm']=='original':
                        assert pickle.dumps(before['method'].inner,protocol=4)==pickle.dumps(after['method'].inner,protocol=4)
                    else:
                        # Filter only candidate arrays; fitted quantities are shared.
                        info=f.library_info(row['r16_cell'],job['mode'],before)
                        expected=f.candidates(before,info['kept_episode_ids']);f.vf.unchanged(before,expected)
                        assert pickle.dumps(expected['method'].inner,protocol=4)==pickle.dumps(after['method'].inner,protocol=4)
                    serving.append(row['arm'])
            if job['arm']=='original' and job['pipeline']=='R15':
                assert all(r['original_bytes_identical'] for r in read(Path(job.get('artifacts',root.parent/'ARTIFACTS.json')))['rows'])
            results.append(dict(name=job['name'],startup='PASS',native_plugin_rows=len(rows),planner_episodes=job['episodes'],
                original_serving_tables_equal=True if job['arm']=='original' else None,candidate_rows_checked=len(serving),
                native_sync_plan_checked=(f.ROOT/'plans'/job['name']/'audit.json').exists(),**identity))
        dump(f.ROOT/'cpu_checks'/(job['name']+'.json'),results[-1])
        active={j['name'] for j in jobs}
        dump(f.ROOT/'CPU_CHECKS.json',dict(jobs=[read(p) for p in sorted((f.ROOT/'cpu_checks').glob('*.json')) if p.stem in active],cpu_only=True))
        print('CPU_PASS',job['name'],flush=True)


def inventory(unused=None):
    from . import formal_a_queue as queue
    if queue.INDEX.exists():
        queue.publish();print(json.dumps(read(f.ROOT/'QUEUE_TOTALS.json'),indent=2));return
    jobs=read(f.ROOT/'CAMPAIGNS.json')['jobs'];totals=[]
    rates={(r['model'],r['mode']):r for r in read(f.ROOT/'THROUGHPUT.json')['rows']}
    for model in f.MODELS:
        for mode in ('reduced','full'):
            for arm in f.ARMS:
                group=[j for j in jobs if (j['model'],j['mode'],j['arm'])==(model,mode,arm)]
                rate=rates.get((model,mode),{})
                episodes=sum(j['episodes'] for j in group)
                totals.append(dict(model=model,mode=mode,arm=arm,episodes=sum(j['episodes'] for j in group),
                    campaigns=len(group),gpus=sorted({j['gpu'] for j in group}),
                    replicas=group[0]['replicas'],batch=group[0]['batch'],workers=group[0]['workers'],
                    source_episodes_per_hour=rate.get('episodes_per_hour'),
                    gpu_hours=episodes/rate['episodes_per_hour'] if rate else None,
                    estimate_status='source_test_measured' if rate else 'missing_source_test_timing'))
    dump(f.ROOT/'TOTALS.json',dict(rows=totals,total_episodes=sum(r['episodes'] for r in totals)))
    order=sorted(jobs,key=lambda j:(j['mode']!='reduced',f.MODELS.index(j['model']),j['arm']!='original',j['name']))
    dump(f.ROOT/'LAUNCH_ORDER.json',dict(jobs=order,rule='Finish existing queues first; reduced first, original then filtered. Serialize any jobs sharing a worker node or server GPU.',
        pi05_full_hold='Missing an existing same-step pure-A H100 reference; no new reference authorized or defined.',
        conflicts=[dict(a=a['name'],b=b['name'],nodes=sorted(a['workers'].keys()&b['workers'].keys()))
                   for i,a in enumerate(order) for b in order[i+1:] if a['workers'].keys()&b['workers'].keys()]))
    for gpu in ('RTX 4090','H100'):
        (f.ROOT/'commands'/('4090_ORDER.txt' if gpu=='RTX 4090' else 'H100_ORDER.txt')).write_text(''.join(
            f'{j["name"]}\t{j["episodes"]}\tR{j["replicas"]} B{j["batch"]}\t{j["workers"]}\n'
            for j in order if j['gpu']==gpu))
    print(json.dumps(totals,indent=2))
