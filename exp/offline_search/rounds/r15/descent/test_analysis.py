"""Synthetic-only full loaders, unchanged statistical rules, failure rejection."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from . import analyze as a,analyze_test_a as t


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def jl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def fake_run(root,row,pairs,pool,subset=None,loeo=False):
    name=row['arm'];run=root/'runs'/name;manifest=json.loads(Path(row['manifest']).read_text())
    dump(run/'manifest.json',manifest);dump(run/'selection.json',dict(init_pool=pool,manifest=str(run/'manifest.json')))
    # Preserve byte formatting as production does.
    (run/'manifest.json').write_bytes(Path(row['manifest']).read_bytes())
    if pool=='B':
        contract=dict(dev=True,init_pool='B',arm=name,model='pi05',suite='libero_10',
            subset=subset,fit_sha256=a.sha(Path(row['plugin_args'][1]).read_bytes()),
            manifest_sha256=a.sha(json.dumps(pairs,separators=(',',':')).encode()),
            **(dict(leave_own_episode_out=True) if loeo else {}))
        dump(run/'pool_contract.json',contract)
    journal=[];decs=[]
    for task,init in pairs:
        uid=f'{name}:eval:{task}:{init}'
        journal.append(dict(task_uid=uid,init_pool=pool,accepted=True,status='done',error=None,
                            success=not(task==0 and init==pairs[0][1]),attempt=2,ts=1.))
        audit=dict(fit_sha256=a.sha(Path(row['plugin_args'][1]).read_bytes()))
        if loeo:audit.update(leave_own_episode_out=True,query_pool='B',query_init=init,
                            excluded_episode=task*50+init,candidate_episode_ids=[1000],
                            donor_episode_ids=[1000],parent_donor_rows=[1001])
        decs.append(dict(ev='dec',uid=uid,task_id=task,init=init,step=0,attempt=1,
                         hit=True,vision=True,miss_k=None,r14=audit))
        decs.append(dict(decs[-1],attempt=2,hit=False,miss_k=10,ok=True))
    journal.append(dict(journal[0],attempt=3,error='synthetic rejected retry',success=True))
    decs.append(dict(decs[1]))
    jl(run/'client/journal.jsonl',journal);jl(run/'server_synthetic/decisions_0.jsonl',decs)
    count=len(pairs);per=count//10
    dump(run/'summary.json',dict(init_pool=pool,arm=name,model='pi05',complete=count,success=count-1,
         sr=(count-1)/count,journal_rows=count+1,error_rows=1,retried=count,
         per_task={str(task):dict(n=per,success=per-1 if task==0 else per) for task in range(10)},
         cost_ledger=dict(full_stage_costs=dict(s1=.152,s2=.410,s3=.438,k=10),decisions=count,
             vision_decisions=count,misses=count,controls=count*5,ir_per_five_controls=1.,
             l_per_request=dict(counts={'5.0':count}),stage1_modes={'full':count})))
    return contract if pool=='B' else None


def seal(root,cell,pool,rows):
    dump(root/'ROOT.json',dict(cell=cell,init_pool=pool,episodes_per_arm=300 if pool=='B' else 500,
        arm_count=len(rows),arms=[r['arm'] for r in rows],definitions_sha256={
            n:a.sha((root/n).read_bytes()) for n in ('arms.json','dev_selection.json') if (root/n).exists()},
        fit_sha256={r['plugin_args'][1]:a.sha(Path(r['plugin_args'][1]).read_bytes()) for r in rows}))


def fake_b(tmp_path,cell):
    pairs=[(task,i) for task in range(10) for i in range(20,50)]
    subsets={str(n):dict(episode_ids_by_task={str(task):list(range(task*50,task*50+n//10)) for task in range(10)},
              excluded_inits_by_task={str(task):list(range(n//10)) for task in range(10)}) for n in (100,200,500)}
    selection=dict(model_suites={'pi05_l10':dict(pairs=pairs,served_subsets=subsets)})
    p=dict(trace_parent=str(tmp_path),input_sha256={str((a.HERE/'selection.json').relative_to(a.REPO)):'selection'},
           cost=dict(coefficients={'pi05':[.152,.848],'groot':[.148,.852]}))
    caps=dict(cells={f'pi05_l10_{n}':dict(fit_sha256_by_level={}) for n in (100,200,500)})
    for shared in (True,False):
        source_cell='pi05_l10_100' if shared else cell
        root=tmp_path/('r15_descent_c10_pi05_l10' if shared else 'r15_descent_'+cell)
        manifest=root/'dev_selection.json'
        dump(manifest,dict(dev=True,init_pool='B',pairs=pairs,selection_sha256='selection'))
        levels=('c10',) if shared else a.ORDER[1:];rows=[];contracts={}
        for level in levels:
            fit=tmp_path/f'{source_cell}_{level}.bin';fit.write_bytes(b'synthetic')
            digest=a.sha(fit.read_bytes());caps['cells'][source_cell]['fit_sha256_by_level'][level]=digest
            row=dict(arm=a.arm_name(cell,level),dev=True,init_pool='B',episodes=300,model='pi05',
                     r14_descent_level=level,plugin_args=['--os-fit-artifact',str(fit)],manifest=str(manifest))
            rows.append(row)
            contracts[row['arm']]=fake_run(root,row,pairs,'B',subsets[source_cell.split('_')[-1]],
                                          loeo=not shared and cell.endswith('_500'))
        dump(root/'arms.json',rows);dump(root/'dev_contracts.json',contracts);seal(root,source_cell,'B',rows)
    return tmp_path/('r15_descent_'+cell),p,selection,caps

@pytest.mark.parametrize('cell',('pi05_l10_100','pi05_l10_500'))
def test_full_b_loader_shared_reference_and_retry(cell,tmp_path):
    root,p,s,c=fake_b(tmp_path,cell)
    arms=a.load_cell(root,cell,p,s,c,a.Reader())
    assert set(arms)==set(a.ORDER)
    assert all(x.y.sum()==299 and x.n.sum()==300 for x in arms.values())
    assert arms['ir45'].audit['leave_own_episode_out']==cell.endswith('_500')
    path=next(root.glob('runs/*/server_*/decisions_*.jsonl'))
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    if cell.endswith('_500'):
        for row in rows:row['r14']['donor_episode_ids']=[row['r14']['excluded_episode']]
    else:rows=[r for r in rows if not (r['uid'].endswith(':0:20') and r['attempt']==2)]
    jl(path,rows)
    with pytest.raises(ValueError):a.load_cell(root,cell,p,s,c,a.Reader())


def test_original_selection_math_and_synthetic_adapter():
    rng=np.random.default_rng(7);y=rng.integers(0,2,size=(10,30));n=np.ones_like(y)*8
    arms={}
    for j,level in enumerate(a.ORDER):
        z=y.copy()
        if level=='ir40':z[0]=0
        arms[level]=a.Episodes(z,n,n//2,n//4,.152*(n//2)+.848*(n//4),dict(full_stage_costs={}))
    p=dict(frozen_at_utc='synthetic',primary_tolerance_pp=5,tolerances_pp=[0,2,5],cells=['pi05_l10_500'],
           bootstrap=dict(seed=20261005,replicates=5000))
    result=a.analyze_cells({'pi05_l10_500':arms},p,{},[],[])
    out=a.outputs(result,p)
    assert result['cells']['pi05_l10_500']['leave_own_episode_out']
    assert out['levels_tol0pp.json']==out['levels_tol2pp.json']
    stats={l:a.paired(np.ones(30,int),np.ones(30,int),identical=l=='c10') for l in a.ORDER}
    stats['ir40'].update(successes=28,pure_successes=30,sr=28/30)
    assert a.descend(stats,5)['level']=='ir45'
    assert 'off' in a.descend(stats,5)['passing_below_first_failure']
    assert a.paired(np.ones(30,int),np.ones(30,int))['lower95']<-.11


def test_a_loader_shared_reference_and_measurement_math(tmp_path,monkeypatch):
    cell='pi05_l10_100';monkeypatch.setattr(t,'TRACE',tmp_path)
    levels={str(task):'ir32' for task in range(10)};uniform='ir32';template=(json.dumps([list(x) for x in t.EXPECTED])+'\n').encode()
    caps=dict(cells={cell:dict(periodic_cap_by_level={'ir32':1.},test_fit_sha256_by_level={'ir32':a.sha(b'synthetic')})})
    sharedroot=tmp_path/'r14_reference'
    roots=[sharedroot,tmp_path/('r15_descent_test_'+cell)]
    refs={};fitdict={}
    for root in roots:
        root.mkdir();manifest=root/'eval500.json';manifest.write_bytes(template)
        labels=('pure_c10',) if root==sharedroot else ('uniform','task_descent');rows=[]
        for label in labels:
            name='r14dt_pi05_l10_50_pure_c10' if label=='pure_c10' else f'r15dt_{cell}_'+('uniform_ir32' if label=='uniform' else label)
            fit=tmp_path/(label+'.bin');fit.write_bytes(b'synthetic')
            row=dict(arm=name,model='pi05',dev=False,init_pool='A',episodes=500,
                 r14_descent_level='c10' if label=='pure_c10' else 'task_descent' if label=='task_descent' else 'ir32',
                 replan_steps=5,full_model=True,recording=False,manifest=str(manifest),
                 plugin_args=['--os-fit-artifact',str(fit)],kwargs=dict(task_levels=levels,task_caps=[1.]*10))
            rows.append(row);fake_run(root,row,t.EXPECTED,'A')
            fitdict[label]=dict(path=str(fit),sha256=a.sha(fit.read_bytes()))
        dump(root/'arms.json',rows);seal(root,cell,'A',rows)
    shared=dict(root=str(sharedroot),arm='r14dt_pi05_l10_50_pure_c10',
                arms_sha256=a.sha((sharedroot/'arms.json').read_bytes()),fit_sha256=a.sha(b'synthetic'))
    p=dict(test_references={'pi05_l10':shared},cost=dict(coefficients={'pi05':[.152,.848],'groot':[.148,.852]}))
    dump(roots[1]/'descent_table.json',dict(levels=levels,uniform=uniform,off_tasks=[],caps=[1.]*10,
        template_sha256=a.sha(template),source_artifacts={'ir32':a.sha(b'synthetic')},
        shared_c10_reference=shared,fit=fitdict['task_descent']))
    loaded=t.load_a(cell,levels,uniform,p,caps,a.Reader(),template)
    assert set(loaded)==set(t.ARMS) and all(ep.y.sum()==499 for ep in loaded.values())
    # Four-cell groups retain R14 pooling and do not triple-count C10 across sizes.
    cells=tuple(f'{m}_{s}_{n}' for n in (100,200,500) for m in ('pi05','groot') for s in ('l10','spatial'))
    allarms={c:copy.deepcopy(loaded) for c in cells};b=dict(cells={})
    for c in cells:
        stats=allarms[c]['pure_c10'].stats()|dict(full_stage_costs=allarms[c]['pure_c10'].audit['full_stage_costs'])
        b['cells'][c]=dict(arm_stats={'c10':stats},tolerances={'5':dict(savings={'uniform':stats,'task':stats},library={})},matched_ir={})
    monkeypatch.setattr(t,'CELLS',cells)
    tables={c:levels for c in cells};uniforms={c:uniform for c in cells}
    result=t.calculate(allarms,b,p,tables,uniforms)
    assert set(result['groups_by_library_size'])=={'100','200','500'}
    assert len(result['cells'])==12
    for group in result['groups_by_library_size'].values():
        assert group['pooled']['arms']['pure_c10']['n_episodes']==2000
        assert group['pooled']['arms']['pure_c10']['sr']==499/500
    assert 'Library size 500' in t.render_test(result)
    assert '冻结' in t.render_report(result)
