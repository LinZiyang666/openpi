"""Synthetic-only model selection and subset emission/verification/plan/install."""
import ast
import copy
import functools
import importlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from . import analyze as a, prepare as prep, verify as verify
from .fits import CELLS
from .recipe import R15Knob


@pytest.fixture(scope='module')
def synthetic_results():
    p=dict(frozen_at_utc='synthetic',primary_tolerance_pp=5,tolerances_pp=[0,2,5],
           cells=list(CELLS),bootstrap=dict(seed=20261005,replicates=64))
    all_arms={}
    for i,cell in enumerate(CELLS):
        pure=np.zeros((10,30),dtype=int)
        pure[:,:22]=1
        arms={}
        for j,level in enumerate(a.ORDER):
            y=pure.copy()
            if j:
                for task in range(10):
                    losses=(task+i+j)%5
                    y[task,:losses]=0
                    y[task,22:22+(task+i)%2]=1
            n=np.full_like(y,10)
            v=np.full_like(y,6)
            m=np.full_like(y,max(1,6-j))
            arms[level]=a.Episodes(y,n,v,m,.152*v+.848*m,dict(synthetic=True))
        # First-failure recovery and the discrete 5 pp boundary.
        arms['ir45'].y[0]=pure[0];arms['ir45'].y[0,0]=0
        arms['ir40'].y[0]=pure[0];arms['ir40'].y[0,:2]=0
        for level in ('ir32','ir25','off'):
            arms[level].y[0]=pure[0]
        all_arms[cell]=arms
    full=a.analyze_cells(all_arms,p,{},[],[])
    partials={model:a.analyze_cells({c:v for c,v in all_arms.items() if c.startswith(model+'_')},
                                  p,{},[],[],models=[model]) for model in a.MODELS}
    return p,all_arms,full,partials


@pytest.mark.parametrize('model',a.MODELS)
def test_partial_full_cell_and_table_bytes_identical(model,synthetic_results):
    p,arms,full,partials=synthetic_results
    partial=partials[model]
    assert partial['analysis_scope']['partial']
    assert partial['analysis_scope']['simultaneous_family']==200
    assert 'partial' in partial['analysis_scope']['cross_cell_reporting']
    assert len(partial['cells'])==6
    for cell,row in partial['cells'].items():
        assert a.output_text(row).encode()==a.output_text(full['cells'][cell]).encode()
        selected=row['tolerances']['5']['task_decisions']['0']
        assert selected['level']=='ir45' and selected['first_failure']=='ir40'
        assert 'off' in selected['passing_below_first_failure']
    files=a.outputs(partial,p)
    assert {'levels_'+model+'.json','uniform_'+model+'.json','analysis_'+model+'.json',
            'DESCENT_'+model+'.md'}<=set(files)
    assert not {'levels.json','uniform.json','analysis.json','DESCENT.md'} & set(files)
    assert 'PARTIAL reporting' in files['DESCENT_'+model+'.md']
    assert files[f'levels_tol0pp_{model}.json']==files[f'levels_tol2pp_{model}.json']
    fullfiles=a.outputs(full,p)
    for kind in ('levels','uniform'):
        for name in (kind+'.json',*[f'{kind}_tol{tol}pp.json' for tol in p['tolerances_pp']]):
            projected={c:v for c,v in fullfiles[name].items() if c.startswith(model+'_')}
            assert a.output_text(projected).encode()==a.output_text(files[a.output_name(name,partial['analysis_scope'])]).encode()


def test_missing_requested_cells_and_unrelated_cells(synthetic_results):
    p,arms,full,partials=synthetic_results
    with pytest.raises(ValueError,match='Missing/extra'):
        a.analyze_cells({c:v for c,v in arms.items() if c.startswith('pi05_')},p,{},[],[])
    broken=dict(arms);broken.pop('pi05_l10_100')
    with pytest.raises(ValueError,match='Missing/extra'):
        a.analyze_cells(broken,p,{},[],[],models=['pi05'])
    changed=copy.deepcopy(arms)
    for cell,cell_arms in changed.items():
        if cell.startswith('groot_'):
            for ep in cell_arms.values():
                ep.y[:]=0;ep.n[:]=17;ep.cost[:]=2
    other_full=a.analyze_cells(changed,p,{},[],[])
    for cell,row in partials['pi05']['cells'].items():
        assert a.output_text(row)==a.output_text(other_full['cells'][cell])


def test_equivalence_guard_checks_all_alternates_and_freezes_reapply(tmp_path,synthetic_results):
    p,arms,full,partials=synthetic_results
    for result in partials.values():
        a.write_outputs(a.outputs(result,p),tmp_path)
    reader=a.Reader()
    check=a.check_partial_equivalence(full,p,reader,tmp_path)
    assert check['PASS'] and check['checked_models']==['pi05','groot']
    assert len(check['checks'])==16
    reader.stable()
    assert a.check_partial_equivalence(partials['pi05'],p,a.Reader(),tmp_path)['checked_models']==['pi05']
    for name in ('levels_pi05.json','uniform_pi05.json','levels_tol0pp_pi05.json',
                 'uniform_tol2pp_pi05.json','levels_groot.json'):
        path=tmp_path/name;original=path.read_bytes()
        path.write_bytes(original+b' ')
        with pytest.raises(ValueError,match='byte-for-byte'):
            a.check_partial_equivalence(full,p,a.Reader(),tmp_path)
        path.write_bytes(original)
    (tmp_path/'uniform_pi05.json').rename(tmp_path/'hidden_uniform.json')
    with pytest.raises(ValueError,match='Incomplete frozen partial'):
        a.check_partial_equivalence(full,p,a.Reader(),tmp_path)


@pytest.mark.parametrize('model',a.MODELS)
def test_apply_cli_reads_only_requested_model_and_full_checks_before_writes(model,tmp_path,monkeypatch,synthetic_results):
    p,arms,full,partials=synthetic_results
    protocol=dict(p,trace_parent='/home/weiland/trace_runs/os_closed_loop',root_prefix='r15_descent_')
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'selection.json').write_text('{}')
    (tmp_path/'cap_tables.json').write_text('{}')
    monkeypatch.setattr(a,'HERE',tmp_path)
    monkeypatch.setattr(a,'verify_protocol',lambda reader:protocol)
    reads=[]
    def load(root,cell,*unused):
        reads.append(cell)
        return arms[cell]
    monkeypatch.setattr(a,'load_cell',load)
    a.main(['apply','--models',model])
    expected=[c for c in p['cells'] if c.startswith(model+'_')]
    assert reads==expected
    selected={name:(tmp_path/name).read_bytes() for name in a.outputs(partials[model],p)
              if name.startswith(('levels','uniform'))}
    assert not (tmp_path/'levels.json').exists()
    provenance=json.loads((tmp_path/f'provenance_{model}.json').read_bytes())
    assert provenance['analysis_scope']['cells']==expected and provenance['test_a_read'] is False
    a.main(['apply'])
    assert selected=={name:(tmp_path/name).read_bytes() for name in selected}
    check=json.loads((tmp_path/'partial_equivalence.json').read_bytes())
    assert check['checked_models']==[model] and len(check['checks'])==8
    full_before=(tmp_path/'levels.json').read_bytes()
    (tmp_path/f'levels_{model}.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError,match='byte-for-byte'):
        a.main(['apply'])
    assert (tmp_path/'levels.json').read_bytes()==full_before


@pytest.mark.parametrize('argv',(['apply','--models','bad'],['verify-protocol','--models','pi05']))
def test_invalid_model_cli_never_reads(argv,monkeypatch):
    monkeypatch.setattr(a,'verify_protocol',lambda reader:pytest.fail('unexpected protocol/outcome read'))
    with pytest.raises(SystemExit):
        a.main(argv)


def test_protocol_adapter_preserves_frozen_registration_and_math():
    reader=a.Reader();p=a.verify_protocol(reader)
    assert reader.read(a.HERE/'PROTOCOL.md')==reader.read(a.R14/'descent_astra/PROTOCOL.md')
    original=reader.read(a.HERE/'partial_apply_original_analyze.py.txt')
    assert a.sha(original)==p['input_sha256'][str((a.HERE/'analyze.py').relative_to(a.REPO))]
    source=str((a.HERE/'analyze.py').relative_to(a.REPO))
    with pytest.raises(ValueError,match='Unregistered source change'):
        a.verify_source_adapter(reader,source,a.sha(original),reader.read(a.HERE/'analyze.py')+b'\n',
                                reader.read(a.HERE/'protocol.json'))
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(reader.read(path)).body
                if isinstance(n,ast.FunctionDef)}
    before=functions(a.R14/'descent_astra/analyze.py');after=functions(a.HERE/'analyze.py')
    for name in ('cp_low','cp_high','paired','cell_pair','descend','stitch','bootstrap_indices',
                 'sampled_sum','interval','savings','accepted_rows'):
        assert before[name]==after[name]
    reader.stable()


def test_coordinator_append_preserves_exact_frozen_prefix():
    reader=a.Reader();protocol=reader.read(a.HERE/'protocol.json')
    p=json.loads(protocol)
    path=str((a.R15/'COORDINATOR_DECISIONS.md').relative_to(a.REPO))
    original=reader.read(a.HERE/'partial_apply_original_coordinator_decisions.md')
    a.verify_source_adapter(reader,path,p['input_sha256'][path],original+b'\nsynthetic appended note\n',protocol)
    with pytest.raises(ValueError,match='Frozen coordinator decisions changed'):
        a.verify_source_adapter(reader,path,p['input_sha256'][path],b'changed\n'+original,protocol)
    path=str((a.HERE/'prepare.py').relative_to(a.REPO))
    with pytest.raises(ValueError,match='Frozen input changed'):
        a.verify_source_adapter(reader,path,p['input_sha256'][path],b'synthetic change',protocol)


def test_adapter_rejects_rule_change_even_with_updated_source_digest(tmp_path,monkeypatch):
    original=(a.HERE/'partial_apply_original_analyze.py.txt').read_bytes()
    adapter=json.loads((a.HERE/'partial_apply_adapter.json').read_bytes())
    changed=original.replace(b'100 * (row',b'101 * (row')
    assert changed!=original
    protocol=b'synthetic protocol'
    adapter.update(source=str((tmp_path/'analyze.py').relative_to(a.REPO)),
                   protocol_sha256=a.sha(protocol),adapted_sha256=a.sha(changed))
    (tmp_path/'partial_apply_original_analyze.py.txt').write_bytes(original)
    (tmp_path/'partial_apply_adapter.json').write_text(json.dumps(adapter))
    monkeypatch.setattr(a,'HERE',tmp_path)
    with pytest.raises(ValueError,match='Frozen numerical/loader function changed: descend'):
        a.verify_source_adapter(a.Reader(),adapter['source'],a.sha(original),changed,protocol)


@pytest.fixture
def synthetic_emitter(tmp_path,monkeypatch):
    """Replace every template, source fit, YAML and store with generated fixtures."""
    def synthetic_atomic(path,data):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    monkeypatch.setattr(prep,'atomic',synthetic_atomic)
    trace=tmp_path/'trace';store=tmp_path/'store';source=tmp_path/'yaml'
    monkeypatch.setattr(prep,'TRACE',trace)
    monkeypatch.setattr(prep.devset,'STORE',store)
    monkeypatch.setattr(importlib.import_module(prep._emit.__module__),'SRC',source)
    template=(json.dumps([[t,i] for t in range(10) for i in range(50)])+'\n').encode()
    for i in range(1,5):
        prep.atomic(trace/f'r11_knob_{i}/eval500.json',template)
    records={}
    for cell in CELLS:
        model,suite,size=cell.split('_')
        native=tmp_path/f'{model}_{suite}_native.pkl'
        prep.atomic(native,pickle.dumps({'synthetic':True}))
        yaml_text=('checkpoints:\n  cp1:\n    gate: {type: always_search}\n'
                   '    judge: {type: threshold, threshold: -1000000}\n'
                   'write_policy: {type: never}\nbackend:\n  in_memory:\n'
                   f'    preload_path: {native}\n')
        prep.atomic(source/f'tr_{model}_{"sp" if suite=="spatial" else "l10"}_cache.yaml',yaml_text.encode())
        current=store/'library'/f'{model}_{suite}'/'current'
        for name in ('action.npy','task_id.npy'):
            prep.atomic(current/name,b'synthetic payload')
        for name in ('ids.json','manifest.json'):
            prep.write_json(current/name,{})
        for level in a.ORDER:
            cap=0. if level in ('off','c10') else float(int(level[-2:]))
            method=R15Knob(library='synthetic',method='off' if level=='off' else 'periodic',target=None if level=='off' else .5)
            method.knob_settings={'setting':cap};method.knob_model=model
            kwargs=dict(library='synthetic',episode_subset=None)
            spec='exp.offline_search.rounds.r15.descent.recipe:R15Knob'
            blob=dict(cell=f'{model}_{suite}_cache',spec=spec,kwargs=kwargs,method=method)
            path=tmp_path/'fits'/f'{cell}_{level}.pkl'
            prep.atomic(path,pickle.dumps(blob,protocol=4))
            records[cell,level]=dict(path=str(path),sha256=prep.devset.sha(path),spec=spec,kwargs=kwargs)
    monkeypatch.setattr(prep,'frozen',lambda cell,level,**unused:records[cell,level])
    monkeypatch.setattr(prep,'test_reference',lambda cell:dict(root='synthetic_reference',arm='synthetic_c10'))
    monkeypatch.setattr(verify,'level_cap',prep.level_cap)
    assets=importlib.import_module(verify.control.build_plan.__module__)
    monkeypatch.setattr(verify.control,'build_plan',functools.partial(assets.build_plan,store=store))
    return tmp_path,trace,template


def test_both_model_subset_cli_emit_verify_plan_and_local_install(synthetic_emitter,synthetic_results,monkeypatch):
    tmp_path,trace,template=synthetic_emitter
    p,arms,full,partials=synthetic_results
    stages=[]
    def prepare_cli(*args):
        monkeypatch.setattr(sys,'argv',['prepare',*map(str,args)])
        prep.main()
    def verify_cli(*args):
        monkeypatch.setattr(sys,'argv',['verify',*map(str,args)])
        verify.main()
    for model,host in (('pi05','h100'),('groot','local')):
        files=a.outputs(partials[model],p)
        for kind in ('levels','uniform'):
            prep.write_json(tmp_path/f'{kind}_{model}.json',files[f'{kind}_{model}.json'])
        stage=tmp_path/f'test_a_{model}';stages.append(stage)
        prepare_cli('emit-test-a','--levels',tmp_path/f'levels_{model}.json',
                    '--uniform',tmp_path/f'uniform_{model}.json','--out',stage)
        records=json.loads((stage/'test_a_roots.json').read_bytes())
        assert len(records)==6 and {r['cell'] for r in records}==set(partials[model]['cells'])
        queue=(stage/'queue_test_a.txt').read_text().splitlines()
        assert len(queue)==6 and all(line.startswith('@'+host+' ') for line in queue)
        assert all(r['intended_host']==host and r['arm_count']==2 for r in records)
        assert all((Path(r['root'])/'eval500.json').read_bytes()==template for r in records)
        verify_cli('verify-test-a','--out',stage)
        assert json.loads((stage/'verification.json').read_bytes())['PASS']
        verify_cli('plan-test-a','--out',stage)
        plans=json.loads((stage/'plans/audit.json').read_bytes())
        assert plans['PASS'] and plans['remote_operations']==plans['servers']==0
        assert len(plans['records'])==6 and all(r['host']==host for r in plans['records'])
        # Exercise the install CLI, redirecting placement to synthetic scratch.
        install=prep.install_test_a
        with monkeypatch.context() as patch:
            patch.setattr(prep,'install_test_a',lambda output:install(output,trace))
            prepare_cli('install-test-a','--out',stage)
        targets=[trace/Path(r['final_root']).name for r in records]
        verify.verify_records([prep.check_seal(path) for path in targets])
        with pytest.raises(FileExistsError):
            prep.install_test_a(stage,trace)
    assert stages[0]!=stages[1]
    assert len(list(trace.glob('r15_descent_test_*')))==12
    assert not list(trace.glob('**/runs'))


def test_emitter_rejects_incomplete_or_mixed_tables_before_reading_sources(tmp_path):
    with pytest.raises(ValueError,match='same cells'):
        prep.emit_test_a({'pi05_l10_100':{str(t):'c10' for t in range(10)}},
                         {'groot_l10_100':'c10'},tmp_path/'bad')
    with pytest.raises(ValueError,match='string task IDs'):
        prep.emit_test_a({'pi05_l10_100':{'0':'c10'}},{'pi05_l10_100':'c10'},tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
