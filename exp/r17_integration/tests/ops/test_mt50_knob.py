"""CPU stubs for direct seed-7 MT50 knob batches; no launch or model load."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.metaworld.tasks import TASK_NAMES, DIFFICULTY
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.common.contracts import SamplerSpec, SampleResult
from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_provider, fake_spec
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.provider import binding_for, register_provider
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.ops import cache_knob as knob, cache_study as study, formal_eval
from exp.r17_integration.tests.ops.test_cache_knob import fake_campaign
from exp.r17_integration.tests.ops.test_cache_study import freeze
from exp.offline_search.rounds.r17.knob import common as c, descent, test as analysis, ir_formal


def sampler(model, steps):
    profile = study.read(study.ROOT/'exp/r17_integration/profile/results_mt50'/(model+'_mt50.json'))
    return SamplerSpec(**profile['cells'][str(steps)]['sampler'])


@pytest.fixture(scope='module', params=study.MW_MODELS)
def source(request, tmp_path_factory):
    model = request.param
    root = tmp_path_factory.mktemp(model)
    full = sampler(model, knob.MW_STEPS[model][0])
    spec = replace(fake_spec(action_dim=4, views=1, model_id=model), default_sampler=full,
                   allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    library = root/'libraries'/(model+'_mt50')/'bpool'
    library.mkdir(parents=True)
    study.atomic_json(library/'READY.json', dict(binding=dict(model=asdict(spec),sampler=asdict(full),
                      benchmark_protocol=FORMAL_SPEC.identity), content_digest='cpu-parent'))
    for size in knob.MW_SIZES:
        view = library/'subsets'/('size_'+str(size//50))
        view.mkdir(parents=True)
        study.atomic_json(view/'VIEW.json', dict(size=size//50, content_digest='cpu-view-'+str(size)))
    group = study.full_group(model)
    study.generate(root/'source', library_root=root/'libraries', models=(model,), groups=(group,), sizes=knob.MW_SIZES)
    gaps = root/'gaps.jsonl'
    gaps.write_text(''.join(json.dumps(dict(model=model,suite='mt50',size=size,miss_steps=mode,
        levels={str(target):dict(gap=float(j+1+(10 if mode=='reduced' else 0)))
                for j,target in enumerate((.5,.4,.3,.2,.1))}))+'\n'
        for mode in ('full','reduced') for size in knob.MW_SIZES))
    return model, root/'source/STUDY.json', gaps


@pytest.mark.parametrize('mode', ('full','reduced'))
def test_batch_all_sizes_seed7_no_loeo_no_pure(source, tmp_path, mode):
    model, src, gaps = source
    path = knob.build(tmp_path/'batch', model=model, sources=[src], gaps=gaps, miss_steps=mode)
    manifest = study.validate_manifest(path)
    rows = manifest['models'][model]
    assert manifest['eval_pool'] == 'A' and manifest['inits'] == 50 and len(rows) == 36
    assert set(manifest['library_sizes']) == set(knob.MW_SIZES)
    k = knob.MW_STEPS[model][mode=='reduced']
    for row in rows:
        doc = yaml.safe_load((path.parent/row['file']).read_text())
        assert doc['adapter_binding']['num_steps'] == row['num_steps'] == k
        assert doc['adapter_binding']['sampler_identity'] == sampler(model,k).identity
        assert doc['adapter_binding']['commit_horizon'] == (5 if model=='pi05_mw' else 10)
        assert doc['recipe']['corrector'] and doc['recipe']['miss_mode'] == 'stall_only'
        assert not doc['recipe']['serve_loeo'] and not row['serve_loeo']
        assert row['group'].startswith(study.full_group(model)+'_')
        assert not row['group'].endswith('_pure')
        if row['group'].endswith('_koff'):assert doc['recipe']['periodic_gap'] is None
        else:assert doc['recipe']['periodic_gap'] >= (11 if mode=='reduced' else 1)
        assert doc['recipe']['store_path'].endswith('/size_'+str(row['library_size']//50))
    assert sum(50*len(yaml.safe_load((path.parent/r['file']).read_text())['recipe']['ordered_tasks']) for r in rows)==90000
    # Exercise the complete 90k planner once; other model/mode combinations
    # use two tasks per cell to keep the CPU suite bounded while checking every cell.
    full_plan = model=='dp_mw' and mode=='full'
    plan_rows = copy.deepcopy(rows)
    if not full_plan:
        for row in plan_rows:row['tasks']=list(TASK_NAMES[:2])
    campaign = fake_campaign(path,model,tmp_path/'assets',rows=plan_rows)
    campaign.update(model=model.removesuffix('_mw'),domain='metaworld')
    _,_,tasks,endpoints = study.plan(campaign)
    assert len(tasks) == len({t.task_uid for t in tasks}) == (90000 if full_plan else 3600)
    assert {t.extra['seed'] for t in tasks} == {7}
    assert {t.extra['partition'] for t in tasks} == {'test'}
    assert {t.orig_init_state_idx for t in tasks} == set(range(50))
    assert {t.extra['num_steps'] for t in tasks} == {k}
    assert endpoints[0].allowed_k == (k,)
    assert all(not t.extra['serve_loeo'] and t.extra['init_pool']=='A' for t in tasks)


@pytest.mark.parametrize('mode', ('full','reduced'))
def test_cli_compatibility_and_partial_task_plans(source, tmp_path, mode, capsys):
    model, src, gaps = source
    assert knob.model_name(model) == model
    assert knob.main(['dev','--model',model,'--source',str(src),'--gaps',str(gaps),
        '--miss-steps',mode,'--levels','k50','--out',str(tmp_path/'wave')]) == 0
    assert '15000 episodes' in capsys.readouterr().out
    assert study.read(tmp_path/'wave/STUDY.json')['eval_pool'] == 'A'
    assert knob.main(['test','--model',model,'--source',str(src),'--gaps',str(gaps),
        '--miss-steps',mode,'--out',str(tmp_path/'test_batch')]) == 0
    assert '90000 episodes' in capsys.readouterr().out
    assert study.read(tmp_path/'test_batch/STUDY.json')['eval_pool'] == 'A'
    table = tmp_path/'partial.json'
    study.atomic_json(table,{c.model_label(model):{'mt50':{'250':{TASK_NAMES[0]:'k40'}}}})
    path = knob.build(tmp_path/'partial',sources=[src],gaps=gaps,model=model,
                      table=table,partial=True,miss_steps=mode)
    rows = study.read(path)['models'][model]
    assert len(rows)==1 and rows[0]['tasks']==[TASK_NAMES[0]]
    campaign = fake_campaign(path,model,tmp_path/'assets')
    campaign.update(model=model.removesuffix('_mw'),domain='metaworld')
    tasks = study.plan(campaign)[2]
    assert len(tasks)==50 and all(t.extra['seed']==7 for t in tasks)


@pytest.mark.parametrize('mode', ('full','reduced'))
def test_calibration_selected_library_loeo_proxy_and_mode(source,tmp_path,monkeypatch,mode):
    model,src,_ = source
    from exp.r17_integration.search import fit, teacher_library
    from exp.r17_integration.common import ir
    seen=[]
    monkeypatch.setattr(fit,'load_fitted_recipe',lambda *a,**kw:SimpleNamespace(commit=5 if model=='pi05_mw' else 10))
    monkeypatch.setattr(teacher_library,'RowIndexStore',lambda path:Path(path))
    def schedule(fitted,view,config):
        seen.append((view,config.adapter_binding['num_steps']))
        # The real library_schedule owns LOEO; this stub exercises its caller and solver.
        return lambda gap:(10000,10000,round(10000/(1+gap)))
    monkeypatch.setattr(ir,'library_schedule',schedule)
    path=knob.calibrate(tmp_path/'gaps.jsonl',sources=[src],model=model,miss_steps=mode)
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==6 and {r['size'] for r in rows}==set(knob.MW_SIZES)
    assert all(r['miss_steps']==mode and r['num_steps']==knob.MW_STEPS[model][mode=='reduced'] for r in rows)
    assert all(set(r['levels'])=={'0.5','0.4','0.3','0.2','0.1'} for r in rows)
    assert {p.name for p,_ in seen}=={'size_5','size_10','size_20','size_30','size_40','size_50'}
    assert all('LOEO' in r['calibration_scope'] for r in rows)


@pytest.mark.parametrize('model', study.MW_MODELS)
@pytest.mark.parametrize('mode', ('full','reduced'))
@pytest.mark.parametrize('trigger', ('stall','periodic','pure'))
def test_every_call_uses_bound_native_or_reduced_sampler(tmp_path,model,mode,trigger):
    import uuid
    k=knob.MW_STEPS[model][mode=='reduced']
    full=sampler(model,knob.MW_STEPS[model][0])
    spec=replace(fake_spec(action_dim=4,views=1,model_id=model),default_sampler=full)
    provider=fake_provider('mwknob.'+uuid.uuid4().hex,spec)
    received=[]
    class Stub(FakeAdapter):
        def sample(self,s2,bound,rng,capture):
            received.append(bound)
            assert bound.identity==sampler(model,k).identity and not capture.enabled
            return SampleResult(np.zeros_like(s2.data),k,bound)
    provider=replace(provider,provider_id=provider.provider_id+'.live',load_adapter=lambda:Stub(spec),
                     sampler_for_steps=lambda n:sampler(model,n))
    register_provider(provider)
    cfg,_=freeze(provider,tmp_path,5 if model=='pi05_mw' else 10,gated=True)
    cfg.adapter_binding=binding_for(provider,spec.allowed_benchmark_protocols[0],num_steps=k,
                                    commit_horizon=5 if model=='pi05_mw' else 10)
    cfg.recipe.periodic_gap=None if trigger=='stall' else 0. if trigger=='pure' else 1.
    cfg.recipe.serve_loeo=False
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    class Tracker:
        def __init__(self,*args):pass
        def observe(self,*args):pass
        def status(self):return dict(state='slow_confirmed' if trigger=='stall' else 'ok',W=2)
    policy.recipe.tracker_class=Tracker
    policy.on_episode_start(extra_metadata=dict(task_name='reach',task_id=0,original_init_idx=0,seed=7))
    for seq in range(3):
        obs=dict(state=np.zeros(8,np.float32))
        if seq:obs['__execution__']=dict(decision_seq=seq-1,executed_count=5)
        hit=policy.infer(obs)['__hit_meta__']
        if hit['source']=='policy':
            assert hit['actual_nfe']==k and hit['sampler_identity']==sampler(model,k).identity
    assert received


def lane(path,model,entries,*,control=False,base=False,ts=1):
    """Use real MT50 names and journal/step parsing, including accepted retry filtering."""
    path.mkdir(parents=True, exist_ok=True)
    study.atomic_json(path/'campaign.json',dict(model=c.MODEL_IDS[model],domain='metaworld'))
    tasks=[];journal=[];steps=[]
    for task,level,idx,success in entries:
        group='control' if control else study.full_group(c.MODEL_IDS[model]+'_mw')+( '' if base else '_'+level)
        uid=f'{group}:{task}:{idx}'
        extra=dict(task_name=TASK_NAMES[task],seed=7,init_pool='A',partition='test')
        if not control:extra.update(cache_group=group,library_size=250)
        tasks.append(dict(task_uid=uid,experiment='metaworld_mt50',orig_init_state_idx=idx,extra=extra))
        terminal=dict(task_uid=uid,accepted=True,attempt=2,success=success,ts=ts)
        journal.extend([dict(terminal,accepted=False,attempt=1,success=not success),terminal])
        sources=['policy'] if control else ['policy','library']
        for seq,source in enumerate(sources):
            fields=dict(task_uid=uid,accepted=True,attempt=2,decision_seq=seq)
            steps += [dict(fields,source=source,actual_nfe=2 if source=='policy' else 0),
                      dict(fields,_kind='execution_ack',executed_count=5,terminal=seq==len(sources)-1)]
    study.atomic_json(path/'EPISODE_PLAN.json',dict(tasks=tasks))
    (path/'driver').mkdir()
    for name,rows in (('journal',journal),('per_step',steps)):
        (path/'driver'/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.mark.parametrize('model', ('DP','Fast-WAM','π0.5','GR00T'))
@pytest.mark.parametrize('mode', ('full','reduced'))
def test_direct_descent_final_measured_results_and_references(tmp_path,model,mode):
    controls=tmp_path/'controls'
    control_names=c.MW_REDUCED_CONTROL_NAMES if mode=='reduced' else c.MW_CONTROL_NAMES
    lane(controls/control_names[model][-1],model,[(t,'pure',i,True) for t in range(50) for i in range(50)],control=True)
    entries=[]
    for t in range(50):
        for level in c.LEVELS[1:]:
            # Task0: first failure k40; later recovery cannot lower k50.
            # Task1: two fewer successes passes; k30 fails by three -> k40.
            deficit=3 if (t==0 and level=='k40') or (t==1 and level=='k30') else 2
            entries += [(t,level,i,i>=deficit) for i in range(50)]
    run=tmp_path/'batch';lane(run,model,entries)
    result=descent.main(['--lane',str(run),'--miss-steps',mode,'--control-root',str(controls),'--out',str(tmp_path/'out')])
    row=result['cells'][0]
    assert row['tasks'][TASK_NAMES[0]]['selected']=='k50'
    assert row['tasks'][TASK_NAMES[1]]['selected']=='k40'
    assert row['complete_tasks']==50 and row['n']==2500 and not row['no_knob']
    assert row['group_mean']==pytest.approx(.96) and row['task_mean']==pytest.approx(.96)
    assert row['ir']>0 and row['control']['successes']==2500
    other=analysis.main(['--lane',str(run),'--controls',mode,'--control-root',str(controls),'--out',str(tmp_path/'same')])
    assert other==result
    assert not (tmp_path/'out/raise.json').exists()
    assert study.read(tmp_path/'out/levels.json')[model]['mt50']['250'][TASK_NAMES[0]]=='k50'


def test_group_mean_no_knob_shortcut_and_missing_lower_levels(tmp_path):
    keys={('mt50',task,i):dict(success=DIFFICULTY[task]=='easy',cost=dict(policy=1,library=1,controls=10))
          for task in TASK_NAMES for i in range(50)}
    stats=c.stats(keys,'DP')
    assert stats['group_mean']==.25 and stats['task_mean']==.56
    # Only koff observed; the complete cell equals pure -> use directly.
    result=descent.analyze_mt50({('DP','mt50',250):{'koff':keys}},{('DP','mt50'):keys},tmp_path/'off','full')
    row=result['cells'][0]
    assert row['no_knob'] and row['complete_tasks']==50
    assert all(t['selected']=='koff' for t in row['tasks'].values())
    ir_formal.main(['--test','full:'+str(tmp_path/'off/test.json'),'--out',str(tmp_path/'formal_ir')])
    report=(tmp_path/'formal_ir/IR_FORMAL.md').read_text()
    assert '25.0% / 56.0%' in report
    exported=study.read(tmp_path/'formal_ir/ir_formal.json')[0]
    assert exported['ir']==pytest.approx(row['ir'])
    levels={level:dict(n=50,passes={'4':True}) for level in c.LEVELS}
    levels['k40']['passes']['4']=False
    levels['k10']['n']=0
    assert descent.descend(levels,4,count=50)['selected']=='k50'
    levels['k50']['n']=49
    assert descent.descend(levels,4,count=50)['selected'] is None


def test_existing_mt50_koff_study_reuse(tmp_path):
    controls=tmp_path/'controls'
    entries=[(t,'koff',i,True) for t in range(50) for i in range(50)]
    lane(controls/c.MW_CONTROL_NAMES['π0.5'][-1],'π0.5',entries,control=True)
    lane(tmp_path/'study','π0.5',entries,base=True)
    result=analysis.main(['--koff-lane',str(tmp_path/'study'),'--control-root',str(controls),'--out',str(tmp_path/'out')])
    assert result['cells'][0]['no_knob'] and result['cells'][0]['n']==2500


def test_commands_parse_topology_and_fixed_groot_startup(source,tmp_path,monkeypatch):
    model=source[0]
    # All command generation is text; any attempted GPU/network launch is fatal.
    _,src,gaps=source
    path=knob.build(tmp_path/'commands',model=model,sources=[src],gaps=gaps)
    prep,run=knob.commands(path,model,miss_steps='full',run_name='cpu_mt50')
    tokens=shlex.split(prep);tokens=tokens[tokens.index('exp.r17_integration.ops.formal_eval')+1:]
    args=formal_eval.parse_args(tokens)
    p=knob.MW_PROFILES[model]
    assert args.server_node==p['node'] and args.replicas==p['replicas'] and args.max_batch==p['batch']
    assert args.worker_counts==dict((node,int(count)) for node,count in (item.split('=') for item in p['workers'].split(',')))
    assert args.worker_counts['timan1']==20
    assert args.worker_max_episodes==0 and args.prepare_only and args.mps
    assert '--eval-pool B' not in prep and '--denoising-steps' not in prep
    assert '--prepared-run /data/mw_finetune/eval/cpu_mt50' in run
    if model=='dp_mw':
        prep,_=knob.mw_commands(None,model,reference='test',run_name='tdp7h_full')
        assert '--denoising-steps 100' in prep and '--init-count 50' in prep and '--server-node h100' in prep
        tokens=shlex.split(prep)
        baseline=formal_eval.parse_args(tokens[tokens.index('exp.r17_integration.ops.formal_eval')+1:])
        assert baseline.prepare_only and baseline.seconds==172800
    if model!='groot_mw':return
    from exp.r17_integration.models.fastwam.ops import mps
    monkeypatch.setattr(mps,'paths',lambda *a,**kw:{})
    _,src,gaps=source
    for mode in ('full','reduced'):
        path=knob.build(tmp_path/mode,sources=[src],gaps=gaps,model=model,miss_steps=mode)
        monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path/'formal')
        seen=[]
        monkeypatch.setattr(formal_eval,'prepare',lambda campaign,directory:seen.append(dict(campaign)))
        def forbid(*a,**kw):raise AssertionError('CPU test attempted launch')
        monkeypatch.setattr(formal_eval,'run_campaign',forbid)
        monkeypatch.setattr(formal_eval,'open_tunnels',forbid)
        prep,_=knob.commands(path,model,run_name='cpu_'+mode)
        argv=shlex.split(prep);argv=argv[argv.index('exp.r17_integration.ops.formal_eval')+1:]
        assert formal_eval.main(argv)==0
        assert seen[0]['live_k']==knob.MW_STEPS[model][mode=='reduced']


def test_mt50_formal_stage_costs_and_cpu_text_exclusion():
    for model in ('DP','Fast-WAM','π0.5','GR00T'):
        system,pure=ir_formal.coefficients(model,'mt50')
        assert system['vision']+system['full_miss']==pytest.approx(1)
        assert 0<pure['vision']+pure['reduced_miss']<1
    p=study.read(study.ROOT/'exp/r17_integration/profile/results_mt50/dp_mw_mt50.json')
    system,_=ir_formal.coefficients('DP','mt50')
    v=p['stage1_ms']-p['dp_cpu_text_encoder_ms']['median']
    assert system['vision']==pytest.approx(v/(v+p['stage2_ms']+p['stage3_full_loop_ms']))
