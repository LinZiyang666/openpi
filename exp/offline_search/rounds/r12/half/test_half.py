"""Binding contract §8(a)-(h), genuine B rows and dry final serving argv."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace as NS
import numpy as np
import pytest
from exp.offline_search.harness import api
from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.blind import BlindResult
from exp.offline_search.closed_loop.lookhalf import NeedCamera, _View
from exp.offline_search.closed_loop.lookhalf_cameras import ledger
from exp.offline_search.closed_loop.half_everywhere import R12HalfEverywhere
from exp.offline_search.closed_loop.half_everywhere_config import SPEC, SCHEMA, load_table, validate_fitted
from exp.offline_search.closed_loop.half_everywhere_ledger import summarize, validate_receipt
from exp.offline_search.closed_loop.lookhalf_config import value
from exp.offline_search.closed_loop.ops.h100 import assets, control, local_server, node
from exp.offline_search.closed_loop.test_r12_test_a import dry_transport  # noqa: F401
from exp.offline_search.rounds.r02.g3_recovery.g3_core import PseudoQuery, episode_rows
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary
from .build import HERE, R12, CELLS, STORE, load, sha, write_json
from .prepare import wrap
from .knob import calibrate

CELL_NAMES = [f'{m}_{s}_{n}' for m,s,n in CELLS]
RECORDS = json.loads((HERE/'provisional/pilot_roots.json').read_text())+json.loads((HERE/'provisional/test_a_roots.json').read_text())


@pytest.fixture(scope='session')
def argv_evidence():
    rows=[]
    yield rows
    write_json(HERE/'ARGV_VERIFICATION.json',rows)


def binding(tmp_path, cell, mode):
    p = tmp_path/f'{cell}_{mode}.json'
    write_json(p, dict(schema=SCHEMA,cell=cell,camera={str(t):mode for t in range(10)}))
    return dict(path=str(p),sha256=sha(p))


def fixture(cell, tmp_path, mode='wrist', knob=False):
    model,suite,_ = cell.split('_')
    source = (R12.parent/'r11/knob/artifacts'/f'r11_{cell}_periodic_ir32.pkl' if knob else
              R12.parent/'r10/recipe/artifacts'/f'r10_recipe_{cell}.pkl')
    base = load(source)['method']
    guard = load(R12/'guard/artifacts'/cell/'h2.pkl')['method']
    fit = load(HERE/'fits'/cell/'wrist.pkl')
    kw = binding(tmp_path,cell,mode)
    method,_ = wrap(base,guard,fit,kw,knob_method='periodic' if knob else 'off',target=.32 if knob else None)
    lib = _Library(STORE,f'{model}_{suite}',base.library,base.episode_subset)
    return base,method,guard,lib


def query(lib, table, rows, step, model, *, prev_hit=True, gap=False):
    q = PseudoQuery(lib,table,rows,step,model,prev_hit)
    ep=q.episode
    return _View(q,episode=api.EpisodeView(ep.uid,ep.task,ep.task_id,0,ep.index,ep.seed),
                 hist_has_vision=(np.arange(step)%2==0 if gap else np.ones(step,bool)))


class NoAgent:
    def __init__(self,q): self.q=q
    def __getattr__(self,name):
        if name in ('key_v0','hist_key_v0','tok_v0','hist_tok_v0'):
            raise AssertionError('wrist path read key_v0')
        return getattr(self.q,name)


def equal(a,b,*,extras=True):
    for name in ('topk','scores','action'):
        assert getattr(a,name).tobytes()==getattr(b,name).tobytes(),name
    assert a.confidence==b.confidence and a.library==b.library
    if extras: assert a.extras==b.extras


@pytest.mark.parametrize('cell,knob',[(c,False) for c in CELL_NAMES]+[('pi05_l10_50',True)])
def test_a_all_both_byte_exact_deployed(cell,knob,tmp_path):
    base,m,_,lib=fixture(cell,tmp_path,'both',knob)
    episodes=list(episode_rows(base.inner.C).values())
    for task in range(10):
        rows=next(r for r in episodes if base.inner.C.task[r[0]]==task)
        q0=query(lib,base.inner.C,rows,0,base.inner.base.model)
        base.reset(q0.episode);m.reset(q0.episode)
        for step in range(min(10,len(rows))):
            q=query(lib,base.inner.C,rows,step,base.inner.base.model)
            a,b=base.query(q),m.query(q)
            equal(a,b)
            assert base.inner._s==m.inner.deployed._s
            assert base.inner._vision_progress==m.inner._vision_progress
            assert base.inner._noprog_span==m.inner._noprog_span
            assert base.inner.base._anchor['action'].tobytes()==m.inner.base._anchor['action'].tobytes()
            assert m.plan_cameras(q,look_reason=6,executed_steps=0,execution_audit=False,exec_ok=False)=='full'
            # Actual accepted-decision camera work is the deployed price exactly.
            miss=bool(a.extras.get('os_force_miss',0))
            cost=ledger(lib.model,vision=True,wrist_first=False,completions=0,miss=miss)
            assert cost==ledger(lib.model,vision=True,wrist_first=False,completions=0,miss=bool(b.extras.get('os_force_miss',0)))


@pytest.mark.parametrize('cell',CELL_NAMES)
def test_b_all_wrist_disabled_byte_exact_ablation_chunks_and_tails(cell,tmp_path):
    base,m,_,lib=fixture(cell,tmp_path)
    reference=load(R12/'ablate/artifacts'/cell/'wrist_sub.pkl')['method']
    m.guard_enabled=False;m.inner.base.corrector_enabled=False
    episodes=list(episode_rows(base.inner.C).values())
    for task in range(10):
        rows=next(r for r in episodes if base.inner.C.task[r[0]]==task)
        q0=query(lib,base.inner.C,rows,0,lib.model)
        m.reset(q0.episode);reference.reset(q0.episode)
        for step in range(0,min(10,len(rows)-1),2):
            q=query(lib,base.inner.C,rows,step,lib.model,gap=True)
            a,b=m.query(NoAgent(q)),reference.query(NoAgent(q));equal(a,b)
            bq=query(lib,base.inner.C,rows,step+1,lib.model,gap=True)
            bq=_View(bq,blind_age=0,executed_steps=5)
            ba,bb=m.blind_step(NoAgent(bq)),reference.blind_step(NoAgent(bq))
            assert isinstance(ba,BlindResult) and isinstance(bb,BlindResult)
            assert ba.action.tobytes()==bb.action.tobytes()
            assert ba.rows.tobytes()==bb.rows.tobytes() and ba.weights.tobytes()==bb.weights.tobytes()


@pytest.mark.parametrize('cell',CELL_NAMES)
def test_c_h2_identical_inputs_features_flags_and_preserved_artifacts(cell,tmp_path):
    base,m,guard,lib=fixture(cell,tmp_path)
    assert pickle.dumps(m.risk_fit,protocol=4)==pickle.dumps(guard.risk_fit,protocol=4)
    assert pickle.dumps(m.half_fit,protocol=4)==pickle.dumps(guard.half_fit,protocol=4)
    rows=next(iter(episode_rows(base.inner.C).values()))
    q0=query(lib,base.inner.C,rows,0,lib.model)
    m.reset(q0.episode);guard.reset(q0.episode)
    for step in range(0,min(12,len(rows)),2):
        # Identical accepted states, actions, previous served progress and guard
        # history; the independent opus_w guard metric is never replaced by sub.
        q=NoAgent(query(lib,base.inner.C,rows,step,lib.model,gap=True))
        m._guard_history=copy.deepcopy(guard._guard_history)
        pa,pb=m._prepare_guard(q),guard._prepare_guard(q)
        assert pa['guard_call']==pb['guard_call'] and pa['scalars']==pb['scalars']
        assert np.array_equal(pa['features'],pb['features'])
        assert np.array_equal(pa['W']['all_distances'],pb['W']['all_distances'])
        guard._guard_history.append(dict(step=step,rs=pb['rs'],served_prog=float(guard.inner.C.prog[pb['W']['rows'][0]])))
        guard._guard_history=guard._guard_history[-3:]
        m._pending_guard=guard._pending_guard=None


@pytest.mark.parametrize('target,expected',[(.25,2.6125),(.32,1.0657),(.40,.4386)])
def test_d_knob_all_both_exact_R11(target,expected):
    base=load(R12.parent/'r10/recipe/artifacts/r10_recipe_pi05_l10_50.pkl')['method']
    result=calibrate('pi05','l10',50,{str(t):'both' for t in range(10)},base.library,target)
    assert result['setting']==expected
    from exp.offline_search.rounds.r11.knob.calibration import tables
    S=tables('pi05','l10',50)[3]
    x,ir,status=S.solve('periodic',target)
    assert result['unrounded_setting']==x and result['pred_IR_lib']==ir and result['solve_status']==status


@pytest.mark.parametrize('cell',CELL_NAMES)
def test_e_wrist_every_look_no_hidden_key_corrector_and_continuity(cell,tmp_path):
    base,m,_,lib=fixture(cell,tmp_path)
    rows=next(iter(episode_rows(base.inner.C).values()))
    for step in range(0,min(12,len(rows)),2):
        q=query(lib,base.inner.C,rows,step,lib.model,prev_hit=step!=2,gap=True)
        if step==0:m.reset(q.episode)
        mode=m.plan_cameras(q,look_reason=6+step%3,executed_steps=0,execution_audit=False,exec_ok=False)
        assert mode=='wrist_only'
        m.set_camera_mode(mode)
        result=m.query(NoAgent(q))
        if isinstance(result,NeedCamera):
            assert result.reason=='h2_call'
            assert m.inner.decision is None
            m.set_camera_mode('full')
            result=m.query(NoAgent(q))
            assert result.extras['os_force_miss']
        assert np.isfinite(result.action).all()
        if step==0: assert m.lookhalf_diagnostics['factor']==0
        if step==2:
            T,_,reg,_,_,_,_,d,med,c,dt=m.inner.base._dist(NoAgent(q))
            assert reg==1 and c is not None
            np.testing.assert_array_equal(dt,d/med+m.inner.base.lam_c*c/T.s_c)
        equal(result,m.query(NoAgent(q)))
    cloned,_=plugin.clone_method(m,strict=True)
    assert cloned is not m and cloned.inner is not m.inner


@pytest.mark.parametrize('model',['pi05','groot'])
def test_e_ledger_per_mode_engagement_counts(model):
    dd=[]
    for step,mode,vision,miss,initial in [(0,'wrist',True,False,1),(1,'wrist',False,False,0),
        (2,'wrist',True,True,1),(3,'wrist',True,True,2),(4,'both',True,False,2)]:
        completions=int(initial==1 and miss)
        cost=ledger(model,vision=vision,wrist_first=initial==1,completions=completions,miss=miss)
        work=dict(initial_cameras=initial,completions=completions,
                  availability=[True,True] if miss or mode=='both' else [False,True]) if vision else {}
        dd.append(dict(step=step,uid='ep',task_mode=mode,vision=vision,hit=not miss,camera_work=work,lookhalf_cost=cost))
    out=summarize(model,dd)
    assert out['wrist_task_decisions']==4 and out['both_task_decisions']==1
    assert out['wrist_task_looks']==3 and out['wrist_task_two_camera_looks']==2
    assert out['per_mode']['wrist']['per_episode']['ep']['first_call']==2
    assert out['work']==pytest.approx(2+1.5*(.152 if model=='pi05' else .148))
    for mode,miss,initial,completion in [('wrist',False,2,0),('both',False,1,0)]:
        cost=ledger(model,vision=True,wrist_first=initial==1,completions=completion,miss=miss)
        with pytest.raises(ValueError):validate_receipt(mode,vision=True,miss=miss,
            work=dict(initial_cameras=initial,availability=[initial==2,True]),cost=cost)
    with pytest.raises(ValueError):validate_receipt('wrist',vision=True,miss=False,work={},cost=dd[0]['lookhalf_cost'])


@pytest.mark.parametrize('cell',CELL_NAMES)
def test_f_library_fit_audit_loeo_donors_folds(cell):
    fit=load(HERE/'fits'/cell/'wrist.pkl');info=fit['fit_info']
    assert info['library_only'] and len(fit['heads'])==len(fit['scale_sub'])==10
    assert info['head_parameters']==dict(features=153,rff=384,ridge=100.,seed=0,blend=.5,step0=False)
    assert info['fit_reads'] and all('/library/' in p and '/queries/' not in p and 'os_closed_loop' not in p for p in info['fit_reads'])
    base=load(R12/'guard/artifacts'/cell/'h2.pkl')['method']
    served=load(HERE/'provisional/artifacts'/f'{cell}.pkl')['method']
    assert all(not a.flags.writeable for h in served.inner.base.wrist_heads.values() for a in h.values())
    assert all(not a.flags.writeable for metric in served.inner.base.camera_metrics.values()
               for a in vars(metric).values() if isinstance(a,np.ndarray))
    for proof in info['loeo'].values():
        assert proof['own_episode_excluded']
        assert np.all(base.inner.base.lib_ep[proof['query_rows']]!=base.inner.base.lib_ep[proof['donor_top1']])
    for fold in info['distance_folds']:
        assert not set(fold['train_episodes'])&set(fold['heldout_episodes'])
    # Independently force a query's own-episode rows to perfect distance. LOEO
    # must ignore every one, not just the nearest row.
    from exp.offline_search.rounds.r12.lookhalf.build import loeo
    class Metric:
        early=None
        def dist(self,X,initial=False):return np.tile(np.r_[np.zeros(16),np.ones(16)],(len(X),1))
    C=dict(ep=np.r_[np.zeros(16,int),np.ones(16,int)],step=np.zeros(32,int),
        act=np.r_[np.full((16,10,7),999.,np.float32),np.ones((16,10,7),np.float32)],kref=8)
    raw,_,top=loeo(Metric(),np.zeros((32,72)),np.arange(16),np.arange(32),C)
    assert np.all(raw==1) and np.all(top>=16)


@pytest.mark.parametrize('local',[True,False])
@pytest.mark.parametrize('record',RECORDS,ids=lambda r:r['suffix']+'_'+r['init_pool'])
def test_g_every_emitted_arm_real_plan_dry_sync_full_final_argv(record,local,tmp_path,monkeypatch,dry_transport,argv_evidence):
    root=Path(record['root']);host='timan107' if local else 'timan108'
    monkeypatch.setenv('SERVER_HOST','local' if local else 'h100');monkeypatch.setenv('WORKER_HOST',host)
    for key in ('OSCL_MANIFEST','OSCL_INIT_POOL','OSCL_TASKS','OSCL_EPISODES','PI05_CKPT','GROOT_CKPT','GROOT_DENOISING_STEPS'):
        monkeypatch.delenv(key,raising=False)
    before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    work=tmp_path/'plan'
    planned=control.sync(root,record['arms'],plan_only=True,workdir=work)
    synced=control._sync(root,record['arms'],workdir=work)
    assert planned==synced and synced['worker_host']==host
    from exp.offline_search.closed_loop import half_everywhere_config as config
    original_loader=config.load_table
    def mapped_loader(binding,**kw):
        mapped=dict(binding)
        if not Path(mapped['path']).exists():
            mapped['path']=next(f['source'] for f in synced['files'] if str(assets.BASE/f['rel'])==binding['path'])
        return original_loader(mapped,**kw)
    monkeypatch.setattr(config,'load_table',mapped_loader)
    originals=json.loads((root/'arms.json').read_text())
    for original,row in zip(originals,synced['arms'],strict=True):
        mapper=assets.local_remap if local else assets.remap
        expected=mapper(copy.deepcopy(original['plugin_args']),root)
        fitted=value(expected,'--os-fit-artifact')
        fit=next(f for f in synced['files'] if original['arm']+': fitted method' in f['reasons'])
        if local:expected=[fit['source'] if x==fitted else x for x in expected]
        if record['init_pool']=='B':
            contract=next(f for f in synced['files'] if original['arm']+': B recording contract' in f['reasons'])
            expected+=['--os-look-contract',contract['source'] if local else str(assets.BASE/contract['rel']),
                       '--os-look-contract-sha256',contract['sha256']]
        assert row['plugin_args']==expected
        spec=control.server_spec(root,row,23360)
        expected_plugin=['--os-method',original['method'],'--os-cell',original['cell'],
                         *expected,'--os-kwargs',json.dumps(mapper(original['kwargs'],root))]
        assert spec['plugin']==expected_plugin
        if local:
            spec['run_root']=str(work/'argv');spec['out']=str(work/'argv'/original['arm'])
        argv,env=local_server.command(spec) if local else node.command(spec['model'],spec['suite'],spec['port'],
            spec['yaml'],spec['out'],spec['tag'],spec['plugin'],dict(spec['env']))
        stock=(['--port',str(spec['port']),'--replicas','1','--cache-config',spec['yaml'],'policy:checkpoint',
            '--policy.config','pi05_libero','--policy.dir',local_server.PI05_CKPT if local else '/home/exouser/ckpt/pi05_libero_pytorch']
            if row['model']=='pi05' else ['--checkpoint','/data/ckpt/n15_'+original['suite'] if local else '/home/exouser/ckpt/n15_'+original['suite'],
            '--port',str(spec['port']),'--denoising-steps','8','--concurrent','--allow-dynamic-bundles','--cache-config',spec['yaml']])
        assert argv[5 if local else 2:]==[*expected_plugin,'--os-log-dir',spec['out'],'--os-tag',spec['tag'],*stock]
        opts,rest=plugin.parse_cli(argv[5 if local else 2:]);assert rest==stock
        assert opts.os_lookhalf in ('off','half_everywhere')
        assert opts.os_record_looks==(record['init_pool']=='B') and env['STAGE1_ONLY']=='0'
        execution=control.execution_env(row)
        assert execution.get('OSCL_RECORD_LOOKS')==('1' if record['init_pool']=='B' else None)
        if row['method']==SPEC:
            assert execution['OSCL_EXECUTION_AUDIT']=='1'
        frozen=load(fit['source'])
        assert frozen['kwargs']==row['kwargs']
        if row['method']==SPEC:
            table=config.load_table(row['kwargs']['camera_table'])
            assert frozen['method'].camera_table==row['kwargs']['camera_table']
            assert frozen['method'].camera==table['camera']
        argv_evidence.append(dict(root=str(root),arm=row['arm'],init_pool=record['init_pool'],
            server_host='local' if local else 'h100',worker_host=host,argv=argv,
            exact_argv=True,transport='mocked',cuda_visible_devices=''))
    assert before=={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not (root/'runs').exists() and not (root/'state').exists()


def test_h_missing_changed_wrong_cell_and_tampered_fitted_table_refused(tmp_path):
    with pytest.raises(ValueError):R12HalfEverywhere()
    b=binding(tmp_path,'pi05_l10_50','both')
    with pytest.raises(ValueError,match='missing camera table'):load_table(dict(b,path=str(tmp_path/'missing')))
    with pytest.raises(ValueError,match='SHA256 mismatch'):load_table(dict(b,sha256='0'*64))
    with pytest.raises(ValueError,match='cell mismatch'):load_table(b,cell='groot_l10_50')
    base,m,_,_=fixture('pi05_l10_50',tmp_path,'both')
    m.camera['0']='wrist'
    with pytest.raises(ValueError,match='configuration mismatch'):
        validate_fitted(m,dict(camera_table=b,knob_method='off',target=None))
    args=['--os-method',SPEC,'--os-cell','pi05_l10_cache','--os-log-dir',str(tmp_path),
          '--os-lookhalf','half_everywhere','--os-blind','--os-judge','guard_only',
          '--os-no-shadow-native','--os-tokens','off','--os-kwargs',json.dumps(dict(camera_table=dict(b,sha256='0'*64)))]
    with pytest.raises(SystemExit,match='SHA256 mismatch'):plugin.parse_cli(args)


def test_provisional_selection_exact_copy_skips_zero_wrist_and_A_not_installed():
    emission=json.loads((HERE/'provisional/EMISSION.json').read_text())
    assert emission['provisional'] and emission['skipped']==[dict(cell='pi05_spatial_50',reason='no wrist tasks; B and A skipped')]
    assert len(emission['B'])==len(emission['A'])==5
    for record in RECORDS:
        name='dev_selection.json' if record['init_pool']=='B' else 'eval500.json'
        assert (Path(record['root'])/name).read_bytes()==Path(record['source_selection']).read_bytes()
        assert record['wrist_tasks'] and record['provisional']
        assert record['episodes_per_arm']==(100 if record['init_pool']=='B' else 500)
        if record['cell'].startswith('groot') and record['init_pool']=='B':
            assert record['server_host']=='h100' and len(record['arms'])==2


@pytest.mark.parametrize('batched',[False,True])
def test_e_cold_pi05_single_real_image_and_idempotent_completion(batched,monkeypatch):
    import torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from openpi.serving import stage_io
    from openpi.models.model import Observation
    from exp.offline_search.rounds.r07.c2_wrist.test_dispatch import Model,stock,raw,keys
    from exp.offline_search.closed_loop.half_everywhere_cameras import HalfPi05Override
    from exp.offline_search.closed_loop.stage_overrides import CameraRequest
    # Preserve every monkey-patched installation hook, including split/stack.
    for cls,name in ((PI0Pytorch,'run_stage1'),(PI0Pytorch,'run_stage2'),(PI0Pytorch,'run_stage2_capture'),
                     (Pi05StageBatcher,'run_stage1_batch'),(stage_io,'split_stage1_output'),(stage_io,'stack_stage1_output')):
        monkeypatch.setattr(cls,name,getattr(cls,name))
    monkeypatch.setattr(PI0Pytorch,'run_stage1',stock)
    monkeypatch.setattr(PI0Pytorch,'_r4_override',None,raising=False)
    def full_batch(batcher,observations):
        return [stock(batcher._model,Observation.from_dict(stage_io.stack_observation([obs],device='cpu'))) for obs in observations]
    monkeypatch.setattr(Pi05StageBatcher,'run_stage1_batch',full_batch)
    override=HalfPi05Override().install();model=Model();audit={}
    if batched:
        out=Pi05StageBatcher.run_stage1_batch(NS(_model=model,_device='cpu'),[CameraRequest(raw(3.),'wrist_only',audit)])[0]
    else:
        obs=Observation.from_dict(stage_io.stack_observation([raw(3.)],device='cpu'))
        out=model.run_stage1(CameraRequest(obs,'wrist_only',audit))
    assert model.paligemma_with_expert.calls==[(1,(6.,))]
    assert audit==dict(looks=1,initial_cameras=1,availability=[False,True],completions=0)
    assert torch.all(keys(out)['vision_0']==0) and torch.all(keys(out)['vision_1']==6)
    completed=override.complete(model,out)
    assert model.paligemma_with_expert.calls==[(1,(6.,)),(1,(3.,))]
    assert audit['completions']==1
    override.complete(model,completed)
    assert len(model.paligemma_with_expert.calls)==2
    assert torch.all(keys(completed)['vision_0']==3) and torch.all(keys(completed)['vision_1']==6)
    # The full request goes through the byte-preserved stock encoder.
    full_audit={}
    obs=Observation.from_dict(stage_io.stack_observation([raw(3.)],device='cpu'))
    full=model.run_stage1(CameraRequest(obs,'full',full_audit))
    assert torch.equal(keys(full)['vision_0'],keys(completed)['vision_0'])
    assert full_audit==dict(looks=1,initial_cameras=2,availability=[True,True],completions=0)


def test_e_initial_session_respects_wrist_table_and_actual_receipt(tmp_path):
    from exp.offline_search.closed_loop.test_look_recording import connection,FakePolicy
    from exp.offline_search.closed_loop.half_everywhere_cameras import receipt
    from exp.offline_search.closed_loop.stage_overrides import CameraRequest
    staged=next(r for r in RECORDS if r['init_pool']=='A')
    row=json.loads((Path(staged['root'])/'arms.json').read_text())[0]
    args=['--os-method',SPEC,'--os-cell',row['cell'],'--os-kwargs',json.dumps(row['kwargs']),
          '--os-log-dir',str(tmp_path),*row['plugin_args']]
    opts,_=plugin.parse_cli(args)
    conn,s,obs,records=connection(plugin,opts)
    s.rt.lookhalf='half_everywhere';s.rt.request_cameras=s.rt.blind=s.rt.r4=True
    s.method.camera={str(t):'wrist' for t in range(10)}
    s.method.allows_wrist_every_look=True
    s.method.lookhalf_diagnostics={}
    def plan(q,**kw):
        s.method.next_camera_mode='wrist_only'
        return 'wrist_only'
    s.method.plan_cameras=plan
    s.method.set_camera_mode=lambda mode:None if mode=='wrist_only' else pytest.fail('initial full forced')
    s.method.invalidate_anchor=lambda:None
    class Policy(FakePolicy):
        def stage1(self,observation):receipt(CameraRequest(observation,s._camera_mode,s._camera_work));return observation
    conn._osp_inner=Policy(s);conn._osp_adapter=plugin._BlindAdapter(conn._osp_inner,s)
    conn.infer({**obs,'__extra__':dict(executed_steps=0,exec_ok=True,decision_id=0)})
    d=next(r for r in records if r['ev']=='dec')
    assert d['hit'] and d['vision'] and d['camera_mode']=='wrist_only'
    assert d['task_mode']=='wrist' and d['lookhalf_cost']['camera_encoded_count']==1
    assert d['owner_cost']==.076


def test_e_h2_completion_before_single_commit_and_periodic_full_plan(tmp_path):
    base,m,_,lib=fixture('pi05_l10_50',tmp_path,knob=True)
    rows=next(iter(episode_rows(base.inner.C).values()))
    q0=query(lib,base.inner.C,rows,0,lib.model)
    m.reset(q0.episode)
    # At a due look both images are acquired before stage one.
    m.knob_settings=dict(m.knob_settings,setting=0.)
    assert m.plan_cameras(q0,look_reason=6,executed_steps=0,execution_audit=True,exec_ok=True)=='full'
    m.set_camera_mode('full');called=m.query(NoAgent(q0))
    assert called.extras['os_r11_knob_call']==1
    # H2 completion must leave the deployed judge and synthesis untouched until
    # the request has both cameras; then commit exactly once, also on retries.
    m.knob_settings=dict(m.knob_settings,setting=40.)
    q=query(lib,base.inner.C,rows,4,lib.model,gap=True)
    m.set_camera_mode('wrist_only')
    pending=m._prepare_guard(NoAgent(q));pending['guard_call']=True;pending['scalars']['call']=True
    before=pickle.dumps(m.inner.deployed)
    assert m.query(NoAgent(q))==NeedCamera(reason='h2_call')
    assert pickle.dumps(m.inner.deployed)==before and m.inner.decision is None
    m.set_camera_mode('full');result=m.query(NoAgent(q))
    assert result.extras['os_reason']==4 and result.extras['os_r11_guard']==1
    assert result.extras['os_r11_knob_call']==0
    count=len(m.inner._vision_progress);equal(result,m.query(NoAgent(q)))
    assert len(m.inner._vision_progress)==count


def test_builder_only_table_and_gap_depend_on_choice_and_skip_all_both(tmp_path):
    from .prepare import prepare
    source=tmp_path/'screen_fixture.json'
    original=json.loads((R12/'opus/out/design/camera_choice_screen.json').read_text())
    for row in original.values():
        row['camera']=['both']*10;row['wrist_tasks']=[]
    write_json(source,original)
    before={cell:sha(HERE/'fits'/cell/'wrist.pkl') for cell in CELL_NAMES}
    dest=tmp_path/'all_both_provisional'
    result=prepare(source,provisional=True,destination=dest)
    assert not result['B'] and not result['A'] and len(result['skipped'])==5
    assert not (dest/'pilots').exists()
    assert before=={cell:sha(HERE/'fits'/cell/'wrist.pkl') for cell in CELL_NAMES}
    bound=load(dest/'artifacts/pi05_l10_50_periodic_ir32.pkl')['method']
    assert bound.knob_settings['setting']==1.0657
    assert bound.camera_table['sha256']==sha(bound.camera_table['path'])
    assert all(v=='both' for v in bound.camera.values())
    with pytest.raises(FileExistsError):prepare(source,provisional=True,destination=dest)
    with pytest.raises(ValueError,match=r'screen \+ camdev'):prepare(source,destination=tmp_path/'invalid_final')


def test_h_standalone_remote_config_without_serving_pythonpath(tmp_path):
    import subprocess,sys
    config=Path(plugin.__file__).with_name('lookhalf_config.py')
    b=binding(tmp_path,'pi05_l10_50','wrist')
    argv=['--os-method',SPEC,'--os-lookhalf','half_everywhere',
          '--os-kwargs',json.dumps(dict(camera_table=b,knob_method='periodic',target=.32))]
    code='import importlib.util,json,sys; s=importlib.util.spec_from_file_location("validator",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.validate(json.loads(sys.argv[2])))'
    env=dict(os.environ,PYTHONPATH='')
    result=subprocess.run([sys.executable,'-B','-c',code,str(config),json.dumps(argv)],cwd=tmp_path,
                          env=env,text=True,capture_output=True,check=True)
    assert result.stdout.strip()=='half_everywhere'
