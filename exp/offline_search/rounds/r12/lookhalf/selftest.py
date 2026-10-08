"""CPU tests on genuine selected B-library rows; no query/test-A store reads."""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.lookhalf import R12LookHalf, NeedCamera, _View
from exp.offline_search.closed_loop.lookhalf_math import atlas_plan
from exp.offline_search.closed_loop.lookhalf_cameras import GrootOverride, ledger, complete_session
from exp.offline_search.closed_loop.lookhalf_ledger import summarize, numerical_report
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary, fit_path
from exp.offline_search.rounds.r02.g3_recovery.g3_core import PseudoQuery, episode_rows
from .build import HERE, R10, STORE, CELLS

EVIDENCE = {}


def load(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)['method']


def equal_result(test, a, b):
    for k in ('topk', 'scores', 'action'):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    test.assertEqual(a.confidence, b.confidence)
    test.assertEqual(a.extras, b.extras)
    test.assertEqual(a.library, b.library)


class LibraryTests(unittest.TestCase):
    def test_off_and_full_fallback_every_cell(self):
        evidence = []
        for model, suite, size in CELLS:
            base = load(R10/'recipe/artifacts'/f'r10_recipe_{model}_{suite}_{size}.pkl')
            off = load(HERE/'artifacts'/f'{model}_{suite}_{size}'/'off.pkl')
            active = load(HERE/'artifacts'/f'{model}_{suite}_{size}'/'opus_m1.pkl')
            lib = _Library(STORE, f'{model}_{suite}', base.library, base.episode_subset)
            episodes = list(episode_rows(base.inner.C).values())
            count = 0
            for task in range(10):
                rows = next(r for r in episodes if base.inner.C.task[r[0]] == task)
                q0 = PseudoQuery(lib, base.inner.C, rows, 0, model, True)
                for method in (base, off, active):
                    method.reset(q0.episode)
                for pos in range(min(len(rows), 12)):
                    q = PseudoQuery(lib, base.inner.C, rows, pos, model, True)
                    q = _View(q, hist_has_vision=np.ones(pos, bool))
                    r = base.query(q)
                    equal_result(self, r, off.query(q))
                    active.set_camera_mode('full')
                    equal_result(self, r, active.query(q))
                    self.assertEqual(base.inner._vision_progress, off.inner._vision_progress)
                    self.assertEqual(base.inner._vision_progress, active.inner._vision_progress)
                    # A repeated active request must not change the guard.
                    before = list(active.inner._vision_progress)
                    equal_result(self, r, active.query(q))
                    self.assertEqual(before, active.inner._vision_progress)
                    count += 1
            evidence.append(dict(model=model, suite=suite, size=size, decisions=count,
                                 rows_weights_actions_guard_equal=True, retries_idempotent=True))
        EVIDENCE['off_and_full_equivalence'] = evidence

    def test_explorer_fit_reproduction(self):
        evidence = []
        for model, suite, size in CELLS:
            method = load(HERE/'artifacts'/f'{model}_{suite}_{size}'/'astra_consensus2.pkl')
            fit = method.half_fit
            path = HERE.parent/'astra/final_atlas'/f'{model}_{suite}_{size}'/'table.npz'
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['row_subset'], method.row_subset)
                np.testing.assert_array_equal(z['safe'], fit['atlas']['safe'])
                np.testing.assert_array_equal(z['next2'], fit['atlas']['next2'])
                maximum = 0.
                for task in range(10):
                    for own, reference in [('astra_f', 'full'), ('astra_h', 'half')]:
                        met = fit['metrics'][own][task][0]
                        for field in ('W', 'shift', 'Z', 'z2'):
                            delta = float(np.max(np.abs(getattr(met, field)-z[f'{reference}_{task}_{field}'])))
                            maximum = max(maximum, delta)
                            np.testing.assert_allclose(getattr(met, field), z[f'{reference}_{task}_{field}'], atol=2e-3, rtol=2e-4)
            with (HERE.parent/'astra/dev_validation/fits'/f'{model}_{suite}_{size}'/'half.pkl').open('rb') as f:
                reference = pickle.load(f)
            self.assertAlmostEqual(reference['scale'], fit['scales']['astra_h'], delta=2e-3)
            head_delta = max(float(np.max(np.abs(reference['heads'][t][k]-fit['heads']['astra_h'][t][k])))
                             for t in range(10) for k in reference['heads'][t])
            self.assertLess(head_delta, 2e-3)
            # Load only inert fit objects; importing Astra's exploration module
            # would install its old isolation audit hook in this test process.
            class Fits(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.endswith('.astra.retrieval') and name == 'Metric':
                        return NS
                    return super().find_class(module, name)
            path = HERE.parent/'astra/directive2/fits'/f'{model}_{suite}_{size}'/'wrist.pkl'
            with path.open('rb') as f:
                wrist = Fits(f).load()
            wrist_delta = max(float(np.max(np.abs(wrist['heads'][t][k]-fit['heads']['astra_w'][t][k])))
                              for t in range(10) for k in wrist['heads'][t])
            self.assertLess(wrist_delta, 2e-3)
            self.assertAlmostEqual(wrist['scale'], fit['scales']['astra_w'], delta=2e-3)
            evidence.append(dict(model=model, suite=suite, size=size, atlas_exact=True,
                max_metric_abs=maximum, max_head_abs=head_delta,
                scale_abs=abs(reference['scale']-fit['scales']['astra_h']),
                wrist_head_max_abs=wrist_delta, wrist_scale_abs=abs(wrist['scale']-fit['scales']['astra_w'])))
        EVIDENCE['astra_fit_reproduction'] = evidence

    def test_opus_metrics_match_reference_arithmetic(self):
        from exp.offline_search.rounds.r12.opus.tools.common import Metric as Reference
        evidence = []
        for model,suite,size in CELLS:
            m = load(HERE/'artifacts'/f'{model}_{suite}_{size}'/'opus_m1.pkl')
            b, fit = m.inner.base, m.half_fit
            lib = _Library(STORE, f'{model}_{suite}', m.library, m.episode_subset)
            act = np.asarray(lib.action[:,:10,:7],np.float32)
            sigma = act[:,:5].astype(np.float64).std((0,1))
            heads = (act[:,:5]/sigma).reshape(len(act),35)
            worst = 0.
            for name in ('opus_w','opus_l2','opus_l4','opus_l6','opus_l8'):
                for task in range(10):
                    own,rows = fit['metrics'][name][task]
                    # Recover the fitting coordinates from affine metric state,
                    # rather than opening an unselected parent key corpus.
                    P,rs = fit['P'],np.asarray(lib.rs[:,:8])
                    if name == 'opus_w':
                        X = np.c_[P[:,64:],rs].astype(np.float64)
                    else:
                        from .build import predecessors
                        past = predecessors(lib,int(name[6:]),opus=True)
                        X = np.c_[P[past,:64],P[:,64:],rs].astype(np.float64)
                    ref = Reference(X[rows],heads[rows],lib.episode[rows],lib.step[rows])
                    np.testing.assert_array_equal(own.mean,ref.mean)
                    np.testing.assert_array_equal(own.std,ref.std)
                    np.testing.assert_array_equal(own.transform,ref.W)
                    distance = own.dist(X[rows[:3]])
                    from exp.offline_search.rounds.r10.analysis_opus.tools.core import pdist
                    expected = pdist(ref.code(X[rows[:3]]),ref.code(X[rows]))
                    worst = max(worst,float(np.max(np.abs(distance-expected))))
                    np.testing.assert_allclose(distance,expected,atol=2e-6,rtol=1e-6)
            evidence.append(dict(model=model,suite=suite,size=size,main_metrics_exact=True,max_distance_abs=worst))
        EVIDENCE['opus_fit_reproduction'] = evidence

    def test_emitted_roots_and_library_disjointness(self):
        from exp.offline_search.closed_loop import devset
        from exp.offline_search.closed_loop.look_recording import validate_root
        from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
        from exp.offline_search.closed_loop.plugin import parse_cli
        roots = json.loads((HERE/'pilot_roots.json').read_text())
        self.assertEqual(len(roots),5)
        total = 0
        for record in roots:
            root = Path(record['root']); validate_root(root)
            self.assertEqual((root/'dev_selection.json').read_bytes(),Path(record['source_selection']).read_bytes())
            manifest = load_manifest(root/'dev_selection.json')
            self.assertEqual(len(manifest['selected']),100)
            for row in json.loads((root/'arms.json').read_text()):
                self.assertEqual(row['server_host'],'local')
                contract = devset.arm_contract(root,row,manifest)
                self.assertEqual(contract['init_pool'],'B')
                cp = HERE/'scratch/contracts'/f"{row['arm']}.json"
                cp.parent.mkdir(parents=True,exist_ok=True)
                cp.write_text(json.dumps(dict(contract,schema='offline_search.dev_looks.v1',cell=row['cell'],
                    method=row['method'],recording_arm_dir=str(root/'runs'/row['arm']),pairs=sorted(manifest['selected'])),sort_keys=True)+'\n')
                argv = ['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),
                        '--os-cell',f"{row['model']}_{row['suite_short']}_cache",'--os-log-dir',str(root/'runs'/row['arm']),
                        # These two flags are appended by assets.build_plan at
                        # coordinator sync, after validating the contract above.
                        '--os-look-contract',str(cp),'--os-look-contract-sha256',devset.sha(cp)]
                opts,rest = parse_cli(argv+row['plugin_args'])
                self.assertFalse(rest)
                self.assertEqual(opts.os_lookhalf,row['kwargs']['method'])
                self.assertEqual(opts.os_request_cameras,opts.os_lookhalf != 'off')
                self.assertEqual(row['kwargs'].get('knob_method','off'),'off')
                self.assertTrue(opts.os_record_looks)
                total += 1
        self.assertEqual(total,17)
        EVIDENCE['pilot_roots'] = dict(roots=5,arms=total,episodes_per_arm=100,exact_K10_selection=True,
                                     B_disjoint=True,local=True,recording=True,launches=0)

    def test_half_transaction_poisoning_and_guard_once(self):
        m = load(HERE/'artifacts/pi05_spatial_50/opus_m1.pkl')
        lib = _Library(STORE, 'pi05_spatial', m.library, m.episode_subset)
        rows = next(r for r in episode_rows(m.inner.C).values() if len(r)>6)
        q2 = _View(PseudoQuery(lib, m.inner.C, rows, 2, 'pi05', True), hist_has_vision=np.array([True, False]))
        q4 = _View(PseudoQuery(lib, m.inner.C, rows, 4, 'pi05', True), hist_has_vision=np.array([True, False, True, False]))
        m.reset(q2.episode); m.set_camera_mode('full'); m.query(q2)
        original, _ = clone_method(m, strict=True)
        m.set_camera_mode('wrist_only')
        before = list(m.inner._vision_progress)
        # Rejection cannot change the guard, action anchor or accepted result.
        with patch.object(m, '_opus', return_value=NeedCamera(reason='consistency')):
            self.assertIsInstance(m.query(q4), NeedCamera)
        self.assertEqual(before, m.inner._vision_progress)
        np.testing.assert_array_equal(m.inner.base._anchor['action'], original.inner.base._anchor['action'])
        m.set_camera_mode('full')
        expected = original.query(q4)
        equal_result(self, expected, m.query(q4))
        self.assertEqual(original.inner._vision_progress, m.inner._vision_progress)
        # Pure one-camera proposal never reads the hidden current key.
        poison = _View(q4, key_v0=np.full(32768, np.nan, np.float32))
        for method in ('opus_m1', 'astra_consensus2', 'astra_strict'):
            a = load(HERE/'artifacts/pi05_spatial_50'/f'{method}.pkl')
            a.reset(q2.episode); a.set_camera_mode('full'); a.query(q2)
            b, _ = clone_method(a, strict=True)
            p1 = a.inner.base.B1T@q4.key_v1-a.inner.base.muB1
            function = '_opus' if method == 'opus_m1' else '_astra'
            left = getattr(a, function)(q4, p1, q4.rs[:8])
            right = getattr(b, function)(poison, p1, q4.rs[:8])
            if isinstance(left, NeedCamera):
                self.assertEqual(left, right)
            else:
                np.testing.assert_array_equal(left['rows'], right['rows'])
                np.testing.assert_array_equal(left['action'], right['action'])
        EVIDENCE['transaction'] = dict(hidden_camera_poisoning=True, completion_before_guard=True,
                                       rejected_proposal_no_mutation=True, full_requery_exact=True)

    def test_accepted_half_tail_and_connections(self):
        m = load(HERE/'artifacts/pi05_spatial_50/opus_m1.pkl')
        lib = _Library(STORE, 'pi05_spatial', m.library, m.episode_subset)
        rows = next(r for r in episode_rows(m.inner.C).values() if len(r)>8)
        q2 = _View(PseudoQuery(lib, m.inner.C, rows, 2, 'pi05', True), hist_has_vision=np.array([True, False]))
        q4 = _View(PseudoQuery(lib, m.inner.C, rows, 4, 'pi05', True), hist_has_vision=np.array([True, False, True, False]))
        m.reset(q2.episode); m.set_camera_mode('full'); m.query(q2)
        other, _ = clone_method(m, strict=True)
        self.assertIs(m.half_fit['P'],other.half_fit['P'])
        self.assertFalse(m.half_fit['P'].flags.writeable)
        self.assertFalse(m.half_fit['metrics']['opus_w'][0][0].Z.flags.writeable)
        old = list(other.inner._vision_progress)
        p1 = m.inner.base.B1T@q4.key_v1-m.inner.base.muB1
        prop = m._query_metric('opus_l2', int(q4.task_id), np.r_[m._memory['p0'], p1, q4.rs[:8]], q4)
        prop.update(action=prop['raw'].copy(), metric_name='opus_l2')
        prop['action'][:10, :6] += np.float32(.0001)
        m.set_camera_mode('wrist_only')
        with patch.object(m, '_opus', return_value=prop):
            r = m.query(_View(q4, key_v0=np.full(32768, np.nan, np.float32)))
        np.testing.assert_array_equal(r.action, prop['action'])
        np.testing.assert_array_equal(m.inner.base._anchor['action'], prop['action'])
        np.testing.assert_array_equal(r.action[:, 6:], prop['raw'][:, 6:])
        self.assertEqual(len(m.inner._vision_progress), len(old)+1)
        self.assertEqual(other.inner._vision_progress, old)
        bq = NS(step=5, task_id=q4.task_id, episode=q4.episode, rs=q4.rs, raw_state=np.zeros(8),
                prev_hit=True, executed_steps=5, blind_age=0, hist_rs=np.zeros((5,32), np.float32),
                hist_has_vision=np.array([True,False,True,False,True]), hist_hit=np.ones(5,np.int8),
                hist_a_exec=np.zeros((5,50,32),np.float32), prev_a_exec=r.action)
        # Test the authoritative base tail (guard veto is separately unchanged).
        tail = m.inner.base.blind_step(bq)
        np.testing.assert_array_equal(tail.action[:5, :7], r.action[5:10, :7])
        np.testing.assert_array_equal(tail.action[:, 7:], np.zeros_like(tail.action[:,7:]))
        EVIDENCE['half_tail_and_sessions'] = dict(corrected_tail=True, gripper_padding_unchanged=True,
                                                 guard_updates=1, independent_sessions=True)

    def test_library_boundary(self):
        lib = STORE/'library/pi05_spatial/bpool_cs'
        with library_boundary(STORE, lib):
            with self.assertRaises(ValueError):
                (STORE/'queries/pi05_spatial_cache/meta.json').open('rb')
        with self.assertRaises(ValueError):
            fit_path('/home/weiland/trace_runs/os_closed_loop/r12_devlooks_50L/arms.json', STORE)


class GateLedgerTests(unittest.TestCase):
    def test_plugin_completes_then_commits_one_guard(self):
        from exp.offline_search.closed_loop import plugin
        from exp.offline_search.closed_loop.test_look_recording import connection,FakeMethod
        opts,_ = plugin.parse_cli(['--os-method','fixture:Recipe','--os-cell','pi05_l10_cache',
                                  '--os-log-dir',str(HERE/'scratch/integration')])
        conn,s,obs,records = connection(plugin,opts)
        class Method(FakeMethod):
            next_camera_mode = 'wrist_only'
            mode = 'wrist_only'
            lookhalf_diagnostics = {}
            def plan_cameras(self,q,**metadata):return 'wrist_only'
            def set_camera_mode(self,mode):self.mode=mode
            def invalidate_anchor(self):self._anchor=None
            def query(self,q):
                if self.mode == 'wrist_only':
                    return NeedCamera(reason='consistency')
                self.assert_completed = bool(q.key_v0[0] == 123.)
                return super().query(q)
        s.method = Method();s.method.reset(None)
        s._camera_mode='wrist_only';s._camera_work={}
        s.rt.lookhalf='opus_m1';s.rt.request_cameras=True;s.rt.ctrl=None
        s.rt.judge=plugin.JudgeSpec.parse('guard_only')
        def complete(session,ctx,*,reason):
            self.assertEqual(session.method._noprog_span,0)
            session.b_v0.a[session.step] = 123.
            ctx.query_keys['vision_0'] = session.b_v0.a[session.step]
            session._camera_work.update(completions=1,availability=[True,True],completion_reason=reason)
        with patch('exp.offline_search.closed_loop.lookhalf_cameras.complete_session',side_effect=complete) as complete_mock:
            conn.infer(obs)
        self.assertEqual(complete_mock.call_count,1)
        self.assertTrue(s.method.assert_completed)
        self.assertEqual(s.method._noprog_span,1)
        dec = next(r for r in records if r['ev']=='dec')
        self.assertEqual(dec['lookhalf_cost']['camera_encoded_count'],2)
        self.assertEqual(dec['owner_cost'],.152)
        EVIDENCE['plugin_transaction']=dict(completion_before_guard=True,guard_commits=1,completions=1,full_keys_refreshed=True)

    def test_compute_share_full_call_and_completion_once(self):
        for model, price in [('pi05', .076), ('groot', .074)]:
            self.assertEqual(ledger(model,vision=True,wrist_first=True,completions=0,miss=False)['compute_share_work'],price)
            self.assertAlmostEqual(ledger(model,vision=True,wrist_first=True,completions=1,miss=True)['compute_share_work'],1.)
            self.assertAlmostEqual(ledger(model,vision=True,wrist_first=False,completions=0,miss=True)['compute_share_work'],1.)
            with self.assertRaises(ValueError):
                ledger(model,vision=True,wrist_first=True,completions=2,miss=True)
            dd = [dict(uid='B',step=0,vision=True,hit=False,extras={}, lookhalf_cost=ledger(model,vision=True,wrist_first=False,completions=0,miss=True)),
                  dict(uid='B',step=1,vision=False,hit=True,extras={}, lookhalf_cost=ledger(model,vision=False,wrist_first=False,completions=0,miss=False)),
                  dict(uid='B',step=2,vision=True,hit=True,extras={}, lookhalf_cost=ledger(model,vision=True,wrist_first=True,completions=0,miss=False))]
            self.assertAlmostEqual(summarize(model,dd)['ir_per_five_controls'],(1+price)/3)

    def test_atlas_mass_and_age(self):
        table=dict(next2=np.array([2,-1,2]),safe=np.ones(3,bool),rs=np.zeros((3,8)),state_scale=np.ones(8))
        mem=dict(step=2,rows=np.array([0,1]),weights=np.array([.98,.02],np.float32),rs=np.zeros(8))
        self.assertFalse(atlas_plan(step=4,memory=mem,rs=np.zeros(8),table=table,lifecycle=True,due=False,span=0)['half'])
        mem['weights']=np.array([1.,0.],np.float32)
        self.assertTrue(atlas_plan(step=4,memory=mem,rs=np.zeros(8),table=table,lifecycle=True,due=False,span=0)['half'])
        for step,due,span in [(5,False,0),(4,True,0),(4,False,1)]:
            self.assertFalse(atlas_plan(step=step,memory=mem,rs=np.zeros(8),table=table,lifecycle=True,due=due,span=span)['half'])

    def test_periodic_peek_and_policy_memory(self):
        m=R12LookHalf(method='opus_m1',knob_method='periodic',target=.32)
        m.knob_settings={'setting':1.0657}
        q=NS(step=4,hist_has_vision=np.array([True,False,True,False]),hist_hit=np.array([0,1,1,1]),
             task_id=0,episode=NS(init=9))
        before=pickle.dumps(m)
        self.assertEqual(m.periodic_due(q),m.periodic_due(q))
        self.assertEqual(before,pickle.dumps(m))


class EncoderTests(unittest.TestCase):
    def test_groot_exact_pixel_matched_noise_recording(self):
        import torch
        from exp.offline_search.closed_loop.lookhalf_numerics import record
        from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
        pixels = torch.tensor([[[[2.]]],[[[4.]]]],dtype=torch.bfloat16)
        ids = torch.cat([torch.ones(1,2,dtype=torch.long),torch.full((1,256),7),
                         torch.ones(1,1,dtype=torch.long),torch.full((1,256),7)],1)
        model = torch.nn.Linear(1,1)
        def prepare(obs):
            torch.rand(1)  # Even incidental transforms must not change serving RNG.
            return dict(eagle_input_ids=ids,eagle_attention_mask=torch.ones_like(ids),eagle_pixel_values=pixels), {}
        model.prepare_input = prepare
        calls,noises = [],[]
        def encode(x):
            calls.append(len(x))
            return (x.flatten()+len(x))[:,None,None].expand(-1,256,2048).clone()
        eagle = NS(image_token_index=7,extract_feature=encode,
                   language_model=NS(get_input_embeddings=lambda:lambda tokens:torch.zeros((*tokens.shape,2048),dtype=torch.bfloat16)))
        def stage3(s2,noise):
            noises.append(noise.clone())
            chunk = noise+s2.backbone_features.mean().float()
            noise.add_(99)  # Production code may consume/mutate the passed tensor.
            return NS(action_pred=chunk)
        runner = NS(_model=model,_eagle=eagle,session=torch.inference_mode,_require_session=lambda stage:None,
                    run_stage2_llm=lambda s1:NS(backbone_features=s1.input_embeds),run_stage3=stage3,
                    sample_noise=lambda s2,generator:torch.randn(1,10,32,generator=generator))
        session = NS(_camera_observation={},ep=NS(seed=19),_dec={'step':4},kb=GrootLiberoCP1SpatialPool16KeyBuilder())
        before = torch.random.get_rng_state().clone()
        arrays,info = record(session,NS(policy=NS(_runner=runner)))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(calls,[2,1,1])
        self.assertTrue(torch.equal(*noises))
        np.testing.assert_array_equal(arrays['groot_encoder_pixels'],pixels.float().numpy())
        np.testing.assert_array_equal(arrays['groot_policy_noise'],noises[0].numpy())
        self.assertGreater(info['policy_delta']['max_abs'],0)
        self.assertTrue(info['exact_input'] and info['diagnostic_only'])
        report = numerical_report([dict(meta={'uid':'B:toy'},step=4,groot_numerical_path=json.dumps(info))])
        self.assertEqual(report['samples'],1)
        EVIDENCE['numerical_recording'] = dict(exact_pixels=True,matched_noise=True,production_RNG_unchanged=True,
                                              joint_single_deltas_detected=True,cpu_fake=True)

    def test_pi05_genuine_wrist_and_idempotent_live_completion(self):
        import torch
        from exp.offline_search.rounds.r07.c2_wrist.test_dispatch import Model, raw, stock
        from exp.offline_search.closed_loop.stage_overrides import Pi05Override
        from openpi.serving import stage_io
        from openpi.models.model import Observation
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        from exp.offline_search.closed_loop.plugin import _Buf
        model=Model(); obs=Observation.from_dict(stage_io.stack_observation([raw(2.)],device=torch.device('cpu')))
        override=Pi05Override('per_request'); model._r4_override=override
        full=stock(model,obs); override.seed_full_dummy(model,obs,full)
        model.paligemma_with_expert.calls.clear()
        half=override.stage1(model,obs,mode='wrist_only'); audit={}; half.camera_audits=[audit]
        self.assertEqual(len(model.paligemma_with_expert.calls),1)
        self.assertEqual(model.paligemma_with_expert.calls[0][1],(4.,))
        s=NS(rt=NS(model='pi05'),_camera_adapter=NS(policy=NS(_model=model)),_camera_stage1=half,
             _camera_work=audit,kb=CP1SpatialPool16KeyBuilder(),step=0,_dec=None,
             b_v0=_Buf((32768,),np.float32),b_v1=_Buf((32768,),np.float32))
        for buf in (s.b_v0,s.b_v1):buf.append(np.zeros(32768,np.float32))
        ctx=NS(query_keys={})
        complete_session(s,ctx,reason='consistency'); complete_session(s,ctx,reason='call')
        self.assertEqual(len(model.paligemma_with_expert.calls),2)
        self.assertEqual(audit['completions'],1)
        torch.testing.assert_close(half.prefix_embs,full.prefix_embs,rtol=0,atol=0)
        self.assertTrue(np.isfinite(s.b_v0.a[0]).all())
        self.assertTrue(torch.equal(half.to('cpu').prefix_embs,full.prefix_embs))

    def test_groot_genuine_wrist_and_separate_prefix(self):
        import torch
        from openpi.cache.groot.staged import GrootStage1Output
        class Eagle:
            image_token_index=7
            language_model=NS(get_input_embeddings=lambda: lambda ids: torch.zeros((*ids.shape,2048)))
            def __init__(self):self.calls=[]
            def extract_feature(self,pixels):
                self.calls.append(tuple(pixels.flatten().tolist()))
                return (pixels.flatten()+len(pixels))[:,None,None].expand(-1,256,2048).clone()
        eagle=Eagle()
        ids=torch.cat([torch.tensor([[1,2]]),torch.full((1,256),7),torch.tensor([[3]]),torch.full((1,256),7)],1)
        inputs=dict(eagle_input_ids=ids,eagle_attention_mask=torch.ones_like(ids),eagle_pixel_values=torch.tensor([[[[2.]]],[[[4.]]]]))
        runner=NS(_eagle=eagle,_require_session=lambda stage:None,
                  _model=NS(prepare_input=lambda observation:(inputs,{'state':torch.zeros(1,1,32),'state_mask':torch.ones(1,1,32,dtype=torch.bool)})))
        override=GrootOverride(); audit={}
        half=override.partial(runner,{},audit)
        self.assertEqual(eagle.calls,[(4.,)])
        saved=half.input_embeds.clone()
        override.complete(runner,half); override.complete(runner,half)
        self.assertEqual(eagle.calls,[(4.,),(2.,)])
        self.assertEqual(audit['completions'],1)
        self.assertEqual(audit['availability'],[True,True])
        self.assertTrue(torch.equal(saved[:,259:],half.input_embeds[:,259:]))
        # B=2 deliberately differs from separate B=1: completion must retain
        # the actual separate prefix, never silently run the joint path.
        joint=eagle.extract_feature(inputs['eagle_pixel_values'])
        self.assertFalse(torch.equal(joint[1],half.input_embeds[0,259:]))
        EVIDENCE['encoders']=dict(pi05_wrist_tower_calls=1,pi05_completion_calls=1,
            groot_wrist_batch=1,groot_completion_batch=1,groot_joint_reencode_on_fallback=False)


def main():
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__,fromlist=['*']))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),cpu_only=True)
    (HERE/'SELFTEST.json').write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
