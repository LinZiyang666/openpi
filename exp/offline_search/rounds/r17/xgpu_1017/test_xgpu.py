"""CPU contract and failure-mode tests. No model load, CUDA init, network or lanes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from artifacts import digest, load_tree, save_tree, sha, tree_identity, write_json
from capture import Recorder, from_archive, from_steps, read_inputs
from compare import compare, difference, budgets
from replay import draws, verify_runtime, select_sampler

ROOT = Path(__file__).parent


def workspace():
    base = ROOT / 'test_artifacts'
    base.mkdir(exist_ok=True)
    # Retained for review; no cleanup/deletion command runs.
    return Path(tempfile.mkdtemp(prefix='cpu_', dir=base))


def request():
    return dict(campaign='c', benchmark_protocol='b', task_name='task', task_id=0,
                original_init_idx=0, decision_seq=0, seed=7, explicit_seed=99)


def observation():
    return {'observation/image': np.zeros((4, 4, 3), np.uint8),
            'observation/state': np.zeros(8, np.float32), 'prompt': 'move'}


def replay_fixture(root, *, delta=0., noise_delta=0., repeats=3, cache=True):
    root.mkdir()
    meta = dict(status='COMPLETE', input_identity='inputs', runtime_identity='runtime',
        binding_identity='spec', sampler={'num_steps': 2}, cache_identity={'fit':'fit'} if cache else None,
        model='dp', model_spec={'layout': {'replan_steps': 1}}, commit_horizon=1, rows=[])
    for i in range(2):
        for r in range(repeats):
            stages = dict(stage1={'features':np.ones(2, np.float32)},
                stage2={'conditioning':np.ones(2, np.float32)},
                noise={'0':np.ones(2, np.float32) + np.float32(noise_delta)},
                denoise={'001':np.ones((2,2),np.float32)},
                final_normalized=np.ones((2,2),np.float32)+np.float32(delta),
                final_wire=np.ones((2,2),np.float32)+np.float32(delta))
            c = dict(hit=True, top1=1, corrector=False, stall=False,
                corrected_action=np.ones((2,2),np.float32),score_margin=None,neighbour_margin=.1) if cache else None
            name=f'row_{i}_{r}'
            seal=save_tree(root/name,dict(stages=stages,cache=c,request=request(),rng_receipt={}))
            meta['rows'].append(dict(input=i,repeat=r,file=name,sha256=seal))
    write_json(root/'REPLAY.json',meta)


def change_row(root, index, mutate):
    p=root/'REPLAY.json';m=json.loads(p.read_text());e=m['rows'][index]
    value=load_tree(root/e['file']);mutate(value)
    e['sha256']=save_tree(root/e['file'],value);write_json(p,m)


class Contracts(unittest.TestCase):
    def test_cache_sampler_overrides_default_and_checks_identity(self):
        from types import SimpleNamespace as NS
        p=NS(sampler_for_steps=lambda k:NS(num_steps=k,identity=str(k)))
        self.assertEqual(select_sampler(p,5,{'execution_binding':{'num_steps':2,'sampler_identity':'2'}}).num_steps,2)
        with self.assertRaisesRegex(ValueError,'sampler'):
            select_sampler(p,5,{'execution_binding':{'num_steps':2,'sampler_identity':'5'}})

    def test_tree_identity_survives_json_key_sorting(self):
        p=workspace()/'tree';x={'z':np.ones(2),'a':np.zeros(3)}
        save_tree(p,x);self.assertEqual(tree_identity(x),tree_identity(load_tree(p)))

    def test_portable_roundtrip(self):
        p=workspace()/'tree'; x={'nested':[np.arange(4,dtype=np.float32),{'mask':np.array([True])}],'text':'碗'}
        save_tree(p,x); y=load_tree(p)
        np.testing.assert_array_equal(y['nested'][0],x['nested'][0]);self.assertEqual(y['text'],'碗')

    def test_corruption_rejected(self):
        p=workspace()/'tree';save_tree(p,{'x':np.ones(2)})
        with p.with_suffix('.npz').open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'digest'):load_tree(p)

    def test_object_array_rejected(self):
        with self.assertRaisesRegex(ValueError,'object'):save_tree(workspace()/'tree',np.array([{}],object))

    def test_real_record_and_seed_roundtrip(self):
        p=workspace()/'inputs';r=Recorder(p,provenance={'real':True});r.add(observation(),request());r.close()
        _,rows=read_inputs(p);self.assertEqual(rows[0]['request']['explicit_seed'],99)

    def test_missing_seed_rejected(self):
        r=Recorder(workspace()/'inputs',provenance={});q=request();q.pop('explicit_seed')
        with self.assertRaisesRegex(ValueError,'seed'):r.add(observation(),q)

    def test_noninitial_history_required(self):
        r=Recorder(workspace()/'inputs',provenance={});q=request();q['decision_seq']=1
        with self.assertRaisesRegex(ValueError,'history'):r.add(observation(),q)

    def test_log_without_observation_rejected(self):
        p=workspace();(p/'steps.jsonl').write_text(json.dumps({'seed':7,'score':.5})+'\n')
        with self.assertRaisesRegex(ValueError,'no complete'):from_steps(p/'steps.jsonl',p/'inputs')

    def test_log_links_actual_observation(self):
        p=workspace();save_tree(p/'obs',observation())
        (p/'steps.jsonl').write_text(json.dumps({'observation_file':'obs','request':request()})+'\n')
        from_steps(p/'steps.jsonl',p/'inputs');self.assertEqual(len(read_inputs(p/'inputs')[1]),1)

    def test_synthetic_archive_rejected(self):
        p=workspace();write_json(p/'OBSERVATIONS.json',{'synthetic':True})
        with self.assertRaisesRegex(ValueError,'synthetic'):from_archive(p,p/'inputs','b','t',0)

    def test_nonfinite_and_shape_rejected(self):
        with self.assertRaises(ValueError):difference(np.ones(2),np.ones(3))
        with self.assertRaises(ValueError):difference(np.array([np.nan]),np.array([np.nan]))
        with self.assertRaises(ValueError):difference(np.array([np.inf]),np.array([np.inf]))

    def test_zero_relative_floor_finite(self):
        d=difference(np.zeros(2,np.float32),np.full(2,1e-8,np.float32))
        self.assertTrue(np.isfinite(d['relative_max']))

    def pair(self, **kwargs):
        p=workspace();replay_fixture(p/'left');replay_fixture(p/'right',**kwargs);return p

    def test_exact_screen_not_deployment_certificate(self):
        p=self.pair();r=compare(p/'left',p/'right')
        self.assertEqual(r['status'],'SCREEN_PASS');self.assertFalse(r['deployment_equivalence_established'])
        self.assertFalse(r['layer_b']['statistical_certificate'])

    def test_cross_error_over_cap_fails(self):
        p=self.pair(delta=.02);r=compare(p/'left',p/'right');self.assertFalse(r['layer_a_within_budget'])

    def test_noise_mismatch_fails_even_if_small(self):
        p=self.pair(noise_delta=1e-7);r=compare(p/'left',p/'right');self.assertFalse(r['noise_identical'])

    def test_missing_stage_fails(self):
        p=self.pair();change_row(p/'right',0,lambda r:r['stages'].pop('stage1'))
        with self.assertRaises(ValueError):compare(p/'left',p/'right')

    def test_missing_denoise_steps_fails(self):
        p=self.pair()
        for side in ('left','right'):
            change_row(p/side,0,lambda r:r['stages'].update(denoise={}))
        with self.assertRaisesRegex(ValueError,'denoising'):compare(p/'left',p/'right')

    def test_later_repeat_unknown_cache_fails(self):
        p=self.pair();change_row(p/'right',2,lambda r:r['cache'].update(stall=None))
        self.assertFalse(compare(p/'left',p/'right')['layer_b']['eligible'])

    def test_execution_ack_roundtrip(self):
        p=workspace()/'inputs';r=Recorder(p,provenance={});q=request();q['decision_seq']=1
        r.add(observation(),q,history={'executed_count':5},episode_start=False,executed_controls_since_last_query=5)
        r.close();row=read_inputs(p)[1][0]
        self.assertEqual(row['executed_controls_since_last_query'],5);self.assertFalse(row['episode_start'])

    def test_noninitial_missing_ack_rejected(self):
        r=Recorder(workspace()/'inputs',provenance={});q=request();q['decision_seq']=1
        with self.assertRaisesRegex(ValueError,'acknowledgment'):
            r.add(observation(),q,history={})

    def test_identity_mismatch_fails(self):
        p=self.pair();m=json.loads((p/'right/REPLAY.json').read_text());m['cache_identity']={'fit':'other'}
        write_json(p/'right/REPLAY.json',m)
        with self.assertRaisesRegex(ValueError,'identity'):compare(p/'left',p/'right')

    def test_repeat_coverage_fails(self):
        p=self.pair(repeats=1)
        with self.assertRaisesRegex(ValueError,'coverage'):compare(p/'left',p/'right')

    def test_cache_mismatch_and_unknown_margin(self):
        p=self.pair();change_row(p/'right',0,lambda r:r['cache'].update(hit=False))
        result=compare(p/'left',p/'right');self.assertEqual(result['layer_b']['status'],'MISMATCH')
        self.assertEqual(result['layer_b']['mismatches'][0]['near_threshold'],'unknown')

    def test_cache_unknown_is_incomplete(self):
        p=self.pair();change_row(p/'right',0,lambda r:r['cache'].update(stall=None))
        self.assertFalse(compare(p/'left',p/'right')['layer_b']['eligible'])

    def test_tiny_gripper_repeat_sign_flip_rejected(self):
        p=self.pair()
        for side in ('left','right'):
            path=p/side/'REPLAY.json';m=json.loads(path.read_text())
            m['model_spec']['layout']['gripper_channel']=1;write_json(path,m)
            for index in range(6):
                value=-1e-7 if index==1 else 1e-7
                change_row(p/side,index,lambda row,v=value:row['stages']['final_wire'].__setitem__((slice(None),1),v))
        self.assertFalse(compare(p/'left',p/'right')['committed_gripper_sign_agreement'])

    def test_same_gpu_floor_measured(self):
        p=self.pair();change_row(p/'left',1,lambda r:r['stages']['final_wire'].__iadd__(np.float32(1e-5)))
        result=compare(p/'left',p/'right')
        self.assertTrue(any(x['same_gpu_floor']['max_abs']>0 for x in result['layer_a']))

    def test_fw_budget_not_applied_to_dp(self):
        self.assertLess(budgets('dp','/final_normalized')[0],budgets('fastwam','/final_normalized')[0])

    def test_runtime_seal_rejects_changed_source(self):
        import hashlib
        p=workspace();(p/'code.py').write_text('x=1\n')
        m={'schema':'r17.runtime_release.v1','files':{'code.py':sha(p/'code.py')}}
        write_json(p/'RUNTIME_RELEASE.json',m)
        d=hashlib.sha256((json.dumps(m,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
        verify_runtime(p,d);(p/'code.py').write_text('x=2\n')
        with self.assertRaisesRegex(ValueError,'source'):verify_runtime(p,d)

    def test_observed_torch_draws_preserve_rng_and_restore(self):
        import torch
        original=torch.randn
        g=torch.Generator(device='cpu').manual_seed(9)
        expected=torch.randn(3,generator=g)
        with draws(torch) as recorded:
            actual=torch.randn(3,generator=torch.Generator(device='cpu').manual_seed(9))
        self.assertIs(torch.randn,original);self.assertTrue(torch.equal(actual,expected))
        np.testing.assert_array_equal(recorded[0],expected.numpy())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
