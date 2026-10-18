import json
from pathlib import Path
from contextlib import contextmanager
import uuid
import unittest
import numpy as np
from concurrent_clients import ack, decision_seed, drive
from concurrent_compare import errors, compare
from concurrent_server import campaign_for, replica_command, checked_pin, validate_attestation, live_reference, relocate_paths
from paired_sr import paired_interval
from artifacts import write_json


@contextmanager
def retained_test_directory():
    path=Path(__file__).parent/'test_artifacts'/('concurrent_'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path  # retain CPU fixtures; do not delete local files


class FakeClient:
    def __init__(self):
        self.seq=0
    def select_bundle(self,b):return {'__ack__':'select_bundle'}
    def episode_start(self,**kw):return {'__ack__':'episode_start'}
    def infer(self,obs):
        assert 'ack_protocol' not in obs['__request__']
        assert 'partial_tail_protocol' not in obs['__request__']
        if self.seq:
            assert obs['__execution__']['decision_seq']==self.seq-1
            assert obs['__execution__']['executed_count']==5
        else:assert '__execution__' not in obs
        assert obs['__request__']['decision_seq']==self.seq
        self.seq+=1
        return {'actions':np.zeros((10,7),dtype=np.float32),
            '__hit_meta__':{'source':'policy','decision_seed':obs['__request__']['explicit_seed']},
            'server_timing':{'infer_ms':1}}
    def episode_end(self,success,execution):
        assert execution['terminal'] and execution['decision_seq']==self.seq-1
        return {'__ack__':'episode_end'}
    def close(self):pass


class Tests(unittest.TestCase):
    def test_pins_all_models(self):
        for model in ('pi05','groot','dp','fastwam','cosmos'):
            p=checked_pin(Path(__file__).parent/'concurrent/bundles'/model)
            self.assertEqual(p['model'],model)
            if model in ('dp','groot','fastwam'):
                revised=checked_pin(Path(__file__).parent/'concurrent/bundles'/(model+'_longseg_recal1018'))
                self.assertNotEqual(p['identity'],revised['identity'])
                self.assertEqual(p['runtime_identity'],revised['runtime_identity'])
                self.assertEqual(p['bootstrap_sha256'],revised['bootstrap_sha256'])
    def test_actual_formal_bootstrap_and_graph(self):
        b=Path(__file__).parent/'concurrent/bundles/dp'
        c=campaign_for(b,'/tmp/unused','/runtime',29600,'/python')
        cmd=replica_command(c,'/campaign',0,'PINNED_BOOTSTRAP','/ready')
        self.assertIn('--cuda-graph',cmd)
        self.assertEqual(cmd[2],'exp.r17_integration.models.fastwam.ops.mps')
        self.assertEqual(cmd[cmd.index('-c')+1],'PINNED_BOOTSTRAP')
        self.assertEqual(c['replicas'],4)
        self.assertEqual(c['dp_batching']['max_batch'],8)
    def test_graph_failure_rejected(self):
        with self.assertRaises(ValueError):validate_attestation({'model':'dp'},{},{'r17_model_spec':{}})
    def test_graph_off_is_explicit_ablation(self):
        b=Path(__file__).parent/'concurrent/bundles/dp'
        c=campaign_for(b,'/tmp/unused','/runtime',29600,'/python')
        c['xgpu_numerics']=['graphs_off']
        self.assertNotIn('--cuda-graph',replica_command(c,'/campaign',0,'BOOT','/ready'))
        validate_attestation(c,{},{'r17_model_spec':{}})
    def test_direct_attention_not_falsely_unified(self):
        from numerics_bootstrap import run
        with self.assertRaises(ValueError):run(None,['sdpa_math'],'cosmos')
        with self.assertRaises(ValueError):run(None,['sdpa_math'],'groot')
        from numerics_bootstrap import process_environment
        self.assertEqual(process_environment(['te_unfused'],'cosmos'),
            {'NVTE_FLASH_ATTN':'0','NVTE_FUSED_ATTN':'0'})
        with self.assertRaises(ValueError):process_environment(['te_unfused'],'groot')
    def test_cosmos_local_selector_cannot_reenable_flash_in_math_arm(self):
        from attention_control import cosmos_math
        from types import SimpleNamespace
        import torch
        module=SimpleNamespace(FLASH_ATTN_3_AVAILABLE=True)
        with retained_test_directory() as root:
            cosmos_math(root,module)
            with module.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION],set_priority_order=True):
                self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                self.assertFalse(torch.backends.cuda.mem_efficient_sdp_enabled())
            self.assertFalse(module.FLASH_ATTN_3_AVAILABLE)
            self.assertTrue((root/'ATTENTION_OVERRIDE_VISITED.json').exists())
    def test_actual_deployed_reference_is_pinned(self):
        w=Path(__file__).parent
        pin=checked_pin(w/'concurrent/bundles/pi05')
        declared=json.loads((w/'concurrent/bundles/pi05/assets/model_spec.json').read_text())
        live,receipt=live_reference(w/'concurrent/references/pi05.json',pin,declared)
        self.assertTrue(receipt['differs_from_declared'])
        self.assertEqual(live['checkpoint_identity'],declared['checkpoint_identity'])
        with self.assertRaises(ValueError):live_reference(w/'concurrent/references/pi05.json',{**pin,'runtime_identity':'wrong'},declared)
    def test_path_relocation_is_prefix_bounded(self):
        self.assertEqual(relocate_paths(['/data/a/x','/data/ab/x','sha256:123'],{'/data/a':'/owned/a'}),
                         ['/owned/a/x','/data/ab/x','sha256:123'])
    def test_ack_partial_tail(self):
        self.assertEqual(ack(3,np.zeros((2,7)),5)['executed_count'],2)
        with self.assertRaises(ValueError):ack(0,np.full((2,7),np.nan),5)
    def test_fixed_cosmos_noise(self):
        self.assertEqual(decision_seed({'seed_mode':'fixed','seed':195},'task',0,25,999),195)
    def test_nested_stall_and_top1_observability(self):
        from concurrent_compare import metadata
        m=metadata({'response':{'__hit_meta__':{'donor_rows':[6,9], 'knob':False,
            'stall_status':{'state':'inactive','delta_hat':None}}}})
        self.assertEqual(m['top1_neighbour'],6);self.assertEqual(m['stall_state'],'inactive')
        self.assertFalse(m['knob_call']);self.assertIsNone(m['stall_delta_hat'])
        m=metadata({'response':{'__hit_meta__':{'stall_status':{'delta_hat':.2,'e90':.1,'a10':.4}}}})
        self.assertAlmostEqual(m['stall_confirmed_margin'],-.1)
        self.assertAlmostEqual(m['stall_ambiguous_margin'],-.2)
    def test_no_nan_or_shape_mismatch(self):
        with self.assertRaises(ValueError):errors([1],[1,2])
        with self.assertRaises(ValueError):errors([float('nan')],[1])
    def test_observer_preserves_native_call_and_records_members(self):
        from types import SimpleNamespace
        from batch_telemetry import install
        calls=[]
        class Adapter:
            def _input(self,value):return SimpleNamespace(value=value)
        class Batcher:
            @staticmethod
            def _run(items,call):
                calls.append('native')
                return [SimpleNamespace(value=v,size=len(items)) for v in call(None,[x.value for x in items])]
        class Core:
            def submit_to_stage(self,stage,_,items):
                return Batcher._run(items,lambda _,xs:xs)[0]
        class Runtime:
            def wrap_connection_factory(self,factory):return factory
        module=SimpleNamespace(BatchedAdapter=Adapter,_StageBatcher=Batcher,BatchRuntime=Runtime)
        install(module,Core)
        a=Adapter()
        def factory():
            def infer(obs):
                value=Core().submit_to_stage(1,'native',[a._input(9),a._input(10)])
                return {'actions':value.value,'__hit_meta__':{}}
            return SimpleNamespace(infer=infer)
        conn=Runtime().wrap_connection_factory(factory)()
        result=conn.infer({'__request__':{'explicit_seed':27}})
        self.assertEqual(result['actions'],9)
        self.assertEqual(calls,['native'])
        receipt=result['__hit_meta__']['xgpu_batch_composition'][0]
        self.assertEqual(receipt['size'],2)
        self.assertEqual(receipt['members'],[{'explicit_seed':27}]*2)
    def test_cache_observer_preserves_native_action_and_decision(self):
        from types import SimpleNamespace
        from cache_telemetry import install
        calls=[]
        original=SimpleNamespace(action=np.ones((5,7)),rows=np.array([2,9]),
            distances=np.array([.5,.501]),weights=np.array([.6,.4]),
            force_miss=True,extras={'raw_anchor_distance':.4})
        class Recipe:
            distance_scale=.2
            def query(self,q):calls.append(q);return original
        install(Recipe)
        result=Recipe().query('query')
        self.assertIs(result,original);self.assertEqual(calls,['query'])
        self.assertTrue(result.force_miss)
        self.assertEqual(result.extras['corrector_ratio'],2.)
        self.assertAlmostEqual(result.extras['neighbour_gap'],.001)
        original.distances=np.array([.5])
        self.assertIsNone(Recipe().query('k1').extras['neighbour_gap'])
    def test_cosmos_observer_reads_flags_after_native_stage_and_only_once(self):
        from batch_telemetry import observe_after_stage1
        from types import SimpleNamespace
        calls=[]; original=object()
        adapter=SimpleNamespace(flag='loader')
        def native(query):
            calls.append(query);adapter.flag='inference';return original
        adapter.run_stage1=native
        def collect():
            calls.append('collect');return {'flag':adapter.flag}
        with retained_test_directory() as root:
            observe_after_stage1(adapter,root,collect)
            self.assertIs(adapter.run_stage1('first'),original)
            self.assertIs(adapter.run_stage1('second'),original)
            self.assertEqual(calls,['first','collect','second'])
            self.assertEqual(json.loads((root/'NUMERICS_AFTER_STAGE1.json').read_text())['flag'],'inference')
    def test_zero_discordance_interval_nonzero(self):
        r=paired_interval([1]*100,[1]*100)
        self.assertLess(r['CI95'][0],0);self.assertGreater(r['CI95'][1],0)
        self.assertFalse(r['equivalent']);self.assertEqual(r['mcnemar_p'],1)
    def test_harness_snapshot_survives_later_source_update(self):
        from harness_snapshot import seal
        with retained_test_directory() as root:
            source=root/'source';source.mkdir();(source/'client.py').write_text('old')
            first=seal(source,root/'snapshots',('client.py',))
            self.assertEqual(first,seal(source,root/'snapshots',('client.py',)))
            (source/'client.py').write_text('new')
            second=seal(source,root/'snapshots',('client.py',))
            self.assertNotEqual(first,second)
            self.assertEqual((first/'client.py').read_text(),'old')
    def test_harness_snapshot_refuses_changed_sealed_code(self):
        from harness_snapshot import seal
        with retained_test_directory() as root:
            source=root/'source';source.mkdir();(source/'client.py').write_text('code')
            target=seal(source,root/'snapshots',('client.py',))
            path=target/'client.py';path.chmod(0o644);path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'snapshot was changed'):
                seal(source,root/'snapshots',('client.py',))
    def test_historical_size_effect(self):
        r=paired_interval([1]*50+[0]*50,[1]*70+[0]*30)
        self.assertEqual(r['difference'],-.2)
        self.assertLess(r['mcnemar_p'],.001)
    def test_paired_groups_and_censored_divergence(self):
        from paired_analysis import sr,divergence
        base=dict(model='dp',suite='spatial',arm='cache',task_id=0,init_idx=0,env_seed=7,policy_seed=7)
        outcomes=[dict(base,accepted=True,success=True)]
        self.assertEqual(sr(outcomes,outcomes,100)[0]['task_clusters'],1)
        left=[dict(base,decision_step=s,env_step=5*s,actions=[0],state=[0],cache_decision='hit') for s in range(3)]
        right=[dict(r) for r in left];right[1]['actions']=[.1]
        row=divergence(left,right,.001,1e-6)['episodes'][0]
        self.assertEqual(row['first_action'],1);self.assertTrue(row['state_right_censored'])
        self.assertFalse(row['images_right_censored'])
        left[0]['camera_sha256']={'front':'same'};right[0]['camera_sha256']={'front':'different'}
        self.assertEqual(divergence(left,right,.001,1e-6)['episodes'][0]['first_images'],0)
        right[2].pop('state')
        row=divergence(left,right,.001,1e-6)['episodes'][0]
        self.assertFalse(row['state_right_censored']);self.assertEqual(row['unavailable']['state'],[2])
        row=divergence(left,right[:1]+right[2:],.001,1e-6)['episodes'][0]
        self.assertFalse(row['cache_right_censored']);self.assertEqual(row['unmatched_steps'],[1])
        with self.assertRaises(ValueError):sr(outcomes,[],100)
        # Real C10 tails share a segment decision index. Execution-clock
        # alignment must preserve their actions rather than dropping tails.
        tails=[dict(base,decision_step=i//2,env_step=i*5,actions=[0],state=[0],cache_decision='hit') for i in range(4)]
        other=[dict(r) for r in tails];other[1]['actions']=[.1]
        with self.assertRaises(ValueError):divergence(tails,other,.001,1e-6)
        row=divergence(tails,other,.001,1e-6,clock='env_step')['episodes'][0]
        self.assertEqual(row['first_action'],5)
    def test_client_observer_keeps_native_result_and_records_ack_clock(self):
        from client_trace import install
        from capture import read_inputs
        class Native:
            def episode_start(self,**kwargs):return {'__ack__':'episode_start'}
            def episode_end(self,**kwargs):
                self.end_result={'__ack__':'episode_end'};return self.end_result
            def infer(self,obs):
                self.native_result={'actions':np.ones((5,4),np.float32),
                    '__hit_meta__':{'source':'policy'},'server_timing':{}}
                return self.native_result
        req=dict(campaign='test',benchmark_protocol='proto',task_name='task',task_id=0,
            original_init_idx=0,decision_seq=0,seed=7,explicit_seed=8)
        with retained_test_directory() as root:
            install(root,Native,record_limit=2);client=Native();client.episode_start(task='task')
            obs={'observation/image':np.zeros((2,2,3),np.uint8),'observation/state':np.zeros(4),'__request__':req}
            response=client.infer(obs);self.assertIs(response,client.native_result)
            obs=dict(obs,__request__=dict(req,decision_seq=1),__execution__=dict(executed_count=5,decision_seq=0,terminal=False))
            client.infer(obs)
            rows=[json.loads(line) for line in next(p for p in root.glob('*.jsonl')if not p.name.endswith('.executed.jsonl')).read_text().splitlines()]
            self.assertEqual([r['env_step'] for r in rows],[0,5])
            manifest,recorded=read_inputs(next(root.glob('*_inputs')))
            self.assertEqual(len(recorded),2);self.assertEqual(recorded[1]['executed_controls_since_last_query'],5)
            ended=client.episode_end(success=False,execution=dict(decision_seq=1,executed_count=2,terminal=True))
            self.assertIs(ended,client.end_result)
            executed=[json.loads(line)for line in next(root.glob('*.executed.jsonl')).read_text().splitlines()]
            self.assertEqual([r['executed_count']for r in executed],[5,2])
            self.assertEqual([r['env_step']for r in executed],[0,5])
            self.assertEqual(len(executed[-1]['actions']),2)
            self.assertTrue(executed[-1]['terminal_ack'])
    def test_frozen_terminal_control_bypass_is_recorded_once(self):
        import runpy
        from client_trace import install
        class Native:
            def episode_start(self,**kwargs):return {}
            def infer(self,obs):return {'actions':np.arange(35,dtype=np.float32).reshape(5,7)}
            def _send_ctrl(self,message,**kwargs):
                self.original_message=message;self.native_return={};return self.native_return
            def episode_end(self,success=False,execution=None):
                return self._send_ctrl(dict(__ctrl__='episode_end',__execution__=execution))
        runtime=json.loads((Path(__file__).parent/'concurrent/bundles/dp/source_campaign.json').read_text())['runtime']
        end_episode=runpy.run_path(str(Path(runtime)/'exp/r17_integration/benchmarks/execution.py'))['end_episode']
        with retained_test_directory() as root:
            install(root,Native);client=Native()
            for iteration in range(2):
                client.episode_start();client.infer({'observation/state':np.zeros(8),
                    '__request__':dict(decision_seq=0)})
                ack=dict(decision_seq=0,executed_count=2,terminal=True)
                result=end_episode(client,dict(success=False),[ack]) if iteration==0 else client.episode_end(execution=ack)
                self.assertIs(result,client.native_return)
                self.assertEqual(client.original_message['__execution__'],ack)
            rows=[json.loads(line)for path in root.glob('*.executed.jsonl')for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['terminal_ack']and len(r['actions'])==2 for r in rows))
    def test_concurrent_floor_real_protocol_semantics(self):
        # Fake transport CPU test checks stateful ACK ordering/barrier/round
        # identity, not model arithmetic or a fabricated GPU qualification.
        row=dict(observation={'observation/image':np.zeros((2,2,3),dtype=np.uint8)},
            request=dict(task_name='task',original_init_idx=0,benchmark_protocol='proto',task_id=0))
        with retained_test_directory() as tmp:
            root=Path(tmp)
            for rep in range(2):
                drive(FakeClient,[row],dict(benchmark='libero',replan_steps=5),clients=4,
                    deliveries=3,round_index=rep,bundle_id='pure',output=root/f'round_{rep}/group_0')
            write_json(root/'RUN.json',dict(status='COMPLETE',rounds=2,input_identity='inputs',
                pin={'identity':'pin'},clients=4,deliveries=3,campaign={'xgpu_probe':{}},
                artifacts=['round_0/group_0','round_1/group_0']))
            write_json(root/'BUNDLE_ACKS.json',[dict(group=dict(bundle_id='pure'))])
            r=compare(root,root,.001)
            self.assertEqual(r['requests'],12)
            self.assertEqual(r['max_abs'],0)
            self.assertEqual(r['status'],'MEASURED_NOT_CERTIFIED')
            self.assertIsNone(r['agreement']['stall_call']['agreement'])
            self.assertFalse(r['SR_evidence'])
            with self.assertRaises(ValueError):compare(root,root,.001,left_repeats=[root])
            # A whole cache group missing on both rounds is also incomplete.
            write_json(root/'BUNDLE_ACKS.json',[dict(group=dict(bundle_id='pure')),
                dict(group=dict(bundle_id='cache'))])
            with self.assertRaisesRegex(ValueError,'incomplete concurrent request population'):
                compare(root,root,.001)
            write_json(root/'BUNDLE_ACKS.json',[dict(group=dict(bundle_id='pure'))])
            # Dropping the same request from both rounds must not look like a
            # perfectly matched smaller population.
            for rep in range(2):
                path=root/f'round_{rep}/group_0/ROWS.json';doc=json.loads(path.read_text())
                doc['rows'].pop(0);write_json(path,doc)
            with self.assertRaisesRegex(ValueError,'incomplete concurrent request population'):
                compare(root,root,.001)
    def test_long_segment_clock_is_checked_against_ack_counts(self):
        from concurrent_compare import clock_audit
        rows={}
        for i in range(3):
            rows[('cache',0,i)]=dict(response={'__hit_meta__':dict(segment_head=i%2==0,
                env_step_index=5*i,decision_index=i//2)},execution=dict(decision_seq=i,executed_count=5))
        group=clock_audit({0:rows})['0']['cache']
        self.assertEqual((group['env_mismatches'],group['decision_mismatches']),(0,0))
        rows[('cache',0,1)]['response']['__hit_meta__']['decision_index']=1
        self.assertEqual(clock_audit({0:rows})['0']['cache']['decision_mismatches'],1)

    def test_each_gpu_floor_stays_separate_from_cross_error(self):
        from summarize_concurrent import summarize_rows
        class DriftClient(FakeClient):
            level = 1.
            def infer(self, obs):
                result = super().infer(obs)
                result['actions'][:] = self.level
                return result
        row=dict(observation={'observation/image':np.zeros((2,2,3),dtype=np.uint8)},
            request=dict(task_name='task',original_init_idx=0,benchmark_protocol='proto',task_id=0))
        with retained_test_directory() as root:
            for host, levels in [('left',(1.,1.001)), ('right',(1.,1.0001))]:
                directory=root/host
                for repeat,level in enumerate(levels):
                    DriftClient.level=level
                    drive(DriftClient,[row],dict(benchmark='libero',replan_steps=5),clients=2,
                        deliveries=2,round_index=repeat,bundle_id='pure',output=directory/f'round_{repeat}/group_0')
                write_json(directory/'RUN.json',dict(status='COMPLETE',rounds=2,input_identity='inputs',
                    pin={'identity':'pin'},clients=2,deliveries=2,campaign={'xgpu_probe':{}},
                    artifacts=['round_0/group_0','round_1/group_0']))
                write_json(directory/'BUNDLE_ACKS.json',[dict(group=dict(bundle_id='pure'))])
            result=compare(root/'left',root/'right',.0005)
            summary=summarize_rows(result['rows'],result['budget'])
            self.assertAlmostEqual(summary['4090_floor']['max_abs_max'],float(np.float32(1.001))-1)
            self.assertAlmostEqual(summary['h100_floor']['max_abs_max'],float(np.float32(1.0001))-1)
            self.assertEqual(summary['4090_floor']['outside_absolute_budget'],4)
            self.assertEqual(summary['h100_floor']['outside_absolute_budget'],0)
            self.assertEqual(summary['cross']['outside_absolute_budget'],4)
            self.assertEqual(summary['outside_floor'],0)


if __name__=='__main__':unittest.main()
