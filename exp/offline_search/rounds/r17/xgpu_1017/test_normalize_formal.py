import copy
import json
from pathlib import Path
import unittest
import uuid
from artifacts import digest,sha,write_json
from normalize_formal import index_plan,outcomes,trace_index,trace_row,validate_manifest,normalize


class Tests(unittest.TestCase):
    def setUp(self):
        task=dict(task_uid='u',task_id=2,orig_init_state_idx=7,bundle_id='b',
            extra=dict(benchmark_protocol='protocol',task_name='pick'))
        self.plan=dict(model='cosmos',suite='spatial',arm='cache',
            benchmark_protocol='protocol',tasks=[task])
        self.plan['identity']=digest(self.plan)
        self.tasks=index_plan(self.plan,dict(seed=195,seed_mode='fixed',protocol_id='libero_cosmos_official_50_r17_v1'))
        self.attempt=dict(task_uid='u',accepted=True,success=False,attempt=2,error=None)
        self.trace=dict(request=dict(task_id=2,original_init_idx=7,bundle_generation='b',
            seed=195,attempt=2,benchmark_protocol='protocol',task_name='pick'),
            env_step=16,decision_step=0,wire_delivery=1,actual_noise_seed=195)

    def test_failed_success_is_retained_and_rejected_attempt_audited(self):
        a,r=outcomes(self.tasks,[{**self.attempt,'accepted':False,'attempt':1},self.attempt])
        self.assertFalse(a['u']['success']);self.assertEqual(r,{'u':1})
        self.assertEqual(a['u']['policy_seed'],195)
        self.assertEqual(a['u']['env_seed'],0)

    def test_missing_error_and_duplicate_are_not_silently_dropped(self):
        for rows in ([],[{**self.attempt,'error':'timeout'}],[self.attempt,self.attempt]):
            with self.assertRaises(ValueError):outcomes(self.tasks,rows)

    def test_plan_and_seed_mismatch_rejected(self):
        bad=copy.deepcopy(self.plan);bad['tasks'][0]['task_id']=9
        with self.assertRaises(ValueError):index_plan(bad,dict(seed=195))
        accepted,_=outcomes(self.tasks,[self.attempt])
        bad=copy.deepcopy(self.trace);bad['request']['seed']=7
        with self.assertRaises(ValueError):trace_row(bad,self.plan,self.tasks,trace_index(self.tasks),accepted)

    def test_tail_clock_and_accepted_generation_are_preserved(self):
        accepted,_=outcomes(self.tasks,[self.attempt])
        result=trace_row(self.trace,self.plan,self.tasks,trace_index(self.tasks),accepted)
        self.assertEqual(result['env_step'],16);self.assertEqual(result['decision_step'],0)
        self.assertEqual(result['arm'],'cache')
        old=copy.deepcopy(self.trace);old['request']['attempt']=1
        self.assertIsNone(trace_row(old,self.plan,self.tasks,trace_index(self.tasks),accepted))

    def test_driver_population_cannot_change_but_port_can(self):
        task=copy.deepcopy(self.plan['tasks'][0]);task['server_port']=23155
        validate_manifest(self.tasks,dict(tasks=[task]))
        task['orig_init_state_idx']=9
        with self.assertRaises(ValueError):validate_manifest(self.tasks,dict(tasks=[task]))

    def test_complete_normalizer_binds_terminal_execution_to_returned_chunk(self):
        # Synthetic one-episode CPU fixture; no model/runtime/GPU is launched.
        root=Path(__file__).parent/'test_artifacts'/('normalization_'+uuid.uuid4().hex)
        bundle=root/'bundle';assets=bundle/'assets';assets.mkdir(parents=True)
        benchmark=dict(seed=195,protocol_id='libero_cosmos_official_50_r17_v1')
        write_json(assets/'benchmark.json',benchmark);write_json(bundle/'source_campaign.json',{})
        pin=dict(model='cosmos',runtime_identity='fixture_runtime',
            source_campaign_sha256=sha(bundle/'source_campaign.json'),
            files={'benchmark.json':sha(assets/'benchmark.json')})
        pin['identity']=digest(pin);write_json(bundle/'PIN.json',pin)
        plan=copy.deepcopy(self.plan);plan.pop('identity');plan['pin']=pin['identity'];plan['identity']=digest(plan)
        directory=root/'plan';(directory/'assets').mkdir(parents=True)
        write_json(directory/'assets/benchmark.json',benchmark);write_json(directory/'PLAN.json',plan)
        driver=root/'driver';driver.mkdir()
        write_json(driver/'manifest.json',dict(tasks=plan['tasks'],runtime_identity='fixture_runtime'))
        write_json(driver/'DRIVER_RECEIPT.json',dict(status='PASS',all_done=True))
        (driver/'attempts.jsonl').write_text(json.dumps(self.attempt)+'\n')
        query=copy.deepcopy(self.trace);query['request']['decision_seq']=query['wire_delivery']
        query.update(schema='xgpu.client_trace.v1',actions=[[0,1],[.1,1],[99,-1]])
        event={k:query[k]for k in ('request','env_step','decision_step','wire_delivery')}
        event.update(schema='xgpu.client_execution.v1',actions=query['actions'][:2],
            action_width=2,executed_count=2,terminal_ack=True,
            execution=dict(decision_seq=1,executed_count=2,terminal=True))
        (root/'query.jsonl').write_text(json.dumps(query)+'\n');(root/'executed.jsonl').write_text(json.dumps(event)+'\n')
        r=normalize(directory/'PLAN.json',driver/'attempts.jsonl',driver/'DRIVER_RECEIPT.json',
            root/'normalized',bundle,[root/'query.jsonl'],[root/'executed.jsonl'])
        self.assertEqual((r['outcomes'],r['trace_rows'],r['execution_rows']),(1,1,1))
        self.assertEqual(r['queries_without_execution_ack'],0)
        self.assertFalse(json.loads((root/'normalized/outcomes.jsonl').read_text())['success'])
        self.assertEqual(len(json.loads((root/'normalized/executions.jsonl').read_text())['actions']),2)


if __name__=='__main__':unittest.main()
