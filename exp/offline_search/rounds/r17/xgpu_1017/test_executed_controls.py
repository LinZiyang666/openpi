import copy
import unittest
from executed_controls import compare,validate_ack


class Tests(unittest.TestCase):
    def setUp(self):
        self.base=dict(model='dp',suite='spatial',arm='cache',task_id=0,init_idx=0,env_seed=7,policy_seed=7)
    def event(self,start,actions,terminal=False):
        return dict(self.base,env_step=start,actions=actions,executed_count=len(actions),
            action_width=2,terminal_ack=terminal)
    def test_control_offset_and_partial_tail(self):
        left=[self.event(0,[[0,1],[0,1]]),self.event(2,[[0,1],[0,1]],True)]
        right=copy.deepcopy(left);right[1]['actions'][1]=[.1,-1]
        r=compare(left,right)['episodes'][0]
        self.assertEqual(r['first_executed_action'],3)
        self.assertEqual(r['first_executed_gripper'],3)
        self.assertTrue(r['first_executed_action_is_exact'])
    def test_unrecorded_prefix_is_not_an_exact_first_difference(self):
        left=[self.event(2,[[0,1]],True)];right=[self.event(2,[[1,1]],True)]
        r=compare(left,right)['episodes'][0]
        self.assertFalse(r['first_executed_action_is_exact'])
        r=compare(left,left)['episodes'][0]
        self.assertFalse(r['executed_action_right_censored'])
    def test_terminal_censoring_and_missing_terminal_are_distinct(self):
        short=[self.event(0,[[0,1]],True)];long=[self.event(0,[[0,1],[0,1]],True)]
        r=compare(short,long)
        self.assertTrue(r['episodes'][0]['executed_action_right_censored'])
        self.assertIsNone(r['action_distribution']['quantiles']['0.5'])
        open_tail=[self.event(0,[[0,1]])]
        self.assertFalse(compare(open_tail,open_tail)['episodes'][0]['executed_action_right_censored'])
        with self.assertRaisesRegex(ValueError,'duplicate executed control'):compare(short+short,long)
    def test_ack_prefix_binding_rejects_unexecuted_values_or_wrong_identity(self):
        req=dict(decision_seq=4,explicit_seed=19)
        source=dict(env_step=10,actions=[[0,1],[.1,1],[99,-1]],request=req)
        row=dict(schema='xgpu.client_execution.v1',request=req,wire_delivery=4,env_step=10,
            executed_count=2,execution=dict(decision_seq=4,executed_count=2),
            actions=[[0,1],[.1,1]],terminal_ack=True)
        self.assertEqual(validate_ack(row,source).shape,(2,2))
        bad=copy.deepcopy(row);bad['actions'][1]=[99,-1]
        with self.assertRaisesRegex(ValueError,'returned prefix'):validate_ack(bad,source)
        bad=copy.deepcopy(row);bad['request']['explicit_seed']=20
        with self.assertRaisesRegex(ValueError,'identity mismatch'):validate_ack(bad,source)
    def test_missing_both_sides_is_retained_when_population_is_known(self):
        population=[dict(self.base,accepted=True),dict(self.base,init_idx=1,accepted=True)]
        traces=[self.event(0,[[0,1]],True)]
        r=compare(traces,traces,population=population)
        self.assertEqual(len(r['episodes']),2)
        self.assertEqual(r['action_distribution']['unknown_episodes'],1)
    def test_different_frozen_profiles_cannot_enter_paired_comparison(self):
        from paired_identity import check
        a=dict(self.base,plan_identity='plan',serving_pin='fw_full',benchmark_protocol='proto',runtime_identity='runtime')
        b=dict(a,serving_pin='fw_cpu_t5')
        with self.assertRaisesRegex(ValueError,'serving_pin'):check([a],[b])
        self.assertTrue(check([a],[a])[0]['qualified'])
    def test_narrow_sr_interval_without_plan_identity_is_not_qualified(self):
        from paired_analysis import sr
        outcomes=[dict(self.base,init_idx=i,accepted=True,success=True)for i in range(1000)]
        report=sr(outcomes,outcomes,resamples=20)[0]
        self.assertTrue(report['paired_CI_within_margin'])
        self.assertFalse(report['equivalent'])


if __name__=='__main__':unittest.main()
