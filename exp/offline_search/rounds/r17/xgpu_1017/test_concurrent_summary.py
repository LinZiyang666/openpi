import copy
from pathlib import Path
import subprocess
import unittest

from summarize_concurrent import summarize
from analyze_fw_remote import remote_program


def row(head=True):
    metric = dict(max_abs=.0001, relative_l2=.0001)
    decision = dict(available=True, left_agree=True, right_agree=True, cross_agree=True)
    return dict(key=['cache', 0, 0], any_segment_head=head,
                concurrent_floor_left=metric.copy(), concurrent_floor_right=metric.copy(),
                cross=metric.copy(), within_floor=True, gripper_sign_agree=True,
                neighbour_candidate_audit=dict(qualified=True, near_boundary=False),
                decisions={k: decision.copy() for k in ('top1_neighbour', 'donor_rows', 'stall_call')},
                scores={'nearest_distance': {'values': [1., 1.]}})


def report(rows):
    return dict(rows=rows, budget=dict(max_abs=.001, relative_l2=.001),
                agreement={k: {} for k in rows[0]['decisions']})


class SummaryTests(unittest.TestCase):
    def test_relative_failure_is_counted_without_absolute_failure(self):
        r=row();r['cross']['relative_l2']=.002
        g=summarize(report([r]))['by_group']['cache']
        self.assertEqual(g['metrics']['cross']['over_either_budget'], 1)
        self.assertEqual(g['metrics']['cross']['over_absolute_budget'], 0)
        self.assertEqual(g['metrics']['4090']['over_either_budget'], 0)

    def test_unknown_and_tail_are_not_observed_head_agreement(self):
        a=row();a['decisions']['stall_call']['available']=False
        tail=row(False);tail['key'][-1]=1
        g=summarize(report([a,tail]))['by_group']['cache']
        self.assertEqual(g['requests'],2)
        self.assertEqual(g['decisions']['stall_call']['unknown_heads'],1)
        self.assertEqual(g['decisions']['stall_call']['observed_heads'],0)
        self.assertEqual(g['decisions']['top1_neighbour']['observed_heads'],1)

    def test_near_rank_swap_and_far_donor_order_remain_distinct(self):
        a=row();a['neighbour_candidate_audit']['near_boundary']=True
        a['decisions']['top1_neighbour']['cross_agree']=False
        b=copy.deepcopy(a);b['key'][-1]=1;b['neighbour_candidate_audit']['near_boundary']=False
        b['decisions']['top1_neighbour']['cross_agree']=True
        b['decisions']['donor_rows']['cross_agree']=False
        g=summarize(report([a,b]))['by_group']['cache']
        self.assertEqual(g['neighbor_regions']['near']['top1_flips'],1)
        self.assertEqual(g['neighbor_regions']['away']['top1_flips'],0)
        self.assertEqual(g['neighbor_regions']['away']['donor_order_flips'],1)

    def test_remote_program_compiles_without_running(self):
        compile(remote_program(), '<read-only CPU analysis>', 'exec')

    def test_transfer_rejects_old_campaign_port_before_tether(self):
        path=Path(__file__).parent/'pull_fw_reference.sh'
        for port in ('23177', '023177', '0', '65536', 'abc', '123;touch'):
            p=subprocess.run(['bash',str(path),port],capture_output=True,text=True)
            self.assertEqual(p.returncode,2,port)
            self.assertNotIn('some avg10=',p.stdout)


if __name__ == '__main__':
    unittest.main()
