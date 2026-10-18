import unittest
from neighbor_boundary import audit


class Tests(unittest.TestCase):
    def test_same_sorted_gap_can_hide_a_rank_swap(self):
        a=dict(donor_rows=[10,20],xgpu_donor_distances=[1.,1.01])
        b=dict(donor_rows=[20,10],xgpu_donor_distances=[1.,1.01])
        r=audit([a,a],[b,b])
        self.assertTrue(r['qualified']);self.assertTrue(r['near_boundary'])
        self.assertAlmostEqual(r['distance_cross_error'],.01)
        self.assertAlmostEqual(r['boundary_band'],.02)

    def test_missing_identity_support_is_unknown(self):
        a=dict(donor_rows=[10,20],xgpu_donor_distances=[1.,2.])
        b=dict(donor_rows=[30,10],xgpu_donor_distances=[.9,1.])
        self.assertFalse(audit([a],[b])['qualified'])
        self.assertFalse(audit([a],[{}])['qualified'])

    def test_far_stable_candidate_gap_is_observed(self):
        a=dict(donor_rows=[10,20],xgpu_donor_distances=[1.,2.])
        b=dict(donor_rows=[10,20],xgpu_donor_distances=[1.001,2.001])
        self.assertFalse(audit([a,a],[b,b])['near_boundary'])


if __name__=='__main__':unittest.main()
