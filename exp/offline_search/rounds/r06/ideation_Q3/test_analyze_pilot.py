"""Deterministic causal-estimator checks; no simulator or repository test suite."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_pilot import (call_effect, cluster_uncertainty, contrast, estimate, expected_slots,
                           features, interaction, safe_out, weighted_quantile)
from exp.offline_search.rounds.r06.p3_profiling.assignment import assign
from exp.offline_search.rounds.r06.p3_profiling.design import Design


def fixture():
    rows = []
    # Exact assignment-probability quadrature; treatment effects vary by init.
    for task in range(10):
        p = (.125, .25, .5)[task % 3]
        for init in range(2):
            y0, y1 = .2 + .1*init, .6 + .05*init
            for rep in range(8):
                z = rep < 8*p
                rows.append(dict(task_id=task, init=init, arm="synthetic", uid=f"{task}:{init}:{rep}", attempt=1,
                                 episode_weight=1/160, Y=y1 if z else y0,
                                 **{"assignment.coin_call": z, "assignment.nominal_propensity": p}))
    return pd.DataFrame(rows)


class EstimatorTests(unittest.TestCase):
    def test_logged_unequal_propensity_recovers_effect(self):
        r, _ = call_effect(fixture(), "Y")
        self.assertAlmostEqual(r["estimate"], .375, places=12)
        self.assertEqual(r["clusters"], 20)
        self.assertEqual(r["df"], 10)
        self.assertGreater(r["se"], 0)

    def test_copying_anchors_does_not_invent_independent_units(self):
        f = fixture()
        a, _ = call_effect(f, "Y")
        b, _ = call_effect(pd.concat([f]*7, ignore_index=True), "Y")
        self.assertAlmostEqual(a["estimate"], b["estimate"], places=12)
        self.assertAlmostEqual(a["se"], b["se"], places=12)
        self.assertEqual(a["clusters"], b["clusters"])

    def test_single_init_per_task_withholds_fixed_task_interval(self):
        r, _ = call_effect(fixture().query("init == 0"), "Y")
        self.assertEqual(r["df"], 0)
        self.assertTrue(np.isnan(r["lo_95"]))
        self.assertFalse(r["nomination_support"])

    def test_common_cluster_interaction_retains_covariance(self):
        f = fixture()
        r, inf = interaction(call_effect(f.query("init == 1"), "Y"), call_effect(f.query("init == 0"), "Y"))
        self.assertAlmostEqual(r["estimate"], -.05, places=12)
        self.assertEqual(len(inf), 20)

    def test_delay_package_retains_nonexecuted_calls_and_middle_arm(self):
        f = fixture().drop_duplicates(["task_id", "init"]).copy()
        f = pd.concat([f.assign(delay=d, Y=y, executed=d != 2) for d, y in ((0, 1.), (1, .5), (2, .25))], ignore_index=True)
        r, _ = contrast(f, f.delay == 0, 1/3, f.delay == 2, 1/3, "Y")
        self.assertAlmostEqual(r["estimate"], .75)
        self.assertEqual(r["n"], 60)
        self.assertEqual(r["n_b"], 20)  # all of these intentionally never execute

    def test_hold_contrast_marginalizes_other_random_duration(self):
        base = fixture().drop_duplicates(["task_id", "init"])
        f = pd.concat([base.assign(hold=h, duration=d, Y=.1*h+.01*d)
                       for h in (1, 2, 3) for d in (5, 10)], ignore_index=True)
        h, _ = contrast(f, f.hold == 3, 1/3, f.hold == 1, 1/3, "Y")
        d, _ = contrast(f, f.duration == 10, .5, f.duration == 5, .5, "Y")
        self.assertAlmostEqual(h["estimate"], .2)
        self.assertAlmostEqual(d["estimate"], .05)

    def test_sampling_weights_are_in_both_totals(self):
        f = fixture()
        a, _ = call_effect(f, "Y")
        b, _ = call_effect(f, "Y", np.full(len(f), 16.))
        self.assertAlmostEqual(a["estimate"], b["estimate"])
        self.assertAlmostEqual(a["se"], b["se"])

    def test_forced_probability_has_no_local_support(self):
        f = fixture()
        f["assignment.nominal_propensity"] = 1.
        with self.assertRaisesRegex(ValueError, "unsupported"):
            call_effect(f, "Y")

    def test_genuine_delayed_assignment_has_marginal_immediate_p(self):
        found = False
        for seed in range(200):
            d = Design(dict(delays=[0, 1, 2], cap=1))
            a = d.resolve(assign(seed, 0, 0, 0, 0, .5), True)
            if a["scheduled_trigger"] and a["delay_choice"] == 2:
                self.assertAlmostEqual(a["actual_propensity"], .5/3)
                self.assertEqual(a["nominal_propensity"], .5)
                self.assertFalse(a["executed_policy"])
                self.assertEqual(a["override"], "delay_scheduled")
                b = d.resolve(assign(seed, 0, 0, 0, 1, .5), True)
                c = d.resolve(assign(seed, 0, 0, 0, 2, .5), True)
                self.assertEqual(b["actual_propensity"], 0.)
                self.assertEqual(c["actual_propensity"], 1.)
                found = True
                break
        self.assertTrue(found)

    def test_sparse_and_unequal_library_reference_quantile(self):
        self.assertEqual(weighted_quantile([1., 2., 10.], [.45, .45, .1]), 2.)

    def test_manifest_slots_and_own_write_boundary(self):
        self.assertEqual(len(expected_slots(0, "pilot")), 20)
        self.assertEqual(len(expected_slots(0, "continuation")), 230)
        self.assertEqual(len(expected_slots(1, "full")), 150)
        self.assertEqual(len(expected_slots(2, "full")), 100)
        with self.assertRaisesRegex(ValueError, "outputs"):
            safe_out("/tmp/q3_forbidden")

    def test_zero_variance_cannot_certify_noninferiority(self):
        f = fixture()
        r, _ = estimate(f, np.zeros(len(f)))
        self.assertFalse(r["nomination_support"])
        self.assertTrue(np.isnan(r["lo_95"]))

    def test_pooled_physical_ir_is_not_mean_episode_ir(self):
        f = fixture().drop_duplicates(["task_id", "init"]).copy()
        cost = np.where(f.init == 0, 1., 3.)
        controls_in_fives = np.where(f.init == 0, 1., 9.)
        pooled, _ = estimate(f, cost, controls_in_fives)
        episode_mean, _ = estimate(f, cost/controls_in_fives)
        self.assertAlmostEqual(pooled["estimate"], .4)
        self.assertAlmostEqual(episode_mean["estimate"], 2/3)

    def test_diagnostic_B_history_and_outcome_blind_reference(self):
        rows = []
        for ep in range(2):
            for step in range(25):
                rows.append(dict(arm="A", uid=str(ep), attempt=1, cell="toy", cohort="A", split="calibration",
                                 task_id=0, init=0, step=step, Y=ep,
                                 **{"retrieval.d1_loeo_quantile": .5, "retrieval.dispersion_rms": step,
                                    "distance.commit10_rms": step/10, "retrieval.progress": "[0.5]",
                                    "resampling.dispersion_per_step": np.nan, "resampling.selection_p": 1/16,
                                    "assignment.pre_guard_call": False, "assignment.calls_before": 0,
                                    "assignment.anchors_since_call": np.nan, "guards.inputs_outputs.os_force_miss": step == 0,
                                    "calibration.statistics.stuck": 0, "calibration.statistics.progress": 0,
                                    **{"calibration.p_values."+k: .5 for k in ("coverage", "stuck", "lag", "overtime", "progress", "terminal")}}))
        f = pd.DataFrame(rows)
        eps = f.drop_duplicates(["arm", "uid", "attempt"]).assign(anchors=25)
        first, ref1 = features(f, eps)
        second, ref2 = features(f.assign(Y=1-f.Y), eps.assign(Y=1-eps.Y))
        self.assertTrue(np.array_equal(first.neighbour_high, second.neighbour_high))
        self.assertEqual(ref1[0]["threshold"], 22.)
        self.assertEqual(ref1[0]["threshold"], ref2[0]["threshold"])
        self.assertEqual(int(first.neighbour_high.sum()), 4)
        self.assertEqual(first.at_guard.sum(), 0.)
        self.assertEqual(int(first.previous_guard.sum()), 48)
        self.assertTrue(first.k4_high.isna().all())

    def test_pre_guard_randomization_restores_support(self):
        initial = assign(1, 0, 0, 0, 0, .125)
        experimental = Design(dict(pre_guard=True)).resolve(initial, False)
        mandatory = Design(dict(pre_guard=False)).resolve(initial, False)
        self.assertEqual(experimental["actual_propensity"], .125)
        self.assertEqual(experimental["override"], "coin")
        self.assertEqual(mandatory["actual_propensity"], 1.)
        self.assertEqual(mandatory["override"], "baseline_guard")

    def test_unobserved_delay_side_is_not_an_interval(self):
        f = fixture()
        r, _ = contrast(f, np.ones(len(f), bool), 1/3, np.zeros(len(f), bool), 1/3, "Y")
        self.assertEqual(r["status"], "point_only_no_realized_comparison")
        self.assertTrue(np.isnan(r["lo_95"]))
        self.assertFalse(r["nomination_support"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EstimatorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  success=result.wasSuccessful(), scope="deterministic statistical and design invariants; no closed-loop evaluation")
    (Path(__file__).resolve().parent / "analysis_test_results.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
