"""A1 regression checks: statistical/rule boundaries, 44 profiles, R6 raw arms.

No evaluation verdicts are produced. All fixture files are temporary under /tmp;
validation results go to analysis_r7/a1_validation.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.stats import binomtest

from exp.offline_search.rounds.r07.analysis_scripts import a1_common as C
from exp.offline_search.rounds.r07.analysis_scripts import a1_analyze as A


def artificial_pools():
    result = {}
    for key, cells in (("SF1-A", C.CELLS), ("UF1-A", C.CELLS), ("SF1-UF1", C.CELLS),
                       ("SW-A", C.PI05), ("CT30-CU30", C.SPARSE)):
        result[key] = dict(status="complete", delta_SR=dict(estimate=.01, lo=.001, hi=.02),
                           delta_owner_IR_by_cell={c: -.001 for c in cells}, delta_owner_IR_pooled=-.001)
    return result


class AcceptanceTests(unittest.TestCase):
    def verdict(self, pools, rule):
        return A.acceptance(pools, final=True, frontier_complete=True)[rule]

    def test_sf_strict_bound_and_six_of_eight(self):
        p = artificial_pools()
        p["SF1-A"]["delta_SR"]["lo"] = -.010
        self.assertEqual(self.verdict(p, "1")["verdict"], "not supported")
        p["SF1-A"]["delta_SR"]["lo"] = np.nextafter(-.010, 0.)
        for cell in C.CELLS[:2]:
            p["SF1-A"]["delta_owner_IR_by_cell"][cell] = 0.
        self.assertEqual(self.verdict(p, "1")["verdict"], "supported")
        p["SF1-A"]["delta_owner_IR_by_cell"][C.CELLS[2]] = 0.
        self.assertEqual(self.verdict(p, "1")["verdict"], "not supported")

    def test_stage_or_branches_and_no_loss_is_not_NI(self):
        p = artificial_pools()
        p["SF1-A"]["delta_SR"] = dict(estimate=-.002, lo=-.05, hi=0.)
        p["SF1-UF1"]["delta_SR"]["lo"] = 0.
        p["UF1-A"]["delta_SR"]["lo"] = -.010
        self.assertEqual(self.verdict(p, "2")["verdict"], "not supported")
        p["UF1-A"]["delta_SR"]["lo"] = np.nextafter(-.010, -1.)
        self.assertEqual(self.verdict(p, "2")["verdict"], "supported")
        p["SF1-A"]["delta_SR"]["hi"] = -.000001
        self.assertEqual(self.verdict(p, "2")["verdict"], "not supported")
        p["SF1-UF1"]["delta_SR"]["lo"] = .000001
        self.assertEqual(self.verdict(p, "2")["verdict"], "supported")

    def test_wrist_strict_bound_and_four_of_four(self):
        p = artificial_pools()
        p["SW-A"]["delta_SR"]["lo"] = -.015
        self.assertEqual(self.verdict(p, "3")["verdict"], "not supported")
        p["SW-A"]["delta_SR"]["lo"] = np.nextafter(-.015, 0.)
        self.assertEqual(self.verdict(p, "3")["verdict"], "supported")
        p["SW-A"]["delta_owner_IR_by_cell"][C.PI05[0]] = 0.
        self.assertEqual(self.verdict(p, "3")["verdict"], "not supported")

    def test_calls_inclusive_cost_and_strict_SR(self):
        p = artificial_pools()
        p["CT30-CU30"]["delta_owner_IR_by_cell"] = {c: .015 for c in C.SPARSE}
        p["CT30-CU30"]["delta_owner_IR_pooled"] = .015
        self.assertEqual(self.verdict(p, "4")["verdict"], "supported")
        p["CT30-CU30"]["delta_SR"]["lo"] = 0.
        self.assertEqual(self.verdict(p, "4")["verdict"], "not supported")
        p["CT30-CU30"]["delta_SR"]["lo"] = .001
        p["CT30-CU30"]["delta_owner_IR_by_cell"][C.SPARSE[0]] = .016
        self.assertEqual(self.verdict(p, "4")["verdict"], "IR not matched")
        self.assertEqual(self.verdict(p, "4")["unmatched_cells"], [C.SPARSE[0]])
        self.assertEqual(self.verdict(p, "4")["cell_cost_verdict"][C.SPARSE[0]], "IR not matched in that cell")
        p["CT30-CU30"]["delta_owner_IR_pooled"] = 0.
        self.assertEqual(self.verdict(p, "4")["verdict"], "IR not matched")

    def test_phase1_and_missing_do_not_emit_verdict(self):
        p = artificial_pools()
        self.assertTrue(all(r["verdict"] == "deferred_phase1" for r in A.acceptance(p, frontier_complete=True).values()))
        p["SF1-A"] = dict(status="missing")
        self.assertEqual(self.verdict(p, "1")["verdict"], "missing")
        self.assertEqual(self.verdict(p, "2")["verdict"], "missing")


class StatisticsTests(unittest.TestCase):
    def test_single_cell_matches_R6_with_R7_seed(self):
        x = {k: float((k[0]+k[1]) % 3 == 0) for k in C.EXPECTED}
        y = {k: ((k[0]+k[1]) % 4) / 3 for k in C.EXPECTED}
        _, expected = C.R6.boot(x, y, seed=C.SEED, draws=C.DRAWS)
        _, actual = C.bootstrap_differences([(x, y)])
        np.testing.assert_array_equal(actual, expected)

    def test_pool_cells_resample_independently(self):
        x = {(t, i): i % 2 for t in range(2) for i in range(20)}
        y = {k: 0 for k in x}
        _, one = C.bootstrap_differences([(x, y)])
        result, two = C.bootstrap_differences([(x, y), (x, y)])
        self.assertEqual(result["estimate"], .5)
        self.assertLess(two.var(), one.var() * .6)
        self.assertGreater(two.var(), one.var() * .4)

    def test_replicates_are_averaged_before_resampling(self):
        keys = [(t, i) for t in range(2) for i in range(4)]
        records = [dict(id=f"r{i}", status="complete", _outcomes={k: int(i == k[1] % 3) for k in keys},
                        SR=0., owner_IR=.1) for i in range(3)]
        avg = C.average_records(records, "cell", "A")
        self.assertEqual(avg["n"], 8)
        self.assertTrue(all(v == 1/3 for v in avg["_outcomes"].values()))
        with self.assertRaises(ValueError):
            C.exact_pair(avg["_outcomes"], avg["_outcomes"])

    def test_exact_mcnemar_and_CP(self):
        x = {(0, i): int(i < 15) for i in range(100)}
        y = {(0, i): int(10 <= i < 20) for i in range(100)}
        r = C.exact_pair(x, y)
        self.assertEqual((r["wins"], r["losses"]), (10, 5))
        self.assertAlmostEqual(r["mcnemar_two_sided_exact_p"], binomtest(10, 15).pvalue)
        self.assertEqual(r["cp_lower95"], C.R6.ni_lower(x, y)[0])
        self.assertLess(C.exact_pair(x, x)["cp_lower95"], 0.)
        with self.assertRaises(ValueError):
            C.bootstrap_differences([(x, {(0, 0): 0})])

    def test_frontier_ties_and_crossings(self):
        rows = [dict(id="a", owner_IR=.1, SR=.8), dict(id="b", owner_IR=.1, SR=.8),
                dict(id="c", owner_IR=.2, SR=.7), dict(id="d", owner_IR=.5, SR=.9, pure=True)]
        self.assertEqual({r["id"] for r in C.pareto(rows)}, {"a", "b", "d"})
        self.assertIsNone(A.lowest_crossing(rows, .9, include_pure=False))
        self.assertEqual(A.lowest_crossing(rows, .9, include_pure=True)["id"], "d")

    def test_aggregate_IR_is_not_mean_episode_IR(self):
        x = dict(owner_IR=10/11, L=5, episodes=[dict(task=0, init=0, N=1, owner_cost=0.),
                                              dict(task=0, init=1, N=10, owner_cost=10.)])
        y = dict(owner_IR=0., L=5, episodes=[dict(task=0, init=0, N=1, owner_cost=0.),
                                          dict(task=0, init=1, N=1, owner_cost=0.)])
        ci = C.aggregate_cost_interval(x, y)
        self.assertAlmostEqual(ci["estimate"], 10/11)
        self.assertEqual((ci["lo"], ci["hi"]), (0., 1.))
        self.assertEqual(C.aggregate_cost_interval(x, x)["lo"], 0.)
        self.assertEqual(C.aggregate_cost_interval(x, x)["hi"], 0.)

    def test_fractional_mean_CP_uses_binary_constituents(self):
        keys = sorted(C.EXPECTED)
        x = dict(id="x", status="complete", owner_IR=.1, _outcomes={k: int(k[1] < 30) for k in keys})
        reps = [dict(id=f"r{i}", status="complete", owner_IR=.2, SR=.5,
                     _outcomes={k: int((k[1]+i) % 50 < 25) for k in keys}) for i in range(3)]
        y = C.average_records(reps, "cell", "A")
        r = C.contrast(x, y)
        self.assertIsNone(r["exact"])
        target = np.mean([C.R6.ni_lower(x["_outcomes"], rep["_outcomes"], alpha=.05/3)[0] for rep in reps])
        self.assertAlmostEqual(r["replicate_mean_CP"]["lower95"], target)

    def test_complete_and_missing_matrix_frontier_paths(self):
        data, outcomes = A.frozen_frontier()
        table = {}
        for model, short in C.R6.CELLS:
            cell = C.R6.c_cell(model, short)
            ident = C.R6.l10_ref(model, short).replace(":", "/")
            ys = outcomes[ident]
            table[cell, "L10"] = dict(id=ident, status="complete", SR=np.mean(list(ys.values())), _outcomes=ys)
        for spec in C.frozen_arms():
            cell, variant = spec["cell"], spec["variant"]
            ref = table[cell, "L10"]
            table[cell, variant] = dict(id=f"SYNTHETIC/{spec['arm']}", variant=variant, status="complete",
                                       SR=ref["SR"], owner_IR=.001, _outcomes=ref["_outcomes"])
        full = A.frontier_placement(table, data, outcomes)
        self.assertEqual(len(full["placements"]), 28)
        self.assertTrue(all(p["nondominated_by_R6"] for p in full["placements"]))
        self.assertTrue(all(c["status"] == "complete" and c["after_available_nonpure"]["owner_IR"] == .001
                            for c in full["lowest_IR_reaching_L10"]))
        table[C.CELLS[0], "SF1"]["status"] = "missing"
        partial = A.frontier_placement(table, data, outcomes)
        self.assertEqual(sum(p.get("status") == "missing" for p in partial["placements"]), 1)


class AuditTests(unittest.TestCase):
    def test_actual_camera_prices_and_completion_contract(self):
        base = dict(vision=True, hit=True, camera_mode="full", camera_stage1_calls=1,
                    camera_completion_calls=0, camera_completion=False, owner_cost=.152)
        for mode, vision, miss, nc, cost in (("full", True, False, 0, .152), ("full", True, True, 0, 1.),
                ("wrist_only", True, False, 0, .055198), ("wrist_only", True, True, 1, .953088),
                ("blind", False, False, 0, 0.)):
            row = dict(base, camera_mode=mode, vision=vision, hit=not miss, camera_stage1_calls=int(vision),
                       camera_completion_calls=nc, camera_completion=bool(nc), owner_cost=cost)
            self.assertAlmostEqual(C.decision_cost(row, "pi05", True)[0], cost)
        with self.assertRaises(ValueError):
            C.decision_cost(dict(base, camera_mode="wrist_only", hit=False), "pi05", True)
        with self.assertRaises(ValueError):
            C.decision_cost(dict(vision=True, hit=True), "pi05", True)
        with self.assertRaises(ValueError):
            C.decision_cost(dict(base, owner_cost=.1), "pi05", True)

    def test_completion_requires_marker_and_summary(self):
        with tempfile.TemporaryDirectory(prefix="r7_a1_") as tmp:
            root = Path(tmp)
            (root / "runs/x").mkdir(parents=True)
            (root / "state").mkdir()
            (root / "runs/x/summary.json").write_text("{}")
            self.assertEqual(C.completion(root, "x")["status"], "missing")
            (root / "state/x.manifest_hash.DONE").touch()
            self.assertEqual(C.completion(root, "x")["status"], "complete")
            (root / "state/x.manifest_hash.DONE").unlink()
            (root / "state/x.DONE").touch()
            self.assertEqual(C.completion(root, "x")["status"], "complete")
            (root / "runs/x/summary.json").unlink()
            self.assertEqual(C.completion(root, "x")["status"], "missing")

    def test_exception_join_uses_run_and_attempt_and_requires_coverage(self):
        with tempfile.TemporaryDirectory(prefix="r7_a1_") as tmp:
            root = Path(tmp)
            current = dict(task_uid="x:eval:0:0", run_id="new", attempt=1, success=True, status="done", accepted=True)
            old = dict(current, run_id="old", success=False)
            rows = [dict(old, _kind="client_timing", termination_reason="exception"),
                    dict(current, _kind="client_timing", termination_reason="success")]
            (root / "per_step.jsonl").write_text("\n".join(C.json.dumps(r) for r in rows))
            (root / "purged_exceptions.jsonl").write_text(C.json.dumps(old))
            good, _ = C.exception_audit(root, {(0, 0): current})
            self.assertEqual(good["status"], "ok")
            self.assertEqual(good["exception_attempts_all"], 1)
            bad, _ = C.exception_audit(root, {(0, 0): old})
            self.assertEqual(bad["residual_exceptions"], 1)
            unknown, _ = C.exception_audit(root, {(0, 0): dict(current, run_id="unseen")})
            self.assertEqual(unknown["status"], "invalid")

    def test_latest_incarnation_and_gapped_stream(self):
        with tempfile.TemporaryDirectory(prefix="r7_a1_") as tmp:
            root = Path(tmp)
            (root / "server_fixture").mkdir()
            j = dict(task_uid="x:eval:0:0", run_id="new", attempt=1, success=True, ts=10.)
            base = dict(ev="dec", uid=j["task_uid"], attempt=1, task_id=0, init=0, vision=True, hit=True)
            rows = [dict(base, step=0, ts=1., stage1_calls=8), dict(base, step=1, ts=2., stage1_calls=9),
                    dict(base, step=0, ts=5., stage1_calls=10),
                    dict(base, step=1, ts=6., vision=False, stage1_calls=10)]
            path = root / "server_fixture/decisions_fixture.jsonl"
            path.write_text("\n".join(C.json.dumps(r) for r in rows))
            timing = {C.attempt_key(j): dict(infers=2)}
            eps, audit = C.accepted_decisions(root, {(0, 0): j}, timing, "pi05")
            self.assertEqual(audit["discarded_prior_prefix"], 2)
            self.assertEqual(eps[0]["N"], 2)
            self.assertAlmostEqual(eps[0]["IR"], .076)
            rows[-1]["stage1_calls"] = 11
            path.write_text("\n".join(C.json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                C.accepted_decisions(root, {(0, 0): j}, timing, "pi05")
            rows[-1]["stage1_calls"] = 10
            rows[-1]["step"] = 2
            path.write_text("\n".join(C.json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                C.accepted_decisions(root, {(0, 0): j}, timing, "pi05")


def validate_profile():
    path = C.R7 / "profile_results/closed_loop/profile_report.json"
    reports = C.read_json(path)["reports"]
    rows = []
    for old in reports:
        variant = "SW" if old["arm"].startswith("r7_sw_") else old["arm"].split("_")[-2]
        expected = {(e["task"], e["init"]) for e in old["episodes"]}
        actual = C.load_arm(C.RUNS / "r07_profile_bval1", old["arm"], old["cell"], variant, expected=expected)
        errors = []
        if actual["status"] != "complete":
            errors.append(actual.get("error", "not complete"))
        else:
            for field, old_field in (("N", "N"), ("V", "V"), ("M", "M"), ("SR", "SR_descriptive"), ("owner_IR", "IR")):
                if not np.isclose(actual[field], old[old_field], atol=1e-10, rtol=0):
                    errors.append(field)
            previous = {(e["task"], e["init"]): e for e in old["episodes"]}
            for episode in actual["episodes"]:
                prior = previous[episode["task"], episode["init"]]
                for field in ("N", "Y", "IR"):
                    if not np.isclose(episode[field], prior[field], atol=1e-10, rtol=0):
                        errors.append(f"episode {episode['task']}:{episode['init']} {field}")
        rows.append(dict(arm=old["arm"], status="ok" if not errors else "invalid", errors=errors,
                         N=actual.get("N"), exception_audit=actual.get("exception_audit"),
                         decision_audit=actual.get("decision_audit")))
        print(f"profile {old['arm']}: {rows[-1]['status']}", flush=True)
    return dict(source=C.stamp(path, True), arms=len(rows), decisions=sum(r.get("N") or 0 for r in rows),
                all_match=all(r["status"] == "ok" for r in rows), rows=rows,
                chain_exception_audit=C.chain_audit(C.RUNS / "r07_profile_bval1"))


def validate_r6_raw():
    rows = []
    for cell in ("pi05_l10_50", "groot_l10_50"):
        for tag in ("U30", "C30"):
            arm = f"r6c_{cell}_{tag}"
            actual = C.load_arm(C.RUNS / "r06_c_validation", arm, cell, tag, historical=True)
            spec = f"r06_c_validation:{arm}"
            okay = (actual["status"] == "complete" and actual["_outcomes"] == C.R6.journal(spec)
                    and np.isclose(actual["owner_IR"], C.R6.owner_ir(spec), atol=1e-10, rtol=0))
            rows.append(dict(id=actual["id"], status="ok" if okay else "invalid", error=actual.get("error"),
                             N=actual.get("N"), decision_audit=actual.get("decision_audit")))
            print(f"R6 {arm}: {rows[-1]['status']}", flush=True)
    return dict(all_match=all(r["status"] == "ok" for r in rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--out", type=Path, default=C.R7 / "analysis_r7/a1_validation.json")
    args = parser.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (AcceptanceTests, StatisticsTests, AuditTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(unit_tests=result.testsRun, unit_pass=result.wasSuccessful(),
                  failures=[(str(test), trace) for test, trace in result.failures + result.errors])
    if not args.unit_only:
        record["profile"] = validate_profile()
        record["r6_raw"] = validate_r6_raw()
    record["all_pass"] = result.wasSuccessful() and all(record.get(k, {}).get("all_match", True) for k in ("profile", "r6_raw"))
    C.write_json(args.out, record)
    print(C.json.dumps(dict(all_pass=record["all_pass"], unit_tests=result.testsRun, output=str(args.out))))
    if not record["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
