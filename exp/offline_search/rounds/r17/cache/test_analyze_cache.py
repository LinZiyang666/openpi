"""CPU-only safety/statistics/incremental tests; fixtures never read run outcomes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import beta

SPEC = importlib.util.spec_from_file_location("cache_analysis", Path(__file__).with_name("analyze_cache.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class AnalysisTests(unittest.TestCase):
    def test_pending_never_reads_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            put(run / "RECEIPT.json", dict(status="PENDING_EVIDENCE", all_owned_closed=False))
            # Opening this poisoned outcome file would make the test fail.
            (run / "driver").mkdir(); (run / "driver/journal.jsonl").write_text("not json")
            snap = a.receipt_snapshot(run)
            self.assertIsNone(snap["receipt"])
            with self.assertRaisesRegex(ValueError, "unsealed"):
                a.load_lane(run, snap, a.Sources(), {}, [500])

    def test_mixed_size_blocked_before_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            put(run / "RECEIPT.json", dict(status="PASS", all_owned_closed=True))
            put(run / "campaign.json", dict(model="dp", official_50=True, smoke=False))
            put(run / "assets/CACHE_STUDY.json", dict(groups=[dict(library_size=500), dict(library_size=50)]))
            (run / "SUMMARY.json").write_text("OUTCOME MUST NOT BE READ")
            with self.assertRaisesRegex(ValueError, "allowlist"):
                a.load_lane(run, a.receipt_snapshot(run), a.Sources(), {}, [500])

    def test_fixed_exclusions_and_original_500_identities(self):
        flags, ids = a.load_exclusions(a.Sources())
        self.assertEqual(len(flags), 32)
        for suite, left in zip(a.SUITES, (489, 479)):
            keys = {k for k in ids if k[0] == suite}
            a.complete_keys(keys, suite, ids)
            self.assertEqual(len(keys - flags), left)
            with self.assertRaisesRegex(ValueError, "original 500"):
                a.complete_keys(keys - {next(iter(keys))}, suite, ids)

    def test_fit_size_and_nested_views_fail_closed(self):
        parent = dict(content_digest="p")
        view = dict(size=50, episode_ids=list(range(50)), parent_digest="p", content_digest="v50")
        receipt = dict(store_digest="v50", selection_digest="v50", library_size=50, parent_digest="p",
                       config=dict(commit_horizon=10, k=16, kref=8))
        fit = dict(digest="f50", receipt=receipt, fit_evidence=dict(partition="library", source_digest="v50"))
        stall = dict(store_digest="v50", fitted_digest="f50", calibration_partition="B_library_successful_LOEO", library_size=50)
        a.check_fit_selection(view, parent, fit, stall, 50, 10)
        receipt["library_size"] = 500
        with self.assertRaisesRegex(ValueError, "different size"):
            a.check_fit_selection(view, parent, fit, stall, 50, 10)
        sels = {("DP", 50, a.SUITES[0]): dict(parent="p", episodes={1, 2}),
                ("DP", 500, a.SUITES[0]): dict(parent="p", episodes={1, 2, 3})}
        a.nested(sels)
        sels["DP", 50, a.SUITES[0]]["episodes"] = {9}
        with self.assertRaisesRegex(ValueError, "not nested"):
            a.nested(sels)

    def test_retry_and_conflicting_duplicates(self):
        good = dict(task_uid="x", attempt=2, accepted=True, success=False)
        rows = [dict(good, attempt=1, accepted=False, success=True), good, good.copy()]
        self.assertEqual(a.accepted_terminal(rows), {"x": good})
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            a.accepted_terminal(rows + [dict(good, success=True)])

    def test_paired_family_and_bootstrap(self):
        keys = [("s", str(t), i) for t in range(10) for i in range(50)]
        ref = {k: dict(success=True, init_hash=str(k)) for k in keys}
        arm = copy.deepcopy(ref)
        s = a.paired(keys, ref, arm)
        self.assertAlmostEqual(s["cp_lb_family"], -beta.ppf(1-.05/80, 1, 500))
        self.assertLess(s["cp_lb_family"], s["cp_lb_pointwise"])
        self.assertEqual(s["bootstrap_lb_family"], 0)
        for k in keys[:15]:
            arm[k]["success"] = False
        s = a.paired(keys, ref, arm)
        self.assertAlmostEqual(s["diff"], -.03)
        self.assertEqual(s, a.paired(reversed(keys), ref, arm))
        self.assertFalse(s["ni_family"])
        arm[keys[0]]["init_hash"] = "bad"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            a.paired(keys, ref, arm)

    def test_tail_cost_and_early_stall(self):
        task = dict(extra=dict(commit_horizon=10, miss_mode="stall_only", corrector=False,
                               bundle_identity="b", recipe_identity="r", yaml_identity="y"))
        ds, acks = {}, {}
        # A stall MISS may interrupt a library tail. Only two fresh decisions,
        # one policy call; tail request is not another miss or NFE computation.
        for seq, source, cursor in [(0, "library", 0), (1, "policy", 0), (2, "policy_tail", 5)]:
            ds[seq] = dict(task["extra"], source=source, num_steps=100, actual_nfe=100 if source == "policy" else 0,
                           stage_calls=[1, 1, 100] if source == "policy" else [0, 0, 0], stall_call=source == "policy",
                           corrector_enabled=False, capture_enabled=False, cursor=cursor,
                           server_timing=dict(infer_ms=2.))
            count = 2 if seq == 2 else 5
            acks[seq] = dict(source=source, executed_count=count, cursor_end=cursor+count,
                             terminal=seq == 2, decision_seq=seq)
        acks[2]["server_terminal_ack"] = {k: acks[2][k] for k in ("terminal", "decision_seq", "executed_count", "cursor_end")}
        d = dict(controls=12, decisions=ds, acks=acks, timing=dict(infers=3, steps=22))
        a.finish_episode(d, task, is_cache=True, model="DP")
        c = a.cost(["x"], {"x": d})
        self.assertEqual(c["fresh_decisions"], 2)
        self.assertEqual(c["fresh_hit_rate"], .5)
        self.assertEqual(c["nfe"], 100)
        self.assertAlmostEqual(c["nfe_per_control"], 100/12)
        self.assertEqual(c["calls_per_episode"], 1)
        self.assertFalse(c["service_complete"])
        self.assertIsNone(c["service_ms_per_control"])

    def test_aliases_and_fixed_pending_family(self):
        self.assertEqual(a.physical_groups("Cosmos"), ("c", "cr", "cg", "crg"))
        self.assertEqual(sum(len(a.physical_groups(m)) for m in a.MODELS)*2, 40)
        claims = a.claim_table([])
        self.assertEqual(len(claims), 40)  # 20 two-suite claims per family.
        self.assertTrue(all(c["both_suites"] == "PENDING" for c in claims))
        self.assertEqual(a.FAMILY, 40)

    def test_full_synthetic_lane_incremental(self):
        """Exercise admission -> streaming ledgers -> pair/cost -> output.

        A single sealed group/suite has all 500 original identities. Bundle
        admission is tested separately against real N500 metadata, so replace
        only that expensive artifact validator with a fixture here.
        """
        flags, all_ids = a.load_exclusions(a.Sources())
        ids = {k: v for k, v in all_ids.items() if k[0] == "libero_spatial"}
        g = dict(bundle_identity="b", recipe_identity="r", sha256="y", settings=dict(alias=None, commit=5, corrector=False, gated=False))
        tasks, journals, attempts, records = [], [], [], []
        for i, (key, h) in enumerate(sorted(ids.items())):
            uid = f"synthetic:eval:{i//50}:{key[2]}"
            ex = dict(task_name=key[1], init_pool="A", partition="test", library_size=500, cache_group="c",
                       bundle_identity="b", recipe_identity="r", yaml_identity="y", commit_horizon=5,
                       corrector=False, miss_mode="never", num_steps=100)
            tasks.append(dict(task_uid=uid, experiment=key[0], orig_init_state_idx=key[2], extra=ex))
            ok = i % 10 != 0
            journals.append(dict(task_uid=uid, attempt=2, accepted=True, success=ok, status="done" if ok else "failed", error=None))
            attempts.append(dict(journals[-1], n_steps=3, worker_id="w_timan107_0", duration_s=1.))
            d = dict(task_uid=uid, attempt=2, accepted=True, task_name=key[1], original_init_idx=key[2],
                      task_data_hash=h, success=ok, init_pool="A", partition="test", decision_seq=0,
                      source="library", cursor=0, num_steps=100, actual_nfe=0, stage_calls=[1,0,0],
                      corrector_enabled=False, capture_enabled=False, server_timing=dict(infer_ms=2.),
                      bundle_identity="b", recipe_identity="r", yaml_identity="y", commit_horizon=5, miss_mode="never")
            ack = dict(d, _kind="execution_ack", executed_count=3, cursor_end=3, terminal=True,
                        server_terminal_ack=dict(decision_seq=0, executed_count=3, cursor_end=3, terminal=True))
            timing = dict(task_uid=uid, attempt=2, accepted=True, _kind="client_timing", infers=1, steps=13)
            records.extend([dict(d, attempt=1, accepted=False), d, timing, ack, dict(ack)])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/"lane"; out = Path(tmp)/"out"
            put(run/"RECEIPT.json", dict(status="PASS", all_owned_closed=True))
            put(run/"campaign.json", dict(model="dp", official_50=True, smoke=False))
            put(run/"assets/CACHE_STUDY.json", dict(groups=[dict(library_size=500)]))
            put(run/"EPISODE_PLAN.json", dict(tasks=tasks))
            put(run/"SUMMARY.json", dict(status="PASS", completed_episodes=500, expected_episodes=500,
                terminal_errors=0, full_arm_execution_verified=True,
                groups={"c/libero_spatial":dict(completed=500, successes=450)}))
            (run/"driver").mkdir()
            for name, rows in [("journal",journals), ("attempts",attempts), ("per_step",records)]:
                (run/f"driver/{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with patch.object(a, "validate_bundles", return_value=({(500,"libero_spatial","c"):g}, {})):
                cells, _ = a.load_lane(run, a.receipt_snapshot(run), a.Sources(), all_ids, [500])
            rows = a.analyze(cells, flags)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sr"]["sr"], .9)
            self.assertEqual(rows[0]["cost"]["nfe_per_control"], 0)
            self.assertEqual(rows[1]["sr"]["n"], 489)
            self.assertEqual(rows[0]["comparisons"]["control"]["status"], "PENDING_REFERENCE")
            # Newly arriving N50 + control add paired comparisons, without
            # changing F500, the exclusion set or the pending other suite.
            control = copy.deepcopy(cells["DP", 500, "c", "libero_spatial"])
            for d in control["data"].values():
                d["cost"].update(library=0, policy=1, nfe=100)
            cells["DP", 0, "control", "libero_spatial"] = control
            cells["DP", 50, "c", "libero_spatial"] = copy.deepcopy(cells["DP", 500, "c", "libero_spatial"])
            rows = a.analyze(cells, flags)
            small = next(r for r in rows if r["size"] == 50 and r["cohort"] == "full")
            self.assertTrue(small["comparisons"]["N500"]["primary"])
            self.assertEqual(small["comparisons"]["N500"]["family"], "Fsize")
            self.assertFalse(small["comparisons"]["control"]["primary"])
            self.assertEqual(small["comparisons"]["control"]["nfe_per_control_reduction"], 1)
            self.assertFalse(small["comparisons"]["control"]["service_comparable"])
            self.assertEqual(len(rows), 4)
            report = dict(as_of_utc="synthetic", allowed_sizes=[500], lanes=[], rows=rows,
                          controls={}, claims=a.claim_table(rows))
            with patch.object(a, "plot"):
                a.write_report(out, report)
            self.assertTrue((out/"CACHE_RESULTS.csv").exists())
            text = (out/"CACHE_RESULTS.md").read_text()
            self.assertIn("90.00", text)
            self.assertIn("PENDING", text)


if __name__ == "__main__":
    unittest.main()
