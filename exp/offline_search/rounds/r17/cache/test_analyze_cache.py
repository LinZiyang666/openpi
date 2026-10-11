"""CPU-only safety/statistics/incremental tests; fixtures never read run outcomes."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scipy.stats import beta

SPEC = importlib.util.spec_from_file_location("cache_analysis", Path(__file__).with_name("analyze_cache.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class AnalysisTests(unittest.TestCase):
    def test_fixed_exclusions(self):
        flags = a.load_exclusions()
        self.assertEqual(len(flags), 32)
        self.assertEqual(sum(k[0] == "libero_spatial" for k in flags), 11)
        self.assertEqual(sum(k[0] == "libero_10" for k in flags), 21)

    def test_retry_and_conflicting_duplicates(self):
        good = dict(task_uid="x", attempt=2, accepted=True, success=False, ts=2)
        rows = [dict(good, attempt=1, accepted=False, success=True), good, good.copy()]
        self.assertEqual(a.accepted_terminal(rows), {"x": good})
        newer = dict(good, success=True, ts=3)
        self.assertEqual(a.accepted_terminal(rows + [newer, dict(good, ts=1)]), {"x": newer})

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
        del arm[keys[0]]
        self.assertEqual(a.paired(keys, ref, arm)["n"], 499)

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
        a.finish_episode(d)
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
        """Read a running lane without SUMMARY, then produce pair/cost tables."""
        flags = a.load_exclusions()
        with (a.HERE / "CONTAMINATION_DISTANCES.csv").open() as f:
            ids = {(r["suite"], r["task"], int(r["a_init"])): r["a_sha256"]
                   for r in csv.DictReader(f) if r["suite"] == "libero_spatial"}
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
        with tempfile.TemporaryDirectory(dir=a.HERE) as tmp:
            run = Path(tmp)/"lane"; out = Path(tmp)/"out"
            put(run/"RECEIPT.json", dict(status="PARTIAL", all_owned_closed=False))
            put(run/"campaign.json", dict(model="dp", official_50=True, smoke=False))
            put(run/"assets/CACHE_STUDY.json", dict(groups=[dict(library_size=500, suite="libero_spatial", group="c")]))
            put(run/"EPISODE_PLAN.json", dict(tasks=tasks))
            (run/"driver").mkdir()
            for name, rows in [("journal",journals), ("attempts",attempts), ("per_step",records)]:
                (run/f"driver/{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            cells = a.load_lane(run, a.receipt_snapshot(run), a.Sources(), [500])
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
            small_cell = cells["DP", 50, "c", "libero_spatial"]
            removed = next(iter(small_cell["data"]))
            del small_cell["data"][removed]
            del small_cell["data"][next(iter(small_cell["data"]))]["cost"]
            small = next(r for r in a.analyze(cells, flags) if r["size"] == 50 and r["cohort"] == "full")
            self.assertEqual(small["status"], "INTERIM")
            self.assertEqual(small["sr"]["n"], 499)
            self.assertEqual(small["cost"]["episodes"], 498)
            self.assertEqual(small["comparisons"]["N500"]["n"], 499)
            self.assertEqual(small["comparisons"]["N500"]["status"], "INTERIM")

    def test_resume_union_latest_and_missing_cost(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as tmp:
            runs = [Path(tmp) / str(i) for i in range(3)]
            tasks = [dict(task_uid=str(i), experiment="libero_spatial", orig_init_state_idx=i,
                          extra=dict(task_name="task", library_size=50, cache_group="c")) for i in range(3)]
            for run in runs:
                put(run / "campaign.json", dict(model="dp"))
                put(run / "EPISODE_PLAN.json", dict(tasks=tasks))
                (run / "driver").mkdir()
            put(runs[1] / "CACHE_RESUME.json", dict(source=str(runs[0])))
            put(runs[2] / "campaign.json", dict(model="dp", cache_resume_selection=dict(
                source=str(runs[1]), retained_task_uids=[], source_pins={"stale": "bad"})))
            records = [[dict(task_uid="0", accepted=True, success=True, ts=4)],
                       [dict(task_uid="1", accepted=True, success=False, ts=2)],
                       [dict(task_uid="0", accepted=True, success=False, ts=3)]]
            for run, rows in zip(runs, records):
                (run / "driver/journal.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            snap = a.receipt_snapshot(runs[-1])
            cells = a.load_lane(runs[-1], snap, a.Sources(), [50])
            row = a.analyze(cells, set())[0]
            self.assertEqual(row["sr"]["n"], 2)
            self.assertEqual(row["sr"]["sr"], .5)
            self.assertEqual(row["status"], "INTERIM")
            self.assertEqual(row["cost"]["episodes"], 0)
            self.assertEqual(snap["missing_cost_episodes"], 2)
            self.assertEqual(snap["terminal_duplicates"]["journal"]["duplicates"], 1)
            self.assertEqual(len(snap["segments"]), 3)


if __name__ == "__main__":
    unittest.main()
