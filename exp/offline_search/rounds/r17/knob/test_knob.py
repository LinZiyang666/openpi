"""Synthetic campaign-layout fixtures; CPU only, no real outcome mutations."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from exp.offline_search.rounds.r17.knob import common as c, descent, test as a


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def name(suite, task):
    return f"{suite}/task_{task}"


def teacher(root, model, suite="libero_spatial"):
    episodes, rows = [], []
    for task in range(10):
        for init in range(50):
            fields = dict(task_name=name(suite, task), orig_init_idx=init)
            episodes.append(dict(fields, success=not (task == 5 and init < 2), n_decisions=1))
            rows.append(dict(fields, decision_seq=0, source="policy", executed_count=c.C0[model]))
    path = root / f"{c.MODEL_IDS[model]}_libero_{suite}" / "bpool/READY.json"
    put(path, dict(metadata=dict(episodes=episodes), rows=rows))
    return path


def lane(path, model, pool, entries, *, control=False, parent=None, missing_cost=None, pure_size=None):
    """Entries are (task index, level, init, success) in the spatial suite."""
    campaign = dict(model=c.MODEL_IDS[model])
    if parent:
        campaign["resume_from"] = str(parent)
    put(path / "campaign.json", campaign)
    tasks, journal, steps = [], [], []
    groups = set()
    for task, level, init, success in entries:
        base = "crg" if model == "Cosmos" else "chrg"
        group = "control" if control else f"{base}_{level}"
        uid = f"synthetic:{group}:eval:{task}:{init}"
        extra = dict(task_name=name("libero_spatial", task), init_pool=pool)
        if not control:
            extra.update(cache_group=group, library_size=pure_size if level == 'pure' and pure_size is not None else 50)
        tasks.append(dict(task_uid=uid, experiment="libero_spatial", orig_init_state_idx=init, extra=extra))
        groups.add(group)
        terminal = dict(task_uid=uid, accepted=True, attempt=2, success=success, ts=2,
                        status="done" if success else "failed")
        journal.extend([dict(terminal, accepted=False, attempt=1, success=not success), terminal])
        sources = ["policy"] if control or level == "pure" else ["policy", "library", "policy_tail"]
        for seq, source in enumerate(sources):
            fields = dict(task_uid=uid, accepted=True, attempt=2, decision_seq=seq)
            steps.append(dict(fields, source=source, actual_nfe=5 if source == "policy" else 0))
            count = 1 if source == "policy_tail" else c.C0[model]
            if (task, level, init) != missing_cost:
                steps.append(dict(fields, _kind="execution_ack", executed_count=count,
                                  terminal=seq == len(sources)-1))
    put(path / "assets/CACHE_STUDY.json", dict(groups=[dict(group=g, suite="libero_spatial", library_size=50)
                                                       for g in sorted(groups) if g != "control"]))
    put(path / "EPISODE_PLAN.json", dict(tasks=tasks))
    (path / "driver").mkdir(parents=True)
    for filename, records in (("journal", journal), ("attempts", journal), ("per_step", steps)):
        (path / f"driver/{filename}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


class KnobTests(unittest.TestCase):
    def test_reduced_descent_uses_same_lane_size500_pure_for_other_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This teacher succeeds everywhere except task5/init0,1. The
            # reduced pure arm fails init0..5, so k50 passes only against pure.
            teacher(root/'library', 'DP')
            entries = [(t, level, i, i >= 6) for t in range(10) for level in c.LEVELS
                       for i in range(30) if not (t == 9 and level == 'pure' and i == 29)]
            run = root/'dev'
            lane(run, 'DP', 'B', entries, pure_size=500)
            with contextlib.redirect_stdout(io.StringIO()):
                result = descent.main(['--lane', str(run), '--reference', 'pure-arm',
                    '--library-root', str(root/'library'), '--out', str(root/'out')])
            cell = next(row for row in result['cells'] if row['size'] == 50)
            task = cell['tasks'][name('libero_spatial', 0)]
            self.assertEqual(task['levels']['pure']['successes'], 24)
            self.assertEqual(task['levels']['k50']['paired_pure_successes'], 24)
            self.assertEqual(task['selected'], 'koff')
            incomplete = cell['tasks'][name('libero_spatial', 9)]
            self.assertTrue(incomplete['incomplete'])
            self.assertIsNone(incomplete['selected'])
            self.assertEqual(incomplete['levels']['pure']['missing_inits'], [29])

    def test_reduced_test_controls_use_step_study_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for model in c.C0:
                entries = [(t, 'pure', i, i >= 5) for t in range(10) for i in range(50)]
                lane(root/'controls'/c.REDUCED_CONTROL_NAMES[model][-1], model, 'A', entries, control=True)
                lane(root/model, model, 'A', [(t, 'k30', i, i >= 8) for t in range(10) for i in range(50)])
                runs += ['--lane', str(root/model)]
            with contextlib.redirect_stdout(io.StringIO()):
                result = a.main([*runs, '--controls', 'reduced', '--control-root', str(root/'controls'),
                                 '--out', str(root/'out')])
            self.assertEqual(len(result['cells']), 3)
            for cell in result['cells']:
                task = cell['tasks'][name('libero_spatial', 0)]
                self.assertEqual(task['control']['successes'], 45)
                self.assertEqual(task['successes'], 42)
                self.assertEqual(task['retest_level'], 'k40')
            self.assertEqual(result['controls'], c.REDUCED_CONTROL_NAMES)

    def test_descent_synthetic_lanes_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher(root / "library", "DP")
            entries = []
            for task in range(10):
                for level in c.LEVELS[1:]:
                    for init in range(30):
                        if (task == 3 and level == "k20") or (task == 4 and level == "k10" and init == 29):
                            continue
                        success = not (task == 5 and init in (1, 2))
                        if task == 1:
                            success = init >= (2 if level == "k30" else 1 if level == "k40" else 0)
                        if task == 2 and level == "k50":
                            success = init != 0
                        entries.append((task, level, init, success))
            run = root / "dev"
            lane(run, "DP", "B", entries, missing_cost=(6, "koff", 0))
            command = [sys.executable, str(Path(descent.__file__)), "--lane", str(run), "--lane", str(run),
                       "--library-root", str(root / "library"), "--out", str(root / "out")]
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root / "out/descent.json").read_text())
            tasks = result["cells"][0]["tasks"]
            task = lambda i: tasks[name("libero_spatial", i)]
            self.assertEqual(task(0)["selected"], "koff")
            self.assertEqual(task(1)["selected"], "k40")
            self.assertEqual(task(1)["first_failure"], "k30")
            self.assertEqual(task(1)["non_monotone_recoveries"], ["k20", "k10", "koff"])
            self.assertEqual(task(2)["selected"], "koff")
            self.assertEqual(task(2)["tolerances"]["0"]["selected"], "pure")
            for i in (3, 4):
                self.assertTrue(task(i)["incomplete"])
                self.assertIsNone(task(i)["selected"])
            self.assertEqual(task(3)["levels"]["k20"]["n"], 0)
            self.assertEqual(task(4)["levels"]["k10"]["missing_inits"], [29])
            self.assertEqual(task(5)["levels"]["koff"]["gains"], 1)
            self.assertEqual(task(5)["levels"]["koff"]["losses"], 1)
            self.assertAlmostEqual(task(0)["levels"]["koff"]["ir_proxy"], 5/11)
            self.assertIsNone(task(6)["levels"]["koff"]["ir_proxy"])
            table = json.loads((root / "out/levels.json").read_text())["DP"]["libero_spatial"]["50"]
            self.assertEqual(len(table), 8)
            self.assertNotIn(name("libero_spatial", 3), table)
            self.assertEqual((root / "out/levels_tol0pp.json").read_bytes(),
                             (root / "out/levels_tol2pp.json").read_bytes())
            report = (root / "out/DESCENT.md").read_text()
            self.assertIn("8/10", report)
            self.assertNotIn("| SR", report)

    def test_integer_boundaries(self):
        levels = {level: dict(n=30, passes={"5": True}) for level in c.LEVELS}
        self.assertEqual(descent.descend(levels, 5)["selected"], "koff")
        self.assertEqual(a.raised_level("k30", 45, 50, 50), "pure")
        self.assertEqual(a.raised_level("k30", 47, 50, 50), "k40")
        self.assertEqual(a.raised_level("k50", 46, 50, 50), "pure")
        self.assertEqual(a.raised_level("k30", 48, 50, 50), "k30")
        self.assertEqual(a.raised_level("pure", 40, 50, 50), "pure")

    def test_test_lanes_all_models_controls_resume_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            levels = ["koff", "k30", "k50", "pure", "k20", "k10", "k40", "k20", "koff", "koff"]
            deficits = [5, 3, 4, 5, 2, 4, 0, 5, 0, 0]
            for model in c.C0:
                control_entries = [(t, "pure", i, not (t == 6 and i < 2)) for t in range(10) for i in range(50)]
                control_names = c.CONTROL_NAMES[model]
                if model == "Cosmos":
                    original = root / "controls" / control_names[0]
                    lane(original, model, "A", [e for e in control_entries if e[2] < 40], control=True)
                    lane(root / "controls" / control_names[-1], model, "A",
                         [e for e in control_entries if e[2] >= 40], control=True, parent=original)
                else:
                    lane(root / "controls" / control_names[0], model, "A", control_entries, control=True)
                entries = [(t, levels[t], i, i >= deficits[t]) for t in range(10) for i in range(50)
                           if t != 8 and not (t == 7 and i == 49)]
                run = root / c.MODEL_IDS[model]
                lane(run, model, "A", entries, missing_cost=(9, "koff", 0))
                runs.append(run)
            command = [sys.executable, str(Path(a.__file__)), "--control-root", str(root / "controls"),
                       "--out", str(root / "out")]
            for run in runs:
                command.extend(["--lane", str(run)])
            completed = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            raises = json.loads((root / "out/raise.json").read_text())
            result = json.loads((root / "out/test.json").read_text())
            self.assertEqual(len(result["cells"]), 3)
            for row in result["cells"]:
                raised = raises[row["model"]]["libero_spatial"]["50"]
                self.assertEqual(raised, {name("libero_spatial", t): level
                                         for t, level in [(0, "pure"), (1, "k40"), (2, "pure"), (5, "k20")]})
                self.assertEqual(row["n"], 449)
                self.assertEqual(row["comparison"]["n"], 449)
                self.assertEqual(row["comparison"]["gains"], 2)
                self.assertEqual(row["comparison"]["losses"], 28)
                self.assertIsNone(row["ir_proxy"])
                self.assertEqual(row["tasks"][name("libero_spatial", 8)]["n"], 0)
                self.assertAlmostEqual(row["tasks"][name("libero_spatial", 0)]["ir_proxy"],
                                       c.C0[row["model"]] / (2*c.C0[row["model"]]+1))
                self.assertEqual(row["tasks"][name("libero_spatial", 3)]["ir_proxy"], 1)

    def test_bootstrap_matches_r16_within_task_sampling(self):
        names = [name("libero_spatial", t) for t in range(10)]
        ref = {( "libero_spatial", t, i): dict(success=i >= 2) for t in names for i in range(50)}
        data = {k: dict(success=k[2] != 3) for k in ref}
        rng = np.random.default_rng(20261005)
        rng.integers(0, 50, size=(5000, 10, 50))  # R16 long-suite draws first.
        indices = rng.integers(0, 50, size=(5000, 10, 50))
        got = a.paired(data, ref, indices, names)
        delta = np.array([[int(data["libero_spatial", t, i]["success"])-int(ref["libero_spatial", t, i]["success"])
                           for i in range(50)] for t in names])
        draws = delta[np.arange(10)[None, :, None], indices].sum(axis=(1, 2))/500
        self.assertEqual(got["bootstrap_ci95"], np.quantile(draws, [.025, .975]).tolist())
        self.assertEqual(got["gains"], 20)
        self.assertEqual(got["losses"], 10)
        self.assertEqual(got, a.paired(dict(reversed(list(data.items()))), ref, indices, names))

    def test_teacher_exact_counts_and_pure_stitching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in c.C0:
                for suite in c.cache.SUITES:
                    path = teacher(root, model, suite)
                    ref = c.teacher_reference(root, model, suite)
                    self.assertEqual(len(ref), 500)
                    self.assertEqual(c.stats(ref, model)["ir_proxy"], 1)
                    ready = json.loads(path.read_text())
                    ready["metadata"]["episodes"][-1] = ready["metadata"]["episodes"][0]
                    put(path, ready)
                    with self.assertRaisesRegex(ValueError, "exactly one teacher"):
                        c.teacher_reference(root, model, suite)
        mixed = {"pure": dict(success=True, pure=True, cost=dict(controls=6, policy=2)),
                 "knob": dict(success=False, cost=dict(controls=20, policy=1, library=3))}
        self.assertAlmostEqual(c.stats(mixed, "DP")["ir_proxy"], 15/26)

    def test_wrong_pool_and_multiple_test_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "dev"
            lane(run, "DP", "A", [(0, "k10", 0, True)])
            with self.assertRaisesRegex(ValueError, "expected pool B"):
                c.load_knob([run], "B")
            lane(root / "controls" / c.CONTROL_NAMES["DP"][0], "DP", "A",
                 [(t, "pure", i, True) for t in range(10) for i in range(50)], control=True)
            lane(root / "test", "DP", "A", [(0, "k10", 0, True), (0, "k20", 1, True)])
            # Retest rounds only raise levels; the highest observed level is used.
            with contextlib.redirect_stdout(io.StringIO()):
                result = a.main(["--lane", str(root / "test"), "--control-root", str(root / "controls"),
                                 "--out", str(root / "out")])
            task = sorted(result["cells"][0]["tasks"])[0]
            self.assertEqual(result["cells"][0]["tasks"][task]["level"], "k20")


if __name__ == "__main__":
    unittest.main()
