"""Read-only replay of real deployed libraries across all eight selected cells.

These are fixed-observation CPU checks, not closed-loop outcome experiments.
They exercise actual frozen A and wrist metrics without models/simulation.
"""
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import store
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
from exp.offline_search.rounds.r08.ops.emit_arms import CELLS, HERE, STORE, make_specs
from exp.offline_search.rounds.r08.ops.prefit import build_method, load_frozen
from .methods import AnchorCalls, ShiftedAWM
from .test_methods import assert_result, lottery, qview, tape


def mixed_tape(method, lib, rows, episode, oracle=False):
    method.reset(episode)
    h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
    records = []
    for step, row in enumerate(rows):
        q = qview(lib, row, step, episode, h)
        if oracle:
            # The real S2 payload spelling is 'distance', passed by S1 on q.oracle.
            q.oracle = dict(status="partial", privileged=True, objects=[
                dict(object_id="goal0", status="available", distance=.04,
                     lifted=False, satisfied=False, predicate_known=True, in_window=True),
                dict(object_id="unresolved", status="unsupported", reason="body unresolved")])
        is_tail = step > 0 and not h["hit"][-1]
        result = method.policy_tail_step(q) if is_tail else method.blind_step(q)
        look = isinstance(result, LookReason)
        proposal = method.query(q) if look else result
        call = bool(look and proposal.extras.get("os_force_miss", 0))
        served = np.array(lib.action[row], np.float32) if call else proposal.action
        if call:
            method.invalidate_anchor()
        if is_tail and not look:
            if served.tobytes() != policy_tail_chunk(h["action"][-1]).tobytes():
                raise AssertionError("CU policy-tail bytes changed")
        d = method.debug_record()
        json.dumps(d, allow_nan=False)
        records.append(dict(look=look, call=call, action=served.tobytes(), diag=d))
        h["age"] = 0 if look else h["age"] + 1
        for key, value in (("rs", q.rs), ("action", served), ("vision", look), ("hit", not call),
                           ("v0", q.key_v0 if look else np.full_like(q.key_v0, np.nan)),
                           ("v1", q.key_v1 if look else np.full_like(q.key_v1, np.nan))):
            h[key].append(value)
    return records


class LibraryReplayTest(unittest.TestCase):
    def test_all_selected_cells(self):
        lock = json.loads((HERE / "r7_sources.json").read_text())
        specs = make_specs(lock, "/tmp/r8_S5/fits")
        reports = []
        for cell in CELLS:
            start = time.perf_counter()
            model, suite, size = cell.split("_")
            ctx_cell = "{}_{}_cache".format(model, suite)
            frozen = lock["cells"][cell]
            base_blob, follow_blob = load_frozen(frozen["A"], ctx_cell), load_frozen(frozen["SF1"], ctx_cell)
            base, stages = base_blob["method"], follow_blob["method"].follow_table
            lib = store.LibraryView(STORE, "{}_{}".format(model, suite), base.cand_name)
            kwargs = frozen["A"]["kwargs"]
            zero = lottery(base, stages, kwargs, force_e=0)
            random_follow = lottery(base, stages, kwargs)
            probe_zero = AnchorCalls(p=0)
            probe_zero.base, _ = clone_method(base, strict=True)
            shifted_zero, _ = clone_method(base, strict=True)
            shifted_zero.__class__ = ShiftedAWM
            shifted_zero._r8_shifted, shifted_zero._r8_diag = False, {}
            hist_counts = dict(identity_decisions=0, lottery_anchors=0, support=[0, 0, 0], drawn_e=[0, 0, 0],
                               randomized_calls=0, fresh_call_anchors=0, policy_calls=0, policy_tails=0, oracle_calls=0,
                               wrist_queries=0, no_blind_decisions=0)
            cell_rows = [r for r in specs if r["model"] == model and r["suite"] == suite and
                         r["r8"]["library_size"] == int(size)]
            methods = {}
            variants = ["IP", "O5b"] if size == "50" else []
            if model == "pi05":
                variants += ["W10", "W5"]
            if suite == "l10" and size == "50":
                variants += ["A5"]
            blobs = dict(A=base_blob, SF1=follow_blob)
            if model == "pi05":
                blobs["SW"] = load_frozen(frozen["SW"], ctx_cell)
            for variant in variants:
                row = next(r for r in cell_rows if r["r8"]["variant"] == variant)
                methods[variant] = build_method(row, blobs)
            policy_row = next(r for r in specs if r["model"] == model and r["suite"] == suite and r["r8"]["variant"] == "P10")
            methods["P10"] = build_method(policy_row, blobs)
            for task in lib.tasks():
                task_rows = np.flatnonzero(np.asarray(lib.task_id) == task)
                ep_id = int(lib.episode[task_rows[0]])
                rows = task_rows[np.asarray(lib.episode[task_rows]) == ep_id][:24]
                ep = SimpleNamespace(uid=cell + ":library:" + str(ep_id), task_id=int(task), init=ep_id)
                ref, _ = tape(base, lib, rows, ep)
                for method in (zero, probe_zero, shifted_zero):
                    actual, _ = tape(method, lib, rows, ep)
                    for (alook, a), (blook, b) in zip(ref, actual):
                        self.assertEqual(alook, blook)
                        assert_result(self, a, b)
                    hist_counts["identity_decisions"] += len(rows)
                tape(random_follow, lib, rows, ep)
                # Separate replay keeps the diagnostic records at every anchor.
                random_follow.reset(ep)
                for decision, row in enumerate(rows):
                    h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
                    q = qview(lib, row, 0, ep, h)
                    q.step = decision
                    # Query-only records use legal causal previous-key history.
                    if decision:
                        q.hist_key_v0 = np.asarray(lib.key_v0[rows[:decision]])
                        q.hist_key_v1 = np.asarray(lib.key_v1[rows[:decision]])
                        q.hist_has_vision = np.ones(decision, bool)
                    random_follow.query(q)
                    d = random_follow.debug_record()
                    hist_counts["lottery_anchors"] += 1
                    hist_counts["drawn_e"][d["drawn_e"]] += 1
                    for e in d["support"]:
                        hist_counts["support"][e] += 1
                    self.assertIn(d["drawn_e"], d["support"])
                    self.assertAlmostEqual(sum(d["propensities"]), 1.)
                for variant, method in methods.items():
                    if variant in ("IP", "P10", "O5b"):
                        records = mixed_tape(method, lib, rows, ep, oracle=variant == "O5b")
                        if variant == "P10":
                            self.assertEqual([r["call"] for r in records], [j % 2 == 0 for j in range(len(rows))])
                            hist_counts["policy_calls"] += sum(r["call"] for r in records)
                            hist_counts["policy_tails"] += sum(r["diag"]["src"] == "policy_tail" for r in records)
                        elif variant == "O5b":
                            self.assertEqual(sum(r["call"] for r in records), 2)
                            anchors = [r["diag"] for r in records if r["diag"]["fresh"]]
                            self.assertTrue(all(d["oracle_status"] == "partial" for d in anchors))
                            self.assertTrue(all([o["status"] for o in d["oracle_objects"]] ==
                                                ["available", "unsupported"] for d in anchors))
                            self.assertEqual(method._oracle_counts, {"goal0": 2})
                            hist_counts["oracle_calls"] += 2
                        else:
                            anchors = [r for r in records if r["diag"]["fresh"]]
                            hist_counts["fresh_call_anchors"] += len(anchors)
                            hist_counts["randomized_calls"] += sum(r["call"] for r in anchors)
                            self.assertTrue(all(r["diag"]["p_effective"] == .25 and not r["diag"]["cooldown"] for r in anchors))
                    elif variant in ("W5", "W10"):
                        # Simulate the plugin's per-request camera selection.
                        method.reset(ep)
                        h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
                        for step, row in enumerate(rows):
                            q = qview(lib, row, step, ep, h)
                            result = method.blind_step(q)
                            look = isinstance(result, LookReason)
                            if look:
                                expected_camera = "full" if step == 0 else "wrist_only"
                                self.assertEqual(method.next_camera_mode, expected_camera)
                                method.set_camera_mode(method.next_camera_mode)
                                result = method.query(q)
                                if step == 0:
                                    self.assertEqual(result.action.tobytes(), ref[0][1].action.tobytes())
                                    self.assertEqual(result.topk.tobytes(), ref[0][1].topk.tobytes())
                                    self.assertEqual(result.scores.tobytes(), ref[0][1].scores.tobytes())
                                    self.assertEqual(result.confidence, ref[0][1].confidence)
                                hist_counts["wrist_queries"] += step > 0
                            self.assertEqual(look, variant == "W5" or step % 2 == 0)
                            h["age"] = 0 if look else h["age"] + 1
                            for key, value in (("rs", q.rs), ("action", result.action), ("vision", look), ("hit", True),
                                ("v0", q.key_v0 if look else np.full_like(q.key_v0, np.nan)),
                                ("v1", q.key_v1 if look else np.full_like(q.key_v1, np.nan))):
                                h[key].append(value)
                    elif variant == "A5":
                        actual, _ = tape(method, lib, rows, ep)
                        self.assertTrue(all(look for look, _ in actual))
                        hist_counts["no_blind_decisions"] += len(rows)
            if hist_counts["support"][2]:
                self.assertTrue(all(n > 0 for n in hist_counts["drawn_e"]))
            # Every loaded artifact must work with the plugin's ordinary loader.
            with open(frozen["A"]["artifact"], "rb") as f:
                self.assertEqual(fingerprint(pickle.load(f)["method"]), fingerprint(base))
            reports.append(dict(cell=cell, **hist_counts, seconds=time.perf_counter() - start))
            print(json.dumps(reports[-1]), flush=True)
        out = Path("/tmp/r8_S5/library_replay.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(status="PASS", cells=reports), indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
