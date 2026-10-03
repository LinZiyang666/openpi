"""CPU method contracts on a small actual AWM metric and library chains."""
import copy
import json
import pickle
from types import SimpleNamespace
import unittest

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension
from exp.offline_search.rounds.r07.stages.stages import StageTable
from exp.offline_search.rounds.r08.ops.prefit import adapt_cache
from .methods import (AnchorCalls, EveryFiveAWM, FollowLottery, IdentificationProbe,
                      OracleGraspCalls, PolicyEveryTen, ShiftedAWM, WristEveryLook, coin)


def fixture(model="pi05"):
    n, horizon = 64, 10 if model == "pi05" else 16
    rng = np.random.default_rng(3)
    episode, steps = np.repeat(np.arange(4), 16), np.tile(np.arange(16), 4)
    state = (steps[:, None] * .1 + episode[:, None] * .01 + rng.normal(size=(n, 8)) * .005).astype(np.float32)
    v0 = (steps[:, None] * .1 + rng.normal(size=(n, 4)) * .005).astype(np.float32)
    v1 = (steps[:, None] * .1 + rng.normal(size=(n, 4)) * .005).astype(np.float32)
    actions = np.zeros((n, horizon, 32), np.float32)
    actions[:, :, :6] = rng.normal(size=(n, horizon, 6))
    actions[:, :, 6] = np.where(np.arange(n)[:, None] % 16 < 8, -1, 1)
    nxt = np.arange(n, dtype=np.int32) + 1
    nxt[steps == 15] = -1
    lib = SimpleNamespace(action=actions, rs=state, key_v0=v0, key_v1=v1, episode=episode,
        step=steps, next=nxt, success=np.ones(n, bool), task_id=np.zeros(n, np.int32), ep_len=np.full(n, 16), L=n,
        tasks=lambda: [0], rows_of_task=lambda task: np.arange(n))
    kwargs = dict(lib="current", kref=5, serving="anchor_tail", budget=1, gates="budget_only", early=False)
    base = BlindAWM(**kwargs)
    base.model, base.H, base.cand_name, base.prof = model, horizon, "current", api.NULL_PROFILER
    base.act, base.lib_ep, base.lib_step = actions, episode, steps
    base.sig = np.ones(7, np.float32)
    base.B0T = base.B1T = np.eye(2, 4, dtype=np.float32)
    base.mu0 = base.mu1 = np.zeros(4, np.float32)
    base.muB0 = base.muB1 = np.zeros(2, np.float32)
    Z = np.column_stack((v0[:, :2], v1[:, :2], state))
    heads = actions[:, :5, :7].reshape(n, -1)
    table = SimpleNamespace(rows=np.arange(n), Wf=np.eye(12, dtype=np.float32), shift=np.zeros(12, np.float32),
        Z=Z, z2=(Z * Z).sum(1), Z0=None, A0=None, As0=None, W0f=None,
        RS=state, rs2=(state * state).sum(1), s_d=1., s_c=1., HD=heads, h2=(heads * heads).sum(1))
    base.tasks = {0: table}
    base.zmu, base.zsd = dict(d1=0., disp=0., dst=0.), dict(d1=1., disp=1., dst=1.)
    base.zs_sd, base.s_a = 1., 1.
    base._fit_blind(lib)
    manifest = dict(exec_steps=5, H=horizon, act_valid_dims=7, rs_valid_dims=8, gripper_dim=6)
    stages = StageTable.fit(lib, manifest=dict(manifest, _retrieval=base))
    return base, lib, stages, kwargs


def lottery(base, stages, kwargs, **options):
    method = adapt_cache(base, FollowLottery, dict(kwargs, **options))
    method.follow_table = stages
    method.follow_component = FollowExtension(stages, extend_blocks=2, stage_gate=False, state_valve=False)
    return method


def qview(lib, row, step, episode, history):
    hrs = np.asarray(history["rs"], np.float32).reshape(step, lib.rs.shape[1])
    ha = np.asarray(history["action"], np.float32).reshape(step, lib.action.shape[1], 32)
    hv = np.asarray(history["vision"], bool)
    hits = np.asarray(history["hit"], np.int8)
    return SimpleNamespace(step=step, task_id=episode.task_id, episode=episode,
        rs=lib.rs[row], raw_state=lib.rs[row].astype(np.float64), key_v0=lib.key_v0[row], key_v1=lib.key_v1[row],
        hist_key_v0=np.asarray(history["v0"], np.float32).reshape(step, lib.key_v0.shape[1]),
        hist_key_v1=np.asarray(history["v1"], np.float32).reshape(step, lib.key_v1.shape[1]), hist_rs=hrs,
        hist_a_exec=ha, hist_has_vision=hv, hist_hit=hits, prev_hit=bool(hits[-1]) if step else None,
        prev_a_exec=ha[-1] if step else None, blind_age=history["age"], executed_steps=5)


def tape(method, lib, rows, episode=None, observe=False):
    episode = episode or SimpleNamespace(uid="replay", task_id=0, init=4)
    method.reset(episode)
    h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
    result, diag = [], []
    for step, row in enumerate(rows):
        q = qview(lib, row, step, episode, h)
        blind = method.blind_step(q)
        look = isinstance(blind, LookReason)
        r = method.query(q) if look else blind
        h["age"] = 0 if look else h["age"] + 1
        result.append((look, r))
        if observe:
            before = pickle.dumps(method, protocol=4)
            record = method.debug_record()
            json.dumps(record, allow_nan=False)
            # Returning nested state must not give callers a mutable method view.
            record["test_mutation"] = 123
            if record.get("support"):
                record["support"].append(123)
            assert pickle.dumps(method, protocol=4) == before
            diag.append(method.debug_record())
        for name, value in (("rs", q.rs), ("action", r.action), ("vision", look), ("hit", True),
                            ("v0", q.key_v0 if look else np.full_like(q.key_v0, np.nan)),
                            ("v1", q.key_v1 if look else np.full_like(q.key_v1, np.nan))):
            h[name].append(value)
    return result, diag


def assert_result(test, a, b):
    test.assertEqual(type(a), type(b))
    test.assertEqual(a.action.dtype, b.action.dtype)
    test.assertEqual(a.action.shape, b.action.shape)
    test.assertEqual(a.action.tobytes(), b.action.tobytes())
    if isinstance(a, BlindResult):
        test.assertEqual(a.rows.tobytes(), b.rows.tobytes())
        test.assertEqual(a.weights.tobytes(), b.weights.tobytes())
    else:
        test.assertEqual(a.topk.tobytes(), b.topk.tobytes())
        test.assertEqual(a.scores.tobytes(), b.scores.tobytes())
        test.assertEqual(a.confidence, b.confidence)
    test.assertEqual(a.library, b.library)
    test.assertEqual(a.extras, b.extras)


class MethodsTest(unittest.TestCase):
    def test_zero_extension_exact_a_and_observer_read_only(self):
        for model in ("pi05", "groot"):
            base, lib, stages, kw = fixture(model)
            method = lottery(base, stages, kw, force_e=0)
            reference, _ = tape(base, lib, np.tile(np.arange(16), 15))
            actual, diagnostics = tape(method, lib, np.tile(np.arange(16), 15), observe=True)
            self.assertEqual(len(actual), 240)
            for (alook, a), (blook, b) in zip(reference, actual):
                self.assertEqual(alook, blook)
                assert_result(self, a, b)
            self.assertTrue(all(d["drawn_e"] == 0 for d in diagnostics))
            np.testing.assert_array_equal(base._anchor["rows"], method._anchor["rows"])
            self.assertEqual(base._anchor["last_step"], method._anchor["last_step"])

    def test_coins_cover_support_and_do_not_use_rng(self):
        before = pickle.dumps(np.random.get_state())
        values = [coin(7, 0, init, step, "test") for init in range(50) for step in range(50)]
        self.assertEqual(before, pickle.dumps(np.random.get_state()))
        self.assertTrue(all(0 <= c < 1 for c in values))
        buckets = np.bincount((np.asarray(values) * 3).astype(int), minlength=3)
        self.assertTrue(np.all(np.abs(buckets / len(values) - 1/3) < .04))
        self.assertNotEqual(coin(7, 0, 0, 0, "test"), coin(8, 0, 0, 0, "test"))
        self.assertNotEqual(coin(7, 0, 0, 0, "test"), coin(7, 0, 0, 0, "other"))

    def test_structural_support_degrades_without_clamping(self):
        base, lib, table, kw = fixture()
        m = lottery(base, table, kw)
        arows, weights = np.repeat([0], 16), np.full(16, 1/16, np.float32)
        ep = SimpleNamespace(uid="x", task_id=0, init=4)
        for row, expected in ((0, [0, 1, 2]), (13, [0, 1]), (14, [0]), (15, [0])):
            arows[:] = row
            weights[-1] = 0  # structural membership still includes zero-weight rows
            m._remember_anchor(SimpleNamespace(step=0, task_id=0, episode=ep, rs=lib.rs[row]),
                               arows, weights, lib.action[row])
            d = m.debug_record()
            self.assertEqual(d["support"], expected)
            self.assertAlmostEqual(sum(d["propensities"]), 1.)
            self.assertEqual(d["propensities"], [1/len(expected) if e in expected else 0 for e in range(3)])
        arows[-1] = 15
        m._remember_anchor(SimpleNamespace(step=0, task_id=0, episode=ep, rs=lib.rs[0]),
                           arows, weights, lib.action[0])
        self.assertEqual(m.debug_record()["support"], [0])

    def test_stage_and_large_state_valve_only_shadow(self):
        for model in ("pi05", "groot"):
            base, lib, t, kw = fixture(model)
            m = lottery(base, t, kw, force_e=2)
            rows, weights = np.repeat(0, 16), np.full(16, 1/16, np.float32)
            t.mode = np.zeros_like(t.mode)
            t.mode[1] = 1
            ep = SimpleNamespace(uid="x", task_id=0, init=9)
            m._remember_anchor(SimpleNamespace(step=0, task_id=0, episode=ep, rs=lib.rs[0]),
                               rows, weights, lib.action[0])
            self.assertFalse(m.debug_record()["stage_ok_by_e"][2])
            for age in (1, 2, 3):
                q = SimpleNamespace(step=age, task_id=0, episode=ep, rs=lib.rs[0]+1e5, prev_hit=True,
                    hist_rs=np.repeat(lib.rs[0][None], age, 0), blind_age=age-1, executed_steps=5)
                r = m.blind_step(q)
                self.assertIsInstance(r, BlindResult)
                self.assertTrue(m.debug_record()["shadow_valve_fire"])
                expected = (lib.action[0, age*5:age*5+5, :7] if age*5+5 <= base.H else
                            np.tensordot(weights, base.act[t.advance(rows, age), :5, :7], 1))
                self.assertEqual(expected.tobytes(), r.action[:5, :7].tobytes())
            q.step, q.blind_age = 4, 3
            self.assertIsInstance(m.blind_step(q), LookReason)

    def test_budget_zero_and_shift_cadence(self):
        base, lib, _t, kw = fixture("groot")
        every = adapt_cache(base, EveryFiveAWM, dict(kw, budget=0))
        shifted = adapt_cache(base, ShiftedAWM, kw)
        disabled = adapt_cache(base, ShiftedAWM, dict(kw, shifted=False))
        rows = np.tile(np.arange(16), 15)
        reference, _ = tape(base, lib, rows)
        actual, _ = tape(disabled, lib, rows, observe=True)
        for (alook, a), (blook, b) in zip(reference, actual):
            self.assertEqual(alook, blook)
            assert_result(self, a, b)
        self.assertTrue(all(look for look, _ in tape(every, lib, rows)[0]))
        budget_zero, _ = clone_method(base, strict=True)
        budget_zero.budget = 0
        for (_, a), (_, b) in zip(tape(budget_zero, lib, rows)[0], tape(every, lib, rows)[0]):
            assert_result(self, a, b)
        looks = [i for i, (look, _) in enumerate(tape(shifted, lib, rows)[0]) if look]
        self.assertEqual(looks, [0] + list(range(1, len(rows), 2)))

    def test_call_zero_identity_and_no_stall_or_cooldown(self):
        base, lib, _t, _kw = fixture()
        m = AnchorCalls(p=0)
        m.base, _ = clone_method(base, strict=True)
        reference, _ = tape(base, lib, np.tile(np.arange(16), 15))
        actual, _ = tape(m, lib, np.tile(np.arange(16), 15), observe=True)
        for (alook, a), (blook, b) in zip(reference, actual):
            self.assertEqual(alook, blook)
            assert_result(self, a, b)
        m = IdentificationProbe(random_seed=7)
        m.base = base
        ep = SimpleNamespace(uid="x", task_id=0, init=4)
        m.reset(ep)
        h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
        count = 0
        for step in range(1000):
            q = qview(lib, step % 16, 0, ep, h)
            q.step = step * 2
            # Standalone fresh anchors: history is irrelevant for the coin rule.
            p, u, _ = m._assignment(q)
            self.assertEqual(p, .25)
            count += u < p
        self.assertLess(abs(count / 1000 - .25), .04)

    def test_policy_tail_exact_wire_padding_and_lifecycle(self):
        for model in ("pi05", "groot"):
            base, lib, _t, _kw = fixture(model)
            m = PolicyEveryTen()
            m.base = base
            ep = SimpleNamespace(uid="x", task_id=0, init=4)
            m.reset(ep)
            h = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
            q = qview(lib, 0, 0, ep, h)
            proposal = m.query(q)
            self.assertEqual(proposal.extras["os_force_miss"], 1.)
            # Policy output has nonzero padded channels too: preserve every byte.
            policy = np.arange(base.H * 32, dtype=np.float32).reshape(base.H, 32)
            for k, v in (("rs", q.rs), ("action", policy), ("vision", True), ("hit", False),
                         ("v0", q.key_v0), ("v1", q.key_v1)):
                h[k].append(v)
            bq = qview(lib, 1, 1, ep, h)
            m.invalidate_anchor()
            result = m.policy_tail_step(bq)
            self.assertIsInstance(result, BlindResult)
            self.assertEqual(result.action.tobytes(), policy_tail_chunk(policy).tobytes())
            self.assertFalse(m.debug_record()["eligible"])
            self.assertIsInstance(m.policy_tail_step(bq), LookReason)

    def test_oracle_window_lift_goal_and_per_object_cap(self):
        base, lib, _t, _kw = fixture()
        m = OracleGraspCalls(tight=True)
        m.base = base
        ep = SimpleNamespace(uid="x", task_id=0, init=4)
        m.reset(ep)
        q = SimpleNamespace(step=0)
        def assign(objects):
            m.set_oracle(dict(status="available", objects=objects))
            return m._assignment(q)[0]
        obj = dict(object_id="a", in_window=True, distance_m=.05, lifted=False, satisfied=False)
        self.assertEqual(assign([dict(obj, distance_m=.050001)]), 0)
        self.assertEqual(assign([dict(obj, lifted=True)]), 0)
        self.assertEqual(assign([dict(obj, satisfied=True)]), 0)
        self.assertEqual(assign([dict(obj, predicate_known=False)]), 0)
        self.assertEqual([assign([obj]) for _ in range(3)], [1, 1, 0])
        self.assertEqual([assign([dict(obj, object_id="b")]) for _ in range(3)], [1, 1, 0])
        self.assertEqual(m._assignment(q)[0], 0)  # no stale privileged payload reuse
        m.reset(SimpleNamespace(uid="new", task_id=0, init=4))
        self.assertEqual(assign([obj]), 1)
        loose = OracleGraspCalls()
        loose.base = base
        loose.reset(ep)
        for _ in range(5):
            loose.set_oracle(dict(obj, distance_m=.12))
            self.assertEqual(loose._assignment(q)[0], 1)
        q.oracle = dict(status="available", objects=[dict(obj, distance_m=.12)])
        self.assertEqual(loose._assignment(q)[0], 1)  # plugin channel needs no setter
        q.oracle = None
        self.assertEqual(loose._assignment(q)[0], 0)

    def test_oracle_partial_uses_only_available_objects_and_preserves_caps(self):
        resolved = dict(object_id="resolved", status="available", distance=.04,
                        predicate_known=True, in_window=True, lifted=False, satisfied=False)
        # Unresolved objects deliberately look closer and eligible: status wins.
        unsupported = dict(resolved, object_id="unknown", status="unsupported", distance=.001,
                           reason="unresolved goal object")
        failed = dict(resolved, object_id="failed", status="error", distance=.002,
                      reason="predicate evaluation failed")
        payload = dict(status="partial", objects=[unsupported, resolved, failed])
        original = copy.deepcopy(payload)
        for tight in (False, True):
            with self.subTest(tight=tight):
                m = OracleGraspCalls(tight=tight)
                q = SimpleNamespace(oracle=payload)
                assignments = [m._assignment(q) for _ in range(3)]
                self.assertEqual([r[0] for r in assignments], [1, 1, 0] if tight else [1, 1, 1])
                self.assertEqual(m._oracle_counts, {"resolved": 2 if tight else 3})
                d = assignments[0][2]
                self.assertEqual(d["oracle_status"], "partial")
                self.assertEqual(d["oracle_selected_object"], "resolved")
                self.assertEqual([o["status"] for o in d["oracle_objects"]],
                                 ["unsupported", "available", "error"])
                self.assertEqual([o["admissible"] for o in d["oracle_objects"]], [False, True, False])
                self.assertEqual(d["oracle_objects"][0]["reason"], unsupported["reason"])
                self.assertEqual(d["oracle_objects"][2]["reason"], failed["reason"])
                json.dumps(d, allow_nan=False)
        self.assertEqual(payload, original)

    def test_oracle_unavailable_payload_never_calls_even_with_resolved_object(self):
        obj = dict(object_id="resolved", status="available", distance=.01, in_window=True)
        for tight in (False, True):
            m = OracleGraspCalls(tight=tight)
            for status in ("unsupported", "error"):
                with self.subTest(tight=tight, status=status):
                    q = SimpleNamespace(oracle=dict(status=status, objects=[obj]))
                    p, u, d = m._assignment(q)
                    self.assertEqual((p, u), (0, 0))
                    self.assertEqual(d["oracle_status"], status)
                    self.assertEqual(d["override"], "missing_or_unavailable_oracle")
                    self.assertEqual(m._oracle_counts, {})

    def test_oracle_partial_without_usable_truth_reports_each_object(self):
        objects = [dict(object_id="unknown", status="unsupported", reason="body unresolved"),
                   dict(object_id="failed", status="error", reason="predicate unavailable"),
                   dict(object_id="bad", status="available", distance=float("nan"), in_window=True),
                   dict(object_id="missing", status="available", in_window=True), None]
        m = OracleGraspCalls(tight=True)
        p, _u, d = m._assignment(SimpleNamespace(oracle=dict(status="partial", objects=objects)))
        self.assertEqual(p, 0)
        self.assertEqual(d["oracle_status"], "partial")
        self.assertEqual([o["status"] for o in d["oracle_objects"]],
                         ["unsupported", "error", "error", "error", "error"])
        self.assertTrue(all(o["reason"] and not o["admissible"] for o in d["oracle_objects"]))
        self.assertEqual(m._oracle_counts, {})
        json.dumps(d, allow_nan=False)
        self.assertEqual(m._assignment(SimpleNamespace(oracle=dict(status="partial", objects=[])))[0], 0)

    def test_oracle_partial_query_diagnostics_are_detached_and_payload_is_consumed(self):
        base, lib, _table, _kwargs = fixture()
        m = OracleGraspCalls(tight=True)
        m.base = base
        ep = SimpleNamespace(uid="partial", task_id=0, init=4)
        m.reset(ep)
        q = qview(lib, 0, 0, ep, dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0))
        payload = dict(status="partial", objects=[
            dict(object_id="resolved", status="available", distance=.04, in_window=True),
            dict(object_id="unresolved", status="unsupported", reason="body unresolved")])
        m.set_oracle(payload)
        payload["objects"][0]["status"] = "error"  # setter owns a detached payload
        result = m.query(q)
        self.assertEqual(result.extras["os_force_miss"], 1)
        state = pickle.dumps(vars(m), protocol=4)
        diag = m.debug_record()
        self.assertEqual(diag["oracle_status"], "partial")
        self.assertEqual(diag["oracle_objects"][1]["status"], "unsupported")
        diag["oracle_objects"][1]["status"] = "available"
        self.assertEqual(pickle.dumps(vars(m), protocol=4), state)
        self.assertEqual(m.debug_record()["oracle_objects"][1]["status"], "unsupported")
        self.assertEqual(m._assignment(q)[0], 0)
        self.assertEqual(m._oracle_counts, {"resolved": 1})

    def test_oracle_accepts_client_adapter_object_fields_in_partial_payload(self):
        from exp.offline_search.debug.client.adapter import MujocoAdapter
        from exp.offline_search.debug.tests.test_client_capture import FakeEnv

        env = FakeEnv()
        env.goal_state.append(["on", "unresolved", "target"])
        adapter = MujocoAdapter(env)
        adapter.set_reference()
        obs = env.observe()
        obs["robot0_eef_pos"] = env.sim.data.body_xpos[2] + [.02, 0., 0.]
        payload = adapter.oracle(obs)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual([o["status"] for o in payload["objects"]], ["available", "unsupported"])
        m = OracleGraspCalls(tight=True)
        p, _u, d = m._assignment(SimpleNamespace(oracle=payload))
        self.assertEqual(p, 1)
        self.assertEqual(d["oracle_selected_object"], "mug")
        self.assertEqual(d["oracle_objects"][-1]["status"], "unsupported")
        self.assertEqual(m._oracle_counts, {"mug": 1})

    def test_wrist_ignores_stages_but_forced_looks_stay_full(self):
        base, _lib, _t, _kw = fixture()
        m = WristEveryLook()
        m.base = base
        base._anchor = dict(rows=np.array([15]), weights=np.array([1.]), action=base.act[15], rs=np.zeros(8),
                            step=0, last_step=0, task=0, episode="x")
        # No StageTable is needed for the ungated camera decision.
        m._plan(np.full(8, np.nan), 1, 2)
        self.assertEqual(m.next_camera_mode, "wrist_only")
        m.invalidate_anchor()
        self.assertEqual(m.next_camera_mode, "full")
        m.enabled = False
        m.finish_cadence()
        self.assertEqual(m.base.budget, 1)

    def test_disabled_wrist_identity(self):
        base, lib, _t, _kw = fixture()
        m = WristEveryLook(enabled=False)
        m.base, _ = clone_method(base, strict=True)
        m.finish_cadence()
        rows = np.tile(np.arange(16), 15)
        reference, _ = tape(base, lib, rows)
        actual, _ = tape(m, lib, rows, observe=True)
        for (alook, a), (blook, b) in zip(reference, actual):
            self.assertEqual(alook, blook)
            assert_result(self, a, b)

    def test_configuration_rejections(self):
        for ctor, kwargs in ((FollowLottery, dict(force_e=True)), (FollowLottery, dict(stage_gate=True)),
                             (EveryFiveAWM, dict(budget=1)), (WristEveryLook, dict(every_controls=7)),
                             (AnchorCalls, dict(p=float("nan"))), (OracleGraspCalls, dict(tight=1)),
                             (PolicyEveryTen, dict(p=.25)), (FollowLottery, dict(random_seed=-1))):
            with self.assertRaises(ValueError):
                ctor(**kwargs)


if __name__ == "__main__":
    unittest.main()
