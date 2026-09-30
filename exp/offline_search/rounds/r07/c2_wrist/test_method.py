"""Portable hard/unknown gating and owner-cost completeness checks."""
import json
import pickle
import types
import unittest
from pathlib import Path

import numpy as np
from .cost import ledger
from .method import StageWrist
from exp.offline_search.rounds.r07.stages.stages import StageTable


class MethodTests(unittest.TestCase):
    def setUp(self):
        # A different gripper channel, state width, action width and command
        # polarity exercise manifest geometry, not LIBERO numeric conventions.
        self.method = StageWrist()
        self.method.base = types.SimpleNamespace(_anchor=dict(rows=np.array([0, 5]), weights=np.array([1., 0.], np.float32),
            action=np.full((12, 4), 9., np.float32), rs=np.zeros(3), step=0))
        self.method.block_controls, self.method.state_width, self.method.gripper_dim = 3, 3, 2
        t = self.method.stages = StageTable()
        t.mode = np.ones(10, np.int8)
        t.event_near = np.zeros(10, bool)
        t.rows_to_event = np.tile(np.arange(5, 0, -1), 2)
        t.next = np.array([1, 2, 3, 4, -1, 6, 7, 8, 9, -1])
        t.rs = np.zeros((10, 3))
        t.state_scale, t.state_active = np.ones(3), np.ones(3, bool)
        t.valve_radius, t.gripper_threshold = .3, 4.

    def test_easy_next_anchor(self):
        self.method._plan(np.zeros(3), 1, 2)
        self.assertEqual(self.method.next_camera_mode, "wrist_only")

    def test_zero_weight_unknown_is_hard(self):
        self.method.stages.mode[5] = -1
        self.method._plan(np.zeros(3), 1, 2)
        self.assertEqual(self.method.next_camera_mode, "full")

    def test_terminal_member_is_never_clamped(self):
        self.method.base._anchor["rows"][1] = 9
        self.method._plan(np.zeros(3), 1, 2)
        self.assertEqual(self.method.next_camera_mode, "full")

    def test_transition_and_wrong_command_mode_are_hard(self):
        self.method.stages.event_near[2] = True
        self.method._plan(np.zeros(3), 1, 2)
        self.assertEqual(self.method.next_camera_mode, "full")
        self.method.stages.event_near[2] = False
        self.method.base._anchor["action"].fill(-2)
        self.method._plan(np.zeros(3), 1, 2)
        self.assertEqual(self.method.next_camera_mode, "full")

    def test_state_deviation_and_nan_are_hard(self):
        for state in (np.ones(3), np.full(3, np.nan)):
            self.method._plan(state, 1, 2)
            self.assertEqual(self.method.next_camera_mode, "full")

    def test_actual_ledger_all_five_cases(self):
        rows = []
        for mode, hit in (("full", True), ("full", False), ("wrist_only", True), ("wrist_only", False), ("blind", True)):
            vision = mode != "blind"
            nc = int(mode == "wrist_only" and not hit)
            cost = (.055198 if mode == "wrist_only" else .152 if vision else 0) + .049890*nc + .848*(not hit)
            rows.append(dict(camera_mode=mode, hit=hit, vision=vision, camera_completion_calls=nc,
                             camera_stage1_calls=int(vision), owner_cost=cost))
        result = ledger(rows)
        self.assertAlmostEqual(result["total_cost"], 2.160286)
        self.assertEqual(result["completion_calls"], 1)
        self.assertEqual(result["decisions"], 5)
        rows[3]["camera_completion_calls"] = 0
        with self.assertRaises(ValueError):
            ledger(rows)

    def test_fits_have_independent_72_dimensional_metrics(self):
        for cell in ("l10_50", "l10_500", "sp_50", "sp_500"):
            with Path(f"/tmp/r7_C2/fits/r7_sw_pi05_{cell}.pkl").open("rb") as f:
                method = pickle.load(f)["method"]
            self.assertEqual(method.wrist.B1T.shape[0], 64)
            for task in method.wrist.tasks:
                self.assertEqual(method.wrist.tasks[task].Wf.shape, (72, 72))
                self.assertEqual(method.base.tasks[task].Wf.shape, (136, 136))
                self.assertFalse(np.shares_memory(method.wrist.tasks[task].Wf, method.base.tasks[task].Wf))


if __name__ == "__main__":
    unittest.main()
