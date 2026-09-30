"""CPU component contracts, independent of simulator/model execution."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from exp.offline_search.rounds.r07.stages.stages import StageTable
from .methods import FollowExtension, StageFollow


def fixture(stride=5, horizon=10):
    n, width = 32, 4
    action = np.zeros((n, horizon, width), np.float32)
    for r in range(n):
        action[r, :, 0] = r * 100 + np.arange(horizon)
        action[r, :, 1] = -1 if r % 8 < 4 else 1
    episode = np.repeat(np.arange(4), 8)
    step = np.tile(np.arange(8), 4)
    nxt = np.arange(n) + 1; nxt[step == 7] = -1
    success = np.ones(n, bool); success[24:] = False
    rs = np.column_stack((step * .1 + episode * .01, step**2, np.ones(n)))
    lib = dict(action=action, rs=rs, episode=episode, step=step, next=nxt,
               task_id=np.zeros(n, int), success=success)
    manifest = dict(exec_steps=stride, H=horizon, act_valid_dims=2, rs_valid_dims=3, gripper_dim=1)
    task = SimpleNamespace(rows=np.arange(n), Z=rs[:, :2], Z0=None, A0=None, As0=None)
    base = SimpleNamespace(tasks={0: task}, k=4, kref=2, early=False)
    table = StageTable.fit(lib, manifest={**manifest, '_retrieval': base})
    rows = np.array([0, 8, 16, 8], np.int64)
    weights = np.array([.1, .2, .3, .4], np.float32)
    anchor = dict(rows=rows, weights=weights, action=np.tensordot(weights, action[rows], 1),
                  rs=weights @ rs[rows], step=0, last_step=0, task=0, episode='x')
    return lib, manifest, table, anchor


class FollowTests(unittest.TestCase):
    def test_disabled_fit_no_new_library_dependency(self):
        method = StageFollow(lib='current', kref=5, extend_blocks=0, stages_path='/missing/stage.pkl')
        method.finish_follow_fit(None, None)
        self.assertIsNone(method.follow_component)

    def test_segmentation_matches_e1(self):
        from exp.offline_search.rounds.r07.ideation.E1_kinematic_stages.analyze_stages import stable_modes, twomeans
        lib, _manifest, table, _a = fixture()
        cmd = np.median(lib['action'][:, :table.exec_steps, 1], axis=1)
        np.testing.assert_array_equal(table.gripper_centers, twomeans(cmd[lib['success']]))
        for ep in range(3):
            rows = np.flatnonzero(lib['episode'] == ep)
            np.testing.assert_array_equal(table.mode[rows], stable_modes(cmd[rows], table.gripper_threshold))
        np.testing.assert_array_equal(table.mode[24:], -1)
        self.assertEqual(table.rows_to_event[3], 1)
        self.assertEqual(table.rows_to_event[4], 4)
        self.assertTrue(table.event_near[3:6].all())

    def test_three_row_isolated_reversal(self):
        lib, manifest, _table, _a = fixture()
        lib['action'][1, :, 1] = 1
        task = SimpleNamespace(rows=np.arange(32), Z=lib['rs'][:, :2], Z0=None, A0=None, As0=None)
        table = StageTable.fit(lib, manifest={**manifest, '_retrieval': SimpleNamespace(tasks={0:task},k=4,kref=2,early=False)})
        self.assertEqual(table.mode[1], 0)

    def test_save_content_and_library_fingerprints(self):
        lib, manifest, table, _a = fixture()
        path = Path('/tmp/r7_C1/test_stage.pkl'); table.save(path)
        loaded = StageTable.load(path, library=lib, manifest=manifest)
        self.assertEqual(loaded.fingerprint, table.fingerprint)
        self.assertFalse(loaded.mode.flags.writeable)
        bad = dict(lib, rs=lib['rs'] + .01)
        with self.assertRaises(ValueError):
            StageTable.load(path, library=bad, manifest=manifest)
        data = bytearray(path.read_bytes()); data[-20] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValueError):
            StageTable.load(path)

    def test_unknown_even_zero_weight_blocks_unanimity(self):
        _lib, _m, t, a = fixture()
        info = t.online(np.r_[a['rows'], 24], np.r_[a['weights'], 0])
        self.assertFalse(info['unanimous'])
        self.assertEqual(info['unknown_mass'], 0)

    def test_pi_successor_head_original_weights(self):
        lib, _m, t, a = fixture()
        c = FollowExtension(t, stage_gate=False, state_valve=False)
        plan = c.plan(a); self.assertEqual(plan.blocks, 1)
        result = c.serve(a, 2, lib['action'], 'fake', {})
        expected = np.tensordot(a['weights'], lib['action'][t.advance(a['rows'], 2), :5, :2], 1)
        self.assertEqual(result.action[:5, :2].tobytes(), expected.tobytes())
        self.assertEqual(result.weights.tobytes(), a['weights'].tobytes())
        np.testing.assert_array_equal(result.rows, a['rows'] + 2)

    def test_native_third_block_then_successor_head(self):
        lib, _m, t, a = fixture(horizon=16)
        c = FollowExtension(t, extend_blocks=2, stage_gate=False, state_valve=False)
        self.assertEqual(c.plan(a).blocks, 2)
        native = c.serve(a, 2, lib['action'], 'fake', {})
        self.assertEqual(native.action[:5, :2].tobytes(), a['action'][10:15, :2].tobytes())
        successor = c.serve(a, 3, lib['action'], 'fake', {})
        expected = np.tensordot(a['weights'], lib['action'][a['rows'] + 3, :5, :2], 1)
        self.assertEqual(successor.action[:5, :2].tobytes(), expected.tobytes())

    def test_terminal_never_clamped_even_zero_weight(self):
        _lib, _m, t, a = fixture()
        a['rows'][-1] = 7; a['weights'][-1] = 0
        c = FollowExtension(t, stage_gate=False, state_valve=False)
        self.assertEqual(c.plan(a).blocks, 0)
        self.assertEqual(t.advance([7], 2)[0], -1)
        with self.assertRaises(ValueError):
            c.serve(a, 2, _lib['action'], 'fake', {})

    def test_stage_event_veto_and_no_cap_degradation(self):
        _lib, _m, t, a = fixture()
        a['rows'] += 1
        self.assertEqual(FollowExtension(t, extend_blocks=1).plan(a).blocks, 1)
        self.assertEqual(FollowExtension(t, extend_blocks=2).plan(a).blocks, 0)

    def test_valve_current_state_at_every_blind_age(self):
        _lib, _m, t, a = fixture()
        c = FollowExtension(t, extend_blocks=2, stage_gate=False)
        p = c.plan(a)
        for age in (1, 2, 3):
            now = a['rs'] + a['weights'] @ (t.rs[t.advance(a['rows'], age)] - t.rs[a['rows']])
            reason, ex = c.check(a, SimpleNamespace(rs=now), age, p)
            self.assertIsNone(reason); self.assertEqual(ex['os_sf_valve_checked'], 1)
            reason, ex = c.check(a, SimpleNamespace(rs=now + 100*t.state_scale), age, p)
            self.assertEqual(reason.name, 'follow_state_valve'); self.assertEqual(ex['os_sf_valve_fire'], 1)

    def test_manifest_geometry_not_five_or_width(self):
        lib, _m, t, a = fixture(stride=3, horizon=6)
        result = FollowExtension(t, stage_gate=False, state_valve=False).serve(a, 2, lib['action'], 'fake', {})
        self.assertEqual(result.action.shape, (6, 4))
        np.testing.assert_array_equal(result.action[:3, :2], np.tensordot(a['weights'], lib['action'][a['rows']+2, :3, :2], 1))

    def test_cross_episode_or_nonconsecutive_successor_invalid(self):
        lib, manifest, _t, _a = fixture()
        lib['next'][0] = 8; lib['next'][1] = 3
        task = SimpleNamespace(rows=np.arange(32), Z=lib['rs'][:, :2], Z0=None, A0=None, As0=None)
        t = StageTable.fit(lib, manifest={**manifest, '_retrieval':SimpleNamespace(tasks={0:task},k=4,kref=2,early=False)})
        self.assertEqual(t.next[0], -1); self.assertEqual(t.next[1], -1)

    def test_episode_radius_and_task_deviation_are_finite(self):
        _lib, _m, t, _a = fixture()
        self.assertTrue(np.isfinite(t.valve_radius_row))
        self.assertTrue(np.isfinite(t.valve_radius_episode))
        self.assertTrue(np.isfinite(t.deviation_p75[0]))
        self.assertFalse(t.state_active[-1])


if __name__ == '__main__':
    unittest.main()
