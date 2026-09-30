"""Independent tests of the added weight/latch product, not R6 internals."""
import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from exp.offline_search.rounds.r06.ideation_Q1.method_c.budget import (
    replay_cadence, episode_expectation, solve,
)
from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_bridge import call_probability, scheduled_look
from .tilt import factors, lift_deviation_latch, stage_features
from .methods import CallController
from .composition import FollowCacheExtension


class FixtureTracker:
    def __init__(self, model, task):
        self.last = None
        self._window = deque(maxlen=2)

    def observe(self, key, control):
        self.last = key

    def status(self):
        return dict(state=self.last, W=1)


def fixture():
    states = ['slow_ambiguous', 'ok', 'ok', 'slow_ambiguous', 'slow_confirmed', 'ok', 'ok', 'ok']
    tree = replay_cadence(dict(uid='fixture', task=17, init=8, weight=1., block_controls=2,
        commit_controls=4, decisions=[dict(step=j, control_index=2*j, Ehat=.7, key=s)
                                     for j, s in enumerate(states)]), None, FixtureTracker)
    high = [True, False, True, None, None, True, None, False]
    features = [dict(event_mass=j % 3 / 2, h=.25, high=v, h_dev=.25)
                for j, v in enumerate(high)]
    return tree, features


def enumerate_paths(tree, features, parameter):
    def walk(i, latch, probability):
        if i < 0:
            return np.zeros(3)
        node = tree['nodes'][i]
        feature = features[node['j']]
        # Independently state the documented rule, without factors()/lifting.
        event = 1 + feature['event_mass'] * (1 / feature['h'] - 1)
        entry = feature['high'] is True and not latch
        weight = event * (1 / feature['h_dev'] if entry else 1)
        latch = latch if feature['high'] is None else bool(feature['high'])
        p = call_probability(node['state'], node['cooled'], min(1., parameter * weight))
        result = probability * np.array([1., p, (1 - p) * node['look_eligible']])
        for branch, mass in [('call', p), ('nocall', 1 - p)]:
            if mass:
                result += walk(node[branch], latch, probability * mass)
        return result
    return walk(0, False, 1.)


class Tests(unittest.TestCase):
    def test_exact_latch_product(self):
        tree, features = fixture()
        tilted = lift_deviation_latch(tree, features)
        self.assertGreater(len(tilted['nodes']), len(tree['nodes']))
        for parameter in np.linspace(0, 1, 21):
            np.testing.assert_allclose(episode_expectation(tilted, parameter, 'R'),
                enumerate_paths(tree, features, parameter), atol=2e-14, rtol=0)

    def test_disabled_is_original_object(self):
        tree, features = fixture()
        self.assertIs(lift_deviation_latch(tree, features, enabled=False), tree)

    def test_zero_occupancy_is_unit(self):
        self.assertEqual(factors(1., 0., True, 0., False), (1., True, True))

    def test_entry_latch(self):
        latch = False
        entries = []
        weights = []
        for high in [True, True, None, False, True, True]:
            weight, latch, entry = factors(0., .2, high, .25, latch)
            entries.append(entry); weights.append(weight)
        self.assertEqual(entries, [True, False, False, False, True, False])
        self.assertEqual(weights, [4., 1., 1., 1., 4., 1.])

    def test_soft_event_weight_and_overlap(self):
        self.assertEqual(factors(.5, .25, True, .25, False)[0], 10.)
        self.assertEqual(factors(0., .25, False, .25, False)[0], 1.)

    def test_unknown_deviation_retains_latch(self):
        self.assertEqual(factors(0., .25, None, .25, True), (1., True, False))

    def test_invalid_mass_rejected(self):
        for value in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                factors(value, .25, False, .25, False)

    def test_unknown_task_keeps_uniform(self):
        table = SimpleNamespace(online=lambda *_: dict(event_mass=0., unanimous=False, unknown_mass=1.),
            deviation=lambda *_: .4, event_occupancy={}, deviation_occupancy={}, deviation_p75={})
        feature = stage_features(table, [0], [1.], [8., 9.], 902)
        self.assertIsNone(feature['high'])
        self.assertEqual(factors(feature['event_mass'], feature['h'], feature['high'],
                                 feature['h_dev'], False)[0], 1.)

    def test_solve_exact_target_and_infeasible(self):
        tree, features = fixture()
        tree = lift_deviation_latch(tree, features)
        bounds = solve([tree], 0., 'R', .16)
        target = (bounds['floor'] + bounds['ceiling']) / 2
        solution = solve([tree], target, 'R', .16)
        self.assertTrue(solution['feasible'])
        self.assertLess(abs(solution['predicted_IR'] - target), 1e-14)
        self.assertFalse(solve([tree], bounds['ceiling'] + .1, 'R', .16)['feasible'])

    def test_ct_gates(self):
        with self.assertRaises(ValueError):
            CallController(.3, tilt=True, calibration_path='unused')
        with self.assertRaises(ValueError):
            CallController(.3, tilt=True, placement='R', stall_model_path='unused', calibration_path='unused')

    def test_composition_clears_plan_on_call(self):
        component = SimpleNamespace(extend_blocks=1, plan=lambda anchor: 'eligible')
        extension = FollowCacheExtension(component)
        controller = SimpleNamespace(base=SimpleNamespace(_anchor={}))
        result = SimpleNamespace(extras={})
        extension.plan = 'old'
        extension.on_anchor(controller, None, result, call=True)
        self.assertIsNone(extension.plan)
        self.assertEqual(result.extras, {})

    def test_composition_stall_look_priority(self):
        controller = CallController(.3, calibration_path='unused')
        controller._look_due_step = 8
        controller._last_log = {}
        controller.base = SimpleNamespace(last_blind_extras={})
        def must_not_extend(*args):
            self.fail('stall LOOK must bypass extension')
        controller.cache_extension = SimpleNamespace(blind_step=must_not_extend)
        result = controller.blind_step(SimpleNamespace(step=8))
        self.assertEqual(result.name, 'c_slow_ambiguous')
        self.assertIsNone(controller._look_due_step)

    def test_composition_reset_and_invalidation(self):
        extension = FollowCacheExtension(None)
        extension.plan = 'old'
        extension.reset(None)
        self.assertIsNone(extension.plan)
        extension.plan = 'new'
        extension.invalidate_anchor()
        self.assertIsNone(extension.plan)


if __name__ == '__main__':
    unittest.main()
