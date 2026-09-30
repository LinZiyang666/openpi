"""Permanent tests for the consumer's non-projecting metric-code capture."""
from types import MethodType, SimpleNamespace as NS
import unittest

import numpy as np

from exp.offline_search.rounds.r06.ideation_Q1.method_c.methods import CalibratedRescue
from .test_stall_fast import assert_identical


class SmallBase:
    def __init__(self, early, stored):
        self.early = early
        self.tasks = {0: NS(Z=np.arange(12., dtype=np.float32).reshape(4, 3),
            Z0=np.arange(12., dtype=np.float32).reshape(4, 3) if stored else None,
            A0=np.eye(3, dtype=np.float32), As0=np.eye(3, dtype=np.float32))}

    def _dist(self, q):
        t = self.tasks[int(q.task_id)]
        code = np.asarray(q.code, np.float32) - np.float32(.25)
        if self.early and q.step == 0:
            cross = t.Z0 @ code if t.Z0 is not None else t.Z @ (t.A0 @ code)
            if t.Z0 is None: cross = cross + t.Z @ (t.As0 @ code)
        else: cross = t.Z @ code
        return cross

    def query(self, q): return self._dist(q)


class ControllerCaptureTests(unittest.TestCase):
    def controller(self, base):
        c = CalibratedRescue(.3, calibration_path='unused')
        c.base = base
        return c

    def test_exact_main_early_and_unstored_early(self):
        for early in (False, True):
            for stored in (False, True):
                for step in (0, 1, 7):
                    base = SmallBase(early, stored); c = self.controller(base)
                    q = NS(task_id=0, step=step, code=np.array([.125, .5, -2.], np.float32))
                    tasks, table = base.tasks, base.tasks[0]
                    expected = base.query(q); got, key = c._query_with_metric_code(q)
                    assert_identical(got, expected)
                    assert_identical(key['metric_code'], q.code-np.float32(.25))
                    self.assertEqual(key['metric'], 'early' if early and step == 0 else 'main')
                    self.assertIs(base.tasks, tasks); self.assertIs(base.tasks[0], table)
                    self.assertNotIn('_dist', vars(base))
                    self.assertIs(type(table.Z), np.ndarray)

    def test_instance_override_and_exception_restoration(self):
        base = SmallBase(True, False); c = self.controller(base)
        bound = MethodType(SmallBase._dist, base); base._dist = bound
        c._query_with_metric_code(NS(task_id=0, step=0, code=[1, 2, 3]))
        self.assertIs(base._dist, bound)
        def fail(self, q): raise ValueError('bad observation')
        base._dist = MethodType(fail, base); original = base._dist
        with self.assertRaisesRegex(ValueError, 'bad observation'):
            c._query_with_metric_code(NS(task_id=0, step=1))
        self.assertIs(base._dist, original)


if __name__ == '__main__': unittest.main()
