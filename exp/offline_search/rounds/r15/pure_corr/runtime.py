"""A's exact cache with the frozen GC_dist correction, without a judge.

The optional guard mode exists for CPU equivalence proof only. Production artifacts
embed everything and require no fitting, calibration, head loading, or test data.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult
from exp.offline_search.rounds.r10.method import SizeController
from exp.offline_search.rounds.r10.astra.method import DistanceCorrectedBase, multiplier


class LoggedDistanceBase(DistanceCorrectedBase):
    """Use the deployed correction arithmetic unchanged; record scalar telemetry."""

    def _correction(self, q, action):
        corr = super()._correction(q, action)
        strength = 0.0
        if int(q.step) > 0 and self._anchor is not None:
            task, *parts = self._dist(q)
            d = parts[6]
            pos = np.searchsorted(task.rows, self._anchor['rows'])
            strength = self.blend * multiplier(float(np.min(d[pos])) / self.distance_scale,
                                                q.step, self.distance_rule)
        applied = int(strength > 0 and np.any(corr[:, :6] != 0))
        self.pc_count = getattr(self, 'pc_count', 0) + applied
        self.pc_log = dict(os_pc_enabled=1.0, os_pc_applied=float(applied),
                           os_pc_inherited=0.0, os_pc_strength=float(strength),
                           os_pc_count=float(self.pc_count),
                           os_pc_rms=float(np.sqrt(np.mean((self.blend * corr[:, :6]) ** 2))))
        return corr

    def reset(self, episode):
        super().reset(episode)
        self.pc_count = 0
        self.pc_log = dict(os_pc_enabled=1.0, os_pc_applied=0.0, os_pc_inherited=0.0,
                           os_pc_strength=0.0, os_pc_count=0.0, os_pc_rms=0.0)


class PureCorrector(SizeController):
    family = 'r15_pure_corr'

    def __init__(self, size=500, corrector_enabled=True, guard_enabled=False):
        if type(corrector_enabled) is not bool or type(guard_enabled) is not bool:
            raise ValueError('enable switches must be booleans')
        super().__init__(size=size, variant='A')
        self.corrector_enabled = corrector_enabled
        self.guard_enabled = guard_enabled
        self.uses_nonlibrary_action = corrector_enabled

    def fit(self, lib, ctx):
        raise RuntimeError('PureCorrector requires its frozen assembled artifact; refitting is forbidden')

    @classmethod
    def assemble(cls, a, gc, *, corrector_enabled=True, guard_enabled=False):
        if a.variant != 'A' or gc.variant != 'GC_dist' or a.size != gc.size:
            raise ValueError('need matching frozen R10 A and GC_dist')
        if not np.array_equal(a.row_subset, gc.row_subset):
            raise ValueError('different library rows')
        if gc.inner.base.distance_rule != dict(peak=.5, plateau=.75, cutoff=2.0):
            raise ValueError('unexpected frozen GC_dist strength')
        if not all(x.fit_info.get('library_only') for x in (a, gc)):
            raise ValueError('library-only provenance required')
        result = cls(a.size, corrector_enabled, guard_enabled)
        result.row_subset, result.parent_name = a.row_subset, a.parent_name
        result.fit_info = dict(a.fit_info, refitted=False, pure_corr=True,
                               corrector_enabled=corrector_enabled, guard_enabled=guard_enabled)
        if guard_enabled:
            if not corrector_enabled:
                raise ValueError('proof guard mode requires the frozen corrector')
            result.inner = gc.inner
            base = result.inner.base
        else:
            result.inner = a.inner
            base = result.inner
            if corrector_enabled:
                frozen = gc.inner.base
                for key in ('heads', 'head_meta', 'sig_head', 'chans', 'blend', 'correct_gripper',
                            'distance_scale', 'distance_rule'):
                    setattr(base, key, getattr(frozen, key))
        if corrector_enabled:
            base.__class__ = LoggedDistanceBase
            base.invalidate_anchor()
        result.name = f'R15_AC_dist_{a.size}'
        return result

    def _log(self, *, blind=False):
        if not self.corrector_enabled or self.guard_enabled:
            return {}
        base = self.inner.base if self.guard_enabled else self.inner
        ex = dict(getattr(base, 'pc_log', {}))
        if blind:
            ex['os_pc_inherited'] = ex.get('os_pc_applied', 0.0)
            ex['os_pc_applied'] = 0.0
        return ex

    def query(self, q):
        r = super().query(q)
        r.extras = {**self._log(), **r.extras}
        return r

    def blind_step(self, q):
        r = super().blind_step(q)
        if isinstance(r, BlindResult) and self.corrector_enabled:
            return BlindResult(r.action, r.rows, r.weights, r.library, {**self._log(blind=True), **r.extras})
        return r
